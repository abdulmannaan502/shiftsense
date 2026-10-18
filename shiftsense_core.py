"""ShiftSense core: sensor anomaly scoring, manual retrieval, transcription and briefing generation.

Model calls go to OpenAI-compatible endpoints. In the hackathon setup these are vLLM
servers on AMD Developer Cloud (AMD Instinct GPU, ROCm) serving Gemma and Whisper.
If no endpoint is configured, a deterministic rule-based fallback keeps the demo usable.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"

MACHINES = {
    "CNC-07": {"aliases": ["cnc-07", "cnc 07", "cnc7", "cnc-7", "spindle", "cnc"], "type": "CNC machining centre"},
    "Press-02": {"aliases": ["press-02", "press 02", "press 2", "press", "hydraulic"], "type": "Hydraulic press"},
    "Conveyor-C": {"aliases": ["conveyor-c", "conveyor c", "conveyor", "belt"], "type": "Belt conveyor"},
    "Robot-R4": {"aliases": ["robot-r4", "robot r4", "r4", "robot"], "type": "6-axis robot"},
}
SENSORS = ["vibration_mm_s", "temp_c", "current_pct"]
BASE = {  # normal operating mean, std
    "CNC-07": {"vibration_mm_s": (2.2, 0.25), "temp_c": (52, 1.5), "current_pct": (68, 3)},
    "Press-02": {"vibration_mm_s": (1.4, 0.2), "temp_c": (48, 1.5), "current_pct": (60, 3)},
    "Conveyor-C": {"vibration_mm_s": (1.0, 0.15), "temp_c": (38, 1.0), "current_pct": (72, 3)},
    "Robot-R4": {"vibration_mm_s": (0.8, 0.1), "temp_c": (41, 1.0), "current_pct": (55, 3)},
}
SAFETY_TERMS = ["lockout", "guard", "interlock", "pressure", "leak", "smoke", "fire", "injury", "bypass", "seal"]

SCENARIOS = {
    "Line 3 · Night → Morning": {
        "seed": 7,
        "technician": "R. Iyer",
        "faults": [
            ("CNC-07", "vibration_mm_s", 240, 2.6, "drift"),   # bearing wear, starts ~02:00
            ("CNC-07", "temp_c", 300, 14, "drift"),
            ("Press-02", "temp_c", 180, 11, "step"),           # hydraulic temp, partly fixed
        ],
        "start": "22:00",
        "note": (
            "Night shift handover, line three. CNC-07 started making a grinding noise around four thirty, "
            "I think it's the spindle bearing, it got louder before the end of the shift, we did not stop it. "
            "Press-02 hydraulic oil was running hot, I topped up the oil and it came down a bit, but the seal "
            "replacement is still pending, the seal kit is on order. Conveyor C belt was slipping, I re-tensioned "
            "it and closed work order WO-1182. Robot R4 was fine, no alarms."
        ),
    },
    "Line 1 · Day → Evening": {
        "seed": 21,
        "technician": "M. Rossi",
        "faults": [
            ("Robot-R4", "current_pct", 330, 38, "spikes"),     # joint motor spikes
            ("Conveyor-C", "current_pct", 120, 30, "step"),
        ],
        "start": "06:00",
        "note": (
            "Day shift handover. Robot R4 stopped twice with a current alarm on joint three, no collision in the log, "
            "we restarted it, please keep an eye on it. Conveyor C had a jam at the transfer roller around eight, "
            "cleared it, the motor current was high for a while, it is running now but I did not open a work order. "
            "CNC-07 completed the planned grease change, all good. Press-02 normal."
        ),
    },
}


# ----------------------------------------------------------------------------- sensors
def _normal(rng, machine, n):
    return {s: rng.normal(m, sd, n) for s, (m, sd) in BASE[machine].items()}


def generate_shift(scenario: str, minutes: int = 480) -> pd.DataFrame:
    """Simulated per-minute historian export for one 8-hour shift."""
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(cfg["seed"])
    start = pd.Timestamp(f"2026-10-17 {cfg['start']}")
    t = pd.date_range(start, periods=minutes, freq="1min")
    rows = []
    for machine in MACHINES:
        vals = _normal(rng, machine, minutes)
        for (m, sensor, at, mag, kind) in cfg["faults"]:
            if m != machine:
                continue
            idx = np.arange(minutes)
            if kind == "drift":
                vals[sensor] += np.clip((idx - at) / (minutes - at), 0, None) * mag
            elif kind == "step":
                bump = np.where((idx >= at) & (idx < at + 150), mag, np.where(idx >= at + 150, mag * 0.35, 0))
                vals[sensor] += bump
            elif kind == "spikes":
                for k in rng.choice(np.arange(at, minutes), 9, replace=False):
                    vals[sensor][k:k + 2] += mag
        rows.append(pd.DataFrame({"time": t, "machine": machine, **vals}))
    return pd.concat(rows, ignore_index=True)


@dataclass
class MachineRisk:
    machine: str
    level: str               # HIGH / WATCH / OK
    score: float             # 0..1
    sensor: str | None
    since: str | None
    evidence: str


def score_machines(df: pd.DataFrame) -> list[MachineRisk]:
    """IsolationForest per machine, trained on a clean reference shift; scores the actual shift."""
    out = []
    rng = np.random.default_rng(999)
    for machine, g in df.groupby("machine", sort=False):
        ref = pd.DataFrame(_normal(rng, machine, 2000))
        model = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(ref[SENSORS])
        flags = model.predict(g[SENSORS]) == -1
        last2h = flags[-120:].mean()
        whole = flags.mean()
        score = float(min(1.0, 0.65 * last2h * 2.5 + 0.35 * whole * 3))
        # which sensor deviates most (z-score vs reference)
        z = {s: abs((g[s].iloc[-60:].mean() - BASE[machine][s][0]) / BASE[machine][s][1]) for s in SENSORS}
        zmax = {s: float(((g[s] - BASE[machine][s][0]).abs() / BASE[machine][s][1]).max()) for s in SENSORS}
        top = max(SENSORS, key=lambda s: z[s] + 0.2 * zmax[s])
        since = None
        if flags.any():
            first = np.argmax(flags)
            since = g["time"].iloc[first].strftime("%H:%M")
        if score >= 0.45:
            level = "HIGH"
        elif score >= 0.12 or flags.sum() >= 15:
            level = "WATCH"
        else:
            level = "OK"
        unit = {"vibration_mm_s": "mm/s", "temp_c": "°C", "current_pct": "% rated"}[top]
        evidence = (
            f"{top.split('_')[0]} last hour avg {g[top].iloc[-60:].mean():.1f} {unit} "
            f"(normal {BASE[machine][top][0]} {unit}); {int(flags.sum())} anomalous minutes"
            + (f", first at {since}" if since else "")
        ) if level != "OK" else "All sensors within normal range."
        out.append(MachineRisk(machine, level, round(score, 2), top if level != "OK" else None, since, evidence))
    return out


# ----------------------------------------------------------------------------- manual retrieval
class Manual:
    def __init__(self, path: Path = DATA_DIR / "manual.md"):
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"^## ", text, flags=re.M)[1:]
        self.sections = []
        for p in parts:
            title, _, body = p.partition("\n")
            page = re.search(r"\(p\.(\d+)\)", title)
            self.sections.append({"title": title.strip(), "ref": title.split(" ")[0], "page": page.group(1) if page else "?", "body": body.strip()})
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([s["title"] + " " + s["body"] for s in self.sections])

    def search(self, query: str, k: int = 1):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        return [dict(self.sections[i], score=float(sims[i])) for i in sims.argsort()[::-1][:k]]


# ----------------------------------------------------------------------------- model endpoints
def llm_config():
    return {
        "base_url": os.getenv("AMD_LLM_BASE_URL", "").strip(),
        "api_key": os.getenv("AMD_LLM_API_KEY", "EMPTY").strip() or "EMPTY",
        "model": os.getenv("AMD_LLM_MODEL", "google/gemma-3-12b-it").strip(),
        "whisper_base_url": os.getenv("AMD_WHISPER_BASE_URL", "").strip(),
        "whisper_model": os.getenv("AMD_WHISPER_MODEL", "openai/whisper-large-v3").strip(),
    }


def transcribe(audio_bytes: bytes, filename: str) -> str:
    """Whisper served on AMD via vLLM's OpenAI-compatible /v1/audio/transcriptions."""
    cfg = llm_config()
    if not cfg["whisper_base_url"]:
        raise RuntimeError("AMD_WHISPER_BASE_URL is not set")
    from openai import OpenAI
    client = OpenAI(base_url=cfg["whisper_base_url"], api_key=cfg["api_key"])
    r = client.audio.transcriptions.create(model=cfg["whisper_model"], file=(filename, audio_bytes))
    return r.text


SYSTEM = """You are ShiftSense, a maintenance shift-handover assistant in a factory.
You write a briefing for the INCOMING shift from (1) the outgoing technician's voice note,
(2) anomaly scores computed from sensor data, (3) retrieved maintenance-manual sections.
Rules:
- Use only the given evidence. Never invent readings, parts or work orders.
- Do not decide on safety: anything involving lockout, guarding, interlocks, pressure systems or leaks must have escalate=true.
- If evidence conflicts or is weak, set confidence low and say what to check.
- The note may be in any language; always answer in English.
Return ONLY JSON: {"items":[{"machine":str,"status":"HIGH"|"WATCH"|"DONE"|"OK","title":str,
"summary":str,"next_action":str,"manual_ref":str,"evidence":[str],"confidence":float,"escalate":bool,
"escalation_reason":str}],"open_tasks":[str],"headline":str}"""


def _llm_briefing(note, risks, manual_hits):
    cfg = llm_config()
    from openai import OpenAI
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=90)
    payload = {
        "voice_note_transcript": note,
        "sensor_anomaly_scores": [asdict(r) for r in risks],
        "manual_sections": {m: [{"ref": h["ref"], "title": h["title"], "text": h["body"]} for h in hits] for m, hits in manual_hits.items()},
    }
    r = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.1, max_tokens=1800,
    )
    txt = r.choices[0].message.content
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))


def _sentences_for(machine, note):
    al = MACHINES[machine]["aliases"]
    sents = re.split(r"(?<=[.!?])\s+", note)
    return [s for s in sents if any(a in s.lower() for a in al)]


def _rule_briefing(note, risks, manual_hits):
    """Deterministic fallback used when no AMD endpoint is configured."""
    items, open_tasks = [], []
    for r in risks:
        said = " ".join(_sentences_for(r.machine, note))
        low = said.lower()
        done = any(w in low for w in ["closed", "completed", "fixed", "re-tension", "replaced", "cleared"])
        pending = any(w in low for w in ["pending", "on order", "keep an eye", "did not", "not open", "still"])
        hit = manual_hits[r.machine][0]
        status = r.level if r.level != "OK" else ("DONE" if done else "OK")
        if r.level == "OK" and pending:
            status = "WATCH"
        agree = (r.level != "OK") == bool(re.search(r"noise|hot|alarm|slip|jam|stopp|grind|high", low))
        conf = round(0.55 + 0.3 * agree + 0.15 * (r.score if r.level != "OK" else 1 - r.score), 2)
        safety = [t for t in SAFETY_TERMS if t in (low + " " + hit["body"].lower())] if status in ("HIGH", "WATCH") else []
        escalate = bool(safety) or (status == "HIGH") or conf < 0.6
        verbs = ("Stop", "Inspect", "Check", "Re-tension", "Report", "Measure", "Apply", "Do not", "Never", "Clear", "Schedule")
        acts = [x.strip() for x in re.split(r"(?<=\.)\s+", hit["body"]) if x.strip().startswith(verbs) or re.search(r", (check|inspect|report|stop|plan)\b", x)]
        action = " ".join(acts[:2]) if status in ("HIGH", "WATCH") and acts else ("Follow up per manual." if status in ("HIGH", "WATCH") else "No action needed.")
        items.append({
            "machine": r.machine, "status": status,
            "title": f"{r.machine} — {MACHINES[r.machine]['type']}",
            "summary": (said or "Not mentioned in the voice note.") ,
            "next_action": action.rstrip(".") + ".",
            "manual_ref": f"Manual {hit['ref']} p.{hit['page']}",
            "evidence": [r.evidence] + ([f'Voice note: "{said[:140]}"'] if said else []),
            "confidence": min(conf, 0.99),
            "escalate": escalate,
            "escalation_reason": ("Safety-related: " + ", ".join(sorted(set(safety)))) if safety else ("High risk — supervisor sign-off before production" if status == "HIGH" else ("Low confidence" if conf < 0.6 else "")),
        })
        if pending and said:
            open_tasks.append(f"{r.machine}: {said}")
    order = {"HIGH": 0, "WATCH": 1, "DONE": 2, "OK": 3}
    items.sort(key=lambda i: order[i["status"]])
    n_high = sum(i["status"] == "HIGH" for i in items)
    return {"items": items, "open_tasks": open_tasks,
            "headline": f"{n_high} machine(s) need attention before the first batch." if n_high else "No high-risk machines; review watch items."}


def build_briefing(note: str, df: pd.DataFrame, manual: Manual):
    t0 = time.time()
    risks = score_machines(df)
    hits = {}
    for r in risks:
        words = {"vibration_mm_s": "vibration noise bearing", "temp_c": "temperature hot oil", "current_pct": "motor current overload jam"}.get(r.sensor, "")
        q = " ".join([r.machine, MACHINES[r.machine]["type"], words, " ".join(_sentences_for(r.machine, note))])
        hits[r.machine] = manual.search(q, k=1)
    cfg = llm_config()
    backend, error = "rule-based fallback (no AMD endpoint configured)", None
    if cfg["base_url"]:
        try:
            brief = _llm_briefing(note, risks, hits)
            backend = f"{cfg['model']} via vLLM on AMD ({cfg['base_url']})"
        except Exception as e:  # keep the floor tablet working even if the model is down
            error = f"LLM call failed, used fallback: {e}"
            brief = _rule_briefing(note, risks, hits)
    else:
        brief = _rule_briefing(note, risks, hits)
    # hard safety rule enforced outside the model
    for it in brief.get("items", []):
        blob = (it.get("summary", "") + " " + it.get("next_action", "")).lower()
        if any(t in blob for t in SAFETY_TERMS) and not it.get("escalate"):
            it["escalate"], it["escalation_reason"] = True, "Safety keyword detected (enforced rule)"
    return {"brief": brief, "risks": risks, "hits": hits, "backend": backend, "error": error, "seconds": round(time.time() - t0, 2)}
