import os

import pandas as pd
import streamlit as st

# Streamlit Cloud: copy secrets into env so core.py can read them
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

from shiftsense.core import SCENARIOS, Manual, build_briefing, generate_shift, llm_config, transcribe
from shiftsense import evolus

st.set_page_config(page_title="ShiftSense", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
.item{border-radius:12px;padding:16px 18px;margin-bottom:12px;border:1px solid #2a2f37}
.HIGH{background:rgba(255,77,77,.09);border-color:rgba(255,77,77,.45)}
.WATCH{background:rgba(255,176,32,.08);border-color:rgba(255,176,32,.4)}
.DONE{background:rgba(61,220,151,.06);border-color:rgba(61,220,151,.35)}
.tag{font-family:monospace;font-weight:700;font-size:13px;letter-spacing:.06em}
.tHIGH{color:#ff4d4d}.tWATCH{color:#ffb020}.tDONE{color:#3ddc97}.tOK{color:#8b939e}
.small{color:#8b939e;font-size:14px}
.esc{color:#ff4d4d;font-size:14px;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_manual():
    return Manual()


cfg = llm_config()
with st.sidebar:
    st.markdown("### ⚙️ Model backend")
    if cfg["base_url"]:
        st.success(f"Gemma on AMD connected\n\n`{cfg['model']}`")
    else:
        st.warning("No AMD endpoint set — using the rule-based fallback. Set `AMD_LLM_BASE_URL` to use Gemma on AMD Instinct.")
    st.caption("Whisper (AMD): " + ("connected" if cfg["whisper_base_url"] else "not set — paste transcript instead"))
    st.caption("Evolus workflow: " + ("connected" if evolus.enabled() else "not set (optional)"))
    st.divider()
    st.markdown("Sensor data and manual are **simulated sample data** for the demo.")

st.markdown("# 🏭 ShiftSense")
st.markdown("**AI shift-handover copilot for factories.** Voice note + sensor logs + manuals → a cited briefing for the next shift.")

c1, c2 = st.columns([1, 1.3])
with c1:
    scenario = st.selectbox("Shift", list(SCENARIOS))
    sc = SCENARIOS[scenario]
    st.caption(f"Outgoing technician: {sc['technician']} · 4 machines · 8-hour historian export")
    audio = st.file_uploader("Voice note (optional, transcribed by Whisper on AMD)", type=["wav", "mp3", "m4a", "ogg"])
with c2:
    note = st.text_area("Voice-note transcript", sc["note"], height=190, key=f"note_{scenario}")

if audio is not None and cfg["whisper_base_url"]:
    if st.button("Transcribe audio"):
        with st.spinner("Transcribing on AMD…"):
            st.session_state[f"note_{scenario}"] = transcribe(audio.getvalue(), audio.name)
            st.rerun()

df = generate_shift(scenario)

if st.button("Generate handover briefing", type="primary", use_container_width=True):
    with st.spinner("Scoring sensors, searching the manual, writing the briefing…"):
        st.session_state["result"] = (scenario, build_briefing(note, df, get_manual()))

res = st.session_state.get("result")
if res and res[0] == scenario:
    out = res[1]
    b = out["brief"]
    st.divider()
    h1, h2 = st.columns([3, 1])
    h1.markdown(f"## {scenario}")
    h1.markdown(f"**{b.get('headline','')}**")
    h2.metric("Generated in", f"{out['seconds']} s")
    st.caption(f"Backend: {out['backend']}")
    if out["error"]:
        st.error(out["error"])

    left, right = st.columns([1.35, 1])
    with left:
        for it in b.get("items", []):
            s = it.get("status", "OK")
            esc = f"<div class='esc'>⚠ Sent to supervisor: {it.get('escalation_reason','')}</div>" if it.get("escalate") else ""
            ev = "".join(f"<li>{e}</li>" for e in it.get("evidence", []))
            st.markdown(f"""<div class='item {s}'>
<span class='tag t{s}'>{s}</span> &nbsp; <b>{it.get('title', it.get('machine'))}</b>
<span class='small'> · confidence {float(it.get('confidence',0)):.0%}</span>
<div style='margin-top:6px'>{it.get('summary','')}</div>
<div style='margin-top:6px'><b>Next:</b> {it.get('next_action','')} <span style='color:#ffb020'>{it.get('manual_ref','')}</span></div>
<ul class='small' style='margin:6px 0 0 0'>{ev}</ul>{esc}</div>""", unsafe_allow_html=True)
        if b.get("open_tasks"):
            st.markdown("#### Open tasks carried over")
            for t in b["open_tasks"]:
                st.markdown(f"- {t}")
        a1, a2 = st.columns(2)
        if a1.button("✅ Acknowledge & start shift"):
            st.success("Acknowledged. Briefing stored in shift history.")
        if evolus.enabled() and a2.button("Send to Evolus workflow"):
            try:
                st.code(evolus.send_briefing(b, scenario))
            except Exception as e:
                st.error(f"Evolus call failed: {e}")

    with right:
        st.markdown("#### Sensor evidence")
        risks = {r.machine: r for r in out["risks"]}
        for m in [i.get("machine") for i in b.get("items", []) if i.get("status") in ("HIGH", "WATCH")]:
            r = risks.get(m)
            if not r or not r.sensor:
                continue
            g = df[df.machine == m].set_index("time")[[r.sensor]]
            st.caption(f"{m} · {r.sensor} · anomaly score {r.score}")
            st.line_chart(g, height=170)
        st.markdown("#### Manual sections used")
        for m, hits in out["hits"].items():
            h = hits[0]
            with st.expander(f"{m} → {h['ref']} {h['title'].split(' ',1)[1]}"):
                st.write(h["body"])
else:
    st.markdown("#### Shift sensor data (raw)")
    st.line_chart(df.pivot(index="time", columns="machine", values="vibration_mm_s"), height=220)
    st.caption("Vibration (mm/s) per machine. Press **Generate handover briefing** to analyse.")

st.divider()
st.caption("ShiftSense advises; people decide. Safety-related and low-confidence items always go to a supervisor. Built for the AMD Developer Hackathon ACT III.")
