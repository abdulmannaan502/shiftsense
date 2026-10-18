"""Optional Evolus integration: hand the finished briefing to an Evolus workflow
(e.g. distribute to the next shift, file in the knowledge library, open work orders).
Enabled only when EVOLUS_API_KEY, EVOLUS_APP_ID and EVOLUS_WORKFLOW_ID are set."""
import json
import os
import urllib.request

API = "https://api.evolus.ai/api/v1"


def enabled() -> bool:
    return all(os.getenv(k) for k in ("EVOLUS_API_KEY", "EVOLUS_APP_ID", "EVOLUS_WORKFLOW_ID"))


def send_briefing(brief: dict, shift: str) -> str:
    body = json.dumps({"text": json.dumps({"shift": shift, "briefing": brief}, ensure_ascii=False)}).encode()
    req = urllib.request.Request(
        f"{API}/workflows/{os.environ['EVOLUS_WORKFLOW_ID']}/execute", data=body, method="POST",
        headers={"Content-Type": "application/json", "X-ApiKey": os.environ["EVOLUS_API_KEY"],
                 "X-Application-Id": os.environ["EVOLUS_APP_ID"]},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()[:500]
