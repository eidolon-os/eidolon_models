"""Read-only production model and installed Agent/SDK checks; never execute devices."""

import hashlib
import inspect
import json
import time
import urllib.request
from pathlib import Path
import yaml
from eidolon_agent.config.settings import SmartHomeSettings
from eidolon_agent.domain.smarthome.command import SmartHomeCommand
from eidolon_sdk.biz.smarthome import TRAIT_COMMANDS

urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
base = "http://127.0.0.1:8771"
info = json.load(urllib.request.urlopen(base + "/v1/info"))
assert info["revision"] == "e8254243" and info["engine"]["backend"] == "rknn", info
settings = yaml.safe_load(Path("/etc/eidolon/agent.yaml").read_text())
home = SmartHomeSettings.model_validate(settings.get("smarthome", {}))
assert home.interpreter == "laya"
assert home.context_turns == 3
assert "step" in TRAIT_COMMANDS["fan_speed"]
source = Path(inspect.getfile(SmartHomeCommand))
report = {
    "model": info,
    "agent": {
        "interpreter": home.interpreter,
        "context_turns": home.context_turns,
        "source": str(source),
        "command_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    },
    "sdk_fan_speed_step": TRAIT_COMMANDS["fan_speed"]["step"],
    "probes": [],
}
fixtures = [
    json.loads(l) for l in Path(__file__).with_name("smoke-requests.jsonl").read_text().splitlines()
]
for f in fixtures:
    request = urllib.request.Request(
        base + "/v1/systemone", json.dumps(f["body"]).encode(), {"content-type": "application/json"}
    )
    start = time.perf_counter()
    response = json.load(urllib.request.urlopen(request, timeout=10))
    assert response["revision"] == "e8254243"
    assert response["features"]["question_state"] is True
    assert not response["truncated"]
    report["probes"].append(
        {
            "id": f["id"],
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            "response": response,
        }
    )
print(json.dumps(report, ensure_ascii=False, indent=2))
