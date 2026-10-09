"""Compare acceptance decisions as well as argmax choices after a board probe."""

import argparse
import asyncio
import json
from pathlib import Path
from eidolon_sdk.biz.interpretation import InterpretationRequest, InterpretationResult
from eidolon_agent.infra.interpretation.adapters.laya import LayaInterpreter
from eidolon_agent.domain.smarthome.command import SmartHomeCommand
from eidolon_agent.domain.smarthome.tests.conftest import FakeDirectory, FakeExecutor, home_registry


class Saved(LayaInterpreter):
    async def predict(self, body, *, timeout_ms):
        return self.payload


def legacy(answers):
    for name in ("pick", "follow"):
        if name in answers:
            a = answers[name]
            c = a["choice"]
            score = a["probabilities"][c]
            if c == "重新理解":
                return "fallback"
            if c == "取消":
                return "cancel" if name == "pick" and score >= 0.5 else "fallback"
            return c if score >= 0.95 else "fallback"
    if "intent" in answers:
        keys = ["intent"]
        intent = answers["intent"]["choice"]
        if intent != "无关":
            keys.append("device")
        if intent == "控制":
            keys.append("action")
        if any(answers[k]["probabilities"][answers[k]["choice"]] < 0.8 for k in keys):
            return "fallback"
        return {k: answers[k]["choice"] for k in keys}
    return None


async def main(a):
    here = Path(__file__).resolve().parent
    fixtures = [json.loads(l) for l in (here / "requests.jsonl").read_text().splitlines()]
    results = {r["id"]: r for r in map(json.loads, Path(a.responses).read_text().splitlines())}
    baseline = json.loads((here.parent / "model-first-20261008/c10.json").read_text())
    model = Saved("http://127.0.0.1:1")
    directory = FakeDirectory(home_registry())
    command = SmartHomeCommand(
        directory=directory, executor=FakeExecutor(directory), interpreter=model, min_confidence=0.8
    )
    differences = []
    comparisons = 0
    old_new_accepted = [0, 0]
    policy_differences = []
    try:
        for group in ("conversation", "independent", "context_safety"):
            for i, item in enumerate(baseline[group]):
                identity = f"{group}-{i}"
                got = results[identity]
                if got["status"] != 200:
                    differences.append({"id": identity, "http_status": got["status"]})
                    continue
                model.payload = got["response"]
                new = await model.interpret(
                    InterpretationRequest.model_validate(item["model"]["request"])
                )
                old = InterpretationResult.model_validate(item["model"]["result"])
                accepted = [command._confident(r) and r.proposal is not None for r in (old, new)]
                for j in range(2):
                    old_new_accepted[j] += accepted[j]
                comparisons += 1
                if accepted[0] != accepted[1] or (accepted[1] and old.proposal != new.proposal):
                    differences.append(
                        {
                            "id": identity,
                            "old_accepted": accepted[0],
                            "new_accepted": accepted[1],
                            "old": old.model_dump(mode="json"),
                            "new": new.model_dump(mode="json"),
                        }
                    )
        for case in fixtures:
            if not case.get("reference"):
                continue
            got = results[case["id"]]
            if got["status"] != 200:
                policy_differences.append({"id": case["id"], "status": got["status"]})
                continue
            old = {
                k: {"choice": v["pred"], "probabilities": v["probabilities"]}
                for k, v in case["reference"].items()
            }
            before, after = legacy(old), legacy(got["response"]["answers"])
            if before != after:
                policy_differences.append({"id": case["id"], "before": before, "after": after})
        out = {
            "model_first_compared": comparisons,
            "accepted_fp32": old_new_accepted[0],
            "accepted_npu": old_new_accepted[1],
            "model_first_decision_differences": differences,
            "legacy_frozen_decision_differences": policy_differences,
        }
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        await model.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
