"""Freeze request bodies for same-input FP32 versus NPU comparison."""

import asyncio
import json
from pathlib import Path
from eidolon_sdk.biz.interpretation import InterpretationRequest
from eidolon_agent.infra.interpretation.adapters.laya import LayaInterpreter

HERE = Path(__file__).resolve().parent
LAYA = HERE.parents[1]


class Captured(Exception):
    pass


class Capture(LayaInterpreter):
    async def predict(self, body, *, timeout_ms):
        self.body = body
        raise Captured()


async def main():
    data = json.loads((HERE.parent / "model-first-20261008/c10.json").read_text())
    model = Capture("http://127.0.0.1:1")
    fixtures = []
    try:
        for group in ("conversation", "independent", "context_safety"):
            for index, row in enumerate(data[group]):
                rec = row["model"]
                try:
                    await model.interpret(InterpretationRequest.model_validate(rec["request"]))
                except Captured:
                    pass
                answers = json.loads(rec["result"]["diagnostics"]["answers_json"])
                fixtures.append(
                    {
                        "id": f"{group}-{index}",
                        "set": group,
                        "body": model.body,
                        "expected": {k: v["choice"] for k, v in answers.items()},
                    }
                )
        for name in ("target-accept", "target-accept-single", "controls-accept"):
            report = json.loads((LAYA / f"train/runs/c10/acceptance/{name}.json").read_text())
            expected = {(r["record_id"], r["qid"]): r for r in report["rows"]}
            for line in (LAYA / f"train/runs/c10/data/{name}.jsonl").read_text().splitlines():
                row = json.loads(line)
                fixtures.append(
                    {
                        "id": row["id"],
                        "set": name,
                        "body": {k: row[k] for k in ("state", "questions")},
                        "expected": {
                            qid: expected[(row["id"], qid)]["pred"]
                            for qid in row["questions"]
                            if (row["id"], qid) in expected
                        },
                        "reference": {
                            qid: expected[(row["id"], qid)]
                            for qid in row["questions"]
                            if (row["id"], qid) in expected
                        },
                    }
                )
        (HERE / "requests.jsonl").write_text(
            "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in fixtures)
        )
        print("Frozen requests:", len(fixtures))
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
