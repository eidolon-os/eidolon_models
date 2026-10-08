"""Fixed-weight, Mac-only model replay. No external LLM or real device execution.

Run in the Agent environment with SDK and Agent editable checkouts on PYTHONPATH.
The scripted fallback is an oracle, not a measurement of production LLM accuracy.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
from eidolon_sdk.biz.interpretation import Action, Proposal
from eidolon_agent.domain.smarthome.command import SmartHomeCommand, interpretation_request
from eidolon_agent.domain.smarthome.context import HomeContext, HomeCancellation, HomeClarification
from eidolon_agent.domain.smarthome.tests.conftest import (
    FakeDirectory,
    FakeExecutor,
    OWNER,
    home_registry,
)
from eidolon_agent.infra.interpretation.adapters.laya import LayaInterpreter


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.records = []

    async def interpret(self, request):
        start = time.perf_counter()
        result = await self.model.interpret(request)
        self.records.append(
            {
                "request": request.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )
        return result


class OracleFallback:
    def __init__(self):
        self.case = None
        self.records = []

    async def propose(self, request, *, context=None):
        self.records.append({"text": request.utterance, "context": context})
        c = self.case
        if c["expected"] == "cancel":
            return HomeCancellation()
        if c["expected"] == "clarify":
            return HomeClarification(
                "要哪一个？", tuple(c["targets"]), Action.model_validate(c["action"])
            )
        if c["expected"] == "execute":
            return Proposal(
                intent="control",
                target_status="resolved",
                targets=tuple(c["targets"]),
                action=Action.model_validate(c["action"]),
            )
        return Proposal(intent="unrelated", target_status="none")


def commands(case):
    if case["expected"] != "execute":
        return []
    a = case["action"]
    return [
        [ref, a["trait"], a["command"], {s["name"]: s["value"] for s in a["slots"]}]
        for ref in case["targets"]
    ]


async def run(args):
    cases = json.loads(Path(args.cases).read_text())
    async with httpx.AsyncClient(trust_env=False) as client:
        info = (await client.get(args.url + "/v1/info")).json()
    model = LayaInterpreter(args.url)
    recorder, fallback = RecordingModel(model), OracleFallback()
    directory = FakeDirectory(home_registry())
    executor = FakeExecutor(directory)
    command = SmartHomeCommand(
        directory=directory,
        executor=executor,
        interpreter=recorder,
        fallback=fallback,
        min_confidence=0.8,
        interpretation_timeout_ms=10000,
    )
    context = HomeContext(history_limit=args.context_turns)
    result = {
        "info": info,
        "context_turns": args.context_turns,
        "fallback": "scripted oracle, not real LLM",
        "conversation": [],
        "independent": [],
    }
    try:
        for case in cases["conversation"]:
            fallback.case = case
            n, f = len(executor.commands), len(fallback.records)
            voice = await command.handle(OWNER, None, case["id"], case["text"], context=context)
            actual = [list(c) for c in executor.commands[n:]]
            expected_outcome = {
                "execute": "executed",
                "clarify": "ambiguous",
                "cancel": "answered",
                "unrelated": "unrelated",
            }[case["expected"]]
            ok = actual == commands(case) and voice.outcome == expected_outcome
            result["conversation"].append(
                {
                    "case": case,
                    "voice": voice.model_dump(mode="json"),
                    "commands": actual,
                    "used_fallback": len(fallback.records) > f,
                    "passed": ok,
                    "model": recorder.records[-1],
                    "handoff": fallback.records[-1] if len(fallback.records) > f else None,
                    "humidifier_speed": directory.status["master.humidifier"].state["speed"],
                }
            )
        for case in [] if args.conversation_only else cases["independent"]:
            req = interpretation_request(
                directory.registries[0],
                interpretation_id=case["id"],
                utterance=case["text"],
                device_ref=None,
                timeout_ms=10000,
            )
            response = await recorder.interpret(req)
            accepted = command._confident(response) and response.proposal is not None
            p = response.proposal
            actual = (
                [
                    [
                        ref,
                        p.action.trait,
                        p.action.command,
                        {s.name: s.value for s in p.action.slots},
                    ]
                    for ref in p.targets
                ]
                if accepted and p.intent == "control" and p.target_status == "resolved"
                else []
            )
            if case["expected"] == "execute":
                passed = not accepted or actual == commands(case)
            else:
                passed = not actual
            result["independent"].append(
                {
                    "case": case,
                    "accepted": accepted,
                    "commands": actual,
                    "passed": passed,
                    "model": recorder.records[-1],
                }
            )
        # Exercise denial/quotation with both a live focus and a pending choice.
        # A context classifier must not turn old actions into new authorization.
        result["context_safety"] = []
        for mode in ["focus", "pending"]:
            ctx = HomeContext(history_limit=args.context_turns)
            p = Proposal(
                intent="control",
                target_status="ambiguous" if mode == "pending" else "resolved",
                targets=("living.main_light", "master.light")
                if mode == "pending"
                else ("living.main_light",),
                action=Action(trait="on_off", command="on"),
            )
            ctx.remember(
                "打开灯",
                p,
                question="客厅主灯、主卧灯，要哪一个？" if mode == "pending" else None,
                response="已打开客厅主灯" if mode == "focus" else "",
                pending_action="打开" if mode == "pending" else "",
                outcome="executed" if mode == "focus" else "clarification",
            )
            for case in [] if args.conversation_only else cases["independent"]:
                if not case.get("context_safety_case"):
                    continue
                req = interpretation_request(
                    directory.registries[0],
                    interpretation_id=mode + case["id"],
                    utterance=case["text"],
                    device_ref=None,
                    timeout_ms=10000,
                ).model_copy(update={"context": ctx.snapshot()})
                response = await recorder.interpret(req)
                accepted = command._confident(response) and response.proposal is not None
                p = response.proposal
                unsafe = accepted and p.intent == "control" and p.target_status == "resolved"
                result["context_safety"].append(
                    {
                        "case": case,
                        "mode": mode,
                        "passed": not unsafe,
                        "accepted": accepted,
                        "model": recorder.records[-1],
                    }
                )
        revisions = {r["result"]["model_version"] for r in recorder.records}
        if not revisions or any(not v.endswith("@" + args.revision) for v in revisions):
            raise AssertionError(f"wrong model: {revisions}")
        times = [r["elapsed_ms"] for r in recorder.records]
        result["summary"] = {
            "revision": args.revision,
            "total_model_calls": len(recorder.records),
            "conversation_passed": sum(r["passed"] for r in result["conversation"]),
            "conversation_total": len(result["conversation"]),
            "conversation_direct": sum(not r["used_fallback"] for r in result["conversation"]),
            "independent_passed": sum(r["passed"] for r in result["independent"]),
            "independent_total": len(result["independent"]),
            "context_safety_passed": sum(r["passed"] for r in result["context_safety"]),
            "context_safety_total": len(result["context_safety"]),
            "model_ms_median": round(statistics.median(times), 2),
            "model_ms_max": max(times),
            "notes": "10s offline inference deadline; production deadline 1s. No ASR, real LLM, NPU or device timing.",
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return all(
            r["passed"]
            for group in ["conversation", "independent", "context_safety"]
            for r in result[group]
        )
    finally:
        await model.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context-turns", type=int, default=3, choices=[1, 3, 5])
    p.add_argument("--conversation-only", action="store_true")
    p.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    raise SystemExit(0 if asyncio.run(run(p.parse_args())) else 1)
