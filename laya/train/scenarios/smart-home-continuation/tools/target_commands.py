"""Offline command construction audit using unchanged, pinned Agent source.

Loads the baseline SDK DTOs and Agent lexicon from read-only git objects.
Executes the exact _control function and _ACTIONS assignment extracted by AST;
does not call services, simulate Hub acceptance, or change runtime policy.
Currently supports the explicitly authored c5 target-pair layouts only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from target_pairs import dev_homes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-repo", required=True, type=Path)
    ap.add_argument("--sdk-repo", required=True, type=Path)
    ap.add_argument("--agent-ref", default="c22ddd1")
    ap.add_argument("--sdk-ref", default="c375645")
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.agent_ref = subprocess.check_output(
        ["git", "-C", str(a.agent_repo), "rev-parse", a.agent_ref], text=True).strip()
    a.sdk_ref = subprocess.check_output(
        ["git", "-C", str(a.sdk_repo), "rev-parse", a.sdk_ref], text=True).strip()
    provenance = {}

    def source(repo, ref, path):
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])
        provenance[path] = {"sha256": hashlib.sha256(data).hexdigest(), "ref": ref}
        return data.decode()

    with tempfile.TemporaryDirectory(prefix="c5-command-audit-") as tmp:
        root = Path(tmp)
        for package in ("eidolon_sdk", "eidolon_sdk/biz"):
            (root / package).mkdir(parents=True, exist_ok=True)
            (root / package / "__init__.py").write_text("")
        for module in ("interpretation", "smarthome"):
            path = f"eidolon_sdk/biz/{module}/__init__.py"
            dest = root / path
            dest.parent.mkdir(parents=True)
            dest.write_text(source(a.sdk_repo, a.sdk_ref, path))
        sys.path.insert(0, tmp)
        from eidolon_sdk.biz import interpretation, smarthome
        from eidolon_sdk.biz.interpretation import Candidate, InterpretationRequest, Proposal

        for module in (interpretation, smarthome):
            if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                raise RuntimeError("SDK was imported before the pinned snapshot; use a fresh process")

        lex_path = "eidolon_agent/infra/interpretation/adapters/lexicon.py"
        lex_file = root / "pinned_lexicon.py"
        lex_file.write_text(source(a.agent_repo, a.agent_ref, lex_path))
        spec = importlib.util.spec_from_file_location("pinned_lexicon", lex_file)
        lex = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = lex
        spec.loader.exec_module(lex)
        adapter_path = "eidolon_agent/infra/interpretation/adapters/laya.py"
        tree = ast.parse(source(a.agent_repo, a.agent_ref, adapter_path))
        nodes = [n for n in tree.body if (
            isinstance(n, ast.FunctionDef) and n.name == "_control"
        ) or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("_ACTIONS", "NO_DEVICE") for t in n.targets))]
        assert len(nodes) == 3
        scope = {name: getattr(lex, name) for name in (
            "Verb", "command_for", "generic_command", "normalize", "read_values", "take")}
        scope.update(Candidate=Candidate, InterpretationRequest=InterpretationRequest, Proposal=Proposal)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), adapter_path, "exec"), scope)
        records = {r["id"]: r for r in map(json.loads, a.records.read_text().splitlines())}
        report = json.loads(a.report.read_text())
        homes = dev_homes()
        counts, details = Counter(), []
        for row in report["policy_rows"]:
            assert row["qid"] == "follow"
            record = records[row["record_id"]]
            home = {d.name: d for d in homes[record["meta"]["home"]]}
            names = record["state"]["context"]["设备"]
            candidates = tuple(Candidate(ref=n, name=n, kind=home[n].kind) for n in names)
            request = InterpretationRequest(
                interpretation_id=record["id"], domain="smarthome",
                utterance=record["state"]["utterance"], candidates=candidates, timeout_ms=1000)
            options = {c.name: ("", c) for c in candidates}
            qualified = row["pred"] != "重新理解" and row["p_top"] >= .95
            proposals = [scope["_control"](request, options, n, scope["_ACTIONS"][row["pred"]])
                         for n in names] if qualified else []
            constructible = bool(proposals) and all(p is not None for p in proposals)
            constructible = constructible and all(p.action == proposals[0].action for p in proposals)
            wrong = not row["correct"]
            counts["rows"] += 1
            counts["threshold_execute"] += qualified
            counts["threshold_wrong_execute"] += qualified and wrong
            counts["constructible"] += constructible
            counts["constructible_wrong_execute"] += constructible and wrong
            if qualified:
                details.append({"record_id": record["id"], "utterance": request.utterance,
                                "focus": names, "pred": row["pred"], "gold": row["gold"],
                                "p_top": row["p_top"], "wrong": wrong,
                                "constructible": constructible,
                                "proposals": [p.model_dump() if p else None for p in proposals]})
        result = {"scope": "Pinned Agent _control construction only; no Hub or physical execution",
                  "source_report": str(a.report), "source_records": str(a.records),
                  "pinned_sources": provenance, "counts": dict(counts), "details": details}
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
