"""Publish summaries/provenance only; keep audio, vectors and binaries local."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("trial", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    read = lambda path: json.loads(path.read_text())
    result = read(args.trial / "comparison.json")
    assert result["export_passed"]
    for row in result["frozen_policy"]:
        for name in ["exported_same", "exported_pipeline"]:
            r = row["variants"][name]
            assert r["decision_changes"] == r["old_gallery_decision_changes"] == r["discovery_summary_changed_seeds"] == 0
    result["performance"] = {}
    for name in ["native", "public", "exported"]:
        r = read(args.trial / f"benchmark-{name}.json")
        assert len(r["rows"]) == 864
        result["performance"][name] = {k: v for k, v in r.items() if k != "rows"}
    data = read(args.trial / "vectors.json")
    a = np.asarray([x["vector"] for x in data["legacy_same"]["rows"]])
    b = np.asarray([x["vector"] for x in data["public_same"]["rows"]])
    result["direct_export_vs_public_same_features"] = dict(
        cosine_min=float(np.min(np.sum(a * b, axis=1))), unit_abs_max=float(np.max(np.abs(a - b))))
    manifest = read(args.corpus / "assets.json")
    result["corpus"] = {k: v for k, v in manifest.items() if k not in ["models", "code", "code_revision", "export_revision"]}
    result["public_model_source"] = [x for x in manifest["models"] if "campplus" in x["path"]]
    result["exports"] = {k: read(args.trial / f"weights/campplus-{k}.json") for k in ["legacy", "semantic"]}
    reports = sorted((args.trial / "weights").glob("onnx_export_*_pt_export.md"))
    assert len(reports) == 1
    result["dynamo_attempt"] = dict(succeeded=False, model_generated=False,
        error="GuardOnDataDependentSymNode during dynamic segment pooling/slice shape capture",
        report_sha256=sha(reports[0]), report_name=reports[0].name,
        onnxscript=importlib.metadata.version("onnxscript"), onnx_ir=importlib.metadata.version("onnx-ir"))
    test_log = (args.trial / "tests.log").read_text()
    assert "24 passed" in test_log
    result["validation"] = dict(tests_passed=24, pool_lengths=406, pool_cases=812,
                                test_log_sha256=sha(args.trial / "tests.log"),
                                policy_gate_passed=True)
    result["hardware"] = read(args.corpus / "host.json")
    scripts = Path(__file__).parent
    result["script_sha256"] = {x.name: sha(x) for x in sorted(scripts.glob("*.py"))}
    result["reused_evaluator_sha256"] = sha(scripts.parent / "anonymous-20261028/analyze.py")
    model_pkg = Path(importlib.metadata.distribution("modelscope").locate_file("modelscope"))
    result["modelscope"] = dict(version=importlib.metadata.version("modelscope"),
        source_sha256={name: sha(model_pkg / "models/audio/sv" / name) for name in ["DTDNN.py", "DTDNN_layers.py"]})
    result["notes"] = ["Inference timing from separate sequential processes; dependency imports excluded from model load time.",
        "ORT process does not import torch; RSS not directly comparable with the previous experiment.",
        "Discovery comparison is equality of statistics, not per-event assignment trace equality.",
        "Same original recordings recur across duration bins and shuffle seeds; not independent samples.",
        "Public speech only; training overlap and capture-session separation not established.",
        "No deployed model or service changed; no ARM, RKNN or accelerator claim."]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
