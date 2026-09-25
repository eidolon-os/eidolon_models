"""``eidolon-laya-train``: gen | label | augment | assemble | train | calibrate | eval | export | run.

Every stage reads files and writes files; ``run`` chains them from a pipeline.yaml into
``runs/<id>/`` with a manifest per step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .records import read_jsonl, write_jsonl
from .scenario import Scenario


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ stages


def cmd_gen(args) -> int:
    from .generators import generate

    scenario = Scenario.load(args.scenario)
    cfg = _load_yaml(args.config) if args.config else {}
    gens = cfg.get("generators") or [cfg]
    total = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, g in enumerate(gens):
            n = 0
            for r in generate(scenario, g, seed=int(cfg.get("seed", 7)) + i):
                fh.write(r.to_json() + "\n")
                n += 1
            _log(
                f"gen {g.get('kind')}:{g.get('name', g.get('adapter', g.get('module', '')))} -> {n}"
            )
            total += n
    _log(f"wrote {out} ({total} records)")
    return 0


def cmd_label(args) -> int:
    from .label import Teacher, label_records

    stats: dict = {}
    teacher = Teacher(args.teacher, timeout=args.timeout)
    records = read_jsonl(args.input)
    n = write_jsonl(
        args.out,
        label_records(
            records,
            teacher,
            alpha=args.alpha,
            workers=args.workers,
            teacher_name=args.teacher_name,
            stats=stats,
        ),
    )
    _log(f"wrote {args.out} ({n} records; {stats})")
    return 0 if stats.get("failed", 0) == 0 else 3


def cmd_augment(args) -> int:
    from .augment import augment_records

    scenario = Scenario.load(args.scenario)
    cfg = _load_yaml(args.config)
    stats: dict = {}
    n = write_jsonl(
        args.out,
        augment_records(
            read_jsonl(args.input),
            scenario,
            cfg.get("transforms", []),
            seed=int(cfg.get("seed", 7)),
            stats=stats,
        ),
    )
    _log(f"wrote {args.out} ({n} records; derived {stats})")
    return 0


def cmd_assemble(args) -> int:
    from .assemble import assemble

    cfg = _load_yaml(args.config)
    if args.run_dir:  # "{run}" inside the config points at this run's outputs
        cfg = json.loads(json.dumps(cfg).replace("{run}", str(Path(args.run_dir).resolve())))
    manifest = assemble(cfg, Path(args.out), Path(args.config).resolve().parent)
    _log(json.dumps({k: v for k, v in manifest.items() if k != "config"}, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    from .train import train

    cfg = _load_yaml(args.config)
    if cfg.get("init"):  # relative to the config file, like every other path in a stage config
        cfg["init"] = str((Path(args.config).resolve().parent / cfg["init"]).resolve())
    if args.init:
        cfg["init"] = args.init
    if args.device:
        cfg["device"] = args.device
    for k in ("epochs", "batch_size"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    summary = train(cfg, Path(args.dataset), Path(args.out), log=_log)
    _log(f"best epoch {summary['best_epoch']} score {summary['best_score']}")
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate, write_temperatures
    from .model import load_checkpoint

    loaded = load_checkpoint(args.checkpoint, args.device)
    calibrate(loaded, Path(args.calib), min_bucket=args.min_bucket, log=_log)
    write_temperatures(Path(args.checkpoint), loaded.cfg)
    _log(f"temperatures written to {args.checkpoint}/rl_agent_config.json")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, gate, write_report
    from .model import load_checkpoint

    loaded = load_checkpoint(args.checkpoint, args.device)
    out_dir = Path(args.out)
    rc = 0
    for path in args.eval_set:
        report = evaluate(loaded, Path(path))
        name = Path(path).stem
        write_report(report, out_dir / f"{name}.json", keep_rows=not args.no_rows)
        _log(f"{name}: {json.dumps(report['overall'], ensure_ascii=False)}")
        for scn, agg in report["by_scenario"].items():
            _log(f"  {scn}: acc {agg['acc']} ece {agg['ece']} n {agg['n']}")
        if args.baseline:
            base = json.loads((Path(args.baseline) / f"{name}.json").read_text(encoding="utf-8"))
            g = gate(report, base, args.tolerance)
            (out_dir / f"{name}.gate.json").write_text(
                json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _log(f"  gate vs {args.baseline}: {'PASS' if g['passed'] else 'FAIL'}")
            rc = rc or (0 if g["passed"] else 4)
    return rc


def cmd_export(args) -> int:
    """Hand the checkpoint to ``eidolon-laya export-onnx`` through a temporary manifest-less dir."""
    cmd = [
        sys.executable,
        "-m",
        "eidolon_models_laya.cli",
        "export-onnx",
        "--model-dir",
        args.checkpoint,
    ]
    _log(" ".join(cmd))
    return subprocess.call(cmd)


def cmd_run(args) -> int:
    pipeline = _load_yaml(args.pipeline)
    base = Path(args.pipeline).resolve().parent
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    run_dir = (base / pipeline.get("runs_dir", "../runs")).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"pipeline": str(args.pipeline), "run_id": run_id, "steps": []}
    _log(f"run {run_id} -> {run_dir}")
    for step in pipeline["steps"]:
        stage = step["stage"]
        t0 = time.time()
        argv = _step_argv(stage, step, run_dir, base)
        _log(f"[{stage}] {' '.join(argv)}")
        rc = main(argv)
        rec = {"stage": stage, "argv": argv, "rc": rc, "seconds": round(time.time() - t0, 1)}
        manifest["steps"].append(rec)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if rc != 0 and not step.get("continue_on_error"):
            _log(f"[{stage}] failed rc={rc}; stopping")
            return rc
    _log(f"run {run_id} complete")
    return 0


PATH_KEYS = {
    "scenario",
    "config",
    "input",
    "out",
    "dataset",
    "checkpoint",
    "calib",
    "init",
    "pipeline",
    "baseline",
    "eval_set",
}


def _resolve(base: Path, run_dir: Path, value, key: str) -> str:
    v = str(value).replace("{run}", str(run_dir))
    if key not in PATH_KEYS:
        return v
    p = Path(v)
    return str(p if p.is_absolute() else (base / p).resolve())


def _step_argv(stage: str, step: dict, run_dir: Path, base: Path) -> list[str]:
    argv = [stage]
    for k, v in step.items():
        if k in ("stage", "continue_on_error"):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, list):
            for x in v:
                argv += [flag, _resolve(base, run_dir, x, k)]
        else:
            argv += [flag, _resolve(base, run_dir, v, k)]
    if stage in ("assemble",) and "--run-dir" not in argv:
        argv += ["--run-dir", str(run_dir)]
    return argv


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eidolon-laya-train", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen", help="scenario + generator config -> cases.jsonl")
    p.add_argument("--scenario", required=True)
    p.add_argument("--config", help="yaml with `generators: [...]` (or a single generator)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gen)

    p = sub.add_parser("label", help="soft targets from a /v1/systemone teacher")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher", required=True, help="base URL of the teacher service")
    p.add_argument("--teacher-name", default="teacher")
    p.add_argument(
        "--alpha", type=float, default=0.7, help="weight of gold vs teacher when both exist"
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("augment", help="derived records from transforms")
    p.add_argument("--scenario", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("assemble", help="mix sources -> dataset/{train,val,calib}.jsonl")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", help="substitutes {run} inside the config")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("train", help="fine-tune -> checkpoint dir loadable by eidolon-laya serve")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", help="override the checkpoint to start from")
    p.add_argument("--device")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="fit temperatures on calib.jsonl into the checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--min-bucket", type=int, default=30)
    p.add_argument("--device")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "eval", help="score eval sets; optionally gate against a baseline report dir"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-set", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", help="directory of a previous eval (same set names)")
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--no-rows", action="store_true")
    p.add_argument("--device")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export", help="ONNX via eidolon-laya export-onnx")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("run", help="execute a pipeline.yaml into runs/<id>/")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--run-id")
    p.set_defaults(func=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
