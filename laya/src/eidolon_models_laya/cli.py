"""``eidolon-laya``: fetch | export-onnx | doctor | serve | predict."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

from .artifacts import Manifest, fetch_torch, verify_onnx, verify_torch
from .config import BACKENDS, DEVICES, Settings


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    return settings.with_overrides(
        backend=getattr(args, "backend", None),
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        device=getattr(args, "device", None),
        threads=getattr(args, "threads", None),
        max_len=getattr(args, "max_len", None),
        head_max_len=getattr(args, "head_max_len", None),
        model_dir=Path(args.model_dir).resolve() if getattr(args, "model_dir", None) else None,
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = Manifest.load(_settings(args).model_dir)
    fetch_torch(manifest, endpoint=args.endpoint, use_xet=args.xet)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_onnx

    manifest = Manifest.load(_settings(args).model_dir)
    problems = verify_torch(manifest, checksums=False)
    if problems:
        print(
            "PyTorch weights missing; run `eidolon-laya fetch` first:\n  " + "\n  ".join(problems)
        )
        return 2
    export_onnx(manifest, force=args.force)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    manifest = Manifest.load(settings.model_dir)
    source = f"{manifest.repo_id}@{manifest.revision[:8]}/{manifest.subfolder}"
    threads = settings.threads or "auto"
    auth = "key" if settings.api_key else "none"
    print(f"model      {manifest.name} {source}")
    print(f"model_dir  {settings.model_dir}")
    print(f"backend    {settings.backend} (device={settings.device}, threads={threads})")
    print(f"listen     {settings.host}:{settings.port} auth={auth}")
    ok = True
    for label, problems, extra in (
        ("torch", verify_torch(manifest), "torch"),
        ("onnx", verify_onnx(manifest), "onnxruntime"),
    ):
        installed = importlib.util.find_spec(extra) is not None
        state = "ok" if not problems else "; ".join(problems)
        runtime = "installed" if installed else "missing"
        print(f"{label:<10} files: {state} | runtime {extra}: {runtime}")
        if label == settings.backend and (problems or not installed):
            ok = False
    try:
        settings.validate_exposure()
    except ValueError as exc:
        print(f"exposure   {exc}")
        ok = False
    print("ready" if ok else "NOT ready for the configured backend")
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from aiohttp import web

    from .engine import load_engine
    from .service import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = _settings(args)
    settings.validate_exposure()
    engine, manifest = load_engine(settings, log=logging.getLogger("eidolon_laya").info)
    model_info = {
        "model": manifest.name,
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "subfolder": manifest.subfolder,
    }
    web.run_app(
        create_app(engine, settings, model_info),
        host=settings.host,
        port=settings.port,
        print=lambda msg: logging.getLogger("eidolon_laya").info(msg.strip()),
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .engine import load_engine

    request = json.loads(Path(args.file).read_text(encoding="utf-8") if args.file else args.json)
    engine, _ = load_engine(_settings(args), log=lambda m: print(m, file=sys.stderr))
    options = request.get("options") or {}
    prediction = engine.predict(
        request["state"],
        request["questions"],
        truncate_left=bool(options.get("truncate_left", False)),
    )
    print(
        json.dumps(
            prediction.as_response(engine.name, engine.backend.name), ensure_ascii=False, indent=2
        )
    )
    return 0


def _runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=BACKENDS)
    p.add_argument("--device", choices=DEVICES, help="torch backend only")
    p.add_argument("--threads", type=int, help="intra-op threads (0 = runtime default)")
    p.add_argument("--max-len", type=int, help="token budget per question (default: checkpoint's)")
    p.add_argument(
        "--head-max-len", type=int, help="budget for instruction + options (default: checkpoint's)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eidolon-laya", description=__doc__)
    parser.add_argument(
        "--model-dir", help="model version directory (default: EIDOLON_LAYA_MODEL_DIR)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the pinned PyTorch checkpoint and verify it")
    p.add_argument("--endpoint", help="Hugging Face endpoint, e.g. https://hf-mirror.com")
    p.add_argument(
        "--xet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use Xet storage (default: only against huggingface.co)",
    )
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("export-onnx", help="export ONNX from the PyTorch weights (export extra)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("doctor", help="check files, checksums, runtimes and exposure")
    _runtime_args(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="run the HTTP API")
    _runtime_args(p)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("predict", help="one request in-process, no server")
    _runtime_args(p)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="JSON file with state/questions[/options]")
    group.add_argument("--json", help="the same, inline")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eidolon-laya: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
