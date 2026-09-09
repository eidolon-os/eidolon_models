"""`eidolon-models-tts serve` — and nothing else yet.

A subcommand rather than a bare entry point because the ASR service's CLI grew
a benchmark and a one-shot next to its `serve`, and this one will too.
"""

from __future__ import annotations

import argparse
import logging
import sys

from eidolon_models_tts.config import apply_cpu_affinity, load_settings
from eidolon_models_tts.service import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eidolon-models-tts")
    parser.add_argument("command", choices=("serve",), help="serve this Host's own synthesis")
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    # Pin before the engine is spawned: the child inherits the mask, and
    # which cores it gets is the difference between rtf 0.94 and an
    # audible gap when the chat model is decoding (HOST-RK3588.md 2.29).
    apply_cpu_affinity()
    serve(load_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
