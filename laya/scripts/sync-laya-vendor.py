#!/usr/bin/env python3
"""Refresh src/eidolon_models_laya/vendor/laya from an upstream git tag.

    scripts/sync-laya-vendor.py v0.3.20 [--repo https://github.com/NandhaKishorM/laya]

Upstream deletes old releases from PyPI, so the package is carried in-tree and followed by
tag, not by pin. The copy is the upstream ``laya/`` package with its absolute self-imports
rewritten to relative ones (``from laya.common`` -> ``from .common``), plus LICENSE and a
VENDOR_VERSION file naming the tag and commit. After syncing, the upgrade gate is:

    uv run --extra torch pytest tests/test_parity.py        # our torch-free port still matches
    evals/smart-home/run_all.sh                             # the 182-case numbers do not move

Both must pass before the bump is committed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "src" / "eidolon_models_laya" / "vendor" / "laya"
DEFAULT_REPO = "https://github.com/NandhaKishorM/laya"

ABS_IMPORT = re.compile(r"^(\s*)(from|import)\s+laya(\.[\w.]+)?(\s+import\s+.*|\s+as\s+\w+)?\s*$")


def rewrite_imports(path: Path, depth: int) -> int:
    """Make ``laya`` self-imports relative. ``depth`` is the file's nesting under the package
    (0 for laya/x.py, 1 for laya/mcp/x.py). Returns the number of lines rewritten."""
    dots = "." * (depth + 1)
    out, n = [], 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        m = ABS_IMPORT.match(line.rstrip("\n"))
        if not m:
            out.append(line)
            continue
        indent, kind, sub, tail = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
        if kind == "from":
            new = f"{indent}from {dots}{sub.lstrip('.')} import{tail.split('import', 1)[1]}\n"
        else:  # import laya [as x]  ->  from .. import __init__ is not a thing; bind the package
            alias = tail.strip().split()[-1] if tail.strip() else "laya"
            pkg = "eidolon_models_laya.vendor.laya"
            new = f"{indent}import {pkg}{sub} as {alias}\n"
        out.append(new)
        n += 1
    if n:
        path.write_text("".join(out), encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tag")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call(
            ["git", "clone", "-q", "--depth", "1", "--branch", args.tag, args.repo, tmp]
        )
        sha = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
        src = Path(tmp) / "laya"
        if not (src / "common.py").is_file():
            print(f"{args.repo}@{args.tag} has no laya/common.py", file=sys.stderr)
            return 2
        if VENDOR.exists():
            shutil.rmtree(VENDOR)
        shutil.copytree(src, VENDOR, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for lic in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            if (Path(tmp) / lic).is_file():
                shutil.copyfile(Path(tmp) / lic, VENDOR / "LICENSE")
                break
    rewritten = 0
    for py in VENDOR.rglob("*.py"):
        depth = len(py.relative_to(VENDOR).parts) - 1
        rewritten += rewrite_imports(py, depth)
    (VENDOR / "VENDOR_VERSION").write_text(
        f"repo: {args.repo}\ntag: {args.tag}\ncommit: {sha}\n"
        "synced_by: scripts/sync-laya-vendor.py (absolute self-imports rewritten to relative)\n",
        encoding="utf-8",
    )
    print(f"vendored {args.tag} ({sha[:12]}) into {VENDOR}; {rewritten} import lines rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
