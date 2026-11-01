"""WeSpeaker's own scoring scripts, loaded by file from a pinned checkout.

The package ``__init__`` imports the whole inference CLI (torchaudio, VAD,
model loading), which ``score_norm.py`` and ``score_calibration.py`` do not
need, so the two scripts and ``utils/file_utils.py`` are loaded directly. Their
own imports (fire, kaldiio, termcolor) live in a side directory, not in any
runtime environment.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

REVISION = "9fecd6cb4f47475d01761d87c826298dff4ef18c"
DEFAULT_CHECKOUT = Path("/Users/manson/ai/thirdpart/wespeaker")
DEFAULT_DEPS = Path("/Users/manson/ai/thirdpart/wespeaker-deps")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(checkout: Path = DEFAULT_CHECKOUT, deps: Path = DEFAULT_DEPS):
    """(score_norm, score_calibration) modules of WeSpeaker at REVISION."""
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != REVISION:
        raise SystemExit(f"WeSpeaker checkout is at {head}, expected {REVISION}")
    if str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    file_utils = _load("wespeaker.utils.file_utils", checkout / "wespeaker/utils/file_utils.py")
    package, utils = types.ModuleType("wespeaker"), types.ModuleType("wespeaker.utils")
    utils.file_utils = file_utils
    package.utils = utils
    sys.modules.update({"wespeaker": package, "wespeaker.utils": utils,
                        "wespeaker.utils.file_utils": file_utils})
    return (_load("wespeaker_score_norm", checkout / "wespeaker/bin/score_norm.py"),
            _load("wespeaker_score_calibration", checkout / "wespeaker/bin/score_calibration.py"))
