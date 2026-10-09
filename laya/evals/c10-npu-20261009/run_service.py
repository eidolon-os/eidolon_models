"""Isolated board service; production configuration is never edited."""

from pathlib import Path
from aiohttp import web
from eidolon_models_laya.config import Settings
from eidolon_models_laya.service import create_app
from eidolon_models_laya.engine import load_engine
import argparse

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--port", type=int, default=18774)
p.add_argument("--placement", default="1:128,256,384,512|2:128,256")
a = p.parse_args()
s = Settings(
    service="smart_home",
    backend="rknn",
    model_dir=a.root / "model",
    host="127.0.0.1",
    port=a.port,
    max_len=512,
    head_max_len=512,
    rknn_placement=a.placement,
    max_pending=1,
    rknn_library=a.root / "deploy/rk3588/librknnrt.so",
)
s.validate_exposure()
engine, manifest = load_engine(s)
info = {
    "model": manifest.name,
    "repo_id": manifest.repo_id,
    "revision": manifest.revision,
    "subfolder": manifest.subfolder,
}
web.run_app(create_app(engine, s, info), host=s.host, port=s.port)
