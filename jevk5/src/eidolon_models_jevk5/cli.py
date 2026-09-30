"""Small local prediction entry point; training and generation are separate modules."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--file',type=Path,required=True)
    p.add_argument('--device',default='mps',choices=['mps','cpu']);a=p.parse_args()
    from .engine import Engine
    row=json.loads(a.file.read_text());engine=Engine(a.model,a.device,a.adapter)
    probabilities,tokens=engine.predict(row)
    print(json.dumps(dict(probabilities=probabilities,tokens=tokens),ensure_ascii=False))
