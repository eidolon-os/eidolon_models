"""Standalone replay of the frozen multi-task DEV suite."""
import argparse
from pathlib import Path

from .data import read,validate
from .engine import Engine
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('refuse to overwrite prior predictions')
    rows=[validate(r) for r in read(a.data)]
    if any(r['split']!='dev' for r in rows):raise ValueError('this entry point is DEV only')
    if not {'v7','v8','companion','ip_team'}<={r['meta']['dataset'] for r in rows}:
        raise ValueError('expected the complete frozen four-group DEV suite')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    engine=Engine(a.model,adapter=a.adapter);evaluate(engine,rows,a.out)


if __name__=='__main__':main()
