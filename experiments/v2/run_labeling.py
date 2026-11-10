"""Real local-model labeling. Test labeling requires a frozen model selection."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.pipeline import label_tasks, read_jsonl

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['development','test'])
    args=parser.parse_args()
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['data_generation_frozen_hashes'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Frozen data/generator changed: {name}')
    tasks=read_jsonl(ROOT/'data/new_tasks.jsonl')
    if args.stage=='test':
        frozen=ROOT/'results/frozen_model_selection.json'
        if not frozen.exists(): raise ValueError('Train and freeze models/thresholds before test labeling')
        from train_evaluate_v2 import verify_freeze
        verify_freeze(json.loads(frozen.read_text(encoding='utf-8')))
        tasks=[t for t in tasks if t['split']=='test']
        name='new_test'
        print(json.dumps({'stage':'test','selection_file_sha256':sha(frozen),'n_tasks':len(tasks)}),flush=True)
    else:
        tasks=[t for t in tasks if t['split'] in ['train','validation']]
        name='new_train_validation'
    label_tasks(ROOT,tasks,name)

if __name__=='__main__': main()
