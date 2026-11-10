"""Record genuine answering-model observations; test requires frozen selection."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.pipeline import read_jsonl,label_tasks
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['calibration','test'])
    args=parser.parse_args()
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['fixed_input_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Frozen input changed: {name}')
    if args.stage=='test':
        frozen_path=ROOT/'results/frozen_selection.json'
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        from train_v3 import verify_freeze
        verify_freeze(frozen)
        print(json.dumps({'stage':'test','selection_sha256':sha(frozen_path)}),flush=True)
    tasks=[r for r in read_jsonl(ROOT/'data/fresh_tasks.jsonl') if r['split']==args.stage]
    assert len(tasks)==(160 if args.stage=='calibration' else 200)
    label_tasks(ROOT,tasks,args.stage)
if __name__=='__main__': main()
