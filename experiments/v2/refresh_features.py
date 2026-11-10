"""Rebuild candidate features using the final pre-answer feature implementation."""
import argparse,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.features import extract_features
from cachescope.pipeline import read_jsonl,write_jsonl,dump_json
parser=argparse.ArgumentParser()
parser.add_argument('name',choices=['new_train_validation','new_test'])
args=parser.parse_args()
path=ROOT/'data'/f'{args.name}_candidates.jsonl'
rows=read_jsonl(path)
before=hashlib.sha256(path.read_bytes()).hexdigest()
for r in rows:
    r['features']=extract_features(r['original_context'],r['question'],r['compressed_context'],r['original_tokens'],r['compressed_tokens'])
write_jsonl(path,rows)
summary_path=ROOT/'results'/f'{args.name}_labeling_summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary.setdefault('candidate_file_sha256_before_feature_refresh',before)
    summary['candidate_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    summary['features_recomputed_with_final_schema']=True
    summary['feature_code_sha256']=hashlib.sha256((ROOT/'src/cachescope/features.py').read_bytes()).hexdigest()
    summary['timing_note']='Per-row labeling preprocess time predates feature refresh and is not a V2 deployment timing measurement.'
    dump_json(summary_path,summary)
print(f'Rebuilt {len(rows)} candidate features with final implementation')
