"""Rebuild final feature schema while preserving genuine observations."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.pipeline import read_jsonl,write_jsonl,dump_json
from cachescope.features import extract_features
parser=argparse.ArgumentParser()
parser.add_argument('name',choices=['calibration','test'])
args=parser.parse_args()
path=ROOT/'data'/f'{args.name}_candidates.jsonl'
rows=read_jsonl(path)
before=hashlib.sha256(path.read_bytes()).hexdigest()
for r in rows:
    r['features']=extract_features(r['original_context'],r['question'],r['compressed_context'],r['original_tokens'],r['compressed_tokens'])
write_jsonl(path,rows)
summary_path=ROOT/'results'/f'{args.name}_labeling_summary.json'
summary=json.loads(summary_path.read_text(encoding='utf-8'))
summary.setdefault('candidate_sha256_before_feature_refresh',before)
summary['candidate_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
summary['feature_code_sha256']=hashlib.sha256((ROOT/'src/cachescope/features.py').read_bytes()).hexdigest()
summary['timing_note']='Stored labeling preprocessing times are not deployment-decision benchmarks.'
dump_json(summary_path,summary)
print(f'Refreshed {len(rows)} rows without changing answers or labels')
