"""Audit real responses, exact labels, prompt token counts and frozen data."""
from __future__ import annotations
import hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.pipeline import read_jsonl,dump_json
from cachescope.inference import make_prompt,QwenTokenizer
from cachescope.scoring import is_correct
from cachescope.features import extract_features

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from train_evaluate_v2 import verify_freeze
    verify_freeze(json.loads((ROOT/'results/frozen_model_selection.json').read_text(encoding='utf-8')))
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['data_generation_frozen_hashes'].items():
        assert sha(ROOT/name)==digest,name
    tasks=read_jsonl(ROOT/'data/new_tasks.jsonl')
    assert Counter(t['split'] for t in tasks)=={'train':120,'validation':80,'test':160}
    groups=defaultdict(set)
    for t in tasks: groups[t['group_id']].add(t['split'])
    assert all(len(v)==1 for v in groups.values())
    development=read_jsonl(ROOT/'data/development_candidates.jsonl')
    observed=read_jsonl(ROOT/'data/new_train_validation_candidates.jsonl')+read_jsonl(ROOT/'data/new_test_candidates.jsonl')
    counts=Counter(r['task_id'] for r in observed)
    assert set(counts)=={t['task_id'] for t in tasks} and set(counts.values())=={2}
    assert not ({t['task_id'] for t in tasks}&{r['task_id'] for r in development})
    assert not ({t['source_id'] for t in tasks}&{r['source_id'] for r in development})
    assert not ({r['group_id'] for r in development}&set(groups))
    expected={t['task_id']:t for t in tasks}
    for r in observed:
        for key in ['context','question','answer','answer_aliases','source_id','split','group_id','task_type']:
            assert r[key]==expected[r['task_id']][key],(r['task_id'],key)
    raw={}
    for name in ['development_raw_responses','raw_responses']:
        for r in read_jsonl(ROOT/'data'/f'{name}.jsonl'):
            if r['cache_key'] in raw: assert r==raw[r['cache_key']]
            raw[r['cache_key']]=r
    tokenizer=QwenTokenizer(ROOT/'assets/tokenizer.json')
    done=Counter()
    observed_ids={r['task_id'] for r in observed}
    for r in development+observed:
        context=r.get('original_context',r['context'])
        for prefix,text in [('original',context),('compressed',r['compressed_context'])]:
            response=raw[r[f'{prefix}_cache_key']]
            prompt=make_prompt(text,r['question'])
            assert response['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            assert response['answer']==r[f'{prefix}_answer']
            assert response['model_digest']==protocol['answering_model_digest']
            assert response['generation_options']==protocol['answering_options']
            identity={'model':response['model'],'model_digest':response['model_digest'],'options':response['generation_options'],'prompt':prompt,'repeat':response['repeat']}
            assert response['cache_key']==hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            count=len(tokenizer.encode(prompt))
            assert count==r[f'{prefix}_tokens']==r[f'{prefix}_backend_prompt_tokens']==response['prompt_eval_count']
            assert is_correct(response['answer'],r)==r[f'{prefix}_correct']
            done[response['done_reason']]+=1
        assert r['regression']==int(r['original_correct'] and not r['compressed_correct'])
        if r['task_id'] in observed_ids:
            features=extract_features(context,r['question'],r['compressed_context'],r['original_tokens'],r['compressed_tokens'])
            assert features==r['features'],r['task_id']
            assert all(math.isfinite(x) for x in features.values())
    report={'pass':True,'development_task_records':len({r['task_id'] for r in development}),'new_task_records':len(tasks),'candidate_rows_verified':len(development)+len(observed),'unique_real_cached_responses':len(raw),'new_real_cached_responses':len(read_jsonl(ROOT/'data/raw_responses.jsonl')),'wording_family_splits_disjoint':True,'new_old_task_and_source_ids_disjoint':True,'labels_recomputed_from_real_answers':True,'prompt_hash_cache_identity_digest_options_verified':True,'backend_and_local_full_prompt_tokens_match':True,'new_features_recomputed_exactly':True,'frozen_protocol_hashes_match':True,'done_reason_observations':dict(done)}
    dump_json(ROOT/'results/verification.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
