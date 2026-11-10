"""Independently audit frozen V3 artifacts, real observations and predictions."""
from __future__ import annotations
import hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.pipeline import read_jsonl,dump_json
from cachescope.inference import QwenTokenizer,make_prompt
from cachescope.features import extract_features,preanswer_guard
from cachescope.scoring import is_correct
from cachescope.evaluation import make_decisions,summarize_decisions
from train_v3 import verify_freeze,score_records

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(2)
    frozen=json.loads((ROOT/'results/frozen_selection.json').read_text(encoding='utf-8'))
    verify_freeze(frozen)
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['fixed_input_sha256'].items(): assert sha(ROOT/name)==digest,name
    tasks=read_jsonl(ROOT/'data/fresh_tasks.jsonl')
    assert Counter(t['split'] for t in tasks)=={'calibration':160,'test':200}
    groups=defaultdict(set)
    for task in tasks: groups[task['group_id']].add(task['split'])
    assert all(len(splits)==1 for splits in groups.values()) and len(groups)==72
    development=read_jsonl(ROOT/'data/development_candidates.jsonl')
    calibration=read_jsonl(ROOT/'data/calibration_candidates.jsonl')
    test=read_jsonl(ROOT/'data/test_candidates.jsonl')
    fresh=calibration+test
    by_id={t['task_id']:t for t in tasks}
    counts=Counter(r['task_id'] for r in fresh)
    assert set(counts)==set(by_id) and set(counts.values())=={2}
    for key in ('task_id','source_id','group_id'):
        assert not ({r[key] for r in fresh}&{r[key] for r in development}),key
        assert not ({r[key] for r in calibration}&{r[key] for r in test}),key
    for row in fresh:
        for key in ('context','question','answer','answer_aliases','split','source_id','group_id','task_type'):
            assert row[key]==by_id[row['task_id']][key],(row['task_id'],key)
    raw={}
    for name in ('development_raw_responses','raw_responses'):
        for response in read_jsonl(ROOT/'data'/f'{name}.jsonl'):
            if response['cache_key'] in raw: assert response==raw[response['cache_key']]
            raw[response['cache_key']]=response
    tokenizer=QwenTokenizer(ROOT/'assets/tokenizer.json')
    fresh_ids=set(by_id)
    reasons=Counter()
    for row in development+fresh:
        for prefix,context in [('original',row['original_context']),('compressed',row['compressed_context'])]:
            response=raw[row[f'{prefix}_cache_key']]
            prompt=make_prompt(context,row['question'])
            assert response['model_digest']==protocol['answering_model_digest']
            assert response['generation_options']==protocol['answering_options']
            assert response['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            identity={'model':response['model'],'model_digest':response['model_digest'],'options':response['generation_options'],'prompt':prompt,'repeat':response['repeat']}
            assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==response['cache_key']
            assert response['answer']==row[f'{prefix}_answer']
            assert is_correct(response['answer'],row)==row[f'{prefix}_correct']
            assert len(tokenizer.encode(prompt))==row[f'{prefix}_tokens']==row[f'{prefix}_backend_prompt_tokens']==response['prompt_eval_count']
            reasons[response['done_reason']]+=1
        assert row['regression']==int(row['original_correct'] and not row['compressed_correct'])
        if row['task_id'] in fresh_ids:
            features=extract_features(row['original_context'],row['question'],row['compressed_context'],row['original_tokens'],row['compressed_tokens'])
            assert features==row['features'] and all(math.isfinite(x) for x in features.values())
    saved=read_jsonl(ROOT/'results/fresh_test_scored_candidates.jsonl')
    recomputed=score_records(test)
    saved_by_id={(r['task_id'],r['candidate_id']):r for r in saved}
    for row in recomputed:
        stored=saved_by_id[(row['task_id'],row['candidate_id'])]
        assert stored['structural_guard_allowed']==preanswer_guard(row['features'])
        for key,value in row.items():
            if key.startswith('score_'): assert abs(value-stored[key])<=1e-6,(row['task_id'],key)
    primary=frozen['primary_policy']
    selection=frozen['threshold_selection'][primary]
    decisions=make_decisions(recomputed,primary,'score_'+primary,selection['threshold'])
    metrics=summarize_decisions(decisions)
    reported=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))['primary_metrics']
    for key,value in metrics.items():
        if isinstance(value,(int,float)):
            assert abs(value-reported[key])<=1e-10,key
        else: assert value==reported[key],key
    report={'pass':True,'primary_policy':primary,'development_task_records':960,'new_source_tasks':360,'final_test_tasks':200,
      'candidate_rows_verified':len(development)+len(fresh),'unique_real_cached_responses':len(raw),
      'new_real_cached_responses':len(read_jsonl(ROOT/'data/raw_responses.jsonl')),
      'frozen_code_model_data_hashes_match':True,'family_and_source_splits_disjoint':True,
      'gold_labels_recomputed_from_real_answers':True,'all_full_prompt_tokens_match_backend':True,
      'prompt_cache_identity_digest_options_verified':True,'final_features_recomputed_exactly':True,
      'all_saved_model_scores_and_guard_flags_recomputed':True,'primary_policy_metrics_recomputed':True,
      'done_reason_answer_references':dict(reasons)}
    dump_json(ROOT/'results/verification.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
