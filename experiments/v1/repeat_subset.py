"""Repeat a predeclared pilot subset to assess fixed-settings output variability."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.inference import OllamaAnswerer,ResponseCache,make_prompt
from cachescope.scoring import is_correct,normalize_answer

def main():
    rows=[json.loads(x) for x in (ROOT/'data/pilot_candidates.jsonl').read_text(encoding='utf8').splitlines()]
    ids=sorted({r['task_id'] for r in rows})[:20]
    answerer=OllamaAnswerer()
    metadata=answerer.metadata()
    cache=ResponseCache(ROOT/'data/raw_responses.jsonl',answerer)
    repeated=[]
    for index,task_id in enumerate(ids,1):
        task_rows=[r for r in rows if r['task_id']==task_id]
        reference=task_rows[0]
        variants=[('original',reference['original_context'],reference['original_answer'])]
        variants += [(r['candidate_id'],r['compressed_context'],r['compressed_answer']) for r in task_rows]
        for variant,context,prior in variants:
            result,cached=cache.get(make_prompt(context,reference['question']),task_id=task_id,variant=variant,repeat=1)
            repeated.append({'task_id':task_id,'variant':variant,'original_observation_answer':prior,
                'repeated_answer':result['answer'],'answer_changed':prior!=result['answer'],
                'normalized_answer_changed':normalize_answer(prior,reference['answer_type'])!=normalize_answer(result['answer'],reference['answer_type']),
                'original_correct':is_correct(prior,reference),'repeated_correct':is_correct(result['answer'],reference),
                'cache_key':result['cache_key']})
        if index%5==0:print(f'Repeated pilot tasks {index}/20',flush=True)
    (ROOT/'data/repeated_pilot_responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in repeated),encoding='utf8')
    summary={'selection':'Lexicographically first 20 pilot task IDs, fixed before formal results',
      'n_tasks':20,'n_answer_comparisons':len(repeated),'same_model_same_generation_settings':True,
      'answer_text_changes':sum(r['answer_changed'] for r in repeated),
      'normalized_answer_changes':sum(r['normalized_answer_changed'] for r in repeated),
      'correctness_changes':sum(r['original_correct']!=r['repeated_correct'] for r in repeated),
      'used_for_training_or_threshold_selection':False,'model_metadata':metadata}
    (ROOT/'results/output_variability.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='model_metadata'},indent=2),flush=True)

if __name__=='__main__':main()
