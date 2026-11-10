"""Measure real local compression/feature/model/selection overhead on pilot inputs."""
from pathlib import Path
import json
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.compression import compress_context
from cachescope.features import extract_features
from cachescope.inference import QwenTokenizer,make_prompt
from cachescope.models import load_bundle,predict_scores

def main():
    import torch
    torch.set_num_threads(2)
    tokenizer=QwenTokenizer(ROOT/'assets/tokenizer.json')
    tasks=[json.loads(x) for x in (ROOT/'data/pilot_tasks.jsonl').read_text(encoding='utf8').splitlines()][:20]
    thresholds=json.loads((ROOT/'results/selected_thresholds.json').read_text())
    results={}
    for name in ['mlp','lr','rf']:
        bundle=load_bundle(ROOT/'models',name)
        threshold=thresholds[name]['threshold']
        def decision(task):
            original_tokens=len(tokenizer.encode(make_prompt(task['context'],task['question'])))
            candidate_features=[]
            candidate_tokens=[]
            for ratio in [.6,.3]:
                c=compress_context(task['context'],task['question'],ratio,tokenizer)
                tokens=len(tokenizer.encode(make_prompt(c['context'],task['question'])))
                candidate_features.append(extract_features(task['context'],task['question'],c['context'],original_tokens,tokens))
                candidate_tokens.append(tokens)
            scores=predict_scores(bundle,candidate_features)
            accepted=[tokens for tokens,score in zip(candidate_tokens,scores) if score<=threshold and tokens<original_tokens]
            return min(accepted) if accepted else original_tokens
        for t in tasks[:5]:decision(t)
        elapsed=[]
        for repeat in range(5):
            for task in tasks:
                start=time.perf_counter()
                decision(task)
                elapsed.append(time.perf_counter()-start)
        elapsed.sort()
        results[name]={'n_timed_decisions':len(elapsed),'mean_ms':1000*statistics.mean(elapsed),
            'median_ms':1000*statistics.median(elapsed),'p95_ms':1000*elapsed[int(.95*(len(elapsed)-1))]}
    obj={'scope':'Warm local full decision: both compressions, complete-prompt tokenization, feature extraction, model prediction, candidate selection. Excludes model loading and answering-model calls.',
         'input_selection':'First 20 pilot tasks, repeated five times after five warmups per model',
         'torch_cpu_threads':2,'results':results,'causal_end_to_end_speedup_claim':False}
    (ROOT/'results/decision_overhead.json').write_text(json.dumps(obj,indent=2),encoding='utf8')
    print(json.dumps(obj,indent=2))

if __name__=='__main__':main()
