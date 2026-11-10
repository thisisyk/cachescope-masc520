"""Apply the calibration-selected pre-answer hybrid to a new input."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.compression import compress_context
from cachescope.inference import QwenTokenizer,make_prompt
from cachescope.evaluation import choose_candidate
from train_v3 import score_records,verify_freeze

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-file',type=Path,required=True)
    parser.add_argument('--question',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    frozen=json.loads((ROOT/'results/frozen_selection.json').read_text(encoding='utf-8'))
    verify_freeze(frozen)
    primary=frozen['primary_policy']
    threshold=frozen['threshold_selection'][primary]['threshold']
    context=args.context_file.read_text(encoding='utf-8')
    tokenizer=QwenTokenizer(ROOT/'assets/tokenizer.json')
    original_tokens=len(tokenizer.encode(make_prompt(context,args.question)))
    rows=[]
    for ratio in [.6,.3]:
        compressed=compress_context(context,args.question,ratio,tokenizer)
        rows.append({'original_context':context,'question':args.question,
          'compressed_context':compressed['context'],'original_tokens':original_tokens,
          'compressed_tokens':len(tokenizer.encode(make_prompt(compressed['context'],args.question))),
          'candidate_id':f'keep_{int(ratio*100)}','keep_ratio':ratio})
    rows=score_records(rows)
    selected=choose_candidate(rows,'score_'+primary,threshold)
    result={'primary_policy':primary,'policy_is_hybrid':True,'threshold':threshold,
      'original_input_tokens':original_tokens,'selected':selected['candidate_id'] if selected else 'original',
      'selected_input_tokens':selected['compressed_tokens'] if selected else original_tokens,
      'selected_context':selected['compressed_context'] if selected else context,
      'candidates':[{'candidate_id':r['candidate_id'],'input_tokens':r['compressed_tokens'],
        'risk_score':r['score_'+primary],'structural_guard_allowed':r['structural_guard_allowed'],
        'compressed_context':r['compressed_context']} for r in rows],
      'answering_model_called':False,
      'interpretation':'Learned risk plus explicit JSON structural check. Scores are not correctness guarantees.'}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
if __name__=='__main__': main()
