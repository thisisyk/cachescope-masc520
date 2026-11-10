"""Choose a compressed prompt using the frozen V2 model before any answer call."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cachescope.compression import compress_context
from cachescope.features import extract_features
from cachescope.inference import QwenTokenizer,make_prompt
from cachescope.models import load_bundle,predict_scores

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-file',type=Path,required=True)
    parser.add_argument('--question',required=True)
    parser.add_argument('--model',choices=['mlp','lr','rf'],default='mlp')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    context=args.context_file.read_text(encoding='utf-8')
    tokenizer=QwenTokenizer(ROOT/'assets/tokenizer.json')
    original_tokens=len(tokenizer.encode(make_prompt(context,args.question)))
    bundle=load_bundle(ROOT/'models',args.model)
    frozen=json.loads((ROOT/'results/frozen_model_selection.json').read_text(encoding='utf-8'))
    threshold=frozen['threshold_selection'][args.model]['threshold']
    candidates=[]
    for ratio in [.6,.3]:
        compressed=compress_context(context,args.question,ratio,tokenizer)
        tokens=len(tokenizer.encode(make_prompt(compressed['context'],args.question)))
        features=extract_features(context,args.question,compressed['context'],original_tokens,tokens)
        score=float(predict_scores(bundle,[features])[0])
        candidates.append({'candidate_id':f'keep_{int(ratio*100)}','context':compressed['context'],'input_tokens':tokens,'risk_score':score,'admissible':score<=threshold and tokens<original_tokens})
    passed=[c for c in candidates if c['admissible']]
    selected=min(passed,key=lambda c:(c['input_tokens'],c['candidate_id'])) if passed else None
    result={'model':args.model,'validation_selected_threshold':threshold,'original_input_tokens':original_tokens,
            'selected':selected['candidate_id'] if selected else 'original',
            'selected_input_tokens':selected['input_tokens'] if selected else original_tokens,
            'selected_context':selected['context'] if selected else context,'candidates':candidates,
            'interpretation':'Uncalibrated conditional risk; empirical validation threshold, no correctness guarantee.'}
    output=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__': main()
