"""Choose a compression with saved risk models; no answering-model call needed."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cachescope.compression import compress_context
from cachescope.features import extract_features
from cachescope.inference import QwenTokenizer, make_prompt
from cachescope.models import load_bundle, predict_scores

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-file", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--model", choices=["mlp", "lr", "rf"], default="mlp")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    context = args.context_file.read_text(encoding="utf-8")
    tokenizer = QwenTokenizer(ROOT / "assets/tokenizer.json")
    original_tokens = len(tokenizer.encode(make_prompt(context, args.question)))
    bundle = load_bundle(ROOT / "models", args.model)
    thresholds = json.loads((ROOT / "results/selected_thresholds.json").read_text())
    threshold = thresholds[args.model]["threshold"]
    candidates = []
    for ratio in [.6,.3]:
        compressed = compress_context(context, args.question, ratio, tokenizer)
        tokens = len(tokenizer.encode(make_prompt(compressed["context"],args.question)))
        features = extract_features(context,args.question,compressed["context"],original_tokens,tokens)
        score = float(predict_scores(bundle,[features])[0])
        candidates.append({"retention_target":ratio,"context":compressed["context"],
                           "input_tokens":tokens,"risk_score":score,
                           "admissible":score<=threshold and tokens<original_tokens})
    passed = [c for c in candidates if c["admissible"]]
    selected = min(passed,key=lambda c:c["input_tokens"]) if passed else None
    result={"model":args.model,"validation_selected_threshold":threshold,
            "original_input_tokens":original_tokens,"selected":"compressed" if selected else "original",
            "selected_input_tokens":selected["input_tokens"] if selected else original_tokens,
            "selected_context":selected["context"] if selected else context,
            "candidates":candidates,"interpretation":"Uncalibrated conditional risk scores; no guarantee of correctness."}
    output=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(output+"\n",encoding="utf-8")
    print(output)

if __name__ == "__main__":main()
