"""Run from the project root after installing requirements and pulling Ollama model."""
from pathlib import Path
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from cachescope.pipeline import prepare, label_tasks, train_and_evaluate, read_jsonl

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "label", "train", "all"])
    parser.add_argument("--model", default="qwen2.5-coder:0.5b")
    parser.add_argument("--limit", type=int, default=None, help="Development-only task limit; formal run must omit this")
    parser.add_argument("--n-tasks", type=int, default=500, help="Formal task count (default: 500)")
    args = parser.parse_args()
    if args.stage in ["prepare", "all"]:
        prepare(ROOT, n_tasks=args.n_tasks)
        print(f"Prepared 100 pilot + {args.n_tasks} main + 50 structural-holdout tasks.", flush=True)
    if args.stage in ["pilot", "all"]:
        tasks = read_jsonl(ROOT / "data" / "pilot_tasks.jsonl")
        label_tasks(ROOT, tasks, "pilot", args.model, args.limit)
    if args.stage in ["label", "all"]:
        for name in ["main", "structural_holdout"]:
            tasks = read_jsonl(ROOT / "data" / f"{name}_tasks.jsonl")
            label_tasks(ROOT, tasks, name, args.model, args.limit)
    if args.stage in ["train", "all"]:
        train_and_evaluate(ROOT)
        print("Training and policy evaluation completed.", flush=True)

if __name__ == "__main__":
    main()
