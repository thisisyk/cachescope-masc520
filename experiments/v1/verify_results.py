"""Re-score saved responses and audit data/model/policy results without an LLM."""
from pathlib import Path
import sys
import json
import hashlib
from collections import Counter
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cachescope.scoring import is_correct
from cachescope.features import extract_features
from cachescope.inference import make_prompt

def main():
    rows = [json.loads(x) for x in (ROOT/"data/scored_candidates.jsonl").read_text(encoding="utf8").splitlines()]
    pilot_path=ROOT/"data/pilot_candidates.jsonl"
    if pilot_path.exists():
        rows += [json.loads(x) for x in pilot_path.read_text(encoding="utf8").splitlines() if x.strip()]
    length_path=ROOT/"data/length_stress_scored_candidates.jsonl"
    if length_path.exists():
        rows += [json.loads(x) for x in length_path.read_text(encoding="utf8").splitlines() if x.strip()]
    frozen=json.loads((ROOT/"experiment_config.json").read_text())
    raw = {r["cache_key"]:r for r in map(json.loads,(ROOT/"data/raw_responses.jsonl").read_text(encoding="utf8").splitlines())}
    groups = {}
    for row in rows:
        assert row["original_correct"] == is_correct(row["original_answer"],row)
        assert row["compressed_correct"] == is_correct(row["compressed_answer"],row)
        assert row["regression"] == int(row["original_correct"] and not row["compressed_correct"])
        assert row["features"] == extract_features(row["original_context"],row["question"],row["compressed_context"],row["original_tokens"],row["compressed_tokens"])
        for prefix in ["original","compressed"]:
            response = raw[row[f"{prefix}_cache_key"]]
            assert response["model_digest"] == frozen["answering_model_digest"]
            assert response["answer"] == row[f"{prefix}_answer"]
            assert response["prompt_eval_count"] == row[f"{prefix}_tokens"]
            context=row["original_context"] if prefix=="original" else row["compressed_context"]
            prompt=make_prompt(context,row["question"])
            assert response["prompt_sha256"]==hashlib.sha256(prompt.encode()).hexdigest()
        effective_split=row.get("base_split",row["split"])
        prior = groups.setdefault(row["group_id"],effective_split)
        assert prior == effective_split
    assert all(n==2 for n in Counter(r["task_id"] for r in rows).values())
    metrics=json.loads((ROOT/"results/metrics.json").read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in frozen["sha256"].items())
    if length_path.exists():
        protocol=json.loads((ROOT/"results/length_protocol.json").read_text())
        assert hashlib.sha256((ROOT/"length_stress.py").read_bytes()).hexdigest()==protocol["generator_source_sha256"]
        assert hashlib.sha256((ROOT/"data/length_stress_tasks.jsonl").read_bytes()).hexdigest()==protocol["stress_tasks_sha256"]
    for split,policies in metrics["splits"].items():
        n=len({r["task_id"] for r in rows if r["split"]==split})
        assert all(m["n_tasks"]==n for m in policies.values())
    print(json.dumps({"status":"PASS","candidate_records":len(rows),"task_records_including_paired_derivatives":len({r['task_id'] for r in rows}),"paired_length_derivatives":len({r['task_id'] for r in rows if r['split']=='length_stress'}),"genuine_response_references":True,"labels_recomputed":True,"features_recomputed":True,"split_groups_disjoint_or_paired_heldout":True},indent=2))

if __name__ == "__main__":main()
