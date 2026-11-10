"""Supplementary paired synthetic length stress test; never trains or tunes.

Usage: python length_stress.py prepare|label|evaluate
The prepare command reads only main task text, tokenizer assets, and this script.
It never reads main model answers, scores, thresholds, or performance metrics.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
VERSION = "1.0"
SEED = 520
N_TASKS = 50
TARGET_TOKENS = 1000
TOKEN_BOUNDS = (900, 1100)
MODEL_NAMES = ["mlp", "lr", "rf", "mlp_seed_521", "mlp_seed_522", "mlp_no_numeric", "mlp_no_overlap"]

# Unrelated descriptive prose: no instructions, numerals, project identifiers,
# people, dates, monetary facts, or vocabulary naming the queried attributes.
# All sentences are synthetic scenery; their order is shuffled without outcomes.
FILLER_SENTENCES = [
    "The quiet corridor curved toward a wide hall filled with soft light.",
    "Beyond the broad window, thin clouds drifted above a distant ridge.",
    "An empty wooden bench stood beside a plain wall with a rough surface.",
    "A gentle breeze moved through the leaves and faded into the still air.",
    "The pale curtain hung loosely beside an open archway facing the courtyard.",
    "A shallow pool reflected the bright sky and the shapes of nearby branches.",
    "The narrow path wound between low shrubs before disappearing behind a hedge.",
    "Soft shadows rested beneath the broad eaves along the edge of the walkway.",
    "The painted ceiling showed faint traces of color beneath its smooth finish.",
    "A woven rug lay across the floor beside a plain ceramic vessel.",
    "The heavy door opened into a quiet room with a high arched ceiling.",
    "Light from the doorway stretched across the tiles and softened near the wall.",
    "The distant hills appeared hazy beneath a thin layer of drifting cloud.",
    "A dry leaf rested beside the threshold while the surrounding air remained still.",
    "The stone steps led toward a sheltered terrace overlooking the open valley.",
    "Faint ripples spread across the water beneath an overhanging branch.",
    "The wall had a coarse texture that became visible under slanting light.",
    "A small alcove held an undecorated bowl and a folded piece of fabric.",
    "The garden path followed a gentle bend around a patch of low foliage.",
    "A soft glow spread over the ceiling and touched the upper edge of the doorway.",
    "The shaded passage felt cool despite the bright sunlight beyond the arch.",
    "A broad stone rested near the stream where the water became shallow.",
    "The hanging fabric formed loose folds that shifted with the passing breeze.",
    "A faint scent of damp earth rose from the soil beneath the shrubs.",
    "The far side of the courtyard remained hidden behind a textured screen.",
    "The smooth floor reflected the lower part of the wall in a blurred glow.",
    "A narrow strip of sky could be seen above the enclosed passage.",
    "The weathered railing followed the terrace and ended beside a dark recess.",
    "A sheltered corner contained a simple stool beneath a wide opening.",
    "The surface of the pool darkened as a cloud passed across the sky.",
    "The quiet room contained little furniture and felt open beneath its tall ceiling.",
    "A thin vine climbed along the outer wall and crossed a shallow groove.",
    "The path became softer where fallen leaves gathered beneath the overhanging branches.",
    "A pale patch of light lay against the far wall near the floor.",
    "The low hedge separated the grassy slope from the sheltered stone walkway.",
    "The arch cast a curved shadow that stretched across the worn tiles.",
    "A faint rustling sound came from the foliage beyond the open window.",
    "The ceramic surface was matte and slightly uneven beneath the soft light.",
    "A distant line of trees appeared dark against the brighter background of the hills.",
    "The edge of the curtain brushed the floor and settled into a shallow fold.",
    "A quiet recess opened beside the passage and offered a view of the garden.",
    "The air above the shallow water felt cooler than the surrounding terrace.",
    "A carved pattern ran along the railing without interrupting its smooth upper surface.",
    "The broad leaves overlapped and formed a deep patch of shade beneath the branches.",
    "A narrow doorway connected the sheltered hall with a brighter open room.",
    "The damp stones near the stream looked darker than the dry stones above them.",
    "The valley widened beyond the ridge and faded into a soft band of haze.",
    "A strip of woven cloth rested on a low shelf beside an empty basket.",
    "The quiet atmosphere was broken only by the soft movement of leaves outside.",
    "The painted surface looked smooth from afar but revealed gentle variations nearby.",
    "A shallow groove in the floor guided a thin reflection toward the doorway.",
    "The open courtyard allowed the breeze to pass freely between the surrounding walls.",
    "A dark band of shade lay beneath the ledge and softened its lower edge.",
    "The distant clouds appeared almost still above the broad outline of the hills.",
    "A plain wooden frame surrounded the window and contrasted with the pale wall.",
    "The sheltered path continued past the garden and disappeared beneath an arch.",
    "A loose branch cast a delicate shadow across the quiet surface of the pool.",
    "The fabric on the shelf had a coarse weave and a muted natural color.",
    "The empty hall seemed spacious as the light shifted across its smooth floor.",
    "A gentle curve in the railing followed the outer edge of the terrace.",
    "The sound of moving water grew faint beyond the low garden wall.",
    "A soft layer of haze concealed the smallest details of the distant landscape.",
    "The plain vessel stood near the wall where a narrow beam of light fell.",
    "The leaves nearest the window trembled while the deeper foliage remained still.",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def prepare(root=ROOT):
    """Select sources by ID and add unrelated prose using text lengths only."""
    from cachescope.inference import QwenTokenizer
    root = Path(root)
    source_path = root / "data" / "main_tasks.jsonl"
    tokenizer_path = root / "assets" / "tokenizer.json"
    source = sorted((task for task in read_jsonl(source_path) if task["split"] == "test"), key=lambda task: task["task_id"])
    if len(source) < N_TASKS:
        raise ValueError("At least 50 main-test source tasks are required")
    selected = source[:N_TASKS]
    tokenizer = QwenTokenizer(tokenizer_path)
    forbidden = {"owner", "release", "date", "approved", "budget", "project", "current"}
    filler_words = set(re.findall(r"[a-z]+", " ".join(FILLER_SENTENCES).lower()))
    if filler_words & forbidden or re.search(r"\d", " ".join(FILLER_SENTENCES)):
        raise AssertionError("Filler contains a queried attribute or numeric fact")
    rows = []
    lengths = []
    details = []
    for index, base in enumerate(selected):
        original = base["context"]
        if len(tokenizer.encode(original)) >= TOKEN_BOUNDS[0]:
            raise ValueError("Source already exceeds intended short-context baseline")
        rng = random.Random(SEED + index)
        order = list(FILLER_SENTENCES)
        rng.shuffle(order)
        prefix = index % 2 == 0
        appendix = []
        final = original
        length = len(tokenizer.encode(final))
        for sentence in order:
            appendix.append(sentence)
            prose = " ".join(appendix)
            final = (prose + "\n\n" + original) if prefix else (original + "\n\n" + prose)
            length = len(tokenizer.encode(final))
            if length >= TARGET_TOKENS:
                break
        if not TOKEN_BOUNDS[0] <= length <= TOKEN_BOUNDS[1]:
            raise ValueError(f"Cannot reach context token bounds for {base['task_id']}: {length}")
        # The full original must remain byte-for-byte intact at one boundary.
        if (prefix and not final.endswith(original)) or (not prefix and not final.startswith(original)):
            raise AssertionError("Original source fact text was altered")
        question_entity = re.search(r"for project (.+)\?$", base["question"])
        if question_entity and question_entity.group(1).lower() in prose.lower():
            raise AssertionError("Filler mentions target project identifier")
        row = {**base, "task_id": "length-" + base["task_id"], "context": final,
               "split": "length_stress", "base_task_id": base["task_id"],
               "base_source_id": base["source_id"], "source_id": base["source_id"],
               "base_split": "test", "paired_derivative": True,
               "data_origin": "synthetic paired length-stress derivative of a main-test source",
               "length_stress_version": VERSION, "appendix_position": "prefix" if prefix else "suffix",
               "base_context_tokens": len(tokenizer.encode(original)), "length_context_tokens": length}
        for field in ("question", "answer", "answer_aliases", "group_id"):
            if row[field] != base[field]:
                raise AssertionError(f"Immutable source field changed: {field}")
        rows.append(row)
        lengths.append(length)
        details.append({"task_id": row["task_id"], "base_task_id": base["task_id"],
                        "base_source_id": base["source_id"], "group_id": base["group_id"],
                        "appendix_position": row["appendix_position"], "context_tokens": length,
                        "original_source_text_preserved": True,
                        "base_context_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest()})
    path = root / "data" / "length_stress_tasks.jsonl"
    payload = jsonl_bytes(rows)
    if path.exists() and path.read_bytes() != payload and (root / "data" / "length_stress_candidates.jsonl").exists():
        raise RuntimeError("Refusing to change stress inputs after labeling; use a separate experiment directory")
    path.write_bytes(payload)
    protocol = {
        "version": VERSION, "seed": SEED, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "first 50 main-test tasks sorted lexicographically by task_id",
        "selection_uses_model_outcomes": False, "padding_uses_model_outcomes": False,
        "prepare_reads": ["data/main_tasks.jsonl", "assets/tokenizer.json", "length_stress.py"],
        "supplementary_protocol": True,
        "padding_rule": "shuffle fixed irrelevant prose with seed 520 + subset index; append whole sentences until context reaches 1000 local Qwen tokens",
        "appendix_assignment": "alternating sorted-source index; even prefix, odd suffix",
        "n_paired_derivatives": len(rows), "n_new_independent_sources": 0,
        "n_base_test_sources": len({row["base_source_id"] for row in rows}),
        "n_inherited_template_groups": len({row["group_id"] for row in rows}),
        "appendix_counts": dict(collections.Counter(row["appendix_position"] for row in rows)),
        "target_context_tokens": TARGET_TOKENS, "allowed_context_token_bounds": list(TOKEN_BOUNDS),
        "actual_context_token_range": [min(lengths), max(lengths)],
        "main_tasks_sha256": sha256(source_path), "tokenizer_sha256": sha256(tokenizer_path),
        "generator_source_sha256": sha256(__file__), "stress_tasks_sha256": sha256(path),
        "no_training_or_threshold_tuning": True,
        "threshold_source": "results/selected_thresholds.json from the original main validation set",
        "inference_settings": "reuse the frozen answering model, prompt and options without changes",
        "scope": "approximately 1000-context-token synthetic irrelevant-padding stress, not real-world long-document generalization",
        "dependence": "These reuse 50 existing main-test source facts and inherit their group_id; do not combine them as independent sources with the main test.",
        "tasks": details,
    }
    write_json(root / "results" / "length_protocol.json", protocol)
    print(json.dumps({key: protocol[key] for key in ["n_paired_derivatives", "n_new_independent_sources", "actual_context_token_range", "appendix_counts"]}), flush=True)
    return rows


def _validated_tasks(root):
    tasks_path = root / "data" / "length_stress_tasks.jsonl"
    protocol_path = root / "results" / "length_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(tasks_path) != protocol["stress_tasks_sha256"]:
        raise ValueError("Length-stress task file no longer matches its frozen protocol")
    tasks = read_jsonl(tasks_path)
    if len(tasks) != N_TASKS or len({task["task_id"] for task in tasks}) != N_TASKS:
        raise ValueError("Exactly 50 unique stress tasks required")
    if any(task["split"] != "length_stress" or not task["paired_derivative"] for task in tasks):
        raise ValueError("Stress tasks must be labeled as paired derivatives")
    return tasks, protocol


def label(root=ROOT):
    from cachescope.inference import OllamaAnswerer
    from cachescope.pipeline import label_tasks
    root = Path(root)
    tasks, _ = _validated_tasks(root)
    frozen = json.loads((root / "results" / "main_answering_model.json").read_text(encoding="utf-8"))
    current = OllamaAnswerer(model=frozen["model"]).metadata()
    for field in ("model", "model_digest", "options", "system_instruction", "prompt_template"):
        if current[field] != frozen[field]:
            raise ValueError(f"Answering model configuration drift: {field}")
    return label_tasks(root, tasks, "length_stress", model=frozen["model"])


def evaluate(root=ROOT):
    from cachescope.models import load_bundle, predict_scores
    from cachescope.evaluation import make_decisions, summarize_decisions, grouped_bootstrap
    import torch
    torch.set_num_threads(2)
    root = Path(root)
    tasks, protocol = _validated_tasks(root)
    task_map = {task["task_id"]: task for task in tasks}
    rows = read_jsonl(root / "data" / "length_stress_candidates.jsonl")
    counts = collections.Counter(row["task_id"] for row in rows)
    if set(counts) != set(task_map) or any(count != 2 for count in counts.values()):
        raise ValueError("Every stress task requires exactly two observed candidates")
    for row in rows:
        task = task_map[row["task_id"]]
        for field in ("split", "question", "answer", "answer_aliases", "group_id"):
            if row[field] != task[field]:
                raise ValueError(f"Candidate/source mismatch: {field}")
        if row["original_context"] != task["context"]:
            raise ValueError("Candidate source context changed")
        if row["original_tokens"] != row["original_backend_prompt_tokens"] or row["compressed_tokens"] != row["compressed_backend_prompt_tokens"]:
            raise ValueError("Backend/local prompt token count mismatch")
    thresholds_path = root / "results" / "selected_thresholds.json"
    threshold_hash = sha256(thresholds_path)
    selections = json.loads(thresholds_path.read_text(encoding="utf-8"))
    for name in MODEL_NAMES:
        selection = selections[name]
        if selection["selection_split"] != "validation" or selection["score_key"] != "score_" + name:
            raise ValueError(f"Expected previously fixed validation threshold for {name}")
        scores = predict_scores(load_bundle(root / "models", name), rows)
        for row, score in zip(rows, scores):
            row["score_" + name] = float(score)
    (root / "data" / "length_stress_scored_candidates.jsonl").write_bytes(jsonl_bytes(rows))
    specs = [(policy, None, None) for policy in ("original", "always_60", "always_30", "simple_rule_gate")]
    specs += [(name, selections[name]["score_key"], selections[name]["threshold"]) for name in MODEL_NAMES]
    all_decisions, csv_rows, metrics = [], [], {}
    for policy, score_key, threshold in specs:
        decisions = make_decisions(rows, policy, score_key, threshold)
        for decision in decisions:
            decision["base_task_id"] = task_map[decision["task_id"]]["base_task_id"]
            decision["paired_derivative"] = True
        summary = summarize_decisions(decisions)
        uncertainty = grouped_bootstrap(decisions, n_bootstrap=500, seed=SEED)
        metrics[policy] = {**summary, "paired_group_bootstrap": uncertainty}
        csv_rows.append({"split": "length_stress", "policy": policy, **summary,
                         "accuracy_delta_ci_low": uncertainty["accuracy_delta_95_ci"][0],
                         "accuracy_delta_ci_high": uncertainty["accuracy_delta_95_ci"][1],
                         "token_reduction_ci_low": uncertainty["input_token_reduction_ratio_95_ci"][0],
                         "token_reduction_ci_high": uncertainty["input_token_reduction_ratio_95_ci"][1]})
        all_decisions.extend(decisions)
    if threshold_hash != sha256(thresholds_path):
        raise AssertionError("Original validation thresholds changed during stress evaluation")
    report = {
        "split": "length_stress", "n_paired_derivatives": N_TASKS, "n_new_independent_sources": 0,
        "unit_of_evaluation": "one padded derivative per existing main-test source task",
        "threshold_selection": "none on stress data; reuse original main-validation operating points",
        "selected_thresholds_sha256": threshold_hash,
        "locked_thresholds": {name: selections[name]["threshold"] for name in MODEL_NAMES},
        "protocol_sha256": sha256(root / "results" / "length_protocol.json"),
        "context_token_range": protocol["actual_context_token_range"],
        "token_count_integrity": "all local complete-prompt counts equal backend prompt_eval_count",
        "scope": protocol["scope"], "dependence": protocol["dependence"],
        "policies": metrics,
        "notes": ["No risk model or scaler was fitted or updated.",
                  "No threshold was selected using these outcomes.",
                  "Appendices contain unrelated synthetic descriptive prose with no queried attributes.",
                  "Padding creates a controlled stress test, not natural long-document evidence.",
                  "Bootstrap resamples inherited template groups; do not treat the paired main and stress sets as independent.",
                  "Stored answering-call time is affected by order and cache, not a causal speedup measurement."],
    }
    write_json(root / "results" / "length_stress_metrics.json", report)
    (root / "results" / "length_stress_task_policy_decisions.jsonl").write_bytes(jsonl_bytes(all_decisions))
    with (root / "results" / "length_stress_policy_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"stage": "length_stress_evaluate", "tasks": N_TASKS, "policies": len(metrics),
                      "thresholds_unchanged": True}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "label", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "label": label, "evaluate": evaluate}[args.stage](ROOT)


if __name__ == "__main__":
    main()
