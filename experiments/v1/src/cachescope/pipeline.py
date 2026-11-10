"""End-to-end experiment, with resumable real-model labeling."""
from __future__ import annotations
import collections
import hashlib
import json
import platform
import time
from pathlib import Path
from .compression import compress_context
from .data import generate_tasks, generate_ood_tasks
from .features import extract_features
from .inference import OllamaAnswerer, QwenTokenizer, ResponseCache, make_prompt
from .scoring import is_correct


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in values:
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(root, n_tasks=1000, n_ood=50):
    root = Path(root)
    for directory in ["data", "results", "models", "figures", "docs"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    main = generate_tasks(seed=520, n_tasks=n_tasks)
    pilot = generate_tasks(seed=520, n_tasks=100, pilot=True)
    ood = generate_ood_tasks(seed=520, n_tasks=n_ood)
    for name, rows in [("pilot", pilot), ("main", main), ("structural_holdout", ood)]:
        write_jsonl(root / "data" / f"{name}_tasks.jsonl", rows)
    groups = {}
    for row in main + ood:
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Group crosses data partitions")
        groups[group] = row["split"]
    dump_json(root / "data" / "split_manifest.json", {
        "seed": 520, "partition_unit": "template wording family", "groups": groups,
        "tasks": {r["task_id"]: r["split"] for r in main + ood},
        "counts": dict(collections.Counter(r["split"] for r in main + ood)),
        "pilot_excluded_from_training": True,
        "limitation": "Five high-level structures occur in every main partition; the separate structural holdout uses three different synthetic formats."})
    return pilot, main, ood


def label_tasks(root, tasks, name, model="qwen2.5-coder:0.5b", limit=None):
    root = Path(root)
    answerer = OllamaAnswerer(model=model)
    metadata = answerer.metadata()
    metadata["python"] = platform.python_version()
    dump_json(root / "results" / "answering_model.json", metadata)
    dump_json(root / "results" / f"{name}_answering_model.json", metadata)
    tokenizer = QwenTokenizer(root / "assets" / "tokenizer.json")
    cache = ResponseCache(root / "data" / "raw_responses.jsonl", answerer)
    tasks = tasks[:limit] if limit else tasks
    rows, original_results, started = [], [], time.perf_counter()
    output = root / "data" / f"{name}_candidates.jsonl"
    # Responses are append-only and resumable; candidate table is rebuilt deterministically.
    with output.open("w", encoding="utf-8") as file:
        for index, task in enumerate(tasks, 1):
            original_prompt = make_prompt(task["context"], task["question"])
            original_tokens = len(tokenizer.encode(original_prompt))
            original, original_cached = cache.get(original_prompt, task_id=task["task_id"], variant="original")
            original_correct = is_correct(original["answer"], task)
            original_results.append(original_correct)
            for ratio in [0.6, 0.3]:
                candidate_id = f"keep_{int(ratio * 100)}"
                begin = time.perf_counter()
                compressed = compress_context(task["context"], task["question"], ratio, tokenizer)
                candidate_prompt = make_prompt(compressed["context"], task["question"])
                compressed_tokens = len(tokenizer.encode(candidate_prompt))
                features = extract_features(task["context"], task["question"], compressed["context"],
                                            original_tokens, compressed_tokens)
                local_preprocess_s = time.perf_counter() - begin
                response, was_cached = cache.get(candidate_prompt, task_id=task["task_id"], variant=candidate_id)
                compressed_correct = is_correct(response["answer"], task)
                row = {**task, "original_context": task["context"], "compressed_context": compressed["context"],
                    "candidate_id": candidate_id, "keep_ratio": ratio,
                    "original_tokens": original_tokens, "compressed_tokens": compressed_tokens,
                    "original_context_tokens": compressed["original_context_tokens"],
                    "compressed_context_tokens": compressed["compressed_context_tokens"],
                    "actual_context_retention": compressed["keep_ratio_actual"],
                    "original_backend_prompt_tokens": original["prompt_eval_count"],
                    "compressed_backend_prompt_tokens": response["prompt_eval_count"],
                    "original_answer": original["answer"], "compressed_answer": response["answer"],
                    "original_correct": original_correct, "compressed_correct": compressed_correct,
                    "regression": int(original_correct and not compressed_correct),
                    "features": features, "original_latency_s": original["latency_s"],
                    "compressed_latency_s": response["latency_s"], "preprocess_latency_s": local_preprocess_s,
                    "original_output_tokens": original["eval_count"], "compressed_output_tokens": response["eval_count"],
                    "original_done_reason": original["done_reason"], "compressed_done_reason": response["done_reason"],
                    "original_cache_key": original["cache_key"], "compressed_cache_key": response["cache_key"]}
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                file.flush()
            if index == 1 or index % 10 == 0 or index == len(tasks):
                print(json.dumps({"stage": name, "completed_tasks": index, "total_tasks": len(tasks),
                    "original_accuracy": sum(original_results)/len(original_results),
                    "observed_regressions": sum(r["regression"] for r in rows),
                    "elapsed_s": round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
    counts = collections.Counter(r["split"] for r in rows)
    summary = {"tasks": len(tasks), "candidate_records": len(rows), "split_candidate_counts": dict(counts),
        "original_accuracy": sum(original_results) / len(original_results),
        "observed_regressions": sum(r["regression"] for r in rows),
        "regressions_by_ratio": {str(ratio): sum(r["regression"] for r in rows if r["keep_ratio"] == ratio) for ratio in [.6,.3]},
        "context_token_range": [min(r["original_context_tokens"] for r in rows), max(r["original_context_tokens"] for r in rows)],
        "local_minus_backend_prompt_tokens": sorted(set(r["original_tokens"]-r["original_backend_prompt_tokens"] for r in rows)),
        "label_source": "Real Ollama model responses, normalized exact-match against predefined synthetic facts",
        "candidate_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    dump_json(root / "results" / f"{name}_labeling_summary.json", summary)
    return rows


def train_and_evaluate(root):
    import torch
    from .models import train_models, load_bundle, predict_scores
    from .evaluation import evaluate_all
    root = Path(root)
    torch.set_num_threads(2)
    records = read_jsonl(root / "data" / "main_candidates.jsonl")
    holdout = root / "data" / "structural_holdout_candidates.jsonl"
    if holdout.exists():
        records += read_jsonl(holdout)
    expected_tasks = read_jsonl(root / "data" / "main_tasks.jsonl")
    if holdout.exists():
        expected_tasks += read_jsonl(root / "data" / "structural_holdout_tasks.jsonl")
    if {r["task_id"] for r in records} != {t["task_id"] for t in expected_tasks}:
        raise ValueError("Candidate observations do not cover the complete frozen task dataset")
    count_by_task = collections.Counter(r["task_id"] for r in records)
    if any(count != 2 for count in count_by_task.values()):
        raise ValueError("Exactly two observed compression candidates are required for every task")
    mismatch = [r["task_id"] for r in records
                if r["original_tokens"] != r["original_backend_prompt_tokens"]
                or r["compressed_tokens"] != r["compressed_backend_prompt_tokens"]]
    if mismatch:
        raise ValueError(f"Local tokenizer disagrees with answering backend: {mismatch[:5]}")
    dump_json(root / "results" / "dataset_integrity.json", {
        "expected_tasks": len(expected_tasks), "observed_candidates": len(records),
        "exactly_two_candidates_per_task": True, "complete_frozen_dataset": True,
        "local_and_backend_full_prompt_token_counts_match": True,
        "pilot_excluded": all(r["split"] != "pilot" for r in records)})
    train_models(records, root / "models")
    names = ["mlp", "lr", "rf", "mlp_seed_521", "mlp_seed_522", "mlp_no_numeric", "mlp_no_overlap"]
    for name in names:
        bundle = load_bundle(root / "models", name)
        start = time.perf_counter()
        scores = predict_scores(bundle, records)
        elapsed = time.perf_counter() - start
        for row, score in zip(records, scores):
            row[f"score_{name}"] = float(score)
        dump_json(root / "results" / f"{name}_inference_timing.json", {
            "n_candidates": len(records), "batch_wall_s": elapsed,
            "batch_amortized_s_per_candidate": elapsed / len(records),
            "limitation": "Batch timing excludes compression and does not establish end-to-end speedup."})
    write_jsonl(root / "data" / "scored_candidates.jsonl", records)
    return evaluate_all(records, root / "results", epsilon=0.02)
