"""Two-phase CacheScope V2 training and genuinely held-out policy evaluation.

The train command never opens the fresh-test candidate file. All previous V1
observations are explicitly development data in V2, including former test/OOD
rows. Selection and early stopping use only the new validation partition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler

from cachescope import models
from cachescope.evaluation import (
    make_decisions, summarize_decisions, select_threshold, grouped_bootstrap,
    group_tasks,
)
from cachescope.features import extract_features, FEATURE_NAMES
from cachescope.pipeline import read_jsonl, write_jsonl, dump_json

MODEL_NAMES = ["mlp", "lr", "rf", "mlp_seed_521", "mlp_seed_522", "mlp_v1features"]
EPSILON = 0.025
PRIMARY_SEED = 520
V1 = ROOT.parent / "CacheScope_MASC520"
BASELINE = ROOT / "assets/v1_baseline"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_feature_module():
    spec = importlib.util.spec_from_file_location("cachescope_v1_fixed_features", BASELINE / "features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refreshed(rows):
    result = []
    for row in rows:
        r = dict(row)
        r["features"] = extract_features(
            r["original_context"], r["question"], r["compressed_context"],
            r["original_tokens"], r["compressed_tokens"],
        )
        result.append(r)
    return result


def old_matrix(rows):
    module = old_feature_module()
    fs = [module.extract_features(r["original_context"], r["question"],
          r["compressed_context"], r["original_tokens"], r["compressed_tokens"]) for r in rows]
    return np.asarray([[f[n] for n in module.FEATURE_NAMES] for f in fs], dtype=np.float64)


def validate_rows(rows, allowed_splits):
    if not rows or {r["split"] for r in rows} - set(allowed_splits):
        raise ValueError("Unexpected or empty data partition")
    grouped = group_tasks(rows)
    for task_id, variants in grouped.items():
        if len(variants) != 2 or {r["candidate_id"] for r in variants} != {"keep_60", "keep_30"}:
            raise ValueError(f"Exactly two prescribed candidates required: {task_id}")
        for r in variants:
            if r["original_tokens"] != r["original_backend_prompt_tokens"] or r["compressed_tokens"] != r["compressed_backend_prompt_tokens"]:
                raise ValueError("Tokenizer/backend mismatch")
            if int(r["regression"]) != int(bool(r["original_correct"]) and not bool(r["compressed_correct"])):
                raise ValueError("Incorrect observed regression label")
    models._validate_groups(rows)
    return grouped


def fit_v1_feature_ablation(rows, out_dir):
    """Same expanded train/validation data, original twenty feature definitions."""
    import torch
    train = [r for r in rows if r["split"] == "train" and r["original_correct"]]
    validation = [r for r in rows if r["split"] == "validation" and r["original_correct"]]
    x, xv = old_matrix(train), old_matrix(validation)
    y = np.asarray([r["regression"] for r in train], dtype=int)
    yv = np.asarray([r["regression"] for r in validation], dtype=int)
    scaler = StandardScaler().fit(x)
    model, history, info = models._train_mlp(
        scaler.transform(x), y, scaler.transform(xv), yv,
        seed=PRIMARY_SEED, max_epochs=100, patience=10,
    )
    checkpoint = {
        "state_dict": model.state_dict(), "feature_names": list(old_feature_module().FEATURE_NAMES),
        "seed": PRIMARY_SEED, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "scaler_var": scaler.var_.tolist(),
        "scaler_n_samples": int(scaler.n_samples_seen_),
        "score_interpretation": "Uncalibrated conditional risk; V1 features, V2 development data",
    }
    torch.save(checkpoint, out_dir / "mlp_v1features.pt")
    dump_json(out_dir / "mlp_v1features_history.json", history)
    dump_json(out_dir / "mlp_v1features_training.json", info)
    return info


def score(rows, names=MODEL_NAMES, include_frozen_v1=False):
    for name in names:
        bundle = models.load_bundle(ROOT / "models", name)
        scores = models.predict_scores(bundle, old_matrix(rows) if name == "mlp_v1features" else rows)
        for r, s in zip(rows, scores):
            r[f"score_{name}"] = float(s)
    if include_frozen_v1:
        bundle = models.load_bundle(BASELINE, "mlp")
        scores = models.predict_scores(bundle, old_matrix(rows))
        for r, s in zip(rows, scores):
            r["score_frozen_v1_mlp"] = float(s)
    return rows


def train():
    import torch
    torch.set_num_threads(2)
    frozen_path = ROOT / "results/frozen_model_selection.json"
    if frozen_path.exists():
        raise RuntimeError("Selection is already frozen. Preserve it; do not silently re-tune V2.")
    paths = [ROOT / "data/development_candidates.jsonl", ROOT / "data/new_train_validation_candidates.jsonl"]
    development, fresh = [read_jsonl(p) for p in paths]
    if any(r["split"] != "train" for r in development):
        raise ValueError("Every previous V1 observation must be declared development/train")
    if {r["split"] for r in fresh} != {"train", "validation"}:
        raise ValueError("Fresh train/validation file must not contain test observations")
    n_fresh = Counter(r["split"] for r in fresh)
    if n_fresh != {"train": 240, "validation": 160}:
        raise ValueError(f"Expected 120 fresh train and 80 validation tasks: {n_fresh}")
    rows = refreshed(development + fresh)
    validate_rows(rows, {"train", "validation"})
    report = models.train_models(rows, ROOT / "models", seeds=(520, 521, 522), run_ablations=False)
    report["v1_feature_ablation"] = fit_v1_feature_ablation(rows, ROOT / "models")
    # Compute only validation scores for operating-point selection.
    validation = score([r for r in rows if r["split"] == "validation"])
    selections = {name: select_threshold(validation, f"score_{name}", EPSILON) for name in MODEL_NAMES}
    write_jsonl(ROOT / "results/validation_scored_candidates.jsonl", validation)
    dump_json(ROOT / "results/v2_training_report.json", report)
    v1_threshold_path = BASELINE / "selected_thresholds.json"
    v1_selection = json.loads(v1_threshold_path.read_text(encoding="utf-8"))["mlp"]
    files = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for relative in ("experiment_protocol.json", "data/new_tasks.jsonl", "data/source_registry.jsonl", "data/new_data_audit.json"):
        files[relative] = sha(ROOT / relative)
    files.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "models").glob("*") if p.is_file()})
    files.update({str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts})
    files.update({str(p.relative_to(ROOT)): sha(p) for p in BASELINE.glob("*") if p.is_file() and p.is_relative_to(ROOT)})
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_mlp_seed": PRIMARY_SEED, "primary_model": "mlp",
        "selection_split": "new validation only", "test_file_opened_during_training": False,
        "threshold_selection": selections, "epsilon_empirical_not_guaranteed": EPSILON,
        "old_test_reuse": "All V1 test/OOD/length observations are now development data; V1 holdout claims do not carry over to V2.",
        "counts": {s: {"tasks": len({r["task_id"] for r in rows if r["split"] == s}),
                       "candidates": sum(r["split"] == s for r in rows)} for s in ("train", "validation")},
        "feature_names": list(FEATURE_NAMES), "files_sha256": files,
        "frozen_v1_comparator": {
            "threshold": v1_selection["threshold"],
            "feature_module_sha256": sha(BASELINE / "features.py"),
            "model_sha256": sha(BASELINE / "mlp_seed_520.pt"),
            "selection_file_sha256": sha(v1_threshold_path),
        },
        "predeclared_test_success_criteria": {
            "input_token_reduction_ratio_min": 0.20,
            "accuracy_delta_min": -0.025,
            "harmful_accepted_per_task_max": 0.05,
            "comparison": "Report paired MLP vs always_60 regardless of outcome; seed 520 is never replaced with a better test seed.",
        },
        "test_expected_tasks": 160,
    }
    dump_json(frozen_path, frozen)
    print(json.dumps({"stage": "train_frozen", "counts": frozen["counts"],
          "validation_operating_points": selections}, ensure_ascii=False), flush=True)
    return frozen


def verify_freeze(frozen):
    for relative, expected in frozen["files_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Changed after model selection was frozen: {relative}")
    old = frozen["frozen_v1_comparator"]
    if sha(BASELINE / "features.py") != old["feature_module_sha256"] or sha(BASELINE / "mlp_seed_520.pt") != old["model_sha256"]:
        raise ValueError("Frozen V1 comparator changed")
    if sha(BASELINE / "selected_thresholds.json") != old["selection_file_sha256"]:
        raise ValueError("V1 comparator threshold changed")


def paired_comparison(a, b, n_bootstrap=2000, seed=1111520):
    aa, bb = {d["task_id"]: d for d in a}, {d["task_id"]: d for d in b}
    if aa.keys() != bb.keys():
        raise ValueError("Paired policies cover different tasks")
    keys = sorted(aa)
    wins = sum(aa[k]["correct"] and not bb[k]["correct"] for k in keys)
    losses = sum(bb[k]["correct"] and not aa[k]["correct"] for k in keys)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    grouped = defaultdict(list)
    for k in keys:
        grouped[aa[k]["group_id"]].append(k)
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        sampled = [k for g in rng.choice(groups, len(groups), replace=True) for k in grouped[g]]
        diffs.append(sum(int(aa[k]["correct"]) - int(bb[k]["correct"]) for k in sampled) / len(sampled))
    return {
        "n_tasks": len(keys), "a_correct_b_wrong": wins, "a_wrong_b_correct": losses,
        "accuracy_difference_a_minus_b": (wins - losses) / len(keys),
        "accuracy_difference_group_bootstrap_95_ci": np.quantile(diffs, [.025, .975]).tolist(),
        "bootstrap_group_count": len(groups), "bootstrap_repetitions": n_bootstrap,
        "exact_mcnemar_two_sided_p": p,
        "mcnemar_limitation": "Exact binomial McNemar is per task and does not adjust for template clustering; the group bootstrap addresses that dependence.",
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate():
    import torch
    torch.set_num_threads(2)
    frozen_path = ROOT / "results/frozen_model_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verify_freeze(frozen)
    test_file = ROOT / "data/new_test_candidates.jsonl"
    test = refreshed(read_jsonl(test_file))
    tasks = validate_rows(test, {"test"})
    if len(tasks) != frozen["test_expected_tasks"]:
        raise ValueError("Incomplete fresh test observations")
    # Check source/template separation only now, after all choices are locked.
    development = read_jsonl(ROOT / "data/development_candidates.jsonl") + read_jsonl(ROOT / "data/new_train_validation_candidates.jsonl")
    for key in ("task_id", "source_id", "group_id"):
        overlap = {r[key] for r in test} & {r[key] for r in development}
        if overlap:
            raise ValueError(f"Fresh test {key} overlaps development: {sorted(overlap)[:3]}")
    test = score(test, include_frozen_v1=True)
    write_jsonl(ROOT / "results/fresh_test_scored_candidates.jsonl", test)
    specs = [(p, None, None) for p in ("original", "always_60", "always_30", "simple_rule_gate")]
    specs += [(name, f"score_{name}", frozen["threshold_selection"][name]["threshold"]) for name in MODEL_NAMES]
    specs += [("frozen_v1_mlp", "score_frozen_v1_mlp", frozen["frozen_v1_comparator"]["threshold"])]
    report = {
        "scope": "Fresh synthetic factual-extraction test; eight observed high-level structures with new template families and independent source records",
        "test_tasks": len(tasks), "test_groups": len({r["group_id"] for r in test}),
        "test_candidate_file_sha256": sha(test_file), "model_selection_file_sha256": sha(frozen_path),
        "primary_model": "mlp", "primary_seed": PRIMARY_SEED,
        "operating_points_selected_before_test_file_opened": True,
        "success_criteria": frozen["predeclared_test_success_criteria"],
        "policies": {}, "by_task_type": {}, "by_length": {},
        "length_stratum_rule": "short: original context <400 tokenizer tokens; long: >=400",
        "limitations": [
            "All data are synthetic, and this is one small local answering model.",
            "All eight high-level structures occur in development; this is a template/source-disjoint test, not a new-structure OOD test.",
            "Validation harm tolerance is empirical and does not guarantee fresh-test harm.",
            "Token reduction concerns full input prompt counts; it is not measured API cost or causal wall-time speedup.",
            "Old V1 held-out observations were reused as V2 development and cannot validate V2.",
        ],
    }
    all_decisions, flat, by_type_flat, by_length_flat = [], [], [], []
    policy_decisions = {}
    base_rows = {k: variants[0] for k, variants in tasks.items()}
    for policy, score_key, threshold in specs:
        ds = make_decisions(test, policy, score_key, threshold)
        for d in ds:
            r = base_rows[d["task_id"]]
            d["length_stratum"] = "long" if r["original_context_tokens"] >= 400 else "short"
        metrics = summarize_decisions(ds)
        flat.append({"policy": policy, **metrics})
        metrics["paired_group_bootstrap"] = grouped_bootstrap(ds, n_bootstrap=2000, seed=1111520)
        report["policies"][policy] = metrics
        report["by_task_type"][policy], report["by_length"][policy] = {}, {}
        for key, field, target, flat_rows in [
            ("by_task_type", "task_type", report["by_task_type"][policy], by_type_flat),
            ("by_length", "length_stratum", report["by_length"][policy], by_length_flat),
        ]:
            for value in sorted({d[field] for d in ds}):
                summary = summarize_decisions([d for d in ds if d[field] == value])
                target[value] = summary
                flat_rows.append({"policy": policy, field: value, **summary})
        policy_decisions[policy] = ds
        all_decisions.extend(ds)
    report["paired_comparisons"] = {
        f"mlp_vs_{other}": paired_comparison(policy_decisions["mlp"], policy_decisions[other])
        for other in ("original", "always_60", "always_30", "frozen_v1_mlp", "mlp_v1features", "rf")
    }
    primary = report["policies"]["mlp"]
    criteria = report["success_criteria"]
    report["primary_success_criteria_observed"] = {
        "saves_at_least_20_percent": primary["input_token_reduction_ratio"] >= criteria["input_token_reduction_ratio_min"],
        "accuracy_delta_at_least_minus_2_5pp": primary["accuracy_delta"] >= criteria["accuracy_delta_min"] - 1e-12,
        "harmful_acceptance_at_most_5_percent": primary["harmful_accepted_per_task"] <= criteria["harmful_accepted_per_task_max"] + 1e-12,
    }
    report["all_predeclared_primary_criteria_met"] = all(report["primary_success_criteria_observed"].values())
    seed_metrics = [report["policies"][n] for n in ("mlp", "mlp_seed_521", "mlp_seed_522")]
    report["seed_variation"] = {
        "seeds": [520, 521, 522], "seed_was_not_selected_on_test": True,
        "accuracy_mean": float(np.mean([m["accuracy"] for m in seed_metrics])),
        "accuracy_std": float(np.std([m["accuracy"] for m in seed_metrics], ddof=1)),
        "input_token_reduction_mean": float(np.mean([m["input_token_reduction_ratio"] for m in seed_metrics])),
        "input_token_reduction_std": float(np.std([m["input_token_reduction_ratio"] for m in seed_metrics], ddof=1)),
    }
    conditional = [r for r in test if r["original_correct"]]
    report["conditional_classifier_metrics"] = {
        name: models.classification_metrics([r["regression"] for r in conditional],
              [r[f"score_{name}"] for r in conditional])
        for name in MODEL_NAMES + ["frozen_v1_mlp"]
    }
    write_jsonl(ROOT / "results/task_policy_decisions.jsonl", all_decisions)
    write_csv(ROOT / "results/policy_results.csv", flat)
    write_csv(ROOT / "results/by_task_type.csv", by_type_flat)
    write_csv(ROOT / "results/by_length.csv", by_length_flat)
    write_csv(ROOT / "results/task_policy_decisions.csv", all_decisions)
    dump_json(ROOT / "results/metrics.json", report)
    print(json.dumps({"stage": "fresh_test_evaluated", "policies": report["policies"],
          "all_predeclared_primary_criteria_met": report["all_predeclared_primary_criteria_met"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate"])
    args = parser.parse_args()
    train() if args.stage == "train" else evaluate()
