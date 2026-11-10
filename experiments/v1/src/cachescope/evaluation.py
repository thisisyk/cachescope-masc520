"""Task-level policy evaluation with validation-only risk thresholds.

Candidate rows train classifiers; tasks, not candidates, are the unit of policy
evaluation. Scores are conditional risk rankings, not calibrated guarantees.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REJECT_ALL_THRESHOLD = -1.0
MODEL_SCORE_PREFIX = "score_"
RULE_PARAMETERS = {
    "max_question_overlap_change": 0.0,
    "min_numeric_retention": 0.6,
    "min_date_retention": 0.6,
    "min_negation_retention": 1.0,
}


def group_tasks(records: list[dict]) -> dict[str, list[dict]]:
    """Group and validate observations without using outcomes for selection."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row["task_id"])].append(row)
    for task_id, rows in groups.items():
        ids = [r["candidate_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate for task {task_id}")
        for key in ("split", "group_id", "original_tokens", "original_correct"):
            if len({r[key] for r in rows}) != 1:
                raise ValueError(f"Inconsistent {key} for task {task_id}")
        if any(int(r["original_tokens"]) <= 0 or int(r["compressed_tokens"]) <= 0 for r in rows):
            raise ValueError("Input token counts must be positive")
    return dict(sorted(groups.items()))


def choose_candidate(rows: list[dict], score_key: str, threshold: float) -> dict | None:
    """Select the shortest admissible candidate using only pre-answer data."""
    passed = [
        r for r in rows
        if float(r[score_key]) <= threshold
        and int(r["compressed_tokens"]) < int(r["original_tokens"])
    ]
    return min(passed, key=lambda r: (int(r["compressed_tokens"]), str(r["candidate_id"]))) if passed else None


def _rule_candidate(rows: list[dict]) -> dict | None:
    allowed = []
    for row in rows:
        f = row["features"]
        if (
            f["question_overlap_change"] <= RULE_PARAMETERS["max_question_overlap_change"]
            and f["numeric_retention"] >= RULE_PARAMETERS["min_numeric_retention"]
            and f["date_retention"] >= RULE_PARAMETERS["min_date_retention"]
            and f["negation_retention"] >= RULE_PARAMETERS["min_negation_retention"]
            and row["compressed_tokens"] < row["original_tokens"]
        ):
            allowed.append(row)
    return min(allowed, key=lambda r: (r["compressed_tokens"], r["candidate_id"])) if allowed else None


def make_decisions(
    records: list[dict], policy: str, score_key: str | None = None,
    threshold: float | None = None,
) -> list[dict]:
    decisions = []
    for task_id, rows in group_tasks(records).items():
        original = rows[0]
        if policy == "original":
            selected = None
        elif policy in ("always_60", "always_30"):
            target = 0.6 if policy == "always_60" else 0.3
            selected = min(rows, key=lambda r: abs(float(r["keep_ratio"]) - target))
            if abs(float(selected["keep_ratio"]) - target) > 1e-8:
                raise ValueError(f"Missing {policy} candidate for {task_id}")
        elif policy == "simple_rule_gate":
            selected = _rule_candidate(rows)
        else:
            if score_key is None or threshold is None:
                raise ValueError("Learned policy requires score_key and threshold")
            selected = choose_candidate(rows, score_key, threshold)
        accepted = selected is not None
        original_correct = bool(original["original_correct"])
        correct = bool(selected["compressed_correct"]) if accepted else original_correct
        latency = selected.get("compressed_latency_s") if accepted else original.get("original_latency_s")
        decisions.append({
            "task_id": task_id,
            "split": original["split"],
            "group_id": original["group_id"],
            "task_type": original["task_type"],
            "policy": policy,
            "candidate_id": selected["candidate_id"] if accepted else "original",
            "accepted": accepted,
            "correct": correct,
            "original_correct": original_correct,
            "harmful": accepted and original_correct and not correct,
            "improved": accepted and not original_correct and correct,
            "input_tokens": int(selected["compressed_tokens"] if accepted else original["original_tokens"]),
            "original_tokens": int(original["original_tokens"]),
            "answer": selected.get("compressed_answer") if accepted else original.get("original_answer"),
            "original_answer": original.get("original_answer"),
            "observed_answer_latency_s": float(latency) if latency is not None else None,
            "risk_score": float(selected[score_key]) if accepted and score_key else None,
        })
    return decisions


def summarize_decisions(decisions: list[dict]) -> dict:
    if not decisions:
        raise ValueError("Cannot evaluate an empty task split")
    n = len(decisions)
    n_accepted = sum(d["accepted"] for d in decisions)
    n_harmful = sum(d["harmful"] for d in decisions)
    n_improved = sum(d["improved"] for d in decisions)
    baseline_tokens = sum(d["original_tokens"] for d in decisions)
    selected_tokens = sum(d["input_tokens"] for d in decisions)
    accuracy = sum(d["correct"] for d in decisions) / n
    original_accuracy = sum(d["original_correct"] for d in decisions) / n
    latencies = [d["observed_answer_latency_s"] for d in decisions if d["observed_answer_latency_s"] is not None]
    return {
        "n_tasks": n,
        "n_groups": len({d["group_id"] for d in decisions}),
        "accuracy": accuracy,
        "original_accuracy": original_accuracy,
        "accuracy_delta": accuracy - original_accuracy,
        "total_input_tokens": selected_tokens,
        "original_total_input_tokens": baseline_tokens,
        "input_token_reduction_ratio": 1.0 - selected_tokens / baseline_tokens,
        "accepted_count": n_accepted,
        "acceptance_rate": n_accepted / n,
        "harmful_accepted_count": n_harmful,
        "harmful_accepted_per_task": n_harmful / n,
        "harmful_accepted_per_accepted": n_harmful / n_accepted if n_accepted else None,
        "improvement_count": n_improved,
        "original_correct_selected_correct": sum(d["original_correct"] and d["correct"] for d in decisions),
        "original_correct_selected_wrong": n_harmful,
        "original_wrong_selected_correct": n_improved,
        "original_wrong_selected_wrong": sum(not d["original_correct"] and not d["correct"] for d in decisions),
        "observed_answer_latency_s_mean": float(np.mean(latencies)) if latencies else None,
    }


def select_threshold(validation_records: list[dict], score_key: str, epsilon: float = 0.02) -> dict:
    """Optimize validation savings subject to empirical harm/accuracy limits.

    This is an empirical operating point, not a confidence-bound guarantee.
    No test labels, outcomes, scores, or token counts enter this function.
    """
    if not validation_records or any(r["split"] != "validation" for r in validation_records):
        raise ValueError("Threshold selection accepts validation records only")
    values = [float(r[score_key]) for r in validation_records]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Risk scores must be finite and between zero and one")
    thresholds = [REJECT_ALL_THRESHOLD] + sorted(set(values))
    feasible = []
    for threshold in thresholds:
        metrics = summarize_decisions(make_decisions(validation_records, "gate", score_key, threshold))
        if metrics["harmful_accepted_per_task"] <= epsilon + 1e-12 and metrics["accuracy_delta"] >= -epsilon - 1e-12:
            feasible.append((threshold, metrics))
    # Reject-all is always feasible for nonnegative epsilon.
    threshold, metrics = max(feasible, key=lambda item: (
        item[1]["input_token_reduction_ratio"], item[1]["accuracy"], -item[0]
    ))
    return {
        "threshold": threshold,
        "score_key": score_key,
        "selection_split": "validation",
        "epsilon": epsilon,
        "constraint": "harmful_accepted_per_task <= epsilon and accuracy_delta >= -epsilon",
        "objective": "maximum validation total-input-token reduction",
        "n_thresholds_examined": len(thresholds),
        "validation_metrics": metrics,
        "guarantee": False,
    }


def grouped_bootstrap(decisions: list[dict], n_bootstrap: int = 500, seed: int = 520) -> dict:
    """Paired resampling of template/source groups, retaining all their tasks."""
    grouped = defaultdict(list)
    for d in decisions:
        grouped[str(d["group_id"])].append(d)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = [d for key in rng.choice(keys, size=len(keys), replace=True) for d in grouped[key]]
        metrics = summarize_decisions(sample)
        values.append((metrics["accuracy_delta"], metrics["input_token_reduction_ratio"]))
    quantiles = np.quantile(np.asarray(values), [0.025, 0.975], axis=0)
    return {
        "method": "paired percentile bootstrap over group_id",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_independent_groups": len(keys),
        "accuracy_delta_95_ci": quantiles[:, 0].tolist(),
        "input_token_reduction_ratio_95_ci": quantiles[:, 1].tolist(),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _make_plots(records, report, all_decisions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    test_rows = [r for r in records if r["split"] == "test"]
    if not test_rows:
        return []
    n = len(group_tasks(test_rows))
    plots = out_dir / "figures"
    plots.mkdir(exist_ok=True)
    policies = ["original", "always_60", "always_30", "simple_rule_gate", "lr", "rf", "mlp"]
    available = [p for p in policies if p in report["splits"]["test"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    x = np.arange(len(available))
    axes[0].bar(x, [100 * report["splits"]["test"][p]["accuracy"] for p in available], color="#355c7d")
    axes[1].bar(x, [100 * report["splits"]["test"][p]["input_token_reduction_ratio"] for p in available], color="#4a8c77")
    for ax in axes:
        ax.set_xticks(x, available, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Task accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Total input token reduction (%)")
    fig.suptitle(f"Locked policies on synthetic test tasks (n={n})")
    fig.savefig(plots / "policy_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for name in ("lr", "rf", "mlp"):
        score_key = f"score_{name}"
        if not all(score_key in r for r in test_rows):
            continue
        # Test curves are descriptive only; the operating-point marker was
        # chosen exclusively on validation data.
        grid = [REJECT_ALL_THRESHOLD] + sorted(set(float(r[score_key]) for r in test_rows))
        if len(grid) > 101:
            grid = [grid[i] for i in np.unique(np.linspace(0, len(grid)-1, 101).astype(int))]
        metrics = [summarize_decisions(make_decisions(test_rows, name, score_key, threshold)) for threshold in grid]
        ax.plot([100*m["input_token_reduction_ratio"] for m in metrics], [100*m["accuracy"] for m in metrics], label=f"{name.upper()} descriptive curve", linewidth=1.5)
        locked = report["splits"]["test"][name]
        ax.scatter(100*locked["input_token_reduction_ratio"], 100*locked["accuracy"], s=55, edgecolors="black", zorder=5)
    for name in ("original", "always_60", "always_30", "simple_rule_gate"):
        m = report["splits"]["test"][name]
        ax.scatter(100*m["input_token_reduction_ratio"], 100*m["accuracy"], marker="x", s=70, label=name)
    ax.set(xlabel="Total input token reduction (%)", ylabel="Task accuracy (%)", title=f"Accuracy-token tradeoff (synthetic test n={n})")
    ax.text(0.02, 0.02, "Outlined dots: validation-selected operating points", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(plots / "accuracy_token_tradeoff.png", dpi=170)
    plt.close(fig)

    conditional = [r for r in test_rows if r["original_correct"]]
    y = np.asarray([int(not r["compressed_correct"]) for r in conditional])
    if len(y) and len(np.unique(y)) == 2:
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        for name in ("lr", "rf", "mlp"):
            key = f"score_{name}"
            if not all(key in r for r in conditional):
                continue
            score = [r[key] for r in conditional]
            precision, recall, _ = precision_recall_curve(y, score)
            ap = average_precision_score(y, score)
            ax.plot(recall, precision, label=f"{name.upper()} AP={ap:.3f}")
        ax.axhline(float(y.mean()), color="gray", linestyle="--", label=f"Prevalence={y.mean():.3f}")
        ax.set(xlabel="Regression detection recall", ylabel="Precision", title=f"Conditional risk prediction (test candidates n={len(y)})", xlim=(0,1), ylim=(0,1.05))
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(plots / "risk_precision_recall.png", dpi=170)
        plt.close(fig)

    type_rows = []
    for policy in available:
        selected = [d for d in all_decisions if d["split"] == "test" and d["policy"] == policy]
        for task_type in sorted({d["task_type"] for d in selected}):
            metrics = summarize_decisions([d for d in selected if d["task_type"] == task_type])
            type_rows.append({"policy":policy,"task_type":task_type,**metrics})
    if type_rows:
        with (out_dir / "by_task_type.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(type_rows[0]));w.writeheader();w.writerows(type_rows)
    return [str(p.relative_to(out_dir)) for p in sorted(plots.glob("*.png"))]


def evaluate_all(records: list[dict], out_dir, epsilon: float = 0.02, n_bootstrap: int = 500) -> dict:
    """Write locked-threshold task policies, audit records, metrics, and figures."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_tasks(records)
    formal_group_splits = defaultdict(set)
    for row in records:
        if row["split"] in ("train", "validation", "test"):
            formal_group_splits[str(row["group_id"])].add(row["split"])
    leaking_groups = [key for key, splits in formal_group_splits.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(f"Source/template groups cross formal splits: {leaking_groups[:5]}")
    split_records = {split: [r for r in records if r["split"] == split] for split in ("validation", "test", "ood")}
    validation = split_records["validation"]
    if not validation:
        raise ValueError("Validation data required for operating-point selection")
    score_keys = sorted({key for row in records for key in row if key.startswith(MODEL_SCORE_PREFIX)})
    for key in score_keys:
        if any(key not in row for row in records):
            raise ValueError(f"Incomplete candidate score coverage for {key}")
    selections = {key[6:]: select_threshold(validation, key, epsilon) for key in score_keys}
    report = {
        "unit_of_policy_evaluation": "task, each source task counted once",
        "primary_mlp_seed": 520,
        "threshold_selection": "validation only, before test or structural-holdout evaluation",
        "risk_score_interpretation": "conditional on original-prompt correctness in the classifier training population; not a calibrated absolute error probability",
        "dataset_scope": "synthetic factual extraction with one fixed local Qwen answering model",
        "epsilon_is_empirical_not_guaranteed": epsilon,
        "rule_gate_parameters": RULE_PARAMETERS,
        "latency_scope": "observed stored answering-call wall time; excludes optimizer cost and is affected by cache and request order, so not a causal speedup estimate",
        "savings_scope": "complete input prompt token counts, not API cost or subscription quota",
        "selected_thresholds": selections,
        "splits": {},
        "notes": [
            "Classifier scores must be computed from pre-answer features only.",
            "Original-incorrect tasks remain in task-policy evaluation.",
            "Two candidates from a task are not independent task observations.",
            "Template-group bootstrap uncertainty and training-seed variation answer different questions.",
            "Descriptive test tradeoff curves do not select the deployed threshold.",
            "Synthetic structural holdout was not human-authored and is not evidence of real-world generalization.",
        ],
    }
    all_decisions = []
    csv_rows = []
    for split, rows in split_records.items():
        if not rows:
            continue
        report["splits"][split] = {}
        specs = [(p, None, None) for p in ("original", "always_60", "always_30", "simple_rule_gate")]
        specs += [(name, sel["score_key"], sel["threshold"]) for name, sel in selections.items()]
        for policy, score_key, threshold in specs:
            decisions = make_decisions(rows, policy, score_key, threshold)
            metrics = summarize_decisions(decisions)
            csv_rows.append({"split":split,"policy":policy,**metrics})
            if split in ("test", "ood"):
                metrics["paired_group_bootstrap"] = grouped_bootstrap(decisions, n_bootstrap=n_bootstrap)
            report["splits"][split][policy] = metrics
            all_decisions.extend(decisions)
    seed_rows = [report["splits"].get("test", {}).get(name) for name in ("mlp", "mlp_seed_521", "mlp_seed_522")]
    seed_rows = [row for row in seed_rows if row]
    if seed_rows:
        report["mlp_training_seed_variation_test"] = {
            "n_seeds": len(seed_rows),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in seed_rows])),
            "accuracy_std": float(np.std([r["accuracy"] for r in seed_rows], ddof=1)) if len(seed_rows)>1 else 0.0,
            "input_token_reduction_mean": float(np.mean([r["input_token_reduction_ratio"] for r in seed_rows])),
            "input_token_reduction_std": float(np.std([r["input_token_reduction_ratio"] for r in seed_rows], ddof=1)) if len(seed_rows)>1 else 0.0,
        }
    _write_json(out_dir / "selected_thresholds.json", selections)
    _write_jsonl(out_dir / "task_policy_decisions.jsonl", all_decisions)
    failures = []
    raw_tasks = group_tasks(records)
    for d in all_decisions:
        if d["split"] in ("test", "ood") and (not d["correct"] or d["harmful"]):
            original = raw_tasks[d["task_id"]][0]
            candidate = next((r for r in raw_tasks[d["task_id"]] if r["candidate_id"] == d["candidate_id"]), None)
            failures.append({**d,"question":original["question"],"reference_answer":original.get("reference_answer",original.get("answer")),"original_context":original["original_context"],"selected_context":candidate["compressed_context"] if candidate else original["original_context"]})
    _write_jsonl(out_dir / "failures.jsonl", failures)
    with (out_dir / "policy_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        if csv_rows:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    report["figures"] = _make_plots(records, report, all_decisions, out_dir)
    _write_json(out_dir / "metrics.json", report)
    return report
