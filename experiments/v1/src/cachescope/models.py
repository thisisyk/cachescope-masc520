"""Supervised conditional-regression models for CacheScope.

The risk classifiers learn only from candidates whose original answer was
correct. Their outputs are conditional risk *scores*, not a calibrated absolute
failure probability for all requests. Final policies must be evaluated on all
held-out tasks, including those whose original answer was wrong.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features


def feature_row(record: dict[str, Any]) -> dict[str, float]:
    """Read only explicit features or the five permitted pre-answer fields."""
    if "features" in record:
        return {name: float(record["features"][name]) for name in FEATURE_NAMES}
    return extract_features(
        record["original_context"], record["question"], record["compressed_context"],
        record["original_tokens"], record["compressed_tokens"],
    )


def feature_matrix(rows: Iterable[dict[str, Any]], names=FEATURE_NAMES) -> np.ndarray:
    rows = list(rows)
    if not rows:
        return np.empty((0, len(names)), dtype=np.float64)
    features = [feature_row(row) if "features" in row or "original_context" in row else row for row in rows]
    values = np.asarray([[float(row[name]) for name in names] for row in features], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains non-finite values")
    return values


def _label(record: dict[str, Any]) -> int:
    observed = int(bool(record["original_correct"]) and not bool(record["compressed_correct"]))
    if "regression" in record and int(record["regression"]) != observed:
        raise ValueError("Regression label disagrees with observed answer correctness")
    return observed


def _validate_groups(records: list[dict[str, Any]]) -> None:
    groups: dict[str, str] = {}
    tasks: dict[str, str] = {}
    for record in records:
        split = record["split"]
        for field, mapping in (("group_id", groups), ("task_id", tasks)):
            key = str(record[field])
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Data leakage: {field} {key!r} occurs in {mapping[key]} and {split}")
            mapping[key] = split
        _label(record)


def _json_dump(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def classification_metrics(labels, scores, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(y) != len(p):
        raise ValueError("Label and score lengths differ")
    if not len(y):
        return {"n": 0, "positives": 0, "precision": None, "recall": None,
                "f1": None, "average_precision": None, "roc_auc": None,
                "confusion_matrix": [[0, 0], [0, 0]], "precision_recall_curve": None}
    predictions = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0
    )
    both_classes = len(np.unique(y)) == 2
    if both_classes:
        curve_precision, curve_recall, curve_threshold = precision_recall_curve(y, p)
        curve = {"precision": curve_precision.tolist(), "recall": curve_recall.tolist(),
                 "threshold": curve_threshold.tolist()}
    else:
        curve = None
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "positive_fraction": float(y.mean()), "threshold": threshold,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "precision_recall_curve": curve,
    }


def _network(input_dim: int):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1),
    )


def _train_mlp(x_train, y_train, x_validation, y_validation, seed: int,
               max_epochs: int, patience: int):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _network(x_train.shape[1]).cpu()
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    xv = torch.as_tensor(x_validation, dtype=torch.float32)
    yv = torch.as_tensor(y_validation, dtype=torch.float32).reshape(-1, 1)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    positive_weight = negatives / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        loss_total = 0.0
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(xv), yv))
        history.append({"epoch": epoch, "train_loss": loss_total / len(x),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, stale, best_epoch = validation_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"seed": seed, "epochs_completed": len(history),
                            "best_epoch": best_epoch, "best_validation_loss": best_loss,
                            "positive_weight": positive_weight}


def predict_scores(bundle: dict[str, Any], feature_rows) -> np.ndarray:
    """Return conditional risk scores in [0, 1] for pre-answer feature rows."""
    if isinstance(feature_rows, np.ndarray):
        matrix = np.asarray(feature_rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(bundle["feature_names"]):
            raise ValueError("Array feature count does not match saved feature_names")
    else:
        matrix = feature_matrix(feature_rows, bundle["feature_names"])
    if not len(matrix):
        return np.empty(0, dtype=float)
    matrix = bundle["scaler"].transform(matrix)
    if bundle["kind"] == "torch_mlp":
        import torch
        with torch.inference_mode():
            logits = bundle["model"](torch.as_tensor(matrix, dtype=torch.float32))
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)
    classes = list(bundle["model"].classes_)
    return bundle["model"].predict_proba(matrix)[:, classes.index(1)]


def load_bundle(modeldir, name: str = "mlp") -> dict[str, Any]:
    """Load one saved risk model; aliases lr/rf/mlp name the primary models."""
    modeldir = Path(modeldir)
    if name == "mlp":
        manifest = json.loads((modeldir / "training_report.json").read_text(encoding="utf-8"))
        name = manifest["primary_mlp"]
    if (modeldir / f"{name}.joblib").exists():
        return joblib.load(modeldir / f"{name}.joblib")
    import torch
    checkpoint = torch.load(modeldir / f"{name}.pt", map_location="cpu", weights_only=True)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=float)
    scaler.var_ = np.asarray(checkpoint["scaler_var"], dtype=float)
    scaler.n_features_in_ = len(checkpoint["feature_names"])
    scaler.n_samples_seen_ = int(checkpoint["scaler_n_samples"])
    model = _network(len(checkpoint["feature_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"kind": "torch_mlp", "model": model, "scaler": scaler,
            "feature_names": checkpoint["feature_names"], "seed": checkpoint["seed"],
            "score_interpretation": checkpoint["score_interpretation"]}


def train_models(records: Iterable[dict[str, Any]], out_dir,
                 seeds=(520, 521, 522), *, max_epochs: int = 100,
                 patience: int = 10, run_ablations: bool = True) -> dict[str, Any]:
    """Fit risk models using the fixed split and return their training report.

    No test/OOD outcomes are used for training, early stopping, preprocessing,
    or model selection. Ablations use the first seed and the same protocol.
    """
    import torch
    records = list(records)
    if not seeds:
        raise ValueError("At least one training seed is required")
    _validate_groups(records)
    train = [r for r in records if r["split"] == "train" and r["original_correct"]]
    validation = [r for r in records if r["split"] == "validation" and r["original_correct"]]
    if not train or not validation:
        raise ValueError("Need original-correct records in both fixed train and validation splits")
    y_train = np.asarray([_label(r) for r in train], dtype=int)
    y_validation = np.asarray([_label(r) for r in validation], dtype=int)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Risk training requires both observed regression and no-regression examples; do not invent labels")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interpretation = (
        "Uncalibrated conditional risk score learned from original-correct training candidates; "
        "not an absolute error probability or a safety guarantee."
    )
    report: dict[str, Any] = {
        "primary_mlp": f"mlp_seed_{seeds[0]}", "feature_names": list(FEATURE_NAMES),
        "score_interpretation": interpretation,
        "training_population": "Candidates from train split with original_correct=true only",
        "preprocessing": "StandardScaler fitted on original-correct train candidates only",
        "architecture": "d -> 32 ReLU -> 16 ReLU -> 1 logit",
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
                      "batch_size": 32, "max_epochs": max_epochs, "patience": patience},
        "class_weight_policy": "train-only inverse frequency: pos_weight=N_negative/N_positive; balanced sklearn class weights",
        "split_counts": {}, "models": {},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    import sklearn
    report["versions"]["sklearn"] = sklearn.__version__
    for split in sorted({r["split"] for r in records}):
        subset = [r for r in records if r["split"] == split]
        conditional = [r for r in subset if r["original_correct"]]
        report["split_counts"][split] = {
            "tasks": len({r["task_id"] for r in subset}), "candidate_records": len(subset),
            "original_correct_candidates": len(conditional),
            "observed_regressions": sum(_label(r) for r in conditional),
        }
    prediction_rows = []

    def evaluate_and_record(name, bundle):
        metrics = {}
        for split in ("train", "validation", "test", "ood"):
            subset = [r for r in records if r["split"] == split]
            scores = predict_scores(bundle, subset)
            selected = [i for i, r in enumerate(subset) if r["original_correct"]]
            metrics[split] = classification_metrics(
                [_label(subset[i]) for i in selected], scores[selected]
            )
            for record, score in zip(subset, scores):
                prediction_rows.append({
                    "task_id": record["task_id"], "candidate_id": record.get("candidate_id", record.get("keep_ratio")),
                    "split": split, "model": name, "risk_score": float(score),
                    "original_correct": bool(record["original_correct"]),
                    "compressed_correct": bool(record["compressed_correct"]),
                    "regression": _label(record), "in_conditional_classification_evaluation": bool(record["original_correct"]),
                })
        return metrics

    x_train_all = feature_matrix(train)
    scaler_all = StandardScaler().fit(x_train_all)
    x_train_scaled = scaler_all.transform(x_train_all)
    for name, model in (
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seeds[0])),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                      class_weight="balanced", random_state=seeds[0], n_jobs=1)),
    ):
        model.fit(x_train_scaled, y_train)
        bundle = {"kind": "sklearn", "model": model, "scaler": scaler_all,
                  "feature_names": list(FEATURE_NAMES), "score_interpretation": interpretation}
        joblib.dump(bundle, out_dir / f"{name}.joblib")
        report["models"][name] = {"metrics": evaluate_and_record(name, bundle)}

    configurations = [(f"mlp_seed_{seed}", list(FEATURE_NAMES), seed) for seed in seeds]
    if run_ablations:
        configurations.extend([
            ("mlp_no_numeric", [n for n in FEATURE_NAMES if "numeric" not in n and "number" not in n and "date" not in n], seeds[0]),
            ("mlp_no_overlap", [n for n in FEATURE_NAMES if "question_overlap" not in n], seeds[0]),
        ])
    for name, names, seed in configurations:
        raw_train, raw_validation = feature_matrix(train, names), feature_matrix(validation, names)
        scaler = StandardScaler().fit(raw_train)
        model, history, training_info = _train_mlp(
            scaler.transform(raw_train), y_train, scaler.transform(raw_validation), y_validation,
            seed=seed, max_epochs=max_epochs, patience=patience,
        )
        checkpoint = {
            "state_dict": model.state_dict(), "feature_names": names, "seed": seed,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_var": scaler.var_.tolist(), "scaler_n_samples": int(scaler.n_samples_seen_),
            "score_interpretation": interpretation,
        }
        torch.save(checkpoint, out_dir / f"{name}.pt")
        bundle = {"kind": "torch_mlp", "model": model, "scaler": scaler,
                  "feature_names": names, "score_interpretation": interpretation}
        _json_dump(history, out_dir / f"{name}_history.json")
        report["models"][name] = {
            **training_info, "feature_names": names,
            "metrics": evaluate_and_record(name, bundle),
        }
    _json_dump(report, out_dir / "training_report.json")
    with (out_dir / "candidate_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report
