"""Reproduce the old-data-only V2/V3 feature development diagnostic.

This reads only the bundled 1,920 development candidate rows and feature code.
It does not read calibration or final-test files, select a final policy, or
write model artifacts. Output: results/development_feature_cv.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from cachescope.models import _train_mlp


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    feature = load_module('development_v3_features', ROOT / 'src/cachescope/features.py')
    old = load_module('development_v2_features', ROOT / 'assets/v2_baseline/features.py')
    torch.set_num_threads(2)
    path = ROOT / 'data/development_candidates.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 1920:
        raise ValueError(f'Expected 1,920 old development candidate rows, got {len(rows)}')
    if {r['split'] for r in rows} - {'development', 'train'}:
        raise ValueError('The bundled old-development file must not contain new calibration/test rows')
    matrix = []
    for row in rows:
        arguments = (row['original_context'], row['question'], row['compressed_context'],
                     row['original_tokens'], row['compressed_tokens'])
        current, previous = feature.extract_features(*arguments), old.extract_features(*arguments)
        assert all(current[name] == previous[name] for name in old.FEATURE_NAMES)
        assert feature.preanswer_guard(current) == (not (current['original_json_valid'] and not current['compressed_json_valid']))
        matrix.append([current[name] for name in feature.FEATURE_NAMES])
    x = np.asarray(matrix)
    assert np.isfinite(x).all()
    y = np.asarray([int(r['original_correct'] and not r['compressed_correct']) for r in rows])
    conditional = np.asarray([bool(r['original_correct']) for r in rows])
    groups = np.asarray([r['group_id'] for r in rows])
    report = {
        'old_candidate_rows': len(rows), 'feature_count': len(feature.FEATURE_NAMES),
        'first_38_features_exactly_preserved': True, 'all_features_finite': True,
        'future_v3_tasks_or_outputs_read': False,
        'feature_sha256': hashlib.sha256((ROOT / 'src/cachescope/features.py').read_bytes()).hexdigest(),
        'folds': [],
    }
    scores = {'v2_38': np.zeros(len(rows)), 'v3_54': np.zeros(len(rows))}
    outer = GroupKFold(n_splits=4, shuffle=True, random_state=520)
    for fold, (train, test) in enumerate(outer.split(x, y, groups), 1):
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=520+fold).split(train, groups=groups[train]))
        tr, va = train[tr], train[va]
        tr, va = tr[conditional[tr]], va[conditional[va]]
        assert not (set(groups[train]) & set(groups[test]))
        summary = {'fold': fold, 'test_groups': len(set(groups[test])),
                   'conditional_test_candidates': int(conditional[test].sum()), 'models': {}}
        for name, dimensions in [('v2_38', 38), ('v3_54', 54)]:
            scaler = StandardScaler().fit(x[tr, :dimensions])
            network, history, info = _train_mlp(
                scaler.transform(x[tr, :dimensions]), y[tr],
                scaler.transform(x[va, :dimensions]), y[va],
                seed=520, max_epochs=100, patience=10,
            )
            with torch.inference_mode():
                tensor = torch.tensor(scaler.transform(x[test, :dimensions]), dtype=torch.float32)
                scores[name][test] = torch.sigmoid(network(tensor)).numpy().reshape(-1)
            test_conditional = test[conditional[test]]
            summary['models'][name] = {
                'AP': float(average_precision_score(y[test_conditional], scores[name][test_conditional])),
                'ROC_AUC': float(roc_auc_score(y[test_conditional], scores[name][test_conditional])),
                'best_epoch': info['best_epoch'],
            }
        report['folds'].append(summary)
        print(json.dumps(summary), flush=True)
    report['pooled_conditional_oof'] = {
        name: {'AP': float(average_precision_score(y[conditional], predictions[conditional])),
               'ROC_AUC': float(roc_auc_score(y[conditional], predictions[conditional]))}
        for name, predictions in scores.items()
    }
    output = ROOT / 'results/development_feature_cv.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['pooled_conditional_oof']), flush=True)


if __name__ == '__main__':
    main()
