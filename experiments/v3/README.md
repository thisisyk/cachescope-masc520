# CacheScope V3 — MASC 520

CacheScope predicts factual-answer regression before accepting a compressed prompt. V3 continues training toward the same three empirical targets: at least 20% full-input-token reduction, accuracy no more than 2.5 percentage points below original prompts, and harmful compression on no more than 5% of final test tasks.

Read **[RESULTS.md](RESULTS.md)** or **[实验结果.html](实验结果.html)** for the measured outcome. A target is achieved only if the final report marks it achieved; development or calibration performance does not substitute for final-test performance.

The completed test met all three primary targets, and all three predeclared re-trained single MLPs also met them. Read [the goal check and interpretation](docs/GOAL_CHECK.md): the frozen V2 baseline also met the targets on this same test, so the findings do not establish that re-training or the JSON guard universally improves performance.

## Method and change from V2

The fixed compressor and answering model remain unchanged. Two candidates retain approximately 60% or 30% of source context. A selected risk threshold permits the shortest eligible candidate; otherwise the policy keeps the original.

V3 trains three small MLPs with seeds 520/521/522 and 54 pre-answer features. Thirty-eight features retain V2's exact definitions; sixteen describe question field, competing values/records, source ambiguity and JSON binding/syntax. Each neural model uses two hidden layers of 32 and 16 units. An internal group-disjoint split chooses epoch counts, then each model is initialized again and fitted for that fixed number of epochs using all old development data. The ensemble score is the maximum of the three scores; no best seed is selected on the test.

An explicit structural guard rejects a candidate if the original is valid JSON but the candidate is not. This is a **hybrid neural-plus-rule policy**. Syntax validity does not establish semantic correctness. Pure-neural, guard-only, LR, RF and frozen-V2 comparisons remain visible so that a rule's contribution is not falsely attributed to neural training.

The primary policy is selected using calibration from two predeclared families: the newly trained ensemble plus the guard, or the frozen V2 MLP plus the guard. If the old-model hybrid wins calibration, the report must say so. Re-training by itself is not assumed to improve performance.

## Data and independence

All previously inspected V1/V2 observations are now development data: 960 task records represent 910 independent sources plus 50 paired length derivatives. Former tests cannot validate V3. The original 100-task pilot remains excluded.

360 independent new sources were fixed before model calls: **160 calibration tasks and 200 final test tasks**, in 72 new wording/layout families with five tasks each. The partitions have 32 and 40 families respectively. All eight high-level structures appear in development: direct, header, update, contrast, reference, table, nested JSON and paired directory. This is a new-source and wording/layout-family holdout, not an unseen-structure OOD benchmark.

New contexts contain 153–248 or 511–633 Qwen context tokens. These are dataset-relative length bands, not a real long-document benchmark. Sources are synthetic English owner/date/budget facts, generated with AI assistance. Full source registries and an independent rendered-source audit are included.

## Calibration and the fixed final target

Calibration never fits neural weights. At each threshold, a policy's complete task-level outcome is measured, counting each task once. The predeclared harm buffers are 1.25%, 1.875%, then 2.5%; the first tier with a hybrid saving at least 20% and accuracy delta at least -1.25 percentage points wins. Within that tier, calibration savings select the primary policy. Every attempted tier is retained. If no tier is feasible, no selection is frozen and the final test does not run.

This empirical buffer is not a statistical guarantee. The final test's unchanged limits are **20% savings, accuracy delta ≥-2.5 points, harmful task rate ≤5%**. Weights, features, code, primary policy and operating thresholds are hashed and frozen before final test labeling. All predeclared test policies are reported; test results do not choose a model, seed or threshold.

## Files

| File | Purpose |
|---|---|
| `experiment_protocol.json` | Fixed source hashes, candidates, calibration tiers and test targets |
| `make_v3_data.py` | Reproducible independent new source generator |
| `data/fresh_tasks.jsonl`, `source_registry.jsonl` | New sources and canonical audit facts |
| `data/development_candidates.jsonl` | Old real-model observations, explicitly development |
| `data/*raw_responses.jsonl` | Genuine model answers, prompt hashes, options and backend tokens |
| `train_v3.py` | Separate training, calibration/freezing and final evaluation stages |
| `results/fitted_model_manifest.json` | Old-data training provenance and hashes |
| `results/calibration_attempt.json` | All predeclared safety-buffer attempts |
| `results/frozen_selection.json` | Selected hybrid, thresholds, code/model/data hashes |
| `results/metrics.json`, `policy_results.csv` | Full final-test comparisons and uncertainties |
| `results/task_policy_decisions.jsonl` | Auditable per-task selections and outcomes |
| `assets/v2_baseline/` | Self-contained frozen V2 features, weights and original threshold |
| `verify_v3.py` | Recompute labels, token counts, scores, guard decisions and primary metrics |
| `audit_sources.py` | Parse all rendered new source facts without reading LLM answers |
| `predict.py` | Use the saved calibration-selected hybrid before answering |

## Run and reproduce

Use Python 3.11–3.13 and install `requirements.txt`. Actual execution uses CPU PyTorch 2.8.0, scikit-learn 1.7.2 and the bundled official Qwen tokenizer. Answer labeling uses local Ollama `qwen2.5-coder:0.5b`; its exact digest, options and prompt wrapper are recorded in the protocol. No cloud API credential is required.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_features_v3.py" -v
python audit_sources.py
python verify_v3.py
python train_v3.py evaluate
python build_v3_summary.py
```

The delivered selection is already frozen. The original sequence, in a separate clean experiment copy, is:

```powershell
python train_v3.py train
python run_labeling.py calibration
python refresh_features.py calibration
python train_v3.py calibrate
python run_labeling.py test
python refresh_features.py test
python train_v3.py evaluate
python verify_v3.py
python build_v3_summary.py
```

Replaying cached model observations verifies reproducibility; it is not a new independent statistical experiment. Re-training refuses to overwrite a frozen selection. Use a separate version and fresh test sources for further development. The final model needs one selected answering-model call per request; obtaining original and two compressed answers is an offline labeling cost.

Try the saved pre-answer gate on a small JSON example:

```powershell
python predict.py --context-file examples/context.json --question "What is the current owner for project Demo-520?" --output examples/prediction.json
```

The gate does not call the answering model or use a reference answer. Scores are uncalibrated conditional risk rankings. Prediction should not be interpreted as a correctness guarantee.

The saved manual JSON example exposes a valid-but-empty-container failure mode. See [the smoke-check limitation](examples/README.md); it is separate from the predefined benchmark and is retained for transparency.

## Scope and attribution

Results concern one small local answering model and controlled synthetic English sources. Input-token reductions count the full prompt, including instructions and question; they are not measured monetary savings or causal latency improvements. Group bootstrap intervals reflect the sampled wording families; per-task McNemar values do not adjust for all clustering or multiple comparisons.

This course experiment follows the CacheScope proposal and reuses previous project components, with all inspected older results explicitly assigned to development. Code, synthetic data generation and documentation used AI assistance. Labels come from actual stored local-model outputs and predefined source facts. The bundled tokenizer retains its official license and provenance. The course report should distinguish prior work, new experimental contributions, neural learning, explicit rules and AI assistance.
