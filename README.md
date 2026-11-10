# CacheScope MASC 520 Course Project

CacheScope learns whether a fixed prompt compressor is likely to turn a correct factual answer into an incorrect one. It computes local features before calling the answering model, scores two compression candidates, and selects the shortest candidate permitted by a validation-selected risk threshold. It retains the original prompt when no candidate passes.

This repository contains the complete three-version experiment history: code, synthetic datasets, genuine local-model responses, trained risk models, frozen protocols, evaluation tables, figures, audits and known failures. The original experiment deliveries are preserved byte for byte. ZIP archives are not repeated because their complete expanded contents are already included.

## Start here

- [Final V3 results](experiments/v3/RESULTS.md) and [offline HTML report](experiments/v3/实验结果.html)
- [Goal check and interpretation](experiments/v3/docs/GOAL_CHECK.md)
- [Course report files](reports/)
- [V3 data card](experiments/v3/docs/DATA_CARD.md), [features](experiments/v3/docs/FEATURES_V3.md) and [failure cases](experiments/v3/docs/FAILURE_ANALYSIS.md)
- [Full final metrics](experiments/v3/results/metrics.json), [per-task decisions](experiments/v3/results/task_policy_decisions.csv) and [audit](experiments/v3/results/verification.json)
- [Course deliverable summary](docs/requirements_summary.md), [third-party provenance](LICENSE_NOTE.md) and [publication inventory](upload_manifest.json)

## Final held-out test

The final V3 evaluation uses **200 independent new source tasks in 40 held-out wording/layout families**. All eight high-level structures were already observed in development. The answering model is local Ollama `qwen2.5-coder:0.5b`, with the exact digest and generation options recorded in the protocol. All documents and facts are synthetic English owner/date/budget records.

| Policy on the same 200 tasks | Accuracy | Full input-token reduction | Harmful tasks |
|---|---:|---:|---:|
| Original prompts | 69.00% | 0.00% | 0/200 |
| Fixed retention target 60% | 62.50% | 35.86% | 23/200 |
| Frozen V2 MLP with original threshold | 75.50% | 29.67% | 7/200 |
| **Calibration-selected primary: frozen V2 MLP plus JSON guard** | **75.00%** | **26.23%** | **7/200** |
| Re-trained MLP seed 520, predeclared secondary comparison | 74.50% | 26.12% | 3/200 |
| Re-trained MLP seed 521 | 73.00% | 24.32% | 5/200 |
| Re-trained MLP seed 522 | 74.50% | 21.61% | 3/200 |
| Re-trained maximum-score ensemble | 73.50% | 19.67% | 3/200 |

The primary and all three predeclared individual re-trained MLPs met the empirical targets of at least 20% token reduction, accuracy delta no lower than -2.5 percentage points, and harmful compression on at most 5% of all test tasks. The maximum-score ensemble missed the token target and is not marked successful.

The primary was chosen using **160 separate calibration tasks before final test labeling**. Calibration selected the frozen V2 weights plus a JSON syntax check, rather than the newly trained ensemble. The primary was not replaced with a model or seed chosen from test results.

The frozen V2 pure model also met all targets on this same new test, with higher accuracy and token savings than the selected hybrid and the same harmful-task count. Therefore these results do **not** establish a net benefit from the JSON guard or universal superiority of re-training. Individual re-trained models have fewer harmful accepted tasks at different operating points. Full comparisons, calibration attempts and uncertainties are retained.

## Three-version history

| Version | Scope and observation | Entry point |
|---|---|---|
| V1 | 500 main tasks, a separate 100-task pilot, 50 structural holdouts, and 50 paired length derivatives. Main-test MLP saved 32.60% tokens at 79% accuracy; structural and length failures limited generalization. | [V1 results](experiments/v1/RESULTS.md) |
| V2 | Expanded to 38 pre-answer features and 360 independent new tasks. Its 160-task test saved 25.94% at 71.25% accuracy, but 11/160 harmful tasks exceeded the 5% target. | [V2 results](experiments/v2/RESULTS.md) |
| V3 | Reused all previously inspected observations as development, re-trained 54-feature MLPs, added an explicit JSON syntax guard, and used fresh 160-task calibration plus 200-task final test. | [V3 results](experiments/v3/RESULTS.md) |

Absolute scores from different versions' test sets must not be treated as a controlled training improvement: their source records and wording families differ. Once a test was inspected, it was explicitly assigned to development in later versions. The old 50 length derivatives share sources and are not additional independent sources.

V3 training contains 960 task records representing 910 independent sources plus 50 paired derivatives. The new calibration/test data bring V3's total to 1,320 records and 1,270 independent sources. The original 100-task pilot is excluded from V3 training and these totals.

## Reproduce the frozen final evaluation

Use Python 3.11-3.13. From the repository root:

```sh
python -m venv .venv
# Activate .venv with the command appropriate to your shell.
python -m pip install -r requirements.txt
cd experiments/v3
python -m unittest discover -s tests -p "test_features_v3.py" -v
python audit_sources.py
python verify_v3.py
python train_v3.py evaluate
python build_v3_summary.py
python analyze_cases.py
```

These commands reuse stored real answering-model observations and saved risk models; verification/evaluation does not need a cloud API key or new language-model calls. Replaying a cached dataset is a reproducibility check, not a new independent statistical experiment.

The complete frozen replay was tested on Windows. Some original frozen JSON manifests contain Windows backslashes in their `files_sha256` path keys; native Linux/macOS `Path` does not interpret those as directory separators. On Linux or macOS, run the same preserved scripts through the limited compatibility wrapper, from `experiments/v3`:

```sh
python ../../tools/run_frozen.py audit_sources.py
python ../../tools/run_frozen.py verify_v3.py
python ../../tools/run_frozen.py train_v3.py evaluate
python ../../tools/run_frozen.py build_v3_summary.py
python ../../tools/run_frozen.py analyze_cases.py
```

The wrapper changes only the keys of a top-level `files_sha256` mapping after JSON decoding, in memory. It preserves file bytes, digest values, contexts, scores and neural logic. It leaves Windows decoding unchanged by default. Its standard-library assertions can be run with `python -m unittest discover -s tools -p "test_run_frozen.py" -v` from the repository root. These tests exercise the path shim, not a full Linux/macOS model replay; cross-platform library and rendering compatibility is not guaranteed by this shim.

All old code and manifest hashes are preserved. Keep file bytes unchanged; `.gitattributes` disables Git line-ending normalization so that checkout does not break the frozen SHA-256 checks. Rebuilding reports can change presentation-only prose or layout, but must not change the frozen models, inputs or selected thresholds.

The `run_local.ps1` convenience scripts preserve paths from the original Windows machine and are historical machine-specific helpers. For another machine, use the portable Python commands above after installing the requirements. They do not require the original machine's directory structure.

For generation workflows that inspect older sibling versions under their historical folder names, `python tools/stage_historical_workspace.py` makes a separate byte-preserved workspace under `reproductions/` with the original three directory names. Do not regenerate data or re-train over a frozen delivery. Use a separate experiment copy and fresh held-out sources for further model selection.

## Run a pre-answer prediction

From `experiments/v3`:

```sh
python predict.py --context-file examples/context.json --question "What is the current owner for project Demo-520?" --output examples/prediction.json
```

This command does not call the answering model. The saved manual example exposes a valid-but-empty JSON-container limitation; see [its explanation](experiments/v3/examples/README.md). A JSON syntax guard does not certify that the queried evidence remains, and learned scores are uncalibrated conditional risk rankings.

## Labels and evaluation

A candidate regression label is `original_correct and not compressed_correct`. Risk training uses only candidates whose original answer was correct. Original-incorrect tasks remain in the complete task-level policy evaluation. Gold answers, response text, correctness, source registries and task-type labels are not inputs to risk features.

Token reduction counts complete answering prompts, including instructions and question. It is not a measurement of API spending, subscription quota or causal wall-time speedup. Source/wording-family bootstrap intervals and per-task McNemar comparisons are reported with their limitations. Empirical target attainment is not a deployment risk guarantee.

## Disclosure and inventory

Code, synthetic data generation, analysis and documentation used AI assistance. Labels come from actual recorded local-model outputs matched to predefined facts, rather than fabricated responses. Earlier components and inspected test reuse are documented instead of presented as new held-out evidence.

No author names, student IDs or email addresses are added in this submission overview. The original course syllabus and personal proposal are not republished here; the course requirements are summarized in prose. The prior web demonstration is separate background work: <https://github.com/thisisyk/cachescope-web>.

`upload_manifest.json` records file sizes, SHA-256 and Git blob SHA-1 for every included file except itself. Run `python tools/build_upload_manifest.py` after adding final report artifacts to refresh the inventory and publication scan. Publication scans report categories and file paths, never suspected credential values.
