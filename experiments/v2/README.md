# CacheScope V2 — MASC 520

CacheScope learns whether compressing a factual-extraction prompt is likely to turn a correct answer into an incorrect one. A fixed sentence selector proposes 60% and 30% context-retention candidates. A small neural network chooses the shortest candidate below a validation-selected risk threshold, otherwise it keeps the original. At deployment the answering model receives only the selected prompt; the three observed answers per task are an offline labeling cost.

Open **[RESULTS.md](RESULTS.md)** or **[实验结果.html](实验结果.html)** for the actual new test results. All reported models use the same fresh test tasks. Original V1 artifacts remain in the separate `CacheScope_MASC520` folder.

For a short Chinese interpretation, read **[这轮真正改善了什么](docs/UPGRADE_SUMMARY.md)** and **[失败案例](docs/FAILURE_ANALYSIS.md)**. The primary model met two of three predeclared targets; its 6.875% harmful acceptance rate exceeded the 5% target.

## What changed

- V1's global text-retention statistics were expanded to 38 pre-answer features, including query-specific source units, linked record identifiers, corrections, table headers and JSON fields. Five absolute counts now use log transforms.
- All previous V1 main/OOD/length observations are explicitly development data in V2. The old test results provide no held-out evidence for V2.
- 360 independent new synthetic source tasks were fixed before model labeling: 120 train, 80 validation, 160 final test. Wording/layout families are split as groups. Eight high-level structures occur in every split; the final test is **not** an unseen-structure OOD benchmark.
- Models and thresholds were frozen before running the final test. The primary neural seed is 520; seeds 521/522 are reported without replacing the primary result.

## Contents

| Item | Purpose |
|---|---|
| `experiment_protocol.json` | Pre-labeling dataset hashes, comparator list and predeclared evaluation criteria |
| `data/new_tasks.jsonl` | 360 new fixed source tasks with known source facts |
| `data/source_registry.jsonl` | Independent source-fact registry; not supplied to risk features or compressor |
| `data/development_candidates.jsonl` | 600 previous V1 task records, all reassigned to development/train |
| `data/*raw_responses.jsonl` | Genuine local-model responses, prompt hashes, model digest, generation options and token counts |
| `data/new_*_candidates.jsonl` | Two observed compressed candidates per source task |
| `models/` | Saved MLP/LR/RF models and training histories |
| `assets/v1_baseline/` | Frozen V1 model, original feature definitions and threshold for same-test comparison |
| `results/frozen_model_selection.json` | Validation-selected thresholds and locked code/model/data hashes |
| `results/metrics.json` | All final test metrics, seed variation, grouped intervals and comparisons |
| `results/task_policy_decisions.csv` | One row per task and policy, including selected candidate and observed correctness |
| `docs/` | Data card and feature explanation |
| `verify_v2.py` | Recompute labels, features, prompt/cache hashes and token counts |
| `audit_sources.py` | Independently parse all 360 rendered sources and verify current gold facts |

The 960 modeling task records represent 910 independent source records: 50 old length-stress tasks are paired derivatives. Training has 720 records/670 independent sources; validation and test have 80 and 160 independent sources respectively. The previous 100 pilot tasks are excluded.

## Run

Use Python 3.11–3.13 and install `requirements.txt`. Actual execution used Python 3.13, CPU PyTorch 2.8.0, scikit-learn 1.7.2 and the bundled Qwen tokenizer. The only answering model is the locally installed Ollama `qwen2.5-coder:0.5b`, fixed digest in the protocol. No cloud API key is needed.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_features_v2.py" -v
python audit_sources.py
python verify_v2.py
python train_evaluate_v2.py evaluate
python build_v2_summary.py
```

The delivered model selection is already frozen; `train` intentionally refuses to replace it. For a new reproduction, use a separate copy with a separate results/models directory and follow the original sequence: generate data, fix the protocol, label training/validation, refresh features, train/freeze, label test, evaluate, audit. The original run commands were:

```powershell
python run_labeling.py development
python refresh_features.py new_train_validation
python train_evaluate_v2.py train
python run_labeling.py test
python refresh_features.py new_test
python train_evaluate_v2.py evaluate
python verify_v2.py
python build_v2_summary.py
```

`run_labeling.py` caches genuine model responses and verifies the fixed source/data hashes. Test labeling also verifies that the selected model artifacts and code remain frozen. Replaying cached observations checks reproducibility; it is not another independent experiment.

For a new prompt, use the saved pre-answer gate:

```powershell
python predict.py --context-file examples/context.txt --question "What is the current owner for project Demo-520?" --output examples/prediction.json
```

This command does not call the answering model. Its score is an uncalibrated conditional risk ranking, not a correctness guarantee.

## Interpretation and limits

The main comparison is quality and full-input-token use on the **same** fresh 160-task test. V1 and V2 absolute accuracy on different test sets must not be compared as a controlled improvement. Fixed-compression policies, a rule gate, LR, RF, original V1 and a V1-feature MLP trained on the expanded development data are all retained. The latter helps distinguish feature changes from added data; a single comparison cannot identify every cause.

Candidate features are rebuilt with the final V2 feature implementation before training/evaluation. The stored per-row preprocessing times were collected during labeling before that refresh; they are not measurements of V2 deployment decision overhead.

Validation tolerances are empirical targets. Test confidence intervals use a paired bootstrap over wording families; per-task McNemar values are also reported with their clustering limitation. Seeds and any failed predeclared criteria remain visible. Token reductions use complete prompts including instructions and question. They do not measure cloud charges or establish a causal wall-time speedup.

All sources are synthetic English name/date/budget records. The larger contexts here contain 478–639 Qwen context tokens; this is not a real long-document benchmark. One small answering model cannot establish generalization to other LLMs or document domains.

## Attribution

This course experiment follows the original CacheScope proposal and reuses its V1 experiment components. Data generators, implementation and documentation were developed with AI assistance. Labels come from the recorded local answering-model outputs and predefined source facts, not fabricated predictions. The tokenizer is bundled with its original Apache 2.0 license and provenance. The prior public CacheScope web demo is background work; these V2 experimental artifacts are a separate course contribution. The final course report and presentation should disclose this assistance and distinguish prior work from the new experiment.
