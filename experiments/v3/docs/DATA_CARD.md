# CacheScope V3 data card

## Scope

These data support pre-answer prompt-compression risk prediction for synthetic English factual extraction. Each task asks for the current owner, release date, or approved budget of one named project. Eight known high-level structures are used: direct statements, paragraph headings, updates, contrasts, reference links, tables, nested JSON, and paired directories.

All sources and facts are synthetic, produced by AI-assisted Python code. They are not real organizational documents or independently human-annotated natural text. Results cannot establish performance on real documents, other languages, or other answering models.

## Dataset accounting and prior-test reuse

| Component | Task records | Independent source records | V3 use |
|---|---:|---:|---|
| Reused V1 main and structural sources | 550 | 550 | Development/training |
| Reused V1 paired length derivatives | 50 | 0 additional | Development/training |
| All newly generated V2 sources | 360 | 360 | Development/training |
| New V3 calibration sources | 160 | 160 | Operating-point selection |
| New V3 final-test sources | 200 | 200 | Final evaluation after model selection is frozen |
| **Total** | **1,320** | **1,270** | |

The old development/training set has **960 task records representing 910 independent sources**. The remaining 50 records are paired length derivatives of older sources, not additional independent evidence. The previous V1 100-task pilot is excluded from these modeling totals.

Every V1/V2 result has already been inspected. Their former validation, test, structural-holdout and length-stress labels are therefore explicitly development evidence for V3. They cannot be presented as independent V3 validation. The 160 calibration and 200 final-test sources are new and independent. With two compression candidates per task, 1,320 task records correspond to 2,640 candidate records; candidates are not independent tasks.

## Fresh V3 partition design

The generation seed is **1113520**, fixed without using new answering-model outcomes. `make_v3_data.py` creates 360 new tasks and a separate canonical source-fact registry.

- Each of eight structures contributes 20 calibration and 25 final-test tasks.
- Each wording/layout family contributes five tasks.
- There are **72 new families**: 32 calibration and 40 final-test families.
- Family assignments never cross calibration and test.
- Every structure and both partitions include the two named length bands.

The five prose structures use 45 newly written target templates. Table families place the Project column in an interior position; V2 table families used its first or last position. JSON families use new root/detail keys with an additional visible card wrapper. Paired-directory families use newly written mapping sentences and identifier-to-attribute-card layouts. Dates are in 2028; full target entities, task IDs, source IDs and context strings are distinct from V1/V2 sources.

All eight **high-level structures are already known from development**. The fresh test holds out source records and wording/layout families. It is not an entirely unseen-structure OOD experiment. Structured-family changes are deliberately visible layout variations within the known format types.

## Context lengths

The bundled Qwen tokenizer measured these ranges before answering-model calls:

| Generator band | Context-token range |
|---|---:|
| Short | 153–248 |
| Longer | 511–633 |

There are 184 short-band and 176 longer-band source records across calibration and final test. Context-token counts exclude the system instruction, question and prompt wrapper. Token-savings metrics must use complete prompt counts instead.

Longer sources contain additional project records with their own owner/date/budget facts. They are not created by adding empty repetition. The band name “longer” is relative to this dataset; contexts of at most 633 tokens do not constitute a real long-document benchmark.

## Source facts and independent gold audit

`data/source_registry.jsonl` contains canonical current attributes, obsolete/rejected attributes and distractor records. This registry is audit material. It must not enter the answering prompt, compressor or risk features. The answering prompt receives only the rendered context and question; gold answers are used only for offline correctness scoring.

Each requested current attribute differs from every distractor's corresponding attribute. Update/contrast old values differ from current values. Sources have one unambiguous current answer under their predefined semantics.

`audit_sources.py` independently parses the **rendered source text**, without reading model responses, candidates, scores, or sibling project files:

- For 225 prose sources it isolates the target paragraph and parses the requested name/date/USD value. Fixed update/contrast semantics identify the current rather than obsolete or rejected value.
- For 135 structured sources it recovers the target row, nested JSON record or explicit directory-key join, and verifies all three emitted current attributes against the registry.
- It verifies the requested gold, aliases, uniqueness constraints, family separation, partition counts and frozen source hashes for all 360 sources.

Run from the project directory:

```powershell
python audit_sources.py
```

The script uses only the Python standard library and writes `results/source_audit.json`. It works when V1/V2 sibling folders are absent. New-versus-old task/source/context disjointness was checked during generation and recorded in `data/generation_audit.json`; the independent source audit does not claim to recheck absent older datasets.

## Frozen source identities

`data/fresh_tasks.jsonl` SHA-256:

`2415d5276d6163c1fbe3c3f3f5fb12da875e9483444a750524eef709220c7829`

`data/source_registry.jsonl` SHA-256:

`95334cc0442051bc3201bc2448ffc1a6e8ed6220334bc0ea8c8c6b9bede3c7b2`

The generator and three data files were finalized before new answering-model labels were observed. Fact auditing tests internal consistency of controlled synthetic tasks; it is not external validation of natural-document quality.

## Label and evaluation discipline

Observed regression is `original_correct and not compressed_correct`. Conditional risk-classifier training excludes original-incorrect examples from that particular target. Policy evaluation must retain all source tasks, including original-incorrect tasks, and select a prompt using only pre-answer inputs.

Calibration may choose operating points; final-test observations must not guide model selection. Risk scores are not automatically calibrated failure probabilities. Empirical calibration targets cannot guarantee final-test risk. Uncertainty estimates should account for five tasks sharing each wording/layout family, and paired old length derivatives must not be counted as independent new sources.
