# V3 features and the explicit structural guard

Feature extraction receives only original context, question, compressed context, and the original/compressed full-prompt token counts. It receives no gold answer, observed answer, correctness label, task type, split, task ID, or source registry. Reading a scalar already present in the source for a retention comparison does not supply an external reference answer.

The schema has **54 features**. The first 38 definitions and their order are exactly V2's. Sixteen additional features describe:

- question wording identifying owner/date/budget fields;
- whether the original and compressed texts parse as JSON containers;
- retention of the queried source key/scalar binding inside the same query-matched JSON record;
- currency/date value counts and negation/revision counts in the local source support;
- compressed units mentioning the requested field outside the source support;
- units sharing only part of the queried entity's lexical anchors;
- distance between query-anchor units and explicitly linked units;
- the queried entity's first position in the candidate;
- a lexical count of competing values for the queried field;
- unbalanced JSON delimiters, ignoring delimiters inside quoted strings.

The person-value count is a capitalized two-word lexical proxy, not a person recognizer. Entity anchors and support use the same transparent token, identifier and paragraph heuristics as V2. These features cannot certify semantic attribution or that an answering model will follow a retained correction. Unknown field vocabulary, implicit relationships, malformed originals and novel document formats remain limitations.

## Separate hybrid-policy component

`preanswer_guard(features, mode="structural")` returns whether a candidate is allowed by an explicit structural check. The structural mode rejects exactly this condition:

```text
original_json_valid == 1 and compressed_json_valid == 0
```

It does not reject prose merely because prose is not JSON. It does not use gold values or task-type metadata. Valid JSON with the wrong or missing queried record passes this syntax guard; the neural model must still assess the evidence-retention features. The guard's output is not a semantic safety guarantee.

`mode="none"` always allows the candidate. Unsupported modes raise an error. The guard is distinct from learned neural risk scores: results using it must be called a **hybrid** policy, with a pure-neural and guard-only comparison where available.

## Development evidence, not fresh V3 results

All V1/V2 observations, including the former V2 final test, are development data for V3. An inspection of V2's 11 harmful accepted cases found six nested-JSON candidates with invalid JSON syntax despite V2's lexical evidence-closure feature being 1.0. The remaining cases were two contrast amounts, two table names and one reference amount. These observations motivated syntactic validity and source-record binding features.

Keeping the V2 scores and threshold but adding the structural guard changes its **old V2 test, now development** outcome from 71.25% accuracy / 25.94% token savings / 11 harmful accepts to 75.00% / approximately 22.93% / 5 harmful accepts. This is an inspected development diagnostic; it is not an independent V3 evaluation, and it must not replace new calibration and final-test evidence.

A fixed four-fold grouped development diagnostic used all 1,920 old candidate rows, keeping source/template groups together. Each MLP used train-only scaling, seed 520, the same 32/16 hidden-layer architecture, and group-disjoint inner early-stopping data. It compared the original first 38 features with all 54, without reading any future V3 calibration/test tasks or outputs:

| Feature set | Pooled conditional out-of-fold AP | Pooled conditional ROC-AUC |
|---|---:|---:|
| V2 38 features | 0.7107 | 0.8217 |
| V3 54 features | 0.7263 | 0.8395 |

Not every fold improved. These pooled metrics combine uncalibrated scores from separate fitted folds, and are only a development signal. Complete fold details and the feature-file hash are recorded in [results/development_feature_cv.json](../results/development_feature_cv.json). The delivered JSON is the actual previously completed development diagnostic, copied without rerunning it during calibration. These are not fresh-test claims and do not establish a statistically significant or universal improvement.

## Behavioral checks

Nine portable standard-library tests in `tests/test_features_v3.py` passed. They check valid-to-invalid JSON rejection, valid-record acceptance, missing record binding, prose acceptance, ambiguity counts, shared-name competition, question-derived field indicators, quoted delimiter handling, finite ordered outputs, and unsupported guard modes. On all 1,920 old candidate rows, the first 38 feature values were also verified to equal V2 exactly, and all 54 values were finite.

Run from the project root:

```powershell
python -m unittest discover -s tests -p "test_features_v3.py" -v
```

The portable diagnostic code is [development_cv.py](../development_cv.py). It reads only the bundled `data/development_candidates.jsonl` (1,920 old candidate rows), uses `assets/v2_baseline/features.py` for the original 38-feature definitions and `src/cachescope/features.py` for the 54-feature definitions, and repeats the same fixed grouped-CV algorithm. It never reads new V3 calibration or final-test files and never writes to `models/`. Running it overwrites only the diagnostic result JSON:

```powershell
python development_cv.py
```

The V3 training/evaluation pipeline remains responsible for fitting the final models, selecting thresholds from permitted development/calibration data, freezing artifacts, and evaluating the untouched final test. Recomputing this old-development diagnostic is not another independent V3 test.
