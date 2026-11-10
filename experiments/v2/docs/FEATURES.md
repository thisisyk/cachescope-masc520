# CacheScope V2 pre-answer features

The feature implementation is `src/cachescope/features.py`. V1 remains untouched. No answering model, compressor, gold answer, generated response, label, split, task type, or task ID is an input to feature extraction.

The schema has 38 ordered features. The original 20-feature framework is retained, with five absolute quantities changed to `log1p` and renamed: original/compressed full-prompt token counts, original sentence count, and original/compressed mean words per sentence. Ratios remain ratios. Saved V1 model bundles require V1 features; V2 bundles must be retrained.

The 18 additional features cover query anchor count; whole anchor-unit, anchor-word and anchor-number retention; neighboring-unit retention; explicit identifier and linked-unit retention; linked-unit count; local evidence-closure retention; queried-field-unit retention; presence/retention of correction statements and their neighbors; table-header retention; JSON key retention; queried JSON field binding retention; and change in normalized anchor position.

Query anchors derive from question terms present in the context, excluding function words and common field vocabulary. Rare terms receive greater weight. Evidence closure includes source units containing the anchors, source units sharing visible identifiers, and short single-line paragraphs binding an entity to adjacent references/corrections. Multi-line tables/directories are not assumed to describe one entity. JSON source records use visible field boundaries; queried field retention checks source key/value text only and returns a fraction, never an answer.

Explicit spaced identifiers recognize the generic record words `item`, `record`, `entry`, `ticket`, `key`, `code`, `file`, `slot`, `index`, `unit`, `reference`, `ref`, and `id`, followed by a numeric or alphanumeric identifier. The directory-dependency test is parameterized across all thirteen words. This vocabulary extension was made during code review before model selection and fresh-test evaluation; it did not use new answering-model outcomes.

Six behavioral tests in `tests/test_features_v2.py` pass: deleting an Item attribute record reduces linked support; deleting a correction while retaining an obsolete value reduces correction support; appending an unrelated record cannot repair missing linked support; dropping a table header is detected; dropping the queried JSON field is detected; all 38 features are finite and ordered. The development diagnostic script also recomputed 1,200 V1 candidate rows and checked finite deterministic features plus identity retention.

From the project directory, run the portable feature tests with Python 3.10 or newer. They use only the standard library and do not call the answering model:

```powershell
python -m unittest discover -s tests -p "test_features_v2.py" -v
```

The V2 training script recomputes features through this module. `python train_evaluate_v2.py train` trains on declared development/train observations, uses fresh validation observations for early stopping and threshold selection, and freezes the model choices. `python train_evaluate_v2.py evaluate` opens the fresh-test observations only after checking the frozen hashes. Training and evaluation need the packages in `requirements.txt`; feature-only tests do not.

Old V1 tests are now development evidence, not an independent V2 test. No newly generated V2 task file or outcome was read. On old original-correct candidates, mean evidence-closure retention for regression versus safe rows is .737 versus .994 (main), .684 versus .928 (structural), and .733 versus 1.000 (length). These diagnostics show a useful development signal, not validated future performance.

Limitations: this extractor is a transparent lexical heuristic, not a semantic verifier. An intact source fact may still be misread by the answering model. Punctuation-only delimiters carry no lexical retention weight, so features do not certify valid JSON or a complete table schema. A surviving value may be attributed to the wrong entity. Novel identifiers, field wording, prose dependencies, or differently nested structures may escape the heuristics. New held-out evaluation is necessary.
