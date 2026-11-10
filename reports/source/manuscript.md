# CacheScope Neural Risk Prediction for Prompt Compression

MASC 520 Course Project

## Abstract

CacheScope estimates whether prompt compression will turn a correct factual answer into an incorrect answer. A fixed extractive compressor proposes two candidates, and a lightweight neural risk model accepts a candidate before the answering model runs or falls back to the original. We implement binary labels of answer degradation, grouped data partitions, multilayer perceptron (MLP) training, traditional baselines, and task-level accuracy–token evaluation. All previously inspected experiments become development data; the final study uses 160 new calibration tasks and 200 new test tasks with disjoint wording/layout families. The calibration-selected hybrid, comprising frozen earlier MLP weights and a JSON syntax check, achieves 75.00% accuracy versus 69.00% for original prompts, saves 26.23% of full input tokens, and introduces seven harmful compressions among 200 tasks. All three retrained single MLPs also meet the empirical targets; the maximum-score ensemble misses the token target. The frozen unguarded baseline also meets the targets, so the experiment does not establish a net benefit from the JSON rule or universal superiority from retraining. Results support a controlled feasibility finding, not a deployment risk guarantee.

Index Terms: prompt compression, selective prediction, neural risk estimation, factual extraction, reproducible evaluation.

## I Introduction

Shortening a language-model prompt changes both its input cost and the evidence available for answering. Removing irrelevant records can help factual extraction, while removing a reference, correction, or attribute binding can make a previously correct answer wrong. A fixed retention ratio therefore does not directly specify an acceptable accuracy loss.

Prompt-compression research studies information selection, including selective context removal and learned compression [1]-[3]. CacheScope addresses a complementary decision: should a particular candidate compression be accepted? This resembles selective prediction in allowing a model to decline an uncertain action [4]. Here, declining compression retains the original prompt rather than abstaining from answering. The answering model remains separate from the risk network.

The original proposal specified a two-hidden-layer MLP, features available before answering, factual questions with checkable names/dates/amounts, and comparisons against logistic regression (LR), random forest (RF), original prompts, and fixed compression. The implemented study follows that design. Its main contributions are an auditable labeling pipeline, controlled comparisons of neural and hybrid gates, and an analysis of failures that simple text-retention statistics miss. The initial long-text ambition is narrower in the completed benchmark: final contexts contain at most 633 context tokens, and the sources remain synthetic.

## II Task and Data

For source context x, question q, and candidate c, let O indicate correctness on the original prompt and C indicate correctness on the candidate. The observed regression label is y = 1 when O = 1 and C = 0. A safe conditional training example has O = 1 and C = 1. Candidates with O = 0 are excluded from risk-classifier fitting: failure on the original cannot establish preservation of a correct answer. They remain in every task-level policy evaluation, including cases improved by compression.

Each source asks for a project's current owner, release date, or approved budget. Eight structures cover direct statements, paragraph headers, historical updates, rejected alternatives, reference links, tables, nested JSON, and paired project/attribute directories. Canonical facts are sampled before model calls. An independent parser checks the requested current value in every rendered new source; this is an internal consistency audit, not human validation of natural documents.

Table I records the final data accounting. Earlier V1/V2 results, including their former tests, had already been inspected and are explicitly development data. Fifty older length variants share sources and are not additional independent evidence. The original 100-task pilot is excluded. The 360 new sources use 72 new wording/layout families, five tasks per family, with no family crossing calibration and test. All eight high-level structures were previously observed; this is not unseen-structure evaluation.

**Table I. Final study data accounting.**

| Component | Task records | Independent sources | Wording groups | Use |
|---|---:|---:|---:|---|
| Previous experiments | 960 | 910 | 125 | Development |
| New calibration | 160 | 160 | 32 | Operating-point selection |
| New final test | 200 | 200 | 40 | Held-out evaluation |
| Total | 1,320 | 1,270 | - | 2,640 candidate rows |

New shorter contexts contain 153-248 Qwen tokens; longer contexts contain 511-633. Longer sources add independent project records with distractor attributes. These relative bands do not constitute a long-document benchmark. New task identifiers, sources, and complete contexts do not overlap older records.

## III Compression and Risk Models

### A Fixed compression and answering model

The compressor splits source text into sentence/line units, ranks units by question-word overlap, then overlap density and original order, and greedily fills a token budget. Selected units return to source order. Two targets retain approximately 60% and 30% of context tokens; actual retention varies with unit boundaries. Neither gold facts nor model responses enter this algorithm.

Original and candidate prompts are answered by the same local `qwen2.5-coder:0.5b` model from the Qwen2.5-Coder family [5]. The experiment fixes model digest, raw ChatML wrapper, temperature 0, seed 520, context limit 2,048, and maximum generation length 24 tokens. The system requests the current approved value, a value-only answer, and UNKNOWN when evidence is absent. Predefined normalization and exact matching score names, dates, and currency aliases. Genuine responses, generation settings, prompt hashes, and backend token counts are stored; no simulated-answer fallback is used.

### B Pre-answer features and neural training

V3 uses 54 features. The original 38 describe log lengths, retained lexical/numeric/date content, repetition, question overlap, source-unit retention, correction statements, identifiers, table headers, and JSON field support. Sixteen additions describe the question's field, JSON parse validity and queried-record bindings, competing values/records, revision and negation burden, link distance, target position, and delimiter imbalance. Feature extraction receives only x, q, c, and full-prompt token counts. It receives no gold answer, observed response, correctness label, task type, or split identifier. Source-side scalar retention is an input-text comparison, not access to a separate reference answer.

Each new MLP has layers 54-32-16-1, ReLU hidden activations, and a sigmoid risk score, totaling 2,305 parameters. Weighted binary cross-entropy uses the training negative/positive ratio as positive-class weight. Adam uses learning rate 0.001, weight decay 0.0001, and batches of 32. A grouped internal split contains 735 development task records for training and 225 for checkpoint selection. Scaling is fitted only on the corresponding original-correct training candidates. Maximum training is 100 epochs with patience 10. Seeds 520, 521, and 522 select 11, 10, and 20 epochs, respectively. Each network is then reinitialized and fitted for that fixed epoch count on all prior development observations; its final scaler uses only that fitting population. This yields 1,374 original-correct candidate rows, including 322 regressions. Fresh calibration does not fit weights.

LR and RF provide traditional comparisons. LR uses balanced classes and at most 2,000 iterations; RF uses 300 trees, maximum depth 8, minimum leaf size 3, and balanced classes. Both use the same standardized development features. The neural ensemble uses the maximum of the three fixed-seed scores, a deliberately conservative aggregation rather than a test-selected seed. Scores are uncalibrated conditional rankings, not guaranteed failure probabilities.

### C Hybrid gate and deployment decision

A separate structural check rejects a candidate when the original parses as a JSON container but the candidate does not. It does not establish correct attribution or semantic completeness. Pure-neural and guard-only policies therefore remain separate controls. The frozen V2 baseline retains its own 38-feature extractor and weights; it is not silently evaluated with the new feature schema. Those weights were fitted in V2 on 720 task records and are reused without refitting; the hybrid's threshold is selected on the new calibration set.

For a fixed threshold, the policy chooses the shortest candidate whose score passes and whose full prompt is shorter than the original; otherwise it retains the original. A hybrid also requires the structural check. Selection uses no response or correctness information. Deployment needs one answering-model call on the selected prompt. Running the original and both candidates is an offline labeling expense.

## IV Experimental Protocol and Metrics

All input sources and generation settings were fixed before new labeling. Two primary candidate families were declared: the new neural ensemble with the JSON check and the frozen V2 MLP with that check. Calibration visits empirical harm ceilings of 1.25%, 1.875%, and 2.5%, from strictest upward. The first tier with a candidate saving at least 20% and losing no more than 1.25 accuracy points is used. Within it, savings choose the primary policy, followed by accuracy and a fixed tie rule. If no tier is feasible, the test does not run. These buffers are decision rules, not statistical safety certificates.

The strictest tier was feasible. Calibration selected the frozen V2 hybrid at threshold 0.304512918: 80.625% accuracy, 25.80% savings, and 2/160 harmful tasks. The new 54-feature ensemble was not selected. All models, thresholds, primary identity, and relevant code/data hashes were frozen before final test labeling. Sixteen predefined policies are retained, including individual neural seeds, pure/hybrid ensemble, LR/RF variants, fixed retention, the simple retention rule, guard-only, and the frozen V2 baseline.

Task accuracy counts each question once. Token saving is one minus selected full-input tokens divided by original full-input tokens, aggregated over tasks; it includes instructions, question, and wrapper. Harm is the fraction of all tasks for which the original was correct and selected compression was wrong. Improvement counts the reverse event separately. The final empirical targets are saving at least 20%, accuracy change at least -2.5 percentage points, and harm at most 5%. The denominator of the harm target is all tasks, not only accepted compressions.

Uncertainty is estimated by 2,000 paired bootstrap resamples of wording groups. Per-task exact McNemar values are supplementary because they ignore within-family dependence; multiple comparisons are not corrected. Neither test curves nor favorable test seeds select the primary policy.

## V Results and Observations

### A Final task-level comparisons

Table II shows representative results on exactly the same 200 new tasks. The locked hybrid reaches all three targets: 75.00% accuracy versus 69.00% for original prompts, 26.23% full-input-token saving, and 7/200 harmful cases (3.50%). It accepts compression on 117 tasks, improves 19 originally incorrect answers, and introduces seven regressions, producing 12 additional correct answers overall. Among accepted compressions, harm is 7/117 = 5.98%; this differs from the 3.50% all-task target.

**Table II. Policies on the same final test. All percentages are measured, not guarantees.**

| Policy | Accuracy % | Input saving % | Harmful tasks | Improved tasks |
|---|---:|---:|---:|---:|
| Original | 69.00 | 0.00 | 0 | 0 |
| Fixed keep 60% | 62.50 | 35.86 | 23 | 10 |
| Fixed keep 30% | 61.00 | 61.13 | 39 | 23 |
| JSON guard only | 64.00 | 52.19 | 31 | 21 |
| Frozen V2 MLP | 75.50 | 29.67 | 7 | 20 |
| Locked V2 MLP plus guard | 75.00 | 26.23 | 7 | 19 |
| New MLP seed 520 | 74.50 | 26.12 | 3 | 14 |
| New maximum-score ensemble | 73.50 | 19.67 | 3 | 12 |
| New LR | 75.50 | 24.83 | 3 | 16 |
| New RF | 73.00 | 20.90 | 5 | 13 |

The primary accuracy change has a group-bootstrap 95% interval of +1.00 to +12.00 points; savings have an interval of 20.01%-32.57%. These estimates support the observed tradeoff within sampled families. They do not establish a future harm ceiling. Fixed 60% retention loses accuracy and creates 23 harmful cases; the primary has 25 more correct answers and 16 fewer harmful cases, while sacrificing some compression. This is a quality-token tradeoff, not an equal-budget causal comparison.

The frozen unguarded V2 model also meets all targets and has higher accuracy and savings than the locked hybrid, with the same seven harmful tasks. Thus, this final test does not demonstrate a net benefit from the JSON rule. Its favorable old-development diagnostic motivated evaluation but did not guarantee transfer. The primary identity remains the calibration choice rather than being replaced by the best test row.

### B Retrained models and conservative ensembling

All three preset single MLPs meet the targets: seed 520 reaches 74.50% accuracy, 26.12% savings, and 1.50% harm; seed 521 reaches 73.00%, 24.32%, and 2.50%; seed 522 reaches 74.50%, 21.61%, and 1.50%. The seed-520 risk ranking has AP 0.659 and ROC-AUC 0.834 on 276 original-correct test candidates, of which 62 regress. These are descriptive conditional classification metrics.

The maximum-score ensemble saves only 19.67%, missing the 20% target despite three harmful tasks. Adding the syntax check gives identical policy outcomes at the frozen operating points. Conservative aggregation can therefore reduce efficiency too far. LR is also competitive, so the study does not establish neural superiority over traditional models. Compared with the frozen V2 baseline, seed 520 reduces harmful tasks from seven to three, but loses two correct answers and some token savings; thresholds and coverage differ.

### C Length and failure behavior

For 102 shorter tasks, the primary reaches 84.31% accuracy, saves 40.15%, and harms three tasks; original accuracy is 73.53%. For 98 longer tasks, it reaches 65.31%, saves 20.09%, and harms four; original accuracy is 64.29%. Overall success therefore includes uneven subgroup performance, and the longer-band saving is close to the target.

Failures can occur despite intact evidence. In task `v3-fresh-00122`, the original correctly returns budget 73002. The selected 30% candidate retains both the withdrawn 74111 value and the final 73002 update, yet the answering model returns 74111. Its learned score passes while the correct 60% candidate's score fails. This identifies a nonmonotonic decision error without establishing why the answering model misread the retained update. A separate manual CLI example selects valid but empty JSON. No answer call was made for that example; it documents semantic evidence loss outside the predefined benchmark. Syntax validity alone is insufficient.

## VI Scope Reproducibility and Conclusion

Results use one small quantized answering model and controlled English facts. Eight high-level structures were seen during development, sources use limited vocabulary, and the final contexts are short relative to real long documents. The study does not supply the independently human-authored holdout initially envisaged in the proposal. Differences across the V1, V2, and V3 test sets cannot be interpreted as causal gains from retraining: each version has different sources and wording. Earlier failures remain part of the experimental record.

The repository includes generators, stored tasks and genuine responses, trained weights, calibration attempts, frozen manifests, per-task decisions, figures, and case analyses. Verification recomputes labels, full-prompt token counts, features, model scores, guard flags, and primary metrics for 2,640 candidate rows and 3,960 unique stored responses. Source audits and nine feature/guard behavior tests pass. Cached replay checks reproducibility, not a new independent experiment. Token reduction is not a measured monetary or causal latency saving.

OpenAI Codex [6] assisted implementation, synthetic-source generation, analysis scripts, and manuscript preparation. Numerical claims derive from stored model observations and auditable computations; no response labels were invented. Prior CacheScope interface work is distinguished from the course-specific experiment.

CacheScope achieves its three empirical targets on the final controlled test, and all three retrained single MLPs also achieve them. The useful result is a measured selective-compression tradeoff with transparent failures. The frozen baseline's strength, the ensemble's missed efficiency target, and semantic failures under valid JSON prevent a broader claim of universal neural or structural-rule improvement.
