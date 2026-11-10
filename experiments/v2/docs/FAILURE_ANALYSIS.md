# CacheScope V2 outcome and failure analysis

The frozen primary MLP (seed 520) achieves **71.25% accuracy and 25.94% total input-token reduction** on 160 fresh test tasks. Original prompts achieve 69.375% accuracy. The gate causes **11 harmful acceptances** (6.875% of all tasks), improves 14 original-incorrect cases, and has a net gain of three correct answers. It meets the predeclared token-saving and accuracy-delta criteria but **fails the harmful-acceptance ceiling of 5%**. The complete predeclared success condition is therefore not met.

All eight high-level structures are known from training/development. The fresh test holds out source records and wording/layout families; this is not an unseen-structure OOD claim. All comparisons below use the same new 160-task test.

| Policy | Accuracy | Input-token reduction | Harmful acceptances | Original wrong → correct |
|---|---:|---:|---:|---:|
| Original | 69.375% | 0.00% | 0 | 0 |
| Always keep 60% | 62.500% | 35.57% | 25 | 14 |
| Always keep 30% | 58.125% | 60.89% | 32 | 14 |
| V2 primary MLP | 71.250% | 25.94% | 11 | 14 |
| Same expanded data, old 20 features | 68.125% | 20.68% | 14 | 12 |
| V2 Random Forest | 67.500% | 37.57% | 16 | 13 |
| Frozen original V1 MLP | 58.125% | 59.88% | 32 | 14 |

The V2 primary MLP improves accuracy and token savings relative to the old-feature MLP trained on the same expanded data, and reduces its harmful acceptances from 14 to 11. The frozen V1 model accepts more aggressively and incurs 32 harmful cases. These are different frozen operating points; they do not prove that one classifier dominates all accuracy–cost tradeoffs. Random Forest saves more tokens while making more harmful acceptances.

## Length remains a limitation

On 82 longer-band tasks the MLP scores **64.63%**, below the original **67.07%**, while saving **20.67%** of input tokens. On 78 short-band tasks it scores **78.21%**, compared with the original **71.79%**, while saving **39.30%**. The aggregate net accuracy gain therefore conceals weaker accuracy on the longer band. These are dataset-relative length bands, not a long-context benchmark.

## Auditable case selection

The following cases are the first two task IDs, in ascending order, in each of three primary-MLP outcome categories. They were selected by this explicit rule rather than by how persuasive their explanations appear. Full details and candidate scores are in `results/case_studies.json`. Reported context tokens exclude prompt wrappers; full prompt counts are supplied separately. The acceptance threshold is fixed at **0.3100292683**, selected using validation before opening the fresh test. Risk scores are uncalibrated scores, not guaranteed probabilities.

## Harmful acceptance: original correct, selected wrong

### v2-new-00137 (contrast)

Question: What is the current approved budget for project Redwood Nova-0137?

Gold: `29496`. Original answer: `USD 29496`. Selected answer: `USD 58311`.

Selected variant: `keep_30`; selected score: 0.3053826988; threshold: 0.3100292683. Candidate scores: keep_60=0.4169958234, keep_30=0.3053826988.

Context tokens: **161 → 36**. Full prompt tokens: 237 → 112.

Source excerpt:

```text
For project Redwood Nova-0137, the approved budget is not USD 58311. The verified value is USD 29496.
```

Selected-context excerpt:

```text
For project Redwood Nova-0137, the approved budget is not USD 58311.
The verified value is USD 29496.
```

Both the rejected amount and the verified current amount remain in the selected text. The compressed answer returns the rejected amount. Gold deletion is therefore not the observed failure here; the response does not follow the retained negation and verification statements. The precise reason the answer changes is unestablished.

### v2-new-00155 (contrast)

Question: What is the current approved budget for project Sequoia Nova-0155?

Gold: `54576`. Original answer: `USD 54576`. Selected answer: `USD 55576`.

Selected variant: `keep_30`; selected score: 0.1074827611; threshold: 0.3100292683. Candidate scores: keep_60=0.4583895802, keep_30=0.1074827611.

Context tokens: **159 → 34**. Full prompt tokens: 236 → 111.

Source excerpt:

```text
An audit of Sequoia Nova-0155 accepted USD 54576 for the approved budget and rejected USD 55510.
```

Selected-context excerpt:

```text
An audit of Sequoia Nova-0155 accepted USD 54576 for the approved budget and rejected USD 55510.
```

The selected text retains the entire target audit statement, including the accepted and rejected amounts. The response USD 55576 matches neither amount in that statement. This is an observed answering error despite retained evidence; the experiment does not establish its mechanism.

## Observed improvement: original wrong, selected correct

### v2-new-00029 (direct)

Question: What is the current approved budget for project Elm Nova-0029?

Gold: `39966`. Original answer: `24600`. Selected answer: `USD 39966`.

Selected variant: `keep_60`; selected score: 0.2961403430; threshold: 0.3100292683. Candidate scores: keep_60=0.2961403430, keep_30=0.3607477248.

Context tokens: **486 → 275**. Full prompt tokens: 561 → 350.

Source excerpt:

```text
Project Aster Side-0029-00 has owner Hugo Carter, release date 2027-10-04, and approved budget USD 24600.
In its authorized project file, Elm Nova-0029 records USD 39966 for the approved budget.
```

Selected-context excerpt:

```text
Project Aster Side-0029-00 has owner Hugo Carter, release date 2027-10-04, and approved budget USD 24600.
In its authorized project file, Elm Nova-0029 records USD 39966 for the approved budget.
```

The original response equals the budget of Aster Side-0029-00, an unrelated project. The selected context still includes that distractor and the target sentence, but the selected answer now matches the target. Compression and the output change co-occur; this single observation does not isolate a causal mechanism.

### v2-new-00030 (direct)

Question: What is the current owner for project Larch Nova-0030?

Gold: `Lena Wells`. Original answer: `Iris Carter`. Selected answer: `Lena Wells`.

Selected variant: `keep_30`; selected score: 0.0352570713; threshold: 0.3100292683. Candidate scores: keep_60=0.3056502938, keep_30=0.0352570713.

Context tokens: **148 → 21**. Full prompt tokens: 223 → 96.

Source excerpt:

```text
Project Kestrel Side-0030-02 has owner Iris Carter, release date 2027-11-15, and approved budget USD 62764.
One finalized record specifies the following for Larch Nova-0030: owner = Lena Wells.
```

Selected-context excerpt:

```text
One finalized record specifies the following for Larch Nova-0030: owner = Lena Wells.
```

The original response names the owner of Kestrel Side-0030-02. The selected context contains the target owner statement alone and the response is correct. This is a genuine observed improvement, not evidence that compression always improves answering.

## Remaining error: original and selected wrong

### v2-new-00027 (direct)

Question: What is the current owner for project Acacia Nova-0027?

Gold: `Noah Hayes`. Original answer: `Liam Grant`. Selected answer: `Liam Grant`.

Selected variant: `original`; selected score: not applicable (original retained); threshold: 0.3100292683. Candidate scores: keep_60=0.3665292859, keep_30=0.3510582745.

Context tokens: **482 → 482**. Full prompt tokens: 557 → 557.

Source excerpt:

```text
In its authorized project file, Acacia Nova-0027 records Noah Hayes for the owner.
Project Spruce Side-0027-09 has owner Liam Grant, release date 2027-06-26, and approved budget USD 78990.
```

Selected-context excerpt:

```text
In its authorized project file, Acacia Nova-0027 records Noah Hayes for the owner.
Project Spruce Side-0027-09 has owner Liam Grant, release date 2027-06-26, and approved budget USD 78990.
```

Both candidate scores exceed the frozen threshold, so the policy keeps the full original context. Its incorrect owner matches a different project. The target owner statement is present. Rejecting compression preserves this existing answering error rather than repairing it.

### v2-new-00091 (update)

Question: What is the current release date for project Tern Nova-0091?

Gold: `2027-09-06`. Original answer: `2027-05-28`. Selected answer: `2027-05-28`.

Selected variant: `original`; selected score: not applicable (original retained); threshold: 0.3100292683. Candidate scores: keep_60=0.3323979080, keep_30=0.4726498723.

Context tokens: **505 → 505**. Full prompt tokens: 581 → 581.

Source excerpt:

```text
Project Tern Nova-0091 once listed 2027-05-28 as its release date. The final approved correction replaces it with 2027-09-06.
```

Selected-context excerpt:

```text
Project Tern Nova-0091 once listed 2027-05-28 as its release date. The final approved correction replaces it with 2027-09-06.
```

The policy rejects both candidates and retains the original. The answer repeats the explicitly obsolete release date even though the approved correction remains present. A pre-answer compression-risk gate does not itself fix the original answering model.

## Interpretation limits

A retained correct fact does not ensure the answering model uses it correctly. Conversely, a gate that falls back to the original cannot repair every original-model error. The cases above separate these observations from claims about why a model responded as it did. No post-test feature, threshold, seed, or model change is included in these results. Any further tuning must treat this test as development data and use another untouched evaluation set.

Metrics and uncertainty estimates remain in `results/metrics.json`; the aggregate point estimate alone does not establish a statistically reliable accuracy improvement.
