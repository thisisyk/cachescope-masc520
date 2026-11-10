# Saved-model smoke check and a known limitation

`context.json` and `prediction.json` are a manual pre-answer CLI check, separate from the predefined benchmark. No answering-model call was made, so this example has no measured answer-correctness label.

The selected 30% candidate is a valid but empty JSON records container. The source includes the queried project, while the selected context loses it. The structure guard permits that candidate because it checks JSON syntax, not semantic completeness. The selected frozen V2 risk model also underestimates the loss in this example.

This is retained as a known limitation, not a successful safe-compression demonstration. The 54-feature V3 extractor can measure source-record binding loss, but the calibration-selected primary uses frozen V2 risk weights with the syntax guard. Model selection and final-test choices remain frozen; this manual example does not enter the 200-task benchmark or change any model decision.

The current empirical goals concern the specified synthetic benchmark. They do not establish safe compression for arbitrary JSON layouts. A stronger binding-preservation guard is a possible future improvement that would require a separate validation and fresh evaluation.
