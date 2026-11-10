import unittest

from cachescope.evaluation import choose_candidate, make_decisions, select_threshold, summarize_decisions


def pair(task_id="t1", split="validation", original_correct=True):
    base={"task_id":task_id,"split":split,"group_id":task_id,"task_type":"date","original_tokens":100,"original_correct":original_correct,"original_answer":"A","original_latency_s":1.0,"features":{"question_overlap_change":0.,"numeric_retention":1.,"date_retention":1.,"negation_retention":1.}}
    return [
        {**base,"candidate_id":"keep_60","keep_ratio":.6,"compressed_tokens":60,"score_mlp":.1,"compressed_correct":True,"compressed_answer":"A","compressed_latency_s":.7},
        {**base,"candidate_id":"keep_30","keep_ratio":.3,"compressed_tokens":30,"score_mlp":.3,"compressed_correct":False,"compressed_answer":"B","compressed_latency_s":.5},
    ]


class EvaluationTests(unittest.TestCase):
    def test_shortest_admissible_without_outcome_oracle(self):
        rows=pair()
        self.assertEqual(choose_candidate(rows,"score_mlp",.4)["candidate_id"],"keep_30")
        self.assertEqual(choose_candidate(rows,"score_mlp",.2)["candidate_id"],"keep_60")

    def test_reject_and_count_task_once(self):
        decisions=make_decisions(pair(),"mlp","score_mlp",.05)
        self.assertEqual(len(decisions),1)
        self.assertEqual(decisions[0]["candidate_id"],"original")
        metrics=summarize_decisions(decisions)
        self.assertEqual(metrics["n_tasks"],1)
        self.assertEqual(metrics["total_input_tokens"],100)
        self.assertIsNone(metrics["harmful_accepted_per_accepted"])

    def test_threshold_rejects_test_and_uses_validation_labels_only(self):
        selected=select_threshold(pair(),"score_mlp",epsilon=0.)
        self.assertEqual(selected["threshold"],.1)
        with self.assertRaises(ValueError):
            select_threshold(pair(split="test"),"score_mlp")

    def test_original_incorrect_tasks_are_in_policy_denominator(self):
        rows=pair("a")+pair("b",original_correct=False)
        metrics=summarize_decisions(make_decisions(rows,"mlp","score_mlp",.2))
        self.assertEqual(metrics["n_tasks"],2)
        self.assertEqual(metrics["accuracy"],1.)
        self.assertEqual(metrics["improvement_count"],1)
        self.assertEqual(metrics["accuracy_delta"],.5)


if __name__ == "__main__":
    unittest.main()
