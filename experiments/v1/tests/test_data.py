import unittest

from cachescope.data import generate_tasks, generate_ood_tasks
from cachescope.compression import compress_context
from cachescope.scoring import is_correct


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class DataTests(unittest.TestCase):
    def test_reproducible_group_disjoint_and_complete(self):
        tasks = generate_tasks()
        self.assertEqual(tasks, generate_tasks())
        self.assertEqual(len(tasks), 1000)
        self.assertEqual(len({t['task_id'] for t in tasks}), 1000)
        groups = {}
        for task in tasks:
            groups.setdefault(task['group_id'], set()).add(task['split'])
            self.assertTrue(is_correct(task['answer'], task))
            self.assertTrue(task['answer_aliases'])
        self.assertEqual(len(groups), 50)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        counts = {split: sum(t['split'] == split for t in tasks) for split in ['train','validation','test']}
        self.assertEqual(counts, {'train': 600, 'validation': 200, 'test': 200})

    def test_pilot_and_ood_distinct(self):
        main = generate_tasks()
        pilot = generate_tasks(n_tasks=100, pilot=True)
        ood = generate_ood_tasks()
        self.assertEqual(len(pilot), 100)
        self.assertEqual(len(ood), 50)
        self.assertTrue(all(t['split'] == 'pilot' for t in pilot))
        self.assertTrue(all(t['split'] == 'ood' for t in ood))
        all_tasks = main + pilot + ood
        self.assertEqual(len({t['task_id'] for t in all_tasks}), len(all_tasks))
        self.assertEqual(len({t['source_id'] for t in all_tasks}), len(all_tasks))

    def test_compression_does_not_increase_or_require_gold(self):
        tokenizer = WordTokenizer()
        for task in generate_tasks(n_tasks=100):
            for ratio in [.3, .6]:
                result = compress_context(task['context'], task['question'], ratio, tokenizer)
                self.assertGreater(len(result['context']), 0)
                self.assertLessEqual(result['compressed_context_tokens'], result['original_context_tokens'])
                self.assertLessEqual(result['keep_ratio_actual'], 1)

    def test_strict_scoring(self):
        task = {'answer':'2026-03-04','answer_aliases':['2026-03-04','March 4, 2026'],'answer_type':'date'}
        self.assertTrue(is_correct('Answer: March 4, 2026.', task))
        self.assertFalse(is_correct('2026-03-04 or 2026-03-05', task))
        self.assertFalse(is_correct('2026-03-040', task))
        task = {'answer':'23450','answer_aliases':['23450'],'answer_type':'amount'}
        self.assertTrue(is_correct('The approved budget is USD 23,450.', task))
        self.assertFalse(is_correct('123450', task))
        self.assertFalse(is_correct('23450, not 34560', task))
        task = {'answer':'Ada Reed','answer_aliases':['Ada Reed'],'answer_type':'name'}
        self.assertTrue(is_correct('"Ada Reed"', task))
        self.assertFalse(is_correct('Ada Reed and Mia Cole', task))


if __name__ == '__main__':
    unittest.main()
