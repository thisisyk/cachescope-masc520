"""Pre-answer feature and explicit-guard behavioral checks, with no labels."""
import importlib.util
from pathlib import Path
import math
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v3_features',ROOT/'src/cachescope/features.py')
features=importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

class V3FeatureTests(unittest.TestCase):
    question='What is the current approved budget for project Cedar Demo-520?'

    def extract(self,source,candidate,question=None):
        return features.extract_features(source,question or self.question,candidate,400,200)

    def test_valid_json_becoming_invalid_is_rejected(self):
        source='{"records":[{"project":"Cedar Demo-520","details":{"approved budget":"USD 12000"}}]}'
        candidate='"approved budget":"USD 12000"\n"project":"Cedar Demo-520"'
        row=self.extract(source,candidate)
        self.assertEqual(row['original_json_valid'],1)
        self.assertEqual(row['compressed_json_valid'],0)
        self.assertFalse(features.preanswer_guard(row))
        self.assertTrue(features.preanswer_guard(row,mode='none'))

    def test_valid_record_is_accepted_by_structure_guard(self):
        source='{"records":[{"project":"Cedar Demo-520","details":{"approved budget":"USD 12000"}},{"project":"Other","details":{"approved budget":"USD 99000"}}]}'
        candidate='{"records":[{"project":"Cedar Demo-520","details":{"approved budget":"USD 12000"}}]}'
        row=self.extract(source,candidate)
        self.assertEqual(row['json_target_binding_retention'],1)
        self.assertTrue(features.preanswer_guard(row))

    def test_valid_json_with_wrong_record_does_not_certify_binding(self):
        source='{"records":[{"project":"Cedar Demo-520","details":{"approved budget":"USD 12000"}}]}'
        candidate='{"records":[{"project":"Cedar Other-520","details":{"approved budget":"USD 12000"}}]}'
        row=self.extract(source,candidate)
        self.assertEqual(row['json_target_binding_retention'],0)
        # Syntax guard is intentionally separate from learned evidence risk.
        self.assertTrue(features.preanswer_guard(row))

    def test_prose_is_not_rejected_as_invalid_json(self):
        source='Project Cedar Demo-520 has the approved budget USD 12000.'
        row=self.extract(source,source)
        self.assertEqual(row['original_json_valid'],0)
        self.assertTrue(features.preanswer_guard(row))

    def test_ambiguity_counts_both_rejected_and_accepted_currency(self):
        direct='Project Cedar Demo-520 has approved budget USD 12000.'
        contrast='Project Cedar Demo-520 does not have approved budget USD 99000. The accepted value is USD 12000.'
        simple=self.extract(direct,direct)
        ambiguous=self.extract(contrast,contrast)
        self.assertGreater(ambiguous['target_currency_count_log'],simple['target_currency_count_log'])
        self.assertGreater(ambiguous['target_negation_count_log'],simple['target_negation_count_log'])

    def test_shared_entity_stem_is_competition(self):
        source='Project Cedar Demo-520 has approved budget USD 12000.\n\nProject Cedar Other-2 has approved budget USD 99000.'
        row=self.extract(source,source)
        self.assertGreater(row['compressed_partial_anchor_competitors_log'],0)

    def test_field_indicators_derive_from_question(self):
        source='Project Cedar Demo-520 has owner Ada Reed, release date 2027-01-02 and approved budget USD 12000.'
        name=self.extract(source,source,'Who is the owner for project Cedar Demo-520?')
        date=self.extract(source,source,'What is the release date for project Cedar Demo-520?')
        self.assertEqual((name['query_owner'],name['query_date'],name['query_budget']),(1,0,0))
        self.assertEqual((date['query_owner'],date['query_date'],date['query_budget']),(0,1,0))

    def test_quoted_braces_are_not_structural_errors(self):
        self.assertEqual(features._json_balance_error('{"note":"literal { [ } ]"}'),0)
        self.assertGreater(features._json_balance_error('{"note":"x"'),0)

    def test_schema_finite_ordered_and_guard_mode_validated(self):
        row=self.extract('', '')
        self.assertEqual(tuple(row),features.FEATURE_NAMES)
        self.assertEqual(len(row),54)
        self.assertTrue(all(math.isfinite(x) for x in row.values()))
        with self.assertRaises(ValueError): features.preanswer_guard(row,mode='oracle')

if __name__=='__main__': unittest.main()
