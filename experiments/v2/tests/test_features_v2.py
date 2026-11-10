"""Behavioral tests for pre-answer evidence support; no measured labels used."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_features', ROOT / 'src/cachescope/features.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceFeatureTests(unittest.TestCase):
    question = 'What is the current owner for project Cedar Main-001?'

    def extract(self, original, compressed):
        return module.extract_features(original, self.question, compressed, 200, 100)

    def test_directory_join_requires_attribute_record(self):
        for identifier in ('Item', 'Record', 'Entry', 'Ticket', 'Key', 'Code',
                           'File', 'Slot', 'Index', 'Unit', 'Reference', 'Ref', 'ID'):
            with self.subTest(identifier=identifier):
                mapping = f'{identifier} 3: Cedar Main-001'
                attribute = f'{identifier} 3: owner=Ada Reed; release date=2026-05-08; approved budget=USD 12000'
                unrelated = f'{identifier} 1: owner=Nora Cole; release date=2026-06-09; approved budget=USD 24000'
                source = '\n'.join([mapping, unrelated, attribute])
                complete = self.extract(source, mapping + '\n' + attribute)
                incomplete = self.extract(source, mapping + '\n' + unrelated)
                self.assertEqual(complete['linked_unit_retention'], 1)
                self.assertEqual(incomplete['linked_unit_retention'], 0)
                self.assertLess(incomplete['evidence_closure_retention'], complete['evidence_closure_retention'])

    def test_correction_deleted_even_when_old_value_survives(self):
        old = 'Project Cedar Main-001 has owner Ada Reed.'
        correction = 'A correction replaces that owner with Nora Cole.'
        source = old + ' ' + correction
        complete = self.extract(source, source)
        incomplete = self.extract(source, old)
        self.assertEqual(complete['correction_unit_retention'], 1)
        self.assertEqual(incomplete['correction_unit_retention'], 0)
        self.assertLess(incomplete['evidence_closure_retention'], complete['evidence_closure_retention'])

    def test_unrelated_record_cannot_replace_missing_linked_record(self):
        source = 'Project Cedar Main-001 uses key R12345.\n\nRecord R12345 gives owner Ada Reed.'
        compressed = 'Project Cedar Main-001 uses key R12345.'
        before = self.extract(source, compressed)
        unrelated = '\n\nRecord R98765 gives owner Ada Reed.'
        after = self.extract(source + unrelated, compressed + unrelated)
        self.assertEqual(before['linked_unit_retention'], after['linked_unit_retention'])
        self.assertEqual(before['evidence_closure_retention'], after['evidence_closure_retention'])

    def test_table_needs_source_header(self):
        source = '| Project | Owner | Budget |\n| Cedar Main-001 | Ada Reed | USD 12000 |'
        self.assertEqual(self.extract(source, source)['table_header_retention'], 1)
        self.assertEqual(self.extract(source, '| Cedar Main-001 | Ada Reed | USD 12000 |')['table_header_retention'], 0)

    def test_json_checks_queried_field_binding(self):
        source = '{\n "project": "Cedar Main-001",\n "details": {\n "owner": "Ada Reed",\n "budget": "USD 12000"\n }\n}'
        self.assertEqual(self.extract(source, source)['structured_field_retention'], 1)
        self.assertEqual(self.extract(source, '"project": "Cedar Main-001"\n"budget": "USD 12000"')['structured_field_retention'], 0)

    def test_features_are_finite_and_ordered(self):
        source = 'Project Cedar Main-001 has owner Ada Reed.'
        features = self.extract(source, source)
        self.assertEqual(tuple(features), module.FEATURE_NAMES)
        self.assertEqual(len(features), 38)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))


if __name__ == '__main__':
    unittest.main()
