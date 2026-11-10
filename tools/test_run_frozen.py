"""Standard-library assertions for the limited manifest-path compatibility shim."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from run_frozen import manifest_json_paths, normalize_manifest_paths, run_script


class FrozenPathTests(unittest.TestCase):
    def test_only_manifest_keys_change_and_original_is_unchanged(self):
        context = {'text': r'C:\CacheScope\context', 'score': 0.375}
        digest = r'hash-value-with-\-unchanged'
        original = {'files_sha256': {r'models\model.pt': digest}, 'context': context}
        result = normalize_manifest_paths(original, enabled=True)
        self.assertEqual(result['files_sha256'], {'models/model.pt': digest})
        self.assertEqual(original['files_sha256'], {r'models\model.pt': digest})
        self.assertIs(result['context'], context)
        self.assertEqual(result['context']['text'], r'C:\CacheScope\context')
        self.assertEqual(result['context']['score'], 0.375)

    def test_disabled_and_windows_defaults_leave_object_unchanged(self):
        original = {'files_sha256': {r'models\model.pt': 'abc'}}
        self.assertIs(normalize_manifest_paths(original, enabled=False), original)
        if os.name == 'nt':
            self.assertIs(normalize_manifest_paths(original), original)

    def test_no_recursive_or_general_string_rewrite(self):
        nested = {'context': {'files_sha256': {r'a\b': 'value'}}, 'answer': r'a\b'}
        self.assertIs(normalize_manifest_paths(nested, enabled=True), nested)
        array = [{'files_sha256': {r'a\b': 'value'}}]
        self.assertIs(normalize_manifest_paths(array, enabled=True), array)

    def test_conflicting_paths_fail_instead_of_losing_hashes(self):
        with self.assertRaises(ValueError):
            normalize_manifest_paths({'files_sha256': {r'a\b': 'one', 'a/b': 'two'}}, enabled=True)
        self.assertEqual(normalize_manifest_paths({'files_sha256': {r'a\b': 'one', 'a/b': 'one'}}, enabled=True)['files_sha256'], {'a/b': 'one'})

    def test_json_load_and_loads_are_scoped(self):
        text = json.dumps({'files_sha256': {r'models\a.pt': 'abc'}, 'context': r'keep\this'})
        original_loads = json.loads
        with manifest_json_paths(enabled=True):
            a = json.loads(text)
            b = json.load(io.StringIO(text))
            self.assertEqual(a['files_sha256'], {'models/a.pt': 'abc'})
            self.assertEqual(a['context'], r'keep\this')
            self.assertEqual(a, b)
        self.assertIs(json.loads, original_loads)

    def test_script_execution_preserves_argv_path_cwd_and_restores_state(self):
        previous_argv, previous_path, previous_cwd = sys.argv[:], sys.path[:], Path.cwd()
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        expected_args = ['--probe', r'literal\path']
        payload = json.dumps({'files_sha256': {r'models\a.pt': 'abc'}, 'score': 0.5})
        with tempfile.TemporaryDirectory(prefix='cachescope-shim-', dir=cache) as directory:
            script = Path(directory) / 'probe.py'
            script.write_text(
                'import json, sys\n'
                'from pathlib import Path\n'
                f'assert sys.argv[1:] == {expected_args!r}\n'
                'assert sys.path[0] == str(Path(__file__).parent)\n'
                f'RESULT = json.loads({payload!r})\n',
                encoding='utf-8')
            result = run_script(script, expected_args, enabled=True)
            self.assertEqual(result['RESULT']['files_sha256'], {'models/a.pt': 'abc'})
            self.assertEqual(result['RESULT']['score'], 0.5)
        self.assertEqual(sys.argv, previous_argv)
        self.assertEqual(sys.path, previous_path)
        self.assertEqual(Path.cwd(), previous_cwd)


if __name__ == '__main__':
    unittest.main()
