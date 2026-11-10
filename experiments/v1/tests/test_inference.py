"""Cache integrity tests use fixtures only, never experimental labels."""
import json
import tempfile
import unittest
from pathlib import Path
from cachescope.inference import ResponseCache, make_prompt

class FixtureAnswerer:
    model = "TEST-ONLY"
    digest = "test-weight-a"
    options = {"seed": 520}
    def __init__(self):
        self.calls = 0
    def answer(self, prompt):
        self.calls += 1
        return {"answer": "TEST-FIXTURE", "latency_s": .1}

class CacheIntegrityTest(unittest.TestCase):
    def test_resume_and_weight_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.jsonl"
            backend = FixtureAnswerer()
            cache = ResponseCache(path, backend)
            first, cached = cache.get("fixture prompt", task_id="test", variant="original")
            self.assertFalse(cached)
            resumed = ResponseCache(path, backend)
            second, cached = resumed.get("fixture prompt", task_id="test", variant="original")
            self.assertTrue(cached)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(first["answer"], second["answer"])
            backend.digest = "test-weight-b"
            resumed.get("fixture prompt", task_id="test", variant="original")
            self.assertEqual(backend.calls, 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_repeat_is_a_new_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FixtureAnswerer()
            cache = ResponseCache(Path(directory)/"fixture.jsonl", backend)
            for repeat in [0, 1]:
                cache.get("fixture", task_id="test", variant="original", repeat=repeat)
            self.assertEqual(backend.calls, 2)

if __name__ == "__main__":
    unittest.main()
