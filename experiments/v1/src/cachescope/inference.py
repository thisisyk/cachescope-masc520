"""Real, resumable Ollama inference with a versioned local Qwen tokenizer."""
from __future__ import annotations
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SYSTEM = ("Answer the factual question using only the supplied context. "
          "Use the final approved or corrected value when the context updates a fact. "
          "Return only the requested value, without explanation. If the answer is absent, return UNKNOWN.")


def make_prompt(context, question):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\n")


class QwenTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class OllamaAnswerer:
    def __init__(self, model="qwen2.5-coder:0.5b", host="http://127.0.0.1:11434",
                 seed=520, threads=4, timeout=180):
        self.model, self.host = model, host.rstrip("/")
        self.timeout = timeout
        self.options = {"temperature": 0, "seed": seed, "num_predict": 24,
                        "num_ctx": 2048, "num_thread": threads,
                        "stop": ["<|im_end|>", "<|endoftext|>"]}

    def _post(self, endpoint, payload):
        request = urllib.request.Request(self.host + endpoint,
            data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def metadata(self):
        with urllib.request.urlopen(self.host + "/api/tags", timeout=30) as response:
            tags = json.load(response)
        selected = next((m for m in tags["models"] if m["name"] == self.model), None)
        if selected is None:
            raise RuntimeError(f"Model is not downloaded in Ollama: {self.model}")
        self.digest = selected["digest"]
        shown = self._post("/api/show", {"model": self.model})
        return {"backend": "Ollama /api/generate raw=true", "model": self.model,
                "model_digest": selected["digest"], "details": selected.get("details"),
                "model_info": shown.get("model_info"), "options": self.options,
                "system_instruction": SYSTEM, "prompt_template": "Qwen ChatML, explicitly constructed",
                "observed_at": datetime.now(timezone.utc).isoformat()}

    def answer(self, prompt):
        start = time.perf_counter()
        response = self._post("/api/generate", {
            "model": self.model, "prompt": prompt, "raw": True, "stream": False,
            "options": self.options, "keep_alive": "30m"})
        if not response.get("done"):
            raise RuntimeError("Ollama returned an incomplete response")
        return {"answer": response.get("response", ""),
                "latency_s": time.perf_counter() - start,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "done_reason": response.get("done_reason"),
                "total_duration_ns": response.get("total_duration"),
                "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
                "eval_duration_ns": response.get("eval_duration"),
                "load_duration_ns": response.get("load_duration")}


class ResponseCache:
    """Cache genuine raw responses. No deterministic/mock answer fallback exists."""
    def __init__(self, path, answerer):
        self.path, self.answerer = Path(path), answerer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.values[item["cache_key"]] = item

    def get(self, prompt, *, task_id, variant, repeat=0):
        identity = {"model": self.answerer.model, "model_digest": getattr(self.answerer, "digest", None), "options": self.answerer.options,
                    "prompt": prompt, "repeat": repeat}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in self.values:
            return self.values[key], True
        for attempt in range(3):
            try:
                result = self.answerer.answer(prompt)
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        result.update(cache_key=key, task_id=task_id, variant=variant, repeat=repeat,
                      model=self.answerer.model, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      model_digest=getattr(self.answerer, "digest", None),
                      generation_options=self.answerer.options,
                      observed_at=datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
        self.values[key] = result
        return result, False
