"""Strict normalized exact-match scoring with bounded answer wrappers."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
import unicodedata


DATE_FORMATS = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d"]


def extract_answer(prediction):
    text = unicodedata.normalize("NFKC", str(prediction or "")).strip()
    # Only an entire answer line is scored; explanations with additional
    # values do not become correct merely by containing the gold substring.
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    text = re.sub(r"^(?:answer\s*:\s*|the answer is\s+)", "", text, flags=re.I)
    text = re.sub(r"^(?:the\s+)?(?:current\s+|confirmed\s+|final\s+)?(?:owner|release date|approved budget)(?:\s+for project\s+[A-Za-z0-9 -]+)?\s+(?:is|was)\s+", "", text, flags=re.I)
    text = text.strip().strip('"\'`*').strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def normalize_answer(answer, answer_type=None):
    text = extract_answer(answer)
    text = re.sub(r"\s+", " ", text).strip()
    if answer_type == "amount":
        match = re.fullmatch(r"(?:USD\s*|\$\s*)?([0-9]+(?:,[0-9]{3})*(?:\.0+)?)\s*(?:USD|dollars)?", text, flags=re.I)
        if match:
            try:
                return str(Decimal(match.group(1).replace(",", "")).normalize())
            except InvalidOperation:
                pass
    if answer_type == "date":
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
    return text.casefold()


def is_correct(prediction, task):
    answer_type = task.get("answer_type")
    normalized = normalize_answer(prediction, answer_type)
    if not normalized:
        return False
    aliases = task.get("answer_aliases") or [task["answer"]]
    return normalized in {normalize_answer(alias, answer_type) for alias in aliases}

