"""Features computable before the answering model is called.

This module deliberately accepts no reference answer, model response, or label.
All counts are transparent local text statistics; they do not certify that a
particular fact has survived compression.
"""

from __future__ import annotations

import math
import re


FEATURE_NAMES = (
    "original_tokens",
    "compressed_tokens",
    "removed_fraction",
    "original_repetition",
    "compressed_repetition",
    "original_question_overlap",
    "compressed_question_overlap",
    "question_overlap_change",
    "lexical_retention",
    "numeric_retention",
    "date_retention",
    "negation_retention",
    "original_sentence_count",
    "sentence_retention",
    "original_number_density",
    "compressed_number_density",
    "original_avg_sentence_words",
    "compressed_avg_sentence_words",
    "prefix_word_retention",
    "suffix_word_retention",
)

_WORD = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[,.]\d+)*(?:%)?(?!\w)")
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)
_STOP = {
    "a", "an", "the", "what", "which", "who", "when", "where", "how", "is",
    "was", "are", "were", "be", "been", "of", "for", "to", "in", "on", "at",
    "and", "or", "with", "from", "does", "do", "did", "please", "give", "state",
    "answer", "only", "according", "text", "passage", "context", "its", "it",
}
_NEGATIONS = {"no", "not", "never", "neither", "nor", "without", "except", "excluded", "cancelled", "canceled"}


def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _retention(original: set[str], compressed: set[str]) -> float:
    # An empty original set has no corresponding information to lose.
    return len(original & compressed) / len(original) if original else 1.0


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if x.strip()]


def extract_features(
    context: str,
    question: str,
    compressed_context: str,
    original_tokens: int,
    compressed_tokens: int,
) -> dict[str, float]:
    """Return 20 ordered, finite, pre-answer features.

    ``original_tokens`` and ``compressed_tokens`` must count the complete
    answering prompts with the same tokenizer and prompt wrapper.
    """
    if original_tokens <= 0 or compressed_tokens < 0:
        raise ValueError("Token counts require original_tokens > 0 and compressed_tokens >= 0")
    original = _words(context)
    compressed = _words(compressed_context)
    original_set, compressed_set = set(original), set(compressed)
    relevant = set(_words(question)) - _STOP
    overlap_original = len(relevant & original_set) / max(1, len(relevant))
    overlap_compressed = len(relevant & compressed_set) / max(1, len(relevant))
    original_numbers = {x.replace(",", "") for x in _NUMBER.findall(context)}
    compressed_numbers = {x.replace(",", "") for x in _NUMBER.findall(compressed_context)}
    original_dates = {x.lower() for x in _DATE.findall(context)}
    compressed_dates = {x.lower() for x in _DATE.findall(compressed_context)}
    original_sentences = _sentences(context)
    compressed_sentences = _sentences(compressed_context)
    values = (
        float(original_tokens),
        float(compressed_tokens),
        1.0 - compressed_tokens / original_tokens,
        1.0 - len(original_set) / max(1, len(original)),
        1.0 - len(compressed_set) / max(1, len(compressed)),
        overlap_original,
        overlap_compressed,
        overlap_original - overlap_compressed,
        _retention(original_set, compressed_set),
        _retention(original_numbers, compressed_numbers),
        _retention(original_dates, compressed_dates),
        _retention(original_set & _NEGATIONS, compressed_set & _NEGATIONS),
        float(len(original_sentences)),
        len(compressed_sentences) / max(1, len(original_sentences)),
        len(_NUMBER.findall(context)) / max(1, len(original)),
        len(_NUMBER.findall(compressed_context)) / max(1, len(compressed)),
        len(original) / max(1, len(original_sentences)),
        len(compressed) / max(1, len(compressed_sentences)),
        _retention(set(original[:20]), compressed_set),
        _retention(set(original[-20:]), compressed_set),
    )
    result = dict(zip(FEATURE_NAMES, values))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite feature value")
    return result
