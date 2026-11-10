"""Fixed question-overlap sentence selector, independent of gold answers."""
from __future__ import annotations

import math
import re

STOP = {"a", "an", "the", "is", "are", "was", "were", "what", "who", "when", "how", "for", "of", "to", "in", "on", "and", "as", "its", "it", "this", "that", "with", "project", "current"}


def words(text):
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())) - STOP


def split_units(text):
    # Keep rows and JSON lines intact; punctuation only splits at whitespace.
    # This same deterministic rule is used for every family and holdout type.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def compress_context(context, question, keep_ratio, tokenizer):
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must lie in (0, 1]")
    original = len(_tokens(tokenizer, context))
    if not context.strip() or keep_ratio == 1:
        return {"context": context, "original_context_tokens": original,
                "compressed_context_tokens": original, "keep_ratio_actual": 1.0}
    units = split_units(context)
    if not units:
        return {"context": context, "original_context_tokens": original,
                "compressed_context_tokens": original, "keep_ratio_actual": 1.0}
    query = words(question)
    def rank(index):
        unit_words = words(units[index])
        overlap = len(query & unit_words)
        # Overlap count dominates, then density, then source order.
        return (-overlap, -overlap / max(1, len(unit_words)), index)
    ordered = sorted(range(len(units)), key=rank)
    budget = max(1, math.floor(original * keep_ratio))
    chosen = []
    for index in ordered:
        proposed = sorted(chosen + [index])
        text = "\n".join(units[j] for j in proposed)
        if len(_tokens(tokenizer, text)) <= budget:
            chosen.append(index)
    if chosen:
        candidate = "\n".join(units[index] for index in sorted(chosen))
    else:
        # Sentence boundaries are preserved even if one full sentence exceeds
        # the target budget. Actual retention is reported, never fabricated.
        candidate = units[ordered[0]]
    compressed = len(_tokens(tokenizer, candidate))
    if compressed >= original:
        candidate, compressed = context, original
    return {"context": candidate, "original_context_tokens": original,
            "compressed_context_tokens": compressed,
            "keep_ratio_actual": compressed / original if original else 1.0}

