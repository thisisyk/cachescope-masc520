"""Features computable before the answering model is called.

This module deliberately accepts no reference answer, model response, or label.
All counts are transparent local text statistics; they do not certify that a
particular fact has survived compression. V2 adds query-conditioned source
unit/link support and uses log1p for five absolute count features. The feature
schema is intentionally different from V1 and requires retrained models.
"""

from __future__ import annotations

import math
import re
import json
from collections import Counter


FEATURE_NAMES = (
    "original_log_tokens",
    "compressed_log_tokens",
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
    "original_log_sentence_count",
    "sentence_retention",
    "original_number_density",
    "compressed_number_density",
    "original_log_avg_sentence_words",
    "compressed_log_avg_sentence_words",
    "prefix_word_retention",
    "suffix_word_retention",
    "query_anchor_count_log",
    "anchor_unit_retention",
    "anchor_word_retention",
    "anchor_numeric_retention",
    "anchor_next_unit_retention",
    "anchor_previous_unit_retention",
    "linked_identifier_retention",
    "linked_unit_retention",
    "linked_unit_count_log",
    "evidence_closure_retention",
    "query_field_unit_retention",
    "correction_marker_present",
    "correction_unit_retention",
    "correction_neighbor_retention",
    "table_header_retention",
    "schema_key_retention",
    "structured_field_retention",
    "anchor_position_change_abs",
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
_QUERY_GENERIC = {
    "project", "current", "latest", "final", "record", "entry", "registered",
    "authorized", "authorised", "approved", "owner", "release", "date", "budget",
    "amount", "name", "value", "manager", "lead", "contact", "cost", "deadline",
}
_CORRECTION = re.compile(
    r"\b(?:correct(?:ion|ed)?|replac(?:e[sd]?|ement)|final|current|latest|"
    r"amend(?:ment|ed)?|supersed\w*|instead|obsolete|withdrawn|invalid|"
    r"rejected|not|rather|erroneous|provisional|draft|initial|previously)\b", re.I)
_IDENTIFIER = re.compile(
    r"\b(?:[a-z]{1,5}\d{2,}|[a-z]+[-_]\d[\w-]*|"
    r"(?:item|record|entry|ticket|key|code|file|slot|index|unit|reference|ref|id)\s+[a-z]*\d[\w-]*)\b", re.I)
_JSON_KEY = re.compile(r'"([^"\n]+)"\s*:')


def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _retention(original: set[str], compressed: set[str]) -> float:
    # An empty original set has no corresponding information to lose.
    return len(original & compressed) / len(original) if original else 1.0


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if x.strip()]


def _canonical(text: str) -> str:
    return " ".join(_words(text))


def _unit_present(unit: str, compressed_words: str) -> bool:
    # The fixed compressor is extractive. Match an entire source unit rather
    # than merely finding its words elsewhere in a different record.
    normalized = _canonical(unit)
    # Punctuation-only delimiters have no lexical evidence to lose. Structural
    # correctness is deliberately not inferred from this lexical test.
    return not normalized or f" {normalized} " in compressed_words


def _evidence_features(context: str, question: str, compressed_context: str) -> tuple[float, ...]:
    units = _sentences(context)
    unit_words = [set(_words(unit)) for unit in units]
    query = set(_words(question)) - _STOP
    query_anchors = (query - _QUERY_GENERIC) & set(_words(context))
    if not query_anchors:
        # No entity-specific term was found; use the most selective query
        # terms available in the source. No answer or dataset metadata is read.
        counts = Counter(word for words in unit_words for word in words)
        candidates = query & set(counts)
        if candidates:
            minimum = min(counts[word] for word in candidates)
            query_anchors = {word for word in candidates if counts[word] == minimum}
    counts = Counter(word for words in unit_words for word in words)
    weights = {word: 1.0 / max(1, counts[word]) for word in query_anchors}
    total_weight = sum(weights.values())
    anchors = {
        index for index, words in enumerate(unit_words)
        if total_weight and sum(weights[word] for word in words & query_anchors) / total_weight >= .65
    }
    compressed_words = f" {_canonical(compressed_context)} "
    retained = {index for index, unit in enumerate(units) if _unit_present(unit, compressed_words)}

    def unit_retention(indices):
        return len(set(indices) & retained) / len(set(indices)) if indices else 1.0

    anchor_text = "\n".join(units[index] for index in sorted(anchors))
    anchor_words = set(_words(anchor_text))
    compressed_set = set(_words(compressed_context))
    source_identifiers = {
        match.group(0).lower() for index in anchors for match in _IDENTIFIER.finditer(units[index])
    }
    unit_identifiers = [{match.group(0).lower() for match in _IDENTIFIER.finditer(unit)} for unit in units]
    linked = {index for index, identifiers in enumerate(unit_identifiers) if identifiers & source_identifiers} - anchors
    next_units = {index + 1 for index in anchors if index + 1 < len(units)}
    previous_units = {index - 1 for index in anchors if index > 0}

    # Blank-line paragraphs bind the header/update/reference forms. JSON and
    # tables are handled separately so a whole structured document is not
    # spuriously interpreted as one relevant paragraph.
    closure = set(anchors) | linked
    if not context.lstrip().startswith(("{", "[")):
        position = 0
        for paragraph in re.split(r"\n\s*\n", context):
            paragraph_units = _sentences(paragraph)
            indices = set(range(position, position + len(paragraph_units)))
            # Multi-line directories/tables can contain several independent
            # records without blank lines. Their neighbors are not evidence
            # simply because they share a paragraph; explicit links handle
            # those formats instead.
            if indices & anchors and len(indices) <= 8 and len(paragraph.strip().splitlines()) == 1:
                closure.update(indices)
            position += len(paragraph_units)
    else:
        # Source JSON record boundaries are visible pre-answer. A project/name
        # line is followed by the fields of its own record, ending at the next
        # peer entity line or record closing brace.
        for anchor in anchors:
            for index in range(anchor + 1, min(len(units), anchor + 10)):
                if index in anchors or ('"project"' in units[index].lower()):
                    break
                closure.add(index)
                if re.match(r"^\s*}\s*,?\s*$", units[index]):
                    # A nested object may close here; including its scalar
                    # fields already establishes the local evidence closure.
                    break
    field_terms = query & _QUERY_GENERIC - {"project", "current", "latest", "final", "record", "entry", "value", "name"}
    field_units = {index for index in closure if unit_words[index] & field_terms}
    corrections = {index for index in closure if _CORRECTION.search(units[index])}
    correction_neighbors = {index + 1 for index in corrections if index + 1 < len(units)} & closure
    table_header = set()
    for index, unit in enumerate(units):
        if unit.count("|") >= 2 and re.search(r"[a-z]", unit, re.I):
            table_header.add(index)
            break
    original_keys = set(_JSON_KEY.findall(context))
    compressed_keys = set(_JSON_KEY.findall(compressed_context))
    structured_retention = _structured_field_retention(context, compressed_words, query_anchors, field_terms)
    original_anchor_positions = [index / max(1, len(units) - 1) for index in anchors]
    compressed_units = _sentences(compressed_context)
    compressed_anchor_positions = [index / max(1, len(compressed_units) - 1)
        for index, unit in enumerate(compressed_units) if set(_words(unit)) & query_anchors]
    position_change = abs(sum(original_anchor_positions) / len(original_anchor_positions)
        - sum(compressed_anchor_positions) / len(compressed_anchor_positions)) if original_anchor_positions and compressed_anchor_positions else 0.0
    return (
        math.log1p(len(query_anchors)), unit_retention(anchors),
        _retention(anchor_words, compressed_set),
        _retention(set(_NUMBER.findall(anchor_text)), set(_NUMBER.findall(compressed_context))),
        unit_retention(next_units), unit_retention(previous_units),
        _retention(source_identifiers, {match.group(0).lower() for match in _IDENTIFIER.finditer(compressed_context)}),
        unit_retention(linked), math.log1p(len(linked)), unit_retention(closure),
        unit_retention(field_units), float(bool(corrections)), unit_retention(corrections),
        unit_retention(correction_neighbors), unit_retention(table_header),
        _retention(original_keys, compressed_keys), structured_retention, position_change,
    )


def _structured_field_retention(context: str, compressed_words: str, anchors: set[str], field_terms: set[str]) -> float:
    """Check source JSON field lines within the queried record, without gold.

    This is a text-support heuristic: it reads keys/values already present in
    the prompt and returns a retained fraction, never a reference answer. It
    does not assert that a surviving scalar is still attributed correctly.
    """
    if not context.lstrip().startswith(("{", "[")) or not anchors:
        return 1.0
    try:
        parsed = json.loads(context)
    except (ValueError, TypeError):
        return 1.0
    support = []

    def leaves(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, path + (str(key),))
        elif isinstance(value, list):
            for child in value:
                yield from leaves(child, path)
        else:
            yield path, value

    def visit(value):
        if isinstance(value, dict):
            direct_words = set(word for child in value.values() if isinstance(child, (str, int, float)) for word in _words(str(child)))
            if len(direct_words & anchors) / max(1, len(anchors)) >= .65:
                for path, scalar in leaves(value):
                    key_words = set(_words(" ".join(path)))
                    if key_words & field_terms and path:
                        # Match both field key and scalar, preserving their
                        # textual binding rather than value presence alone.
                        support.append(_unit_present(f"{path[-1]} {scalar}", compressed_words))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(parsed)
    return sum(support) / len(support) if support else 1.0


def extract_features(
    context: str,
    question: str,
    compressed_context: str,
    original_tokens: int,
    compressed_tokens: int,
) -> dict[str, float]:
    """Return 38 ordered, finite, pre-answer features.

    ``original_tokens`` and ``compressed_tokens`` must count the complete
    answering prompts with the same tokenizer and prompt wrapper.
    Token lengths, sentence count, and mean sentence word counts use log1p;
    all corresponding feature names explicitly include ``log``.
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
        math.log1p(original_tokens),
        math.log1p(compressed_tokens),
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
        math.log1p(len(original_sentences)),
        len(compressed_sentences) / max(1, len(original_sentences)),
        len(_NUMBER.findall(context)) / max(1, len(original)),
        len(_NUMBER.findall(compressed_context)) / max(1, len(compressed)),
        math.log1p(len(original) / max(1, len(original_sentences))),
        math.log1p(len(compressed) / max(1, len(compressed_sentences))),
        _retention(set(original[:20]), compressed_set),
        _retention(set(original[-20:]), compressed_set),
    )
    values += _evidence_features(context, question, compressed_context)
    result = dict(zip(FEATURE_NAMES, values))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite feature value")
    return result
