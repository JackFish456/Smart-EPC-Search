from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from epc_smart_search.search_features import normalize_text

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{1,}")
PAREN_ALIAS_RE = re.compile(r"([A-Za-z][A-Za-z0-9/&,\- ]{3,80}?)\s*\(([A-Z][A-Z0-9/\-]{1,15})\)")
REVERSE_PAREN_ALIAS_RE = re.compile(r"([A-Z][A-Z0-9/\-]{1,15})\s*\(([A-Za-z][A-Za-z0-9/&,\- ]{3,80}?)\)")

GENERIC_SYSTEM_TERMS = {
    "system",
    "systems",
    "equipment",
    "package",
    "packages",
    "module",
    "modules",
    "unit",
    "units",
    "train",
    "trains",
    "skid",
    "skids",
}
TECHNICAL_HEADS = GENERIC_SYSTEM_TERMS | {
    "pump",
    "pumps",
    "compressor",
    "compressors",
    "turbine",
    "turbines",
    "generator",
    "generators",
    "motor",
    "motors",
    "valve",
    "valves",
    "blower",
    "blowers",
    "fan",
    "fans",
    "heater",
    "heaters",
    "cooler",
    "coolers",
    "analyzer",
    "analyzers",
    "filter",
    "filters",
    "tank",
    "tanks",
    "vessel",
    "vessels",
    "line",
    "lines",
    "header",
    "headers",
}
LEADING_DESCRIPTOR_TERMS = {
    "selected",
    "existing",
    "new",
    "main",
    "primary",
    "secondary",
    "standby",
    "common",
    "proposed",
    "project",
}
TRAILING_DESCRIPTOR_TERMS = {
    "description",
    "descriptions",
    "installation",
    "installations",
    "requirements",
    "requirement",
    "details",
    "detail",
    "list",
    "lists",
    "listing",
    "schedule",
    "schedules",
    "table",
    "tables",
    "summary",
    "summaries",
    "service",
    "services",
    "data",
    "duty",
    "duties",
    "device",
    "devices",
}
TRAILING_ATTRIBUTE_PHRASES = (
    ("design", "conditions"),
    ("design", "condition"),
    ("operating", "conditions"),
    ("operating", "condition"),
    ("design", "basis"),
    ("horse", "power"),
    ("flow", "rate"),
    ("flow", "rates"),
    ("type", "model"),
    ("motor", "list"),
    ("equipment", "list"),
    ("configuration",),
    ("configured",),
    ("arrangement",),
    ("type",),
    ("model",),
    ("models",),
    ("size",),
    ("sizes",),
    ("capacity",),
    ("capacities",),
    ("pressure",),
    ("pressures",),
    ("temperature",),
    ("temperatures",),
    ("flow",),
    ("function",),
    ("functions",),
    ("responsibility",),
    ("responsibilities",),
)


@dataclass(slots=True, frozen=True)
class SystemVocabularyEntry:
    canonical_phrase: str
    terms: tuple[str, ...]
    significant_terms: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SystemVocabulary:
    entries: tuple[SystemVocabularyEntry, ...]

    def match(self, content_query: str, fallback_phrase: str = "") -> SystemVocabularyEntry | None:
        normalized_query = normalize_text(content_query)
        query_terms = _normalized_terms(normalized_query)
        fallback_terms = _normalized_terms(fallback_phrase)
        best_entry: SystemVocabularyEntry | None = None
        best_score = 0.0
        for entry in self.entries:
            score = _score_entry_match(entry, normalized_query, query_terms, fallback_terms)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry if best_score >= 4.0 else None


def build_contract_system_vocabulary(chunks: Iterable[object]) -> SystemVocabulary:
    registry: dict[str, dict[str, set[str]]] = {}
    for chunk in chunks:
        heading = _value(chunk, "heading")
        full_text = _value(chunk, "full_text")
        for phrase in _extract_heading_phrases(heading):
            _register_phrase(registry, phrase)
        for canonical, alias in _extract_alias_pairs(heading, full_text):
            _register_phrase(registry, canonical, alias)
    entries: list[SystemVocabularyEntry] = []
    for canonical, bucket in registry.items():
        aliases = tuple(
            alias
            for alias in sorted(bucket["aliases"], key=lambda item: (-len(_normalized_terms(item)), item))
            if alias and alias != canonical
        )
        terms = _normalized_terms(canonical)
        if len(terms) < 2:
            continue
        entries.append(
            SystemVocabularyEntry(
                canonical_phrase=canonical,
                terms=terms,
                significant_terms=_significant_terms(terms),
                aliases=aliases,
            )
        )
    entries.sort(key=lambda entry: (-len(entry.significant_terms), -len(entry.terms), entry.canonical_phrase))
    return SystemVocabulary(tuple(entries))


def system_significant_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return _significant_terms(terms)


def _value(chunk: object, key: str) -> str:
    if hasattr(chunk, key):
        return normalize_text(str(getattr(chunk, key) or ""))
    if hasattr(chunk, "keys") and key in chunk.keys():
        return normalize_text(str(chunk[key] or ""))
    return ""


def _extract_heading_phrases(heading: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for segment in re.split(r"[:;|]", heading):
        tokens = _clean_candidate_tokens(TOKEN_RE.findall(segment))
        if len(tokens) < 2:
            continue
        trimmed = _trim_tail(tokens)
        if len(trimmed) < 2:
            continue
        if _looks_like_system_phrase(trimmed):
            candidates.append(" ".join(trimmed))
        if trimmed[-1] not in TECHNICAL_HEADS and len(trimmed) >= 3:
            candidates.append(" ".join(trimmed[-2:]))
    return _dedupe(candidates)


def _extract_alias_pairs(heading: str, full_text: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    text = " ".join(part for part in (heading, full_text[:320]) if part)
    matches = list(PAREN_ALIAS_RE.findall(text)) + [(right, left) for left, right in REVERSE_PAREN_ALIAS_RE.findall(text)]
    for left, right in matches:
        canonical_tokens = _trim_tail(_clean_candidate_tokens(TOKEN_RE.findall(left)))
        if len(canonical_tokens) < 2 or not _looks_like_system_phrase(canonical_tokens):
            continue
        alias = normalize_text(right)
        alias_terms = _normalized_terms(alias)
        if not alias_terms:
            continue
        if len(alias_terms) == 1 and len(alias_terms[0]) <= 2:
            continue
        pairs.append((" ".join(canonical_tokens), alias))
    return tuple(_dedupe_pair_list(pairs))


def _register_phrase(registry: dict[str, dict[str, set[str]]], phrase: str, alias: str = "") -> None:
    canonical = _canonicalize_phrase(phrase)
    terms = _normalized_terms(canonical)
    if len(terms) < 2:
        return
    significant_terms = _significant_terms(terms)
    if len(significant_terms) < 2 and not alias:
        return
    bucket = registry.setdefault(canonical, {"aliases": set()})
    bucket["aliases"].add(canonical)
    singularized = _canonicalize_phrase(" ".join(terms))
    bucket["aliases"].add(singularized)
    if alias:
        normalized_alias = _canonicalize_alias(alias)
        if normalized_alias:
            bucket["aliases"].add(normalized_alias)


def _canonicalize_phrase(phrase: str) -> str:
    tokens = _trim_tail(_clean_candidate_tokens(TOKEN_RE.findall(phrase)))
    if len(tokens) < 2:
        return ""
    if tokens[-1] in TECHNICAL_HEADS:
        tokens[-1] = _singularize_token(tokens[-1])
    return " ".join(tokens)


def _canonicalize_alias(alias: str) -> str:
    terms = _normalized_terms(alias)
    if not terms:
        return ""
    if len(terms) > 1 and terms[-1] in TECHNICAL_HEADS:
        terms = (*terms[:-1], _singularize_token(terms[-1]))
    return " ".join(terms)


def _clean_candidate_tokens(tokens: list[str]) -> list[str]:
    normalized = [normalize_text(token) for token in tokens if normalize_text(token)]
    while normalized and normalized[0] in LEADING_DESCRIPTOR_TERMS:
        normalized = normalized[1:]
    return normalized


def _trim_tail(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    changed = True
    while len(trimmed) >= 2 and changed:
        changed = False
        for suffix in TRAILING_ATTRIBUTE_PHRASES:
            if tuple(trimmed[-len(suffix):]) == suffix:
                trimmed = trimmed[:-len(suffix)]
                changed = True
                break
        if changed:
            continue
        while trimmed and trimmed[-1] in TRAILING_DESCRIPTOR_TERMS:
            trimmed.pop()
            changed = True
    return trimmed


def _looks_like_system_phrase(tokens: list[str]) -> bool:
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    if tokens[-1] in TECHNICAL_HEADS:
        return True
    return len(_significant_terms(tuple(tokens))) >= 2


def _normalized_terms(text: str) -> tuple[str, ...]:
    return tuple(
        cleaned
        for cleaned in (_normalize_query_token(token) for token in TOKEN_RE.findall(normalize_text(text)))
        if cleaned
    )


def _normalize_query_token(token: str) -> str:
    return _singularize_token(normalize_text(token))


def _significant_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    significant = tuple(term for term in terms if term not in GENERIC_SYSTEM_TERMS)
    return significant or terms


def _singularize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _score_entry_match(
    entry: SystemVocabularyEntry,
    normalized_query: str,
    query_terms: tuple[str, ...],
    fallback_terms: tuple[str, ...],
) -> float:
    score = 0.0
    query_term_set = set(query_terms)
    fallback_term_set = set(fallback_terms)

    if entry.canonical_phrase and _contains_phrase(normalized_query, entry.canonical_phrase):
        score += 8.0 + len(entry.terms)
    for alias in entry.aliases:
        if _contains_phrase(normalized_query, alias):
            score += 6.0 + len(_normalized_terms(alias))
            break

    if entry.significant_terms and len(entry.significant_terms) >= 2 and all(term in query_term_set for term in entry.significant_terms):
        score += 5.5 + len(entry.significant_terms)
        if _ordered_term_hits(normalized_query, entry.significant_terms):
            score += 1.0
    elif entry.terms and all(term in query_term_set for term in entry.terms):
        score += 4.5 + len(entry.terms)

    if fallback_term_set and entry.significant_terms and all(term in fallback_term_set for term in entry.significant_terms):
        score += 1.5
    return score


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    wrapped = f" {normalize_text(text)} "
    return f" {normalize_text(phrase)} " in wrapped


def _ordered_term_hits(text: str, terms: tuple[str, ...]) -> bool:
    position = 0
    lowered = normalize_text(text)
    for term in terms:
        next_position = lowered.find(term, position)
        if next_position < 0:
            return False
        position = next_position + len(term)
    return True


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = normalize_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out)


def _dedupe_pair_list(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for left, right in items:
        key = (normalize_text(left), normalize_text(right))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
