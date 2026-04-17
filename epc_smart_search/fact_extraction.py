from __future__ import annotations

import re
from dataclasses import dataclass

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.search_features import normalize_text

ROW_SPLIT_RE = re.compile(r"\s*\|\s*|\t+|\s{2,}")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9\"'])")
CONFIG_VALUE_RE = re.compile(r"\b\d+\s*[xX]\s*\d+(?:\.\d+)?\s*%")

TECHNICAL_HEADS = {
    "analyzer",
    "boiler",
    "blower",
    "compressor",
    "cooler",
    "fan",
    "filter",
    "generator",
    "heater",
    "line",
    "motor",
    "package",
    "panel",
    "pump",
    "skid",
    "system",
    "tank",
    "train",
    "turbine",
    "unit",
    "valve",
    "vessel",
}
LEADING_SYSTEM_STOPWORDS = {
    "a",
    "all",
    "an",
    "each",
    "existing",
    "new",
    "selected",
    "the",
    "these",
    "this",
    "those",
    "vendor",
}
TRAILING_SYSTEM_STOPWORDS = {
    "be",
    "for",
    "has",
    "have",
    "include",
    "includes",
    "including",
    "is",
    "provided",
    "provides",
    "providing",
    "service",
    "shall",
    "using",
    "with",
}
TRAILING_CONTEXT_TERMS = {
    "configuration",
    "configured",
    "capacity",
    "capacities",
    "data",
    "description",
    "descriptions",
    "duty",
    "duties",
    "list",
    "lists",
    "model",
    "models",
    "quantity",
    "quantities",
    "rating",
    "ratings",
    "requirements",
    "schedule",
    "service",
    "services",
    "table",
    "tables",
}
ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "configuration": ("configuration", "configured", "arrangement"),
    "capacity": ("capacity", "capacities"),
    "quantity": ("quantity", "quantities", "number"),
    "rating": ("rating", "ratings", "rated"),
    "model": ("model", "models", "type"),
    "service": ("service", "services", "duty", "duties"),
}


@dataclass(slots=True, frozen=True)
class ContractFactRecord:
    document_id: str
    normalized_system: str
    normalized_attribute: str
    raw_value: str
    evidence_text: str
    page: int
    source_chunk_id: str


@dataclass(slots=True, frozen=True)
class _CandidateFact:
    normalized_system: str
    normalized_attribute: str
    raw_value: str
    evidence_text: str


def extract_contract_facts(chunks: list[ChunkRecord]) -> list[ContractFactRecord]:
    facts: list[ContractFactRecord] = []
    seen: set[tuple[str, str, str, str, int, str]] = set()
    for chunk in chunks:
        for candidate in extract_chunk_facts(chunk):
            key = (
                candidate.normalized_system,
                candidate.normalized_attribute,
                candidate.raw_value,
                candidate.evidence_text,
                chunk.page_start,
                chunk.chunk_id,
            )
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                ContractFactRecord(
                    document_id=chunk.document_id,
                    normalized_system=candidate.normalized_system,
                    normalized_attribute=candidate.normalized_attribute,
                    raw_value=candidate.raw_value,
                    evidence_text=candidate.evidence_text,
                    page=chunk.page_start,
                    source_chunk_id=chunk.chunk_id,
                )
            )
    return facts


def extract_chunk_facts(chunk: ChunkRecord) -> list[ContractFactRecord]:
    candidates = _extract_chunk_fact_candidates(chunk)
    return [
        ContractFactRecord(
            document_id=chunk.document_id,
            normalized_system=candidate.normalized_system,
            normalized_attribute=candidate.normalized_attribute,
            raw_value=candidate.raw_value,
            evidence_text=candidate.evidence_text,
            page=chunk.page_start,
            source_chunk_id=chunk.chunk_id,
        )
        for candidate in candidates
    ]


def _extract_chunk_fact_candidates(chunk: ChunkRecord) -> list[_CandidateFact]:
    heading_system = _normalize_system(chunk.heading)
    candidates: list[_CandidateFact] = []
    seen: set[tuple[str, str, str, str]] = set()
    for evidence in _iter_evidence_units(chunk.full_text):
        for candidate in _extract_from_row(evidence, heading_system):
            key = (
                candidate.normalized_system,
                candidate.normalized_attribute,
                candidate.raw_value,
                candidate.evidence_text,
            )
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        for candidate in _extract_from_patterns(evidence, heading_system):
            key = (
                candidate.normalized_system,
                candidate.normalized_attribute,
                candidate.raw_value,
                candidate.evidence_text,
            )
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _iter_evidence_units(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = _clean_text(value)
        if len(cleaned) < 8:
            return
        key = normalize_text(cleaned)
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    for raw_line in text.splitlines():
        add(raw_line)
        for piece in re.split(r"\s*[;•]\s*", raw_line):
            add(piece)
    flattened = text.replace("\n", " ")
    for sentence in SENTENCE_SPLIT_RE.split(flattened):
        add(sentence)
    return out


def _extract_from_row(evidence: str, heading_system: str) -> list[_CandidateFact]:
    cells = [cell for cell in (_clean_text(part) for part in ROW_SPLIT_RE.split(evidence)) if cell]
    if len(cells) < 2:
        return []
    facts: list[_CandidateFact] = []
    first_attribute = _normalize_attribute(cells[0])
    second_attribute = _normalize_attribute(cells[1]) if len(cells) >= 3 else ""
    if first_attribute and heading_system:
        value = _clean_value(" ".join(cells[1:]))
        candidate = _make_candidate(heading_system, first_attribute, value, evidence)
        if candidate is not None:
            facts.append(candidate)
    if second_attribute:
        system = _normalize_system(cells[0])
        value = _clean_value(" ".join(cells[2:]))
        candidate = _make_candidate(system, second_attribute, value, evidence)
        if candidate is not None:
            facts.append(candidate)
    if len(cells) == 2:
        inferred = _infer_attribute_from_value(cells[1])
        if inferred:
            system = _normalize_system(cells[0]) or heading_system
            candidate = _make_candidate(system, inferred, _clean_value(cells[1]), evidence)
            if candidate is not None:
                facts.append(candidate)
    return facts


def _extract_from_patterns(evidence: str, heading_system: str) -> list[_CandidateFact]:
    facts: list[_CandidateFact] = []

    for attribute, aliases in ATTRIBUTE_ALIASES.items():
        label_pattern = "|".join(re.escape(alias) for alias in aliases)
        explicit = re.search(
            rf"(?P<system>[A-Za-z][A-Za-z0-9/&(),\- ]{{2,90}}?)\s+"
            rf"(?P<label>{label_pattern})\b(?:\s+(?:shall\s+be|is|are|of|for)\b|\s*[:=\-])?\s*"
            rf"(?P<value>[^.;\n]{{1,80}})",
            evidence,
            re.IGNORECASE,
        )
        if explicit:
            candidate = _make_candidate(
                _normalize_system(explicit.group("system")),
                attribute,
                _clean_value(explicit.group("value")),
                evidence,
            )
            if candidate is not None:
                facts.append(candidate)

        implicit = re.search(
            rf"^(?P<label>{label_pattern})\b(?:\s+(?:shall\s+be|is|are|of|for)\b|\s*[:=\-])\s*"
            rf"(?P<value>[^.;\n]{{1,80}})$",
            evidence,
            re.IGNORECASE,
        )
        if implicit and heading_system:
            candidate = _make_candidate(
                heading_system,
                attribute,
                _clean_value(implicit.group("value")),
                evidence,
            )
            if candidate is not None:
                facts.append(candidate)

    for pattern, attribute in (
        (
            re.compile(
                r"(?P<system>[A-Za-z][A-Za-z0-9/&(),\- ]{2,90}?)\b.*?\b(?:in|with)\s+(?:an?\s+)?"
                r"(?P<value>\d+\s*[xX]\s*\d+(?:\.\d+)?\s*%)\s+(?:configuration|arrangement)\b",
                re.IGNORECASE,
            ),
            "configuration",
        ),
        (
            re.compile(
                r"(?P<system>[A-Za-z][A-Za-z0-9/&(),\- ]{2,90}?)\b.*?\brated\s+(?:at|for)\s+"
                r"(?P<value>[^.;\n]{1,60})",
                re.IGNORECASE,
            ),
            "rating",
        ),
        (
            re.compile(
                r"(?P<system>[A-Za-z][A-Za-z0-9/&(),\- ]{2,90}?)\b.*?\bcapacity\b(?:\s+(?:of|is|shall\s+be)\b|\s*[:=\-])?\s*"
                r"(?P<value>[^.;\n]{1,60})",
                re.IGNORECASE,
            ),
            "capacity",
        ),
        (
            re.compile(
                r"(?P<system>[A-Za-z][A-Za-z0-9/&(),\- ]{2,90}?)\b.*?\bfor\s+"
                r"(?P<value>[^.;\n]{1,60}?)\s+service\b",
                re.IGNORECASE,
            ),
            "service",
        ),
    ):
        match = pattern.search(evidence)
        if not match:
            continue
        candidate = _make_candidate(
            _normalize_system(match.group("system")) or heading_system,
            attribute,
            _clean_value(match.group("value")),
            evidence,
        )
        if candidate is not None:
            facts.append(candidate)

    return facts


def _make_candidate(system: str, attribute: str, value: str, evidence: str) -> _CandidateFact | None:
    normalized_system = _normalize_system(system)
    normalized_attribute = _normalize_attribute(attribute)
    cleaned_value = _clean_value(value)
    if not normalized_system or not normalized_attribute or not cleaned_value:
        return None
    return _CandidateFact(
        normalized_system=normalized_system,
        normalized_attribute=normalized_attribute,
        raw_value=cleaned_value,
        evidence_text=_clean_text(evidence),
    )


def _normalize_attribute(value: str) -> str:
    normalized = normalize_text(value)
    for canonical, aliases in ATTRIBUTE_ALIASES.items():
        if normalized in {normalize_text(alias) for alias in aliases}:
            return canonical
    return ""


def _infer_attribute_from_value(value: str) -> str:
    cleaned = _clean_value(value)
    if CONFIG_VALUE_RE.search(cleaned):
        return "configuration"
    return ""


def _normalize_system(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    tokens = [token for token in TOKEN_RE.findall(normalized) if token]
    while tokens and tokens[0] in LEADING_SYSTEM_STOPWORDS:
        tokens.pop(0)
    while tokens and (tokens[-1] in TRAILING_SYSTEM_STOPWORDS or tokens[-1] in TRAILING_CONTEXT_TERMS):
        tokens.pop()
    if len(tokens) >= 2 and tokens[-1] in TECHNICAL_HEADS:
        tokens[-1] = _singularize(tokens[-1])
    elif tokens and tokens[-1].endswith("s") and any(token in TECHNICAL_HEADS for token in tokens):
        tokens[-1] = _singularize(tokens[-1])
    if len(tokens) < 2:
        return ""
    return " ".join(tokens)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" :-\t\r\n")


def _clean_value(value: str) -> str:
    cleaned = _clean_text(value)
    cleaned = re.sub(r"\s+(?:and|which|that)\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(".,;:")
    if len(cleaned) > 80:
        return ""
    return cleaned


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
