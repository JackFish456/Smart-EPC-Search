from __future__ import annotations

import re

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.query_planner import EXACT_VALUE_ATTRIBUTE_LABELS, plan_query
from epc_smart_search.search_features import normalize_text
from epc_smart_search.storage import ContractFactRow
from epc_smart_search.system_vocabulary import SystemVocabulary, build_contract_system_vocabulary

SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
VALUE_TOKEN_RE = re.compile(r"[A-Za-z0-9%./\-]+")
CAPACITY_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?\s*%\s*capacity)\b", re.IGNORECASE)
MULTI_TRAIN_RE = re.compile(r"\b(\d+\s*[xX]\s*\d+(?:\.\d+)?\s*%)\b")
POWER_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:hp|kw|mw))\b", re.IGNORECASE)
PRESSURE_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:psig|psia|psi|bar|kpa))\b", re.IGNORECASE)
TEMPERATURE_VALUE_RE = re.compile(r"\b(-?\d+(?:\.\d+)?\s*(?:degf|degc|°f|°c|f|c))\b", re.IGNORECASE)
FLOW_VALUE_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:gpm|gal/min|gpd|scfm|cfm|acfm|mmscfd|lb/hr|lbm/hr|kg/hr|m3/hr|ft3/min))\b", re.IGNORECASE)
SIZE_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:in(?:ch(?:es)?)?|mm|ft|feet))\b", re.IGNORECASE)
MODEL_VALUE_RE = re.compile(r"\b([A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*)\b")
TYPE_VALUE_RE = re.compile(
    r"\b(?:shall be|is|are|shall use|uses|configured as|provided as)\s+"
    r"((?:electric|gas-fired|steam|centrifugal|reciprocating|duplex|simplex|vertical|horizontal|air-cooled|water-cooled)"
    r"(?:\s+[a-z0-9/\-]+){0,4})\b",
    re.IGNORECASE,
)
ATTRIBUTE_HINTS: dict[str, tuple[str, ...]] = {
    "configuration": ("configuration", "configured", "arrangement", "arranged", "standby", "duty"),
    "capacity": ("capacity", "capacities", "throughput", "duty", "output"),
    "type": ("type", "model", "selected", "manufacturer", "vendor", "using", "used"),
    "size": ("size", "sizes", "diameter", "dimension", "dimensions", "rating"),
    "power": ("power", "horsepower", "hp", "kw", "mw", "motor"),
    "pressure": ("pressure", "pressures", "psig", "psia", "psi", "bar", "kpa"),
    "temperature": ("temperature", "temperatures", "degf", "degc", "°f", "°c"),
    "flow": ("flow", "flowrate", "flow rate", "throughput", "gpm", "scfm", "cfm"),
}
CONFIGURATION_KEYWORDS = ("duplex", "simplex", "standby", "lead", "lag", "arrangement", "configuration")


def build_contract_facts(chunks: list[ChunkRecord]) -> list[ContractFactRow]:
    vocabulary = build_contract_system_vocabulary(chunks)
    facts: list[ContractFactRow] = []
    seen: set[tuple[str, str, str, str, str, int, int]] = set()
    for chunk in chunks:
        systems = _candidate_systems(chunk, vocabulary)
        if not systems:
            continue
        preferred_attribute = _preferred_attribute(chunk, vocabulary)
        for system in systems:
            for attribute, value, evidence_text in _extract_chunk_fact_entries(chunk, system, preferred_attribute):
                dedupe_key = (
                    chunk.document_id,
                    normalize_text(system),
                    attribute,
                    value,
                    evidence_text,
                    chunk.page_start,
                    chunk.page_end,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                facts.append(
                    ContractFactRow(
                        document_id=chunk.document_id,
                        system=system,
                        attribute=attribute,
                        value=value,
                        evidence_text=evidence_text,
                        source_chunk_id=chunk.chunk_id,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                    )
                )
    return facts


def _candidate_systems(chunk: ChunkRecord, vocabulary: SystemVocabulary) -> tuple[str, ...]:
    heading_plan = plan_query(chunk.heading, system_vocabulary=vocabulary)
    candidates: list[str] = []
    if heading_plan.system_phrase and len(heading_plan.system_terms) >= 2:
        candidates.append(heading_plan.system_phrase)
    matched = vocabulary.match(chunk.heading, heading_plan.system_phrase)
    if matched is not None:
        candidates.append(matched.canonical_phrase)
    if not candidates:
        combined_plan = plan_query(f"{chunk.heading} {chunk.full_text[:180]}", system_vocabulary=vocabulary)
        if combined_plan.system_phrase and len(combined_plan.system_terms) >= 2:
            candidates.append(combined_plan.system_phrase)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return tuple(deduped)


def _preferred_attribute(chunk: ChunkRecord, vocabulary: SystemVocabulary) -> str | None:
    heading_plan = plan_query(chunk.heading, system_vocabulary=vocabulary)
    return heading_plan.attribute_label if heading_plan.attribute_label in EXACT_VALUE_ATTRIBUTE_LABELS else None


def _extract_chunk_fact_entries(
    chunk: ChunkRecord,
    system: str,
    preferred_attribute: str | None,
) -> list[tuple[str, str, str]]:
    units = _evidence_units(chunk.full_text)
    heading_text = normalize_text(chunk.heading)
    system_text = normalize_text(system)
    extracted: list[tuple[str, str, str]] = []
    attributes = [preferred_attribute] if preferred_attribute else []
    attributes.extend(attribute for attribute in EXACT_VALUE_ATTRIBUTE_LABELS if attribute not in attributes)
    for evidence_text in units:
        normalized_evidence = normalize_text(evidence_text)
        if not _evidence_matches_system(system_text, heading_text, normalized_evidence):
            continue
        for attribute in attributes:
            value = _extract_attribute_value(attribute, evidence_text)
            if value is None:
                continue
            if not _supports_attribute(attribute, normalized_evidence, heading_text, value):
                continue
            extracted.append((attribute, value, evidence_text))
    for label, value, evidence_text in _fact_row_entries(chunk.full_text):
        normalized_label = normalize_text(label)
        if not _evidence_matches_system(system_text, heading_text, normalize_text(evidence_text)):
            continue
        for attribute in attributes:
            if not _label_supports_attribute(attribute, normalized_label):
                continue
            extracted_value = _extract_attribute_value(attribute, value)
            if extracted_value is not None:
                extracted.append((attribute, extracted_value, evidence_text))
    return _dedupe_entries(extracted)


def _evidence_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in str(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        units.append(line)
        units.extend(
            sentence.strip()
            for sentence in SENTENCE_RE.split(line)
            if sentence and sentence.strip() and sentence.strip() != line
        )
    if units:
        return _dedupe_text(units)
    compact = " ".join(str(text).split()).strip()
    if not compact:
        return []
    return _dedupe_text([compact, *[sentence.strip() for sentence in SENTENCE_RE.split(compact) if sentence.strip()]])


def _fact_row_entries(text: str) -> list[tuple[str, str, str]]:
    lines = [" ".join(line.split()).strip() for line in str(text).splitlines() if line and line.strip()]
    entries: list[tuple[str, str, str]] = []
    for line in lines:
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
        if label and value:
            entries.append((label, value, f"{label}: {value}"))
    if entries:
        return entries
    for index in range(len(lines) - 1):
        label = lines[index]
        value = lines[index + 1]
        if _looks_like_table_label(label) and _looks_like_table_value(value):
            entries.append((label, value, f"{label}: {value}"))
    return entries


def _looks_like_table_label(text: str) -> bool:
    return not re.search(r"\d", text) and 1 <= len(VALUE_TOKEN_RE.findall(text)) <= 8


def _looks_like_table_value(text: str) -> bool:
    lowered = normalize_text(text)
    return bool(re.search(r"\d", text) or any(keyword in lowered for keyword in CONFIGURATION_KEYWORDS))


def _evidence_matches_system(system_text: str, heading_text: str, evidence_text: str) -> bool:
    if system_text and system_text in heading_text:
        return True
    if system_text and system_text in evidence_text:
        return True
    system_terms = tuple(term for term in system_text.split() if term)
    if not system_terms:
        return False
    heading_hits = sum(1 for term in system_terms if f" {term} " in f" {heading_text} ")
    evidence_hits = sum(1 for term in system_terms if f" {term} " in f" {evidence_text} ")
    required = max(1, min(len(system_terms), 2))
    return heading_hits >= required or evidence_hits >= required


def _extract_attribute_value(attribute: str, text: str) -> str | None:
    compact = " ".join(str(text).split())
    if not compact:
        return None
    if attribute == "configuration":
        value = _first_match(compact, (MULTI_TRAIN_RE,))
        if value is not None:
            return value
        lowered = normalize_text(compact)
        for keyword in CONFIGURATION_KEYWORDS:
            if keyword in lowered:
                return compact
        return None
    if attribute == "capacity":
        return _first_match(compact, (MULTI_TRAIN_RE, CAPACITY_PERCENT_RE, FLOW_VALUE_RE))
    if attribute == "power":
        return _first_match(compact, (POWER_VALUE_RE,))
    if attribute == "pressure":
        return _first_match(compact, (PRESSURE_VALUE_RE,))
    if attribute == "temperature":
        return _first_match(compact, (TEMPERATURE_VALUE_RE,))
    if attribute == "flow":
        return _first_match(compact, (FLOW_VALUE_RE,))
    if attribute == "size":
        return _first_match(compact, (SIZE_VALUE_RE,))
    if attribute == "type":
        type_match = _first_match(compact, (MODEL_VALUE_RE, TYPE_VALUE_RE))
        if type_match is None:
            return None
        return type_match
    return None


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return " ".join(match.group(1).split())
    return None


def _supports_attribute(attribute: str, normalized_evidence: str, normalized_heading: str, value: str) -> bool:
    if attribute == "configuration":
        return True
    if attribute == "type" and MODEL_VALUE_RE.fullmatch(value):
        return True
    hints = ATTRIBUTE_HINTS.get(attribute, ())
    combined = f" {normalized_heading} {normalized_evidence} "
    return any(f" {normalize_text(hint)} " in combined for hint in hints)


def _label_supports_attribute(attribute: str, normalized_label: str) -> bool:
    return any(normalize_text(hint) in normalized_label for hint in ATTRIBUTE_HINTS.get(attribute, ()))


def _dedupe_entries(entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for attribute, value, evidence_text in entries:
        key = (attribute, normalize_text(value), normalize_text(evidence_text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((attribute, value, evidence_text))
    return deduped


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(" ".join(value.split()))
    return deduped
