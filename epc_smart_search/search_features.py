from __future__ import annotations

import re
from dataclasses import dataclass

from epc_smart_search.chunking import ChunkRecord

WORD_RE = re.compile(r"[a-z0-9][a-z0-9/&\-]{1,}")
SPACE_RE = re.compile(r"\s+")
DATE_LIKE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")

ACTOR_LEXICON: dict[str, tuple[str, ...]] = {
    "owner": ("owner", "company", "purchaser", "buyer", "nrg"),
    "contractor": ("contractor", "seller", "subcontractor", "vendor"),
    "authority": ("authority", "authorities", "governmental", "municipal", "agency"),
    "engineer": ("engineer", "designer"),
}

ACTION_LEXICON: dict[str, tuple[str, ...]] = {
    "permit": ("permit", "permits", "permitting", "approval", "approvals", "license", "licenses", "consent"),
    "pay": ("pay", "pays", "payment", "payments", "compensate", "compensation", "cost", "costs"),
    "terminate": ("terminate", "termination", "end", "cancel", "convenience", "default"),
    "delay": ("delay", "delays", "late", "lateness", "slip", "slippage"),
    "schedule": ("schedule", "milestone", "milestones", "completion", "substantial completion"),
    "define": ("means", "shall mean", "defined", "definition", "definitions"),
    "indemnify": ("indemnify", "indemnity", "hold harmless", "defend"),
    "design": ("design", "engineering", "engineer"),
    "construct": ("construct", "construction", "build", "install", "erect"),
}

TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "permitting": ("permit", "permits", "permitting", "approval", "approvals", "license", "licenses"),
    "liquidated damages": ("liquidated damages", "late substantial completion", "delay damages", "late completion"),
    "termination": ("termination", "terminate", "convenience", "default"),
    "weather": ("weather", "severe weather", "adverse weather"),
    "delay": ("delay", "delays", "late", "lateness", "schedule", "completion"),
    "payment": ("pay", "payment", "payments", "compensation", "invoice", "cost", "costs"),
    "responsibility": ("responsible", "responsibility", "obligation", "obligations", "shall", "must", "required"),
    "definition": ("means", "shall mean", "defined as", "definition", "definitions"),
}

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "responsible for": ("responsible", "obligation", "obligations", "shall", "must", "required"),
    "who is responsible": ("responsible", "obligation", "shall", "must"),
    "finishes late": ("delay", "delays", "late substantial completion", "liquidated damages"),
    "end the contract": ("terminate", "termination", "convenience", "default"),
    "weather delays": ("weather", "severe weather", "delay", "schedule"),
    "for convenience": ("convenience", "termination"),
}


@dataclass(slots=True, frozen=True)
class ChunkFeatures:
    chunk_id: str
    document_id: str
    section_number: str | None
    heading: str
    parent_heading: str
    search_text: str
    rescue_text: str
    clause_type: str
    actor_tags: str
    action_tags: str
    topic_tags: str
    noise_flags: str


def normalize_text(text: str) -> str:
    lowered = SPACE_RE.sub(" ", text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').lower())
    return lowered.strip()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def expand_query_phrases(query: str) -> list[str]:
    normalized = normalize_text(query)
    phrases: list[str] = []
    for source, targets in QUERY_EXPANSIONS.items():
        if source in normalized:
            phrases.extend(targets)
    return _dedupe_preserve_order(phrases)


def build_chunk_features(chunks: list[ChunkRecord]) -> list[ChunkFeatures]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    features: list[ChunkFeatures] = []
    for chunk in chunks:
        parent_heading = ""
        parent = chunk_by_id.get(chunk.parent_chunk_id or "")
        if parent is not None:
            parent_heading = parent.heading
        normalized_heading = normalize_text(chunk.heading)
        normalized_parent = normalize_text(parent_heading)
        normalized_body = normalize_text(chunk.full_text)
        combined = " ".join(part for part in [chunk.section_number or "", normalized_heading, normalized_parent, normalized_body] if part)
        actor_tags = _detect_tags(combined, ACTOR_LEXICON)
        action_tags = _detect_tags(combined, ACTION_LEXICON)
        topic_tags = _detect_tags(combined, TOPIC_LEXICON)
        noise_flags = _noise_flags(chunk, normalized_heading, normalized_body)
        rescue_text = _build_rescue_text(
            chunk.section_number or "",
            normalized_heading,
            normalized_parent,
            normalized_body,
            actor_tags,
            action_tags,
            topic_tags,
        )
        search_terms = _dedupe_preserve_order(
            [
                chunk.section_number or "",
                normalized_heading,
                normalized_parent,
                normalized_body,
                actor_tags,
                action_tags,
                topic_tags,
                chunk.chunk_type,
            ]
        )
        features.append(
            ChunkFeatures(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_number=chunk.section_number,
                heading=chunk.heading,
                parent_heading=parent_heading,
                search_text=" ".join(term for term in search_terms if term),
                rescue_text=rescue_text,
                clause_type=chunk.chunk_type,
                actor_tags=actor_tags,
                action_tags=action_tags,
                topic_tags=topic_tags,
                noise_flags=" ".join(noise_flags),
            )
        )
    return features


def _detect_tags(text: str, lexicon: dict[str, tuple[str, ...]]) -> str:
    tags: list[str] = []
    normalized = f" {normalize_text(text)} "
    for label, variants in lexicon.items():
        if any(f" {normalize_text(variant)} " in normalized for variant in variants):
            tags.append(label)
    return " ".join(tags)


def _noise_flags(chunk: ChunkRecord, normalized_heading: str, normalized_body: str) -> list[str]:
    flags: list[str] = []
    if "...." in chunk.heading or chunk.full_text.count("....") >= 3:
        flags.append("toc")
    if chunk.page_start <= 20 and chunk.section_number is None:
        flags.append("front_matter")
    if DATE_LIKE_RE.match(chunk.heading.strip()):
        flags.append("date_heading")
    if len(tokenize(normalized_heading)) <= 1 and len(tokenize(normalized_body)) <= 8:
        flags.append("thin")
    return flags


def _build_rescue_text(
    section_number: str,
    normalized_heading: str,
    normalized_parent: str,
    normalized_body: str,
    actor_tags: str,
    action_tags: str,
    topic_tags: str,
) -> str:
    body_tokens = tokenize(normalized_body)[:36]
    parts = [
        section_number,
        normalized_heading,
        normalized_parent,
        " ".join(body_tokens),
        actor_tags,
        action_tags,
        topic_tags,
    ]
    return " ".join(part for part in parts if part)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = SPACE_RE.sub(" ", str(item)).strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out
