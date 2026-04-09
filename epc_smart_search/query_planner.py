from __future__ import annotations

import re
from dataclasses import dataclass

from epc_smart_search.search_features import (
    ACTION_LEXICON,
    ACTOR_LEXICON,
    TOPIC_LEXICON,
    expand_query_phrases,
    normalize_text,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{1,}")
SECTION_QUERY_RE = re.compile(r"\b(?:section|sec\.?)\s*(\d+(?:\.\d+){0,5})\b", re.IGNORECASE)
BARE_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s*$")


@dataclass(slots=True, frozen=True)
class QueryPlan:
    raw_query: str
    normalized_query: str
    intent: str
    section_number: str | None
    count_question: bool
    direct_text_question: bool
    actor_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    topic_terms: tuple[str, ...]
    expansion_terms: tuple[str, ...]

    @property
    def all_terms(self) -> tuple[str, ...]:
        return _dedupe(self.actor_terms + self.action_terms + self.topic_terms + self.expansion_terms)


def plan_query(query: str) -> QueryPlan:
    normalized = normalize_text(query)
    section_number = _extract_section_number(query)
    count_question = normalized.startswith("how many ")
    direct_text_question = normalized.startswith(("show me ", "quote ", "exact ", "what does the contract say about "))
    actor_terms = _match_labels(normalized, ACTOR_LEXICON)
    action_terms = _match_labels(normalized, ACTION_LEXICON)
    topic_terms = _match_labels(normalized, TOPIC_LEXICON)
    expansion_terms = tuple(expand_query_phrases(normalized))

    intent = "general_topic"
    if section_number:
        intent = "section_lookup"
    elif normalized.startswith(("what is ", "define ", "what does ", "meaning of ")) or " definition" in normalized:
        intent = "definition"
    elif count_question or direct_text_question:
        intent = "direct_text"
    elif "responsible for" in normalized or normalized.startswith("who "):
        intent = "responsibility"
    elif "pay" in normalized or "cost" in normalized or "liable" in normalized:
        intent = "payment_liability"
    elif "terminate" in normalized or "termination" in normalized or "end the contract" in normalized:
        intent = "termination"
    elif "delay" in normalized or "late" in normalized or "weather" in normalized or "schedule" in normalized:
        intent = "delay_schedule"

    return QueryPlan(
        raw_query=query,
        normalized_query=normalized,
        intent=intent,
        section_number=section_number,
        count_question=count_question,
        direct_text_question=direct_text_question,
        actor_terms=actor_terms,
        action_terms=action_terms,
        topic_terms=topic_terms,
        expansion_terms=expansion_terms,
    )


def build_match_queries(plan: QueryPlan) -> list[str]:
    queries: list[str] = []
    if plan.section_number:
        queries.append(f'section_number : "{plan.section_number}"')
    heading_terms = list((plan.topic_terms + plan.action_terms + plan.expansion_terms)[:5])
    if heading_terms:
        heading_parts = [f'heading : "{term}"' for term in heading_terms]
        heading_parts.extend(f'parent_heading : "{term}"' for term in heading_terms[:2])
        queries.append(" OR ".join(heading_parts))
    tag_parts: list[str] = []
    tag_parts.extend(f'actor_tags : "{term}"' for term in plan.actor_terms)
    tag_parts.extend(f'action_tags : "{term}"' for term in plan.action_terms)
    tag_parts.extend(f'topic_tags : "{term}"' for term in plan.topic_terms)
    if tag_parts:
        queries.append(" OR ".join(tag_parts))
    text_terms = list(_token_terms(plan))
    if text_terms:
        phrase = " ".join(text_terms[:6])
        parts = [f'"{phrase}"'] if len(text_terms) >= 2 else []
        parts.extend(f'"{term}"' for term in text_terms[:12])
        queries.append(" OR ".join(parts))
    return [query for query in queries if query]


def build_like_fallback(plan: QueryPlan) -> str:
    tokens = list(_token_terms(plan))
    if not tokens:
        return plan.normalized_query
    return " ".join(tokens[:8])


def has_term_overlap(text: str, terms: tuple[str, ...]) -> bool:
    normalized = f" {normalize_text(text)} "
    return any(f" {normalize_text(term)} " in normalized for term in terms if term)


def _match_labels(normalized_query: str, lexicon: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    labels: list[str] = []
    wrapped = f" {normalized_query} "
    for label, variants in lexicon.items():
        if label in normalized_query or any(f" {normalize_text(variant)} " in wrapped for variant in variants):
            labels.append(label)
    return _dedupe(tuple(labels))


def _token_terms(plan: QueryPlan) -> tuple[str, ...]:
    tokens = [token for token in TOKEN_RE.findall(plan.normalized_query) if len(token) > 2]
    return _dedupe(tuple(tokens) + plan.all_terms)


def _extract_section_number(query: str) -> str | None:
    match = SECTION_QUERY_RE.search(query) or BARE_SECTION_RE.match(query.strip())
    return match.group(1) if match else None


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = normalize_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out)
