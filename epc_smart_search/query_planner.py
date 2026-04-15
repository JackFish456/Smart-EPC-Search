from __future__ import annotations

import re
from dataclasses import dataclass

from epc_smart_search.search_features import (
    ACTION_LEXICON,
    ACTOR_LEXICON,
    TOPIC_LEXICON,
    alias_terms_for_text,
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
    content_query: str
    intent: str
    section_number: str | None
    count_question: bool
    direct_text_question: bool
    focus_terms: tuple[str, ...]
    actor_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    topic_terms: tuple[str, ...]
    expansion_terms: tuple[str, ...]

    @property
    def all_terms(self) -> tuple[str, ...]:
        return _dedupe(self.focus_terms + self.actor_terms + self.action_terms + self.topic_terms + self.expansion_terms)


QUERY_PREFIXES = (
    "what does the contract say about ",
    "what does it say about ",
    "does the contract mention ",
    "tell me about ",
    "show me ",
    "quote ",
    "exact ",
    "what is ",
    "what are ",
    "define ",
    "meaning of ",
)

STOPWORDS = {
    "a",
    "an",
    "about",
    "and",
    "are",
    "by",
    "contract",
    "does",
    "exact",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "many",
    "me",
    "mention",
    "provide",
    "provided",
    "of",
    "on",
    "or",
    "quote",
    "required",
    "say",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
}


def plan_query(query: str) -> QueryPlan:
    normalized = normalize_text(query)
    content_query = _extract_content_query(normalized)
    section_number = _extract_section_number(query)
    count_question = normalized.startswith("how many ")
    direct_text_question = normalized.startswith(
        ("show me ", "quote ", "exact ", "what does the contract say about ", "what does it say about ", "does the contract mention ")
    )
    actor_terms = _match_labels(content_query, ACTOR_LEXICON)
    action_terms = _match_labels(content_query, ACTION_LEXICON)
    topic_terms = _match_labels(content_query, TOPIC_LEXICON)
    expansion_terms = _dedupe(tuple(expand_query_phrases(normalized) + alias_terms_for_text(content_query)))
    focus_terms = _focus_terms(content_query)

    intent = "general_topic"
    if section_number:
        intent = "section_lookup"
    elif count_question or direct_text_question:
        intent = "direct_text"
    elif normalized.startswith(("what is ", "what are ", "define ", "meaning of ")) or " definition" in normalized:
        intent = "definition"
    elif "responsible for" in normalized or normalized.startswith("who "):
        intent = "responsibility"
    elif "pay" in normalized or "cost" in normalized or "liable" in normalized:
        intent = "payment_liability"
    elif "terminate" in normalized or "termination" in normalized or "end the contract" in normalized:
        intent = "termination"
    elif "delay" in normalized or "late" in normalized or "weather" in normalized or "schedule" in normalized:
        intent = "delay_schedule"

    topic_terms = _refine_topic_terms(normalized, content_query, topic_terms, intent)

    return QueryPlan(
        raw_query=query,
        normalized_query=normalized,
        content_query=content_query,
        intent=intent,
        section_number=section_number,
        count_question=count_question,
        direct_text_question=direct_text_question,
        focus_terms=focus_terms,
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
    focus_phrases = _focus_phrases(plan.focus_terms)
    if text_terms:
        parts = [f'"{phrase}"' for phrase in focus_phrases]
        parts.extend(f'"{term}"' for term in text_terms[:12])
        queries.append(" OR ".join(parts))
    return [query for query in queries if query]


def build_like_fallback(plan: QueryPlan) -> str:
    tokens = list(_token_terms(plan))
    if not tokens:
        return plan.content_query
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
    return plan.all_terms


def _extract_content_query(normalized_query: str) -> str:
    content = normalized_query
    for prefix in QUERY_PREFIXES:
        if content.startswith(prefix):
            content = content[len(prefix):]
            break
    content = content.strip(" ?.!")
    return content or normalized_query


def _focus_terms(content_query: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in TOKEN_RE.findall(content_query)
        if len(token) > 2 and normalize_text(token) not in STOPWORDS and normalize_text(token) not in ACTOR_LEXICON
    ]
    return _dedupe(tuple(tokens))


def _refine_topic_terms(
    normalized_query: str,
    content_query: str,
    topic_terms: tuple[str, ...],
    intent: str,
) -> tuple[str, ...]:
    if "responsibility" not in topic_terms:
        return topic_terms
    explicit_responsibility = intent == "responsibility" or any(
        term in normalized_query or term in content_query
        for term in ("responsible", "responsibility", "obligation", "obligations")
    )
    if explicit_responsibility:
        return topic_terms
    return tuple(term for term in topic_terms if term != "responsibility")


def _focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
    if len(focus_terms) < 2:
        return ()
    phrases: list[str] = [" ".join(focus_terms[: min(3, len(focus_terms))])]
    phrases.extend(
        " ".join(focus_terms[index:index + 2])
        for index in range(min(len(focus_terms) - 1, 3))
    )
    return _dedupe(tuple(phrases))


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
