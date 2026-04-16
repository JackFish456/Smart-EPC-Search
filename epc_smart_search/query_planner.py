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
    system_phrase: str = ""
    system_terms: tuple[str, ...] = ()
    system_aliases: tuple[str, ...] = ()
    attribute_label: str | None = None
    attribute_terms: tuple[str, ...] = ()

    @property
    def all_terms(self) -> tuple[str, ...]:
        return _dedupe(
            self.system_terms
            + self.attribute_terms
            + self.focus_terms
            + self.actor_terms
            + self.action_terms
            + self.topic_terms
            + self.expansion_terms
        )


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

SYSTEM_FILLER = STOPWORDS | {
    "current",
    "do",
    "does",
    "did",
    "our",
    "using",
    "use",
    "used",
    "work",
    "works",
}

ATTRIBUTE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("design_conditions", ("design conditions", "design condition", "operating conditions", "operating condition", "design basis")),
    ("configuration", ("configuration", "configured", "arrangement", "arranged")),
    ("size", ("size", "sizes", "diameter", "rating", "dimensions")),
    ("capacity", ("capacity", "capacities", "output", "duty", "throughput")),
    ("pressure", ("pressure", "pressures", "psig", "psi")),
    ("temperature", ("temperature", "temperatures")),
    ("flow", ("flow", "flows", "flowrate", "flow rate", "throughput")),
    ("type", ("type", "kind", "model", "selected", "selection")),
    ("responsibility", ("responsible", "provided by", "provide", "furnish", "supply")),
    ("definition", ("defined", "definition", "means", "meaning")),
)

ATTRIBUTE_LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "configuration": ("configuration", "configured", "arrangement"),
    "design_conditions": ("design conditions", "design condition", "design basis", "operating conditions"),
    "type": ("type", "kind", "model", "selected", "manufacturer"),
    "size": ("size", "diameter", "rating", "dimension"),
    "capacity": ("capacity", "output", "duty", "throughput"),
    "pressure": ("pressure", "pressures", "psi", "psig"),
    "temperature": ("temperature", "temperatures"),
    "flow": ("flow", "flow rate", "flowrate", "throughput"),
    "responsibility": ("responsible", "provide", "provided", "supply", "furnish"),
    "definition": ("means", "defined", "definition"),
    "function": ("work", "works", "operate", "operation", "process"),
}

GENERIC_SYSTEM_HEADS = {
    "system",
    "systems",
    "equipment",
    "package",
    "packages",
    "module",
    "modules",
    "unit",
    "units",
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
    expansion_terms = tuple(expand_query_phrases(normalized))
    focus_terms = _focus_terms(content_query)
    attribute_label, attribute_terms = _detect_attribute(normalized, content_query)
    system_phrase = _extract_system_phrase(content_query, attribute_terms, attribute_label)
    system_terms = _focus_terms(system_phrase)
    system_aliases = _system_aliases(system_phrase, system_terms)

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
        system_phrase=system_phrase,
        system_terms=system_terms,
        system_aliases=system_aliases,
        attribute_label=attribute_label,
        attribute_terms=attribute_terms,
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
    if plan.system_phrase and plan.attribute_terms:
        return " ".join([plan.system_phrase, *plan.attribute_terms[:3]]).strip()
    if plan.system_phrase:
        return plan.system_phrase
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


def _detect_attribute(normalized_query: str, content_query: str) -> tuple[str | None, tuple[str, ...]]:
    content = f" {content_query} "
    normalized = f" {normalized_query} "
    if normalized_query.startswith(("how does ", "how do ")) or content_query.endswith((" work", " works")):
        return "function", ATTRIBUTE_LABEL_TERMS["function"]
    if re.search(r"\bwe (?:are )?using\b", normalized_query) or re.search(r"\b(?:using|used)\b", content_query):
        return "type", ATTRIBUTE_LABEL_TERMS["type"]
    for label, patterns in ATTRIBUTE_PATTERNS:
        if any(f" {normalize_text(pattern)} " in normalized or f" {normalize_text(pattern)} " in content for pattern in patterns):
            return label, ATTRIBUTE_LABEL_TERMS[label]
    return None, ()


def _extract_system_phrase(content_query: str, attribute_terms: tuple[str, ...], attribute_label: str | None) -> str:
    if not content_query:
        return ""
    attribute_words = {
        word
        for phrase in attribute_terms
        for word in TOKEN_RE.findall(phrase)
    }
    if attribute_label == "function":
        attribute_words.update({"work", "works", "operate", "operation"})
    tokens = [
        normalize_text(token)
        for token in TOKEN_RE.findall(content_query)
        if normalize_text(token) not in SYSTEM_FILLER and normalize_text(token) not in attribute_words
    ]
    if not tokens:
        return ""
    return " ".join(tokens)


def _system_aliases(system_phrase: str, system_terms: tuple[str, ...]) -> tuple[str, ...]:
    if not system_phrase:
        return ()
    aliases: list[str] = []
    singularized = _singularize_phrase(system_phrase)
    if singularized != system_phrase:
        aliases.append(singularized)
    if len(system_terms) >= 2:
        aliases.append(" ".join(system_terms[-2:]))
    if len(system_terms) >= 2 and system_terms[-1] in GENERIC_SYSTEM_HEADS:
        aliases.append(" ".join(system_terms[:-1]))
    if len(system_terms) == 1:
        aliases.append(system_terms[0])
    return _dedupe(tuple(alias for alias in aliases if alias and alias != system_phrase))


def _singularize_phrase(phrase: str) -> str:
    terms = list(_focus_terms(phrase))
    if not terms:
        return phrase
    tail = terms[-1]
    if len(tail) > 3 and tail.endswith("s") and not tail.endswith("ss"):
        terms[-1] = tail[:-1]
    return " ".join(terms)


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
