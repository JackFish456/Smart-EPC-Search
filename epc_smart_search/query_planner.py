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
from epc_smart_search.system_vocabulary import SystemVocabulary, system_significant_terms

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{1,}")
SECTION_QUERY_RE = re.compile(r"\b(?:section|sec\.?)\s*(\d+(?:\.\d+){0,5})\b", re.IGNORECASE)
BARE_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s*$")
SCOPE_TERM_RE = re.compile(r"\b(?:appendix|exhibit|attachment)(?:\s+[A-Za-z0-9.\-]+)?\b", re.IGNORECASE)

REQUEST_SHAPE_SCALAR = "scalar"
REQUEST_SHAPE_GROUPED_LIST = "grouped_list"
REQUEST_SHAPE_REFERENCE_LOOKUP = "reference_lookup"
REQUEST_SHAPE_DEFINITION = "definition"
REQUEST_SHAPE_RESPONSIBILITY = "responsibility"
REQUEST_SHAPE_DIRECT_TEXT = "direct_text"
REQUEST_SHAPE_BROAD_TOPIC = "broad_topic"

ANSWER_FAMILY_GUARANTEE_OR_LIMIT = "guarantee_or_limit"
RETRIEVAL_MODE_FACT_LOOKUP = "fact_lookup"
RETRIEVAL_MODE_TOPIC_SUMMARY = "topic_summary"
RETRIEVAL_MODE_FALLBACK = "fallback"
EXACT_VALUE_ATTRIBUTE_LABELS = ("configuration", "capacity", "type", "size", "power", "pressure", "temperature", "flow")


@dataclass(slots=True, frozen=True)
class QueryPlan:
    raw_query: str
    normalized_query: str
    content_query: str
    intent: str
    retrieval_mode: str
    request_shape: str
    section_number: str | None
    count_question: bool
    direct_text_question: bool
    focus_terms: tuple[str, ...]
    concept_terms: tuple[str, ...]
    scope_terms: tuple[str, ...]
    actor_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    topic_terms: tuple[str, ...]
    expansion_terms: tuple[str, ...]
    aggregate_requested: bool = False
    answer_family: str | None = None
    system_phrase: str = ""
    system_terms: tuple[str, ...] = ()
    system_aliases: tuple[str, ...] = ()
    attribute_label: str | None = None
    attribute_terms: tuple[str, ...] = ()
    system_significant_terms: tuple[str, ...] = ()

    @property
    def all_terms(self) -> tuple[str, ...]:
        return _dedupe(
            self.concept_terms
            + self.scope_terms
            + self.system_terms
            + self.attribute_terms
            + self.focus_terms
            + self.actor_terms
            + self.action_terms
            + self.topic_terms
            + self.expansion_terms
        )


QUERY_PREFIXES = (
    "give me information about ",
    "information about ",
    "give me details about ",
    "details about ",
    "overview of ",
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
    "any",
    "all",
    "and",
    "are",
    "by",
    "contract",
    "does",
    "details",
    "exact",
    "for",
    "from",
    "give",
    "how",
    "information",
    "in",
    "is",
    "it",
    "many",
    "me",
    "my",
    "list",
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
    "overview",
    "your",
    "to",
    "what",
}

SYSTEM_FILLER = STOPWORDS | {
    "current",
    "do",
    "does",
    "did",
    "site",
    "we",
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
    ("power", ("horsepower", "horse power", "hp", "kw", "kilowatt", "kilowatts", "motor power")),
    ("pressure", ("pressure", "pressures", "psig", "psi")),
    ("temperature", ("temperature", "temperatures")),
    ("flow", ("flow", "flows", "flowrate", "flow rate", "throughput")),
    ("type", ("type", "kind", "model", "selected", "selection")),
    ("definition", ("defined", "definition", "means", "meaning")),
)

QUESTION_TAIL_PATTERNS = (
    r"\bdo we have$",
    r"\bdo we use$",
    r"\bare we using$",
    r"\bis used$",
    r"\bare there$",
    r"\bis there$",
    r"\bdo we need$",
    r"\bdo we require$",
)

ATTRIBUTE_LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "configuration": ("configuration", "configured", "arrangement"),
    "design_conditions": ("design conditions", "design condition", "design basis", "operating conditions"),
    "type": ("type", "kind", "model", "selected", "manufacturer"),
    "size": ("size", "diameter", "rating", "dimension"),
    "capacity": ("capacity", "output", "duty", "throughput"),
    "power": ("horsepower", "horse power", "hp", "kw", "kilowatt", "motor power"),
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


def plan_query(query: str, system_vocabulary: SystemVocabulary | None = None) -> QueryPlan:
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
    scope_terms = _detect_scope_terms(normalized, content_query)
    attribute_label, attribute_terms = _detect_attribute(normalized, content_query)
    answer_family = _detect_answer_family(normalized, content_query, focus_terms)
    aggregate_requested = _detect_aggregate_request(normalized, attribute_label, answer_family, focus_terms)
    extracted_system_phrase = _extract_system_phrase(content_query, attribute_terms, attribute_label)
    matched_system = system_vocabulary.match(content_query, extracted_system_phrase) if system_vocabulary and extracted_system_phrase else None
    broad_topic_requested = _detect_broad_topic_request(
        normalized,
        content_query,
        extracted_system_phrase,
        matched_system.canonical_phrase if matched_system is not None else "",
        attribute_label,
        count_question,
        direct_text_question,
        aggregate_requested,
        answer_family,
        focus_terms,
    )

    intent = "general_topic"
    if section_number:
        intent = "section_lookup"
    elif count_question or direct_text_question:
        intent = "direct_text"
    elif attribute_label == "definition" or (
        attribute_label is None
        and (
            normalized.startswith(("what is ", "define ", "meaning of "))
            or " definition" in normalized
            or " defined as " in normalized
            or " means " in normalized
        )
    ):
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
    request_shape = _determine_request_shape(
        intent,
        section_number,
        count_question,
        direct_text_question,
        aggregate_requested,
        answer_family,
        broad_topic_requested,
    )
    concept_terms = _detect_concept_terms(content_query, scope_terms, attribute_terms, focus_terms, request_shape)

    bind_system_phrase = request_shape not in {REQUEST_SHAPE_GROUPED_LIST, REQUEST_SHAPE_BROAD_TOPIC}
    extracted_system_phrase = extracted_system_phrase if bind_system_phrase else ""
    matched_system = matched_system if bind_system_phrase else None
    system_phrase = matched_system.canonical_phrase if matched_system is not None else extracted_system_phrase
    system_terms = matched_system.terms if matched_system is not None else _focus_terms(system_phrase)
    system_aliases = (
        _dedupe(((extracted_system_phrase,) if extracted_system_phrase else ()) + matched_system.aliases)
        if matched_system is not None
        else _system_aliases(system_phrase, system_terms)
    )
    system_significant = system_significant_terms(system_terms)
    retrieval_mode = _determine_retrieval_mode(
        request_shape=request_shape,
        attribute_label=attribute_label,
        system_phrase=system_phrase,
        count_question=count_question,
        direct_text_question=direct_text_question,
        intent=intent,
    )

    return QueryPlan(
        raw_query=query,
        normalized_query=normalized,
        content_query=content_query,
        intent=intent,
        retrieval_mode=retrieval_mode,
        request_shape=request_shape,
        section_number=section_number,
        count_question=count_question,
        direct_text_question=direct_text_question,
        focus_terms=focus_terms,
        concept_terms=concept_terms,
        scope_terms=scope_terms,
        actor_terms=actor_terms,
        action_terms=action_terms,
        topic_terms=topic_terms,
        expansion_terms=expansion_terms,
        aggregate_requested=aggregate_requested,
        answer_family=answer_family,
        system_phrase=system_phrase,
        system_terms=system_terms,
        system_aliases=system_aliases,
        attribute_label=attribute_label,
        attribute_terms=attribute_terms,
        system_significant_terms=system_significant,
    )


def build_match_queries(plan: QueryPlan) -> list[str]:
    queries: list[str] = []
    if plan.section_number:
        queries.append(f'section_number : "{plan.section_number}"')
    heading_terms = list((plan.scope_terms + plan.concept_terms + plan.topic_terms + plan.action_terms + plan.expansion_terms)[:6])
    if heading_terms:
        heading_parts = [f'heading : "{term}"' for term in heading_terms]
        heading_parts.extend(f'parent_heading : "{term}"' for term in heading_terms[:3])
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
    if plan.concept_terms and plan.scope_terms:
        return " ".join([*plan.scope_terms[:2], *plan.concept_terms[:4]]).strip()
    if plan.concept_terms:
        return " ".join(plan.concept_terms[:6]).strip()
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
    for pattern in QUESTION_TAIL_PATTERNS:
        content = re.sub(pattern, "", content).strip()
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
    if re.search(
        r"\b(who is responsible|who provides|who furnishes|who supplies|provided by|responsible for|required to provide|required to furnish|required to supply)\b",
        normalized_query,
    ):
        return "responsibility", ATTRIBUTE_LABEL_TERMS["responsibility"]
    for label, patterns in ATTRIBUTE_PATTERNS:
        if any(f" {normalize_text(pattern)} " in normalized or f" {normalize_text(pattern)} " in content for pattern in patterns):
            return label, ATTRIBUTE_LABEL_TERMS[label]
    return None, ()


def _detect_scope_terms(normalized_query: str, content_query: str) -> tuple[str, ...]:
    candidates = list(SCOPE_TERM_RE.findall(normalized_query))
    candidates.extend(SCOPE_TERM_RE.findall(content_query))
    normalized_candidates: list[str] = []
    for candidate in candidates:
        cleaned = normalize_text(candidate)
        if not cleaned:
            continue
        normalized_candidates.append(cleaned)
        normalized_candidates.append(cleaned.split()[0])
    return _dedupe(tuple(normalized_candidates))


def _detect_answer_family(normalized_query: str, content_query: str, focus_terms: tuple[str, ...]) -> str | None:
    combined = f" {normalized_query} {content_query} "
    markers = ("guarantee", "guarantees", "guaranteed", "limit", "limits")
    if any(f" {marker} " in combined for marker in markers):
        return ANSWER_FAMILY_GUARANTEE_OR_LIMIT
    if "shall not exceed" in combined and any(term in focus_terms for term in ("emission", "emissions", "nox", "co")):
        return ANSWER_FAMILY_GUARANTEE_OR_LIMIT
    return None


def _detect_aggregate_request(
    normalized_query: str,
    attribute_label: str | None,
    answer_family: str | None,
    focus_terms: tuple[str, ...],
) -> bool:
    if attribute_label is not None:
        return False
    wrapped = f" {normalized_query} "
    if any(phrase in wrapped for phrase in (" all ", " list ", " show me all ", " give me all ")):
        return True
    if normalized_query.startswith("what are ") and answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
        return True
    return any(term in {"guarantees", "limits", "requirements", "values"} for term in focus_terms)


def _determine_request_shape(
    intent: str,
    section_number: str | None,
    count_question: bool,
    direct_text_question: bool,
    aggregate_requested: bool,
    answer_family: str | None,
    broad_topic_requested: bool,
) -> str:
    if section_number:
        return REQUEST_SHAPE_REFERENCE_LOOKUP
    if aggregate_requested and answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
        return REQUEST_SHAPE_GROUPED_LIST
    if broad_topic_requested:
        return REQUEST_SHAPE_BROAD_TOPIC
    if count_question or direct_text_question:
        return REQUEST_SHAPE_DIRECT_TEXT
    if intent == "definition":
        return REQUEST_SHAPE_DEFINITION
    if intent == "responsibility":
        return REQUEST_SHAPE_RESPONSIBILITY
    return REQUEST_SHAPE_SCALAR


def _determine_retrieval_mode(
    *,
    request_shape: str,
    attribute_label: str | None,
    system_phrase: str,
    count_question: bool,
    direct_text_question: bool,
    intent: str,
) -> str:
    if request_shape == REQUEST_SHAPE_BROAD_TOPIC:
        return RETRIEVAL_MODE_TOPIC_SUMMARY
    if (
        request_shape == REQUEST_SHAPE_SCALAR
        and attribute_label in EXACT_VALUE_ATTRIBUTE_LABELS
        and bool(system_phrase)
        and not count_question
        and not direct_text_question
        and intent not in {"definition", "responsibility"}
    ):
        return RETRIEVAL_MODE_FACT_LOOKUP
    return RETRIEVAL_MODE_FALLBACK


def _detect_concept_terms(
    content_query: str,
    scope_terms: tuple[str, ...],
    attribute_terms: tuple[str, ...],
    focus_terms: tuple[str, ...],
    request_shape: str,
) -> tuple[str, ...]:
    scope_tokens = {normalize_text(token) for scope in scope_terms for token in TOKEN_RE.findall(scope)}
    attribute_tokens = {normalize_text(token) for phrase in attribute_terms for token in TOKEN_RE.findall(phrase)}
    content_tokens = [
        normalize_text(token)
        for token in TOKEN_RE.findall(content_query)
        if normalize_text(token) not in STOPWORDS and normalize_text(token) not in scope_tokens
    ]
    if request_shape == REQUEST_SHAPE_GROUPED_LIST:
        return _dedupe(tuple(token for token in content_tokens if token not in attribute_tokens))
    return _dedupe(tuple(token for token in focus_terms if token not in attribute_tokens and token not in scope_tokens))


def _detect_broad_topic_request(
    normalized_query: str,
    content_query: str,
    extracted_system_phrase: str,
    matched_system_phrase: str,
    attribute_label: str | None,
    count_question: bool,
    direct_text_question: bool,
    aggregate_requested: bool,
    answer_family: str | None,
    focus_terms: tuple[str, ...],
) -> bool:
    if count_question or direct_text_question or attribute_label is not None:
        return False
    if aggregate_requested and answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
        return False
    normalized = f" {normalized_query} "
    broad_prefixes = (
        "give me information about ",
        "information about ",
        "give me details about ",
        "details about ",
        "overview of ",
        "what do we have about ",
    )
    if normalized_query.startswith(broad_prefixes):
        return True
    if normalized_query.startswith(("do we have any ", "do we have ")) and len(focus_terms) >= 2:
        return True
    broad_markers = {"requirements", "requirement", "permits", "permit", "permitting", "environmental", "compliance"}
    if not any(marker in focus_terms for marker in broad_markers):
        return False
    canonical_system = matched_system_phrase or extracted_system_phrase
    if canonical_system and len(TOKEN_RE.findall(canonical_system)) >= 3:
        return False
    if " environmental requirements " in normalized or " air permits " in normalized or " permit requirements " in normalized:
        return True
    return any(marker in focus_terms for marker in broad_markers)


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
