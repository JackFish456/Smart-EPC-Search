from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace
from typing import Sequence

from epc_smart_search.name_normalization import build_system_aliases, normalize_attribute_name, normalize_system_name
from epc_smart_search.query_planner import (
    ANSWER_FAMILY_GUARANTEE_OR_LIMIT,
    QueryPlan,
    REQUEST_SHAPE_BROAD_TOPIC,
    REQUEST_SHAPE_DIRECT_TEXT,
    REQUEST_SHAPE_GROUPED_LIST,
    REQUEST_SHAPE_REFERENCE_LOOKUP,
    RETRIEVAL_MODE_FACT_LOOKUP,
    RETRIEVAL_MODE_TOPIC_SUMMARY,
    has_term_overlap,
    plan_query,
)
from epc_smart_search.retrieval import Citation, ExactPageHit, RankedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{0,}")


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    citations: list[Citation]
    refused: bool


@dataclass(slots=True)
class FactAnswerCandidate:
    fact: object
    chunk_row: object | None
    normalized_system: str
    normalized_attribute: str
    normalized_value: str
    has_specific_support: bool


@dataclass(slots=True)
class FactAnswerAssessment:
    confidence: float
    direct_answer_allowed: bool
    reason: str
    unique_system_count: int
    unique_attribute_count: int
    unique_value_count: int
    candidate: FactAnswerCandidate | None


SUMMARY_MAX_NEW_TOKENS = 224
SUMMARY_ENABLE_THINKING = False
SUMMARY_CONTEXT_MAX_SECTIONS = 5
SUMMARY_EXCERPT_LIMIT = 520
EXPAND_MAX_NEW_TOKENS = 320
DEEP_MAX_NEW_TOKENS = 384
DEEP_ENABLE_THINKING = True
DEEP_CONTEXT_MAX_SECTIONS = 6
DEEP_CONTEXT_EXACT_HITS = 2
DEEP_PAGE_CONTEXT_SECTIONS = 3
GENERIC_SYSTEM_TERMS = {
    "system",
    "systems",
    "pump",
    "pumps",
    "valve",
    "valves",
    "motor",
    "motors",
    "turbine",
    "turbines",
    "compressor",
    "compressors",
    "fan",
    "fans",
    "module",
    "modules",
    "unit",
    "units",
    "equipment",
    "package",
    "packages",
}
GENERIC_SUBJECT_TERMS = {
    "design",
    "designs",
    "operating",
    "operation",
    "operations",
    "condition",
    "conditions",
    "basis",
    "pressure",
    "pressures",
    "temperature",
    "temperatures",
    "value",
    "values",
}
SUMMARY_COMPRESSION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "any",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "information",
    "is",
    "me",
    "of",
    "on",
    "or",
    "provide",
    "show",
    "summarize",
    "summarise",
    "tell",
    "the",
    "what",
}


class AnswerPolicy:
    def __init__(self, store, retriever) -> None:
        self.store = store
        self.retriever = retriever

    def answer(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None,
        gemma_client,
        *,
        deep_think: bool = False,
        expand_answer: bool = False,
        previous_answer: str | None = None,
    ) -> AssistantAnswer:
        effective_question = self.resolve_question(question, history)
        grounded_question = self.normalize_grounded_question(effective_question)
        plan = self._plan_query(grounded_question)
        trace = None
        if not deep_think and not expand_answer and plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP:
            trace = self._retrieve_trace(grounded_question, gemma_client, deep_think=deep_think)
        fact_answer = None if deep_think or expand_answer else self.build_fact_answer(grounded_question, plan=plan, trace=trace)
        if fact_answer is not None:
            return fact_answer
        exact_hits = self.retriever.find_exact_page_hits(grounded_question)
        exact_answer = None if expand_answer else self.build_exact_answer(grounded_question, exact_hits)
        if (
            exact_answer is not None
            and not deep_think
            and not (
                plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
                and exact_answer.text.startswith("Direct contract text:\n")
            )
        ):
            return exact_answer

        broad_topic_request = plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
        summary_request = broad_topic_request or self.prefers_generated_answer(question)
        if trace is None:
            trace = self._retrieve_trace(grounded_question, gemma_client, deep_think=deep_think)
        if broad_topic_request and trace is not None:
            ranked = list(trace.merged_ranked)
            citations = self.citations_from_ranked(ranked, grounded_question)
            citations = self._merge_ranked_context_enrichment(ranked, citations)
        elif trace is not None and trace.selected_bundle is not None:
            ranked = list(trace.selected_bundle.ranked_chunks)
            citations = list(trace.selected_bundle.citations)
            if self.should_use_merged_ranked_for_requirements(plan, trace):
                ranked = sorted(
                    trace.merged_ranked,
                    key=lambda chunk: self.score_requirement_candidate(chunk, plan),
                    reverse=True,
                )
                citations = self.citations_from_ranked(ranked, grounded_question)
                citations = self._merge_ranked_context_enrichment(ranked, citations)
        else:
            ranked = (
                self.retriever.retrieve(grounded_question, profile="deep")
                if deep_think
                else self.retriever.retrieve(grounded_question)
            )
            citations = self.retriever.expand_with_context(ranked) if ranked else []
        summary_ranked = ranked
        if self.should_compress_summary_context(plan, trace, summary_request=summary_request):
            summary_ranked = self.compress_context(
                ranked,
                grounded_question,
                plan.system_terms,
                plan.attribute_terms,
            )
        citations = self.limit_citations(citations) if citations else []

        if not deep_think and not ranked:
            return AssistantAnswer("I can't verify that from the contract.", [], True)

        strong_ranked_evidence = self.has_strong_ranked_evidence(plan, ranked)
        evidence_is_weak = not citations or not strong_ranked_evidence
        refusal_citations = citations if not evidence_is_weak else []
        grounded_answer = self.select_grounded_answer(
            grounded_question,
            ranked,
            citations,
            evidence_is_weak=evidence_is_weak,
        )
        broad_topic_answer = self.build_broad_topic_answer(grounded_question, summary_ranked, citations) if broad_topic_request else None
        grounded_broad_topic_answer = None if evidence_is_weak else broad_topic_answer

        if (
            grounded_answer is not None
            and not deep_think
            and not expand_answer
            and self.should_return_grounded_answer_before_generation(
                plan,
                broad_topic_request=broad_topic_request,
                summary_request=summary_request,
            )
        ):
            return grounded_answer

        if deep_think:
            prompt_context = self.build_deep_prompt_context(grounded_question, ranked, exact_hits)
            if not prompt_context and grounded_answer is not None:
                return grounded_answer
            if evidence_is_weak and not exact_hits:
                if grounded_answer is not None:
                    return grounded_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
        elif expand_answer:
            prompt_context = self.build_expand_prompt_context(grounded_question, ranked, previous_answer)
            if not prompt_context:
                if grounded_answer is not None:
                    return grounded_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
        else:
            prompt_context = (
                self.build_summary_prompt_context(grounded_question, summary_ranked)
                if summary_request
                else self._build_standard_prompt_context(grounded_question, ranked, citations, trace)
            )
            if summary_request and not prompt_context:
                if grounded_broad_topic_answer is not None:
                    return grounded_broad_topic_answer
                if grounded_answer is not None:
                    return grounded_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
            if not summary_request and not prompt_context:
                if grounded_answer is not None:
                    return grounded_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)

        gemma_kwargs: dict[str, object] = {
            "enable_thinking": DEEP_ENABLE_THINKING if deep_think else (SUMMARY_ENABLE_THINKING if summary_request else None),
            "max_new_tokens": DEEP_MAX_NEW_TOKENS if deep_think else (EXPAND_MAX_NEW_TOKENS if expand_answer else (SUMMARY_MAX_NEW_TOKENS if summary_request else None)),
            "response_style": "deep_answer" if deep_think else ("expand_answer" if expand_answer else ("detailed_summary" if summary_request else None)),
        }
        if expand_answer and previous_answer is not None:
            gemma_kwargs["previous_answer"] = previous_answer

        try:
            answer_text = gemma_client.ask(question, prompt_context, **gemma_kwargs)
        except Exception:
            if grounded_broad_topic_answer is not None:
                return grounded_broad_topic_answer
            if grounded_answer is not None:
                return grounded_answer
            answer_text = "I can't verify that from the contract."

        answer_text = answer_text.strip() or "I can't verify that from the contract."
        refused = answer_text == "I can't verify that from the contract."
        if (summary_request or deep_think or expand_answer) and (refused or not answer_text):
            if grounded_broad_topic_answer is not None:
                return grounded_broad_topic_answer
            if grounded_answer is not None:
                return grounded_answer
        return AssistantAnswer(answer_text, refusal_citations if refused and evidence_is_weak else citations, refused)

    def _plan_query(self, question: str) -> QueryPlan:
        planner = getattr(self.retriever, "plan_query", None)
        if callable(planner):
            return planner(question)
        return plan_query(question)

    def _retrieve_trace(self, effective_question: str, gemma_client, *, deep_think: bool) -> object | None:
        retrieve_trace = getattr(self.retriever, "retrieve_trace", None)
        if not callable(retrieve_trace):
            return None
        try:
            return retrieve_trace(
                effective_question,
                profile="deep" if deep_think else "normal",
                gemma_client=gemma_client,
            )
        except TypeError:
            return retrieve_trace(effective_question, profile="deep" if deep_think else "normal")

    def _build_standard_prompt_context(self, effective_question: str, ranked, citations, trace) -> str:
        if trace is not None and trace.selected_bundle is not None:
            build_bundle = getattr(self.retriever, "build_bundle_evidence_pack", None)
            if callable(build_bundle):
                return build_bundle(trace.selected_bundle)
        build_evidence = getattr(self.retriever, "build_evidence_pack", None)
        if callable(build_evidence):
            return build_evidence(effective_question, ranked, citations)
        return ""

    @staticmethod
    def should_compress_summary_context(plan: QueryPlan, trace, *, summary_request: bool) -> bool:
        if not summary_request:
            return False
        trace_mode = getattr(trace, "retrieval_mode", None) if trace is not None else None
        active_mode = trace_mode or plan.retrieval_mode
        return active_mode == RETRIEVAL_MODE_TOPIC_SUMMARY

    @classmethod
    def compress_context(
        cls,
        chunks: Sequence[RankedChunk],
        query: str,
        system_terms: Sequence[str],
        attribute_terms: Sequence[str],
    ) -> list[RankedChunk]:
        query_terms = cls.summary_compression_terms(query)
        system_keywords = cls.summary_compression_keywords(system_terms)
        attribute_keywords = cls.summary_compression_keywords(attribute_terms)
        compressed: list[RankedChunk] = []
        for chunk in chunks:
            compact = " ".join(chunk.full_text.split())
            if not compact:
                continue
            compressed_text = cls.compress_chunk_text(
                compact,
                query_terms=query_terms,
                system_keywords=system_keywords,
                attribute_keywords=attribute_keywords,
            )
            if not compressed_text:
                continue
            compressed.append(replace(chunk, full_text=compressed_text))
        return compressed

    @classmethod
    def compress_chunk_text(
        cls,
        text: str,
        *,
        query_terms: set[str],
        system_keywords: tuple[str, ...],
        attribute_keywords: tuple[str, ...],
    ) -> str:
        sentences = cls.split_sentences(text)
        if not sentences:
            return text
        kept: list[str] = []
        for sentence in sentences:
            if cls.should_keep_compressed_sentence(
                sentence,
                query_terms=query_terms,
                system_keywords=system_keywords,
                attribute_keywords=attribute_keywords,
            ):
                kept.append(sentence.strip())
        return " ".join(kept).strip()

    @classmethod
    def should_keep_compressed_sentence(
        cls,
        sentence: str,
        *,
        query_terms: set[str],
        system_keywords: tuple[str, ...],
        attribute_keywords: tuple[str, ...],
    ) -> bool:
        lowered = sentence.lower()
        sentence_terms = cls.summary_compression_terms(lowered)
        system_hits = cls.summary_keyword_hits(lowered, sentence_terms, system_keywords)
        attribute_hits = cls.summary_keyword_hits(lowered, sentence_terms, attribute_keywords)
        query_hits = len(query_terms.intersection(sentence_terms))
        if system_hits > 0 or attribute_hits > 0:
            return True
        if not query_terms:
            return False
        overlap_threshold = 2 if len(query_terms) >= 3 else 1
        if query_hits >= overlap_threshold:
            return True
        return bool(query_hits >= 1 and re.search(r"\d", sentence))

    @staticmethod
    def summary_compression_keywords(terms: Sequence[str]) -> tuple[str, ...]:
        keywords: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = " ".join(str(term).lower().split())
            if not normalized or normalized in SUMMARY_COMPRESSION_STOPWORDS or normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(normalized)
        return tuple(keywords)

    @staticmethod
    def summary_compression_terms(text: str) -> set[str]:
        terms: set[str] = set()
        for token in TOKEN_RE.findall(str(text).lower()):
            normalized = token.lower()
            if len(normalized) <= 2 or normalized in SUMMARY_COMPRESSION_STOPWORDS:
                continue
            terms.add(normalized)
            if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
                terms.add(normalized[:-1])
        return terms

    @staticmethod
    def summary_keyword_hits(lowered_sentence: str, sentence_terms: set[str], keywords: Sequence[str]) -> int:
        hits = 0
        for keyword in keywords:
            if " " in keyword:
                if keyword in lowered_sentence:
                    hits += 1
            elif keyword in sentence_terms:
                hits += 1
        return hits

    def build_page_attribute_answer(
        self,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        if not ranked or self.store is None or self.retriever is None:
            return None
        plan = plan_query(question)
        if plan.attribute_label != "design_conditions":
            return None
        resolve_document_id = getattr(self.retriever, "resolve_document_id", None)
        if not callable(resolve_document_id):
            return None
        document_id = resolve_document_id()
        if not document_id:
            return None
        for chunk in ranked[:4]:
            if not self.is_strong_question_match(plan, chunk):
                continue
            page_rows = self.store.fetch_page_window(
                document_id,
                chunk.page_start,
                chunk.page_end,
                padding=0,
                limit=max(1, min(2, chunk.page_end - chunk.page_start + 1)),
            )
            for row in page_rows:
                entries = self.extract_design_conditions_table_entries(
                    str(row["page_text"]),
                    section_number=chunk.section_number,
                    heading=chunk.heading,
                )
                if len(entries) < 4:
                    continue
                heading = " ".join(chunk.heading.split()) or "Design Conditions"
                label = chunk.section_number or "Unnumbered clause"
                pages = self.format_page_range(chunk.page_start, chunk.page_end)
                lines = "\n".join(f"- {name}: {value}" for name, value in entries[:8])
                body = f"{heading}:\n{lines}\nSection {label} - {heading}\nPages: {pages}"
                return AssistantAnswer(body, citations, False)
        return None

    def build_fact_answer(
        self,
        question: str,
        *,
        plan: QueryPlan | None = None,
        trace=None,
    ) -> AssistantAnswer | None:
        if self.store is None or self.retriever is None:
            return None
        plan = plan or self._plan_query(question)
        if plan.retrieval_mode != RETRIEVAL_MODE_FACT_LOOKUP or not plan.system_phrase or not plan.attribute_label:
            return None
        if trace is not None:
            fact = getattr(trace, "fact_hit", None)
            if fact is not None:
                chunk_row = self.store.fetch_chunk(fact.source_chunk_id)
                return self._fact_answer_from_row(fact, chunk_row)
            return None
        resolve_document_id = getattr(self.retriever, "resolve_document_id", None)
        if not callable(resolve_document_id):
            return None
        document_id = resolve_document_id()
        if not document_id:
            return None
        rows = self.lookup_fact_answer_rows(document_id, plan)
        if not rows:
            return None
        assessment = self.assess_fact_answer(plan, rows)
        if not assessment.direct_answer_allowed or assessment.candidate is None:
            return None
        return self._fact_answer_from_row(assessment.candidate.fact, assessment.candidate.chunk_row)

    def _fact_answer_from_row(self, fact: object, chunk_row: object | None) -> AssistantAnswer:
        section_number = str(self.fact_chunk_value(chunk_row, "section_number", "") or "") or None
        heading = " ".join(str(self.fact_chunk_value(chunk_row, "heading", "Contract evidence")).split())
        citation = Citation(
            chunk_id=str(getattr(fact, "source_chunk_id", "")),
            section_number=section_number,
            heading=heading,
            attachment=None,
            page_start=int(getattr(fact, "page_start", 0) or 0),
            page_end=int(getattr(fact, "page_end", 0) or 0),
            quote=str(getattr(fact, "evidence_text", "") or ""),
        )
        citations = [citation]
        extra_fn = getattr(self.retriever, "fact_hit_context_citations", None)
        if callable(extra_fn):
            for extra in extra_fn(fact):
                if extra.chunk_id == citation.chunk_id:
                    continue
                citations.append(extra)
        page_start = int(getattr(fact, "page_start", 0) or 0)
        page_end = int(getattr(fact, "page_end", 0) or 0)
        value = str(getattr(fact, "value", "") or "")
        evidence = str(getattr(fact, "evidence_text", "") or "")
        section_label = section_number or "Unnumbered clause"
        pages_label = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        return AssistantAnswer(
            f'"{value}"\nSection {section_label} - {heading}\nPages: {pages_label}\nEvidence: "{evidence}"',
            citations,
            False,
        )

    def lookup_fact_answer_rows(self, document_id: str, plan: QueryPlan) -> list[object]:
        rows: list[object] = []
        seen: set[tuple[object | None, str, int, int, str]] = set()
        systems = (plan.system_phrase, *plan.system_aliases)
        for system_name in systems:
            if not system_name:
                continue
            for row in self.store.lookup_facts_by_system_attribute(document_id, system_name, plan.attribute_label):
                key = (
                    getattr(row, "fact_rowid", None),
                    str(getattr(row, "source_chunk_id", "")),
                    int(getattr(row, "page_start", 0) or 0),
                    int(getattr(row, "page_end", 0) or 0),
                    self.normalize_exact_value(str(getattr(row, "value", ""))) or "",
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows

    def assess_fact_answer(self, plan: QueryPlan, rows: Sequence[object]) -> FactAnswerAssessment:
        plan_systems = {
            normalize_system_name(alias)
            for alias in (plan.system_phrase, *plan.system_aliases, *build_system_aliases(plan.system_phrase))
            if alias
        }
        plan_systems.discard("")
        plan_attribute = normalize_attribute_name(plan.attribute_label or "")
        chunk_cache: dict[str, object | None] = {}
        candidates: list[FactAnswerCandidate] = []
        matched_systems: set[str] = set()
        matched_attributes: set[str] = set()
        matched_values: set[str] = set()

        for row in rows:
            normalized_system = normalize_system_name(str(getattr(row, "system_normalized", "") or getattr(row, "system", "")))
            normalized_attribute = normalize_attribute_name(
                str(getattr(row, "attribute_normalized", "") or getattr(row, "attribute", ""))
            )
            normalized_value = self.normalize_exact_value(str(getattr(row, "value", "")))
            if normalized_system in plan_systems:
                matched_systems.add(normalized_system)
            if not plan_attribute or normalized_attribute == plan_attribute:
                matched_attributes.add(normalized_attribute)
            if normalized_value:
                matched_values.add(normalized_value)
            if normalized_system not in plan_systems or (plan_attribute and normalized_attribute != plan_attribute) or not normalized_value:
                continue
            chunk_id = str(getattr(row, "source_chunk_id", ""))
            if chunk_id not in chunk_cache:
                chunk_cache[chunk_id] = self.store.fetch_chunk(chunk_id)
            chunk_row = chunk_cache[chunk_id]
            candidates.append(
                FactAnswerCandidate(
                    fact=row,
                    chunk_row=chunk_row,
                    normalized_system=normalized_system,
                    normalized_attribute=normalized_attribute,
                    normalized_value=normalized_value,
                    has_specific_support=self.fact_candidate_has_specific_support(plan, row, chunk_row, normalized_value),
                )
            )

        strong_candidates = [candidate for candidate in candidates if candidate.has_specific_support]
        unique_systems = {candidate.normalized_system for candidate in candidates}
        unique_attributes = {candidate.normalized_attribute for candidate in candidates}
        unique_values = {candidate.normalized_value for candidate in strong_candidates} or {
            candidate.normalized_value for candidate in candidates
        }
        confidence = 0.0
        if candidates:
            confidence += 0.4
        if len(unique_systems) == 1 and unique_systems:
            confidence += 0.2
        if len(unique_attributes) == 1 and unique_attributes:
            confidence += 0.2
        if len(unique_values) == 1 and unique_values:
            confidence += 0.1
        if strong_candidates:
            confidence += 0.1
        confidence = max(0.0, min(1.0, confidence))

        if not candidates:
            return FactAnswerAssessment(
                confidence=confidence,
                direct_answer_allowed=False,
                reason="weak_system_match",
                unique_system_count=len(matched_systems),
                unique_attribute_count=len(matched_attributes),
                unique_value_count=len(matched_values),
                candidate=None,
            )
        if len(unique_systems) != 1:
            return FactAnswerAssessment(confidence, False, "ambiguous_system_match", len(unique_systems), len(unique_attributes), len(unique_values), None)
        if len(unique_attributes) != 1:
            return FactAnswerAssessment(confidence, False, "ambiguous_attribute_match", len(unique_systems), len(unique_attributes), len(unique_values), None)
        if len(unique_values) != 1:
            return FactAnswerAssessment(confidence, False, "conflicting_values", len(unique_systems), len(unique_attributes), len(unique_values), None)
        if not strong_candidates:
            return FactAnswerAssessment(confidence, False, "generic_only_support", len(unique_systems), len(unique_attributes), len(unique_values), None)

        selected = max(
            strong_candidates,
            key=lambda candidate: (
                len(str(getattr(candidate.fact, "evidence_text", ""))),
                len(str(self.fact_chunk_value(candidate.chunk_row, "full_text", ""))) if candidate.chunk_row is not None else 0,
            ),
        )
        return FactAnswerAssessment(confidence, True, "direct_exact_fact", len(unique_systems), len(unique_attributes), len(unique_values), selected)

    def fact_candidate_has_specific_support(
        self,
        plan: QueryPlan,
        fact: object,
        chunk_row: object | None,
        normalized_value: str,
    ) -> bool:
        evidence_text = str(getattr(fact, "evidence_text", "") or "")
        heading = str(self.fact_chunk_value(chunk_row, "heading", "") or "")
        body = str(self.fact_chunk_value(chunk_row, "full_text", "") or "")
        support_text = " ".join(part for part in [evidence_text, heading, body] if part).strip()
        if not support_text:
            return False
        support_lower = " ".join(support_text.lower().split())
        if plan.attribute_label and not self.matches_attribute(plan, support_lower):
            return False
        significant_terms = plan.system_significant_terms or tuple(
            term for term in plan.system_terms if term and term not in GENERIC_SYSTEM_TERMS
        )
        if significant_terms and not all(term in support_lower for term in significant_terms):
            return False
        return normalized_value.lower() in support_lower

    @staticmethod
    def fact_chunk_value(chunk_row: object | None, key: str, default: object = "") -> object:
        if chunk_row is None:
            return default
        try:
            value = chunk_row[key]  # type: ignore[index]
        except Exception:
            value = getattr(chunk_row, key, default)
        return default if value is None else value

    @classmethod
    def should_use_merged_ranked_for_requirements(cls, plan: QueryPlan, trace) -> bool:
        if trace is None or trace.selected_bundle is None:
            return False
        if plan.request_shape != "scalar":
            return False
        if plan.attribute_label is not None or plan.count_question or plan.direct_text_question:
            return False
        if not cls.is_requirement_question(plan):
            return False
        selected_ranked = list(trace.selected_bundle.ranked_chunks)
        merged_ranked = list(trace.merged_ranked)
        if len(merged_ranked) <= 1:
            return False
        return cls.has_materially_stronger_requirement_candidate(plan, selected_ranked, merged_ranked)

    @classmethod
    def has_materially_stronger_requirement_candidate(
        cls,
        plan: QueryPlan,
        selected_ranked: list[RankedChunk],
        merged_ranked: list[RankedChunk],
    ) -> bool:
        selected_best = max((cls.score_requirement_candidate(chunk, plan) for chunk in selected_ranked[:3]), default=0.0)
        merged_best = max((cls.score_requirement_candidate(chunk, plan) for chunk in merged_ranked[:6]), default=0.0)
        return merged_best >= selected_best + 1.0

    @classmethod
    def score_requirement_candidate(cls, chunk: RankedChunk, plan: QueryPlan) -> float:
        excerpt = cls.extract_ranked_excerpt(chunk.full_text, plan, limit=SUMMARY_EXCERPT_LIMIT)
        if not excerpt:
            return -10.0
        lowered = excerpt.lower()
        score = chunk.total_score
        if plan.system_phrase and plan.system_phrase in lowered:
            score += 1.4
        score += 0.8 * sum(1 for term in plan.concept_terms if term and term in lowered)
        score += 0.35 * cls.numeric_value_count(excerpt)
        if re.search(r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply)\b", lowered):
            score += 1.2
        if re.search(r"\b\d+(?:\.\d+)?\s*%", lowered):
            score += 1.0
        if re.search(r"\bmixture of\b|\bconsist(?:s|ing)? of\b|\bcomprised of\b", lowered):
            score += 1.2
        if re.search(r"\bbasis for design\b|\bdesigned to provide\b|\bmeet the needs of\b", lowered):
            score -= 1.0
        if len(excerpt) <= 220:
            score += 0.35
        return score

    @classmethod
    def citations_from_ranked(cls, ranked: list[RankedChunk], question: str, *, limit: int = 5) -> list[Citation]:
        plan = plan_query(question)
        citations: list[Citation] = []
        seen: set[tuple[str | None, int, int]] = set()
        for chunk in ranked:
            key = (chunk.section_number, chunk.page_start, chunk.page_end)
            if key in seen:
                continue
            excerpt = cls.extract_ranked_excerpt(
                chunk.full_text,
                plan,
                limit=SUMMARY_EXCERPT_LIMIT,
                surrounding_sentences=1,
            )
            if not excerpt or not cls.is_useful_summary_block(chunk, excerpt, plan):
                continue
            seen.add(key)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    section_number=chunk.section_number,
                    heading=" ".join(chunk.heading.split()),
                    attachment=None,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    quote=excerpt.strip().strip('"'),
                )
            )
            if len(citations) >= limit:
                break
        return citations

    def _merge_ranked_context_enrichment(
        self,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        limit: int = 5,
    ) -> list[Citation]:
        merge_fn = getattr(self.retriever, "merge_ranked_enrichment_citations", None)
        if not callable(merge_fn) or not ranked:
            return citations[:limit]
        return merge_fn(ranked, citations, limit=limit)

    @classmethod
    def resolve_question(cls, question: str, history: Sequence[dict[str, str]] | None = None) -> str:
        if not history:
            return question
        normalized = " ".join(question.lower().split())
        plan = plan_query(question)
        if not cls.looks_like_follow_up(question, plan):
            return question
        anchor = cls.find_follow_up_anchor(history)
        if not anchor:
            return question
        normalized_anchor = " ".join(anchor.lower().split())
        if normalized_anchor and normalized_anchor in normalized:
            return question
        stem = question.rstrip(" ?.!")
        return f"{stem} regarding {anchor}"

    @staticmethod
    def normalize_grounded_question(question: str) -> str:
        normalized = " ".join(question.split()).strip()
        if not normalized:
            return question
        cleaned = re.sub(r"^(?:can you\s+)?(?:please\s+)?(?:summari[sz]e|explain)\s+", "", normalized, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+in plain english$", "", cleaned, flags=re.IGNORECASE)
        return cleaned or normalized

    @classmethod
    def find_follow_up_anchor(cls, history: Sequence[dict[str, str]]) -> str:
        for turn in reversed(history):
            if str(turn.get("role", "")).strip() != "user":
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            plan = plan_query(content)
            if cls.looks_like_follow_up(content, plan):
                continue
            anchor = re.sub(r"^(summari[sz]e|explain)\s+", "", plan.content_query.strip(" ?.!"), flags=re.IGNORECASE)
            anchor = re.sub(r"\s+in plain english$", "", anchor, flags=re.IGNORECASE)
            if anchor:
                return anchor
        return ""

    @staticmethod
    def looks_like_follow_up(question: str, plan: QueryPlan) -> bool:
        normalized = " ".join(question.lower().split())
        if not normalized:
            return False
        follow_up_prefixes = (
            "what about",
            "how about",
            "what section",
            "which section",
            "what page",
            "which page",
            "where ",
            "when ",
            "who ",
            "quote that",
            "show me that",
            "does it ",
            "is it ",
            "is that ",
            "are those ",
            "and ",
            "also ",
        )
        referential_terms = (" it ", " that ", " this ", " these ", " those ", " they ", " them ", " same ")
        wrapped = f" {normalized} "
        has_referential_term = any(term in wrapped for term in referential_terms)
        if plan.section_number:
            return False
        if normalized.startswith(follow_up_prefixes):
            return True
        if has_referential_term and (
            not plan.focus_terms or (len(plan.focus_terms) == 1 and plan.focus_terms[0] in {"section", "page"})
        ):
            return True
        if len(plan.focus_terms) >= 2 or len(plan.content_query.split()) >= 4:
            return False
        return has_referential_term

    def build_exact_answer(self, question: str, hits: list[ExactPageHit]) -> AssistantAnswer | None:
        if not hits:
            return None
        lowered = " ".join(question.lower().split())
        plan = plan_query(question)
        if not self.prefer_exact_answer(lowered, plan):
            return None
        citations = self.citations_from_exact_hits(question, hits)
        excerpts = [f"Page {hit.page_num}: {self.trim_page_excerpt(hit.page_text, lowered)}" for hit in hits[:4]]
        if self.is_count_question(lowered):
            quantity = self.extract_count_value(lowered, hits)
            if quantity is not None:
                body = f"Answer: {quantity}\n\nDirect contract text:\n" + "\n\n".join(excerpts)
                return AssistantAnswer(body, citations, False)
        attribute_value = self.extract_attribute_value(question, hits, plan=plan)
        if attribute_value is not None:
            body = f"Answer: {attribute_value}\n\nDirect contract text:\n" + "\n\n".join(excerpts)
            return AssistantAnswer(body, citations, False)
        return AssistantAnswer("Direct contract text:\n" + "\n\n".join(excerpts), citations, False)

    def select_grounded_answer(
        self,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        evidence_is_weak: bool,
    ) -> AssistantAnswer | None:
        reference_answer = self.build_reference_answer(question, ranked, citations)
        if reference_answer is not None:
            return reference_answer
        grouped_answer = self.build_grouped_answer(question, ranked, citations)
        if grouped_answer is not None:
            return grouped_answer
        if evidence_is_weak:
            return None
        page_attribute_answer = self.build_page_attribute_answer(question, ranked, citations)
        if page_attribute_answer is not None:
            return page_attribute_answer
        compact_answer = self.build_compact_answer(question, ranked, citations)
        if compact_answer is not None:
            return compact_answer
        return self.build_extractive_answer(question, ranked, citations)

    @staticmethod
    def should_return_grounded_answer_before_generation(
        plan: QueryPlan,
        *,
        broad_topic_request: bool,
        summary_request: bool,
    ) -> bool:
        if broad_topic_request:
            return False
        if not summary_request:
            return True
        if plan.request_shape in {REQUEST_SHAPE_GROUPED_LIST, REQUEST_SHAPE_REFERENCE_LOOKUP, REQUEST_SHAPE_DIRECT_TEXT}:
            return True
        if plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP:
            return True
        return bool(plan.count_question or plan.direct_text_question or plan.attribute_label is not None)

    @staticmethod
    def is_count_question(question: str) -> bool:
        return question.startswith("how many ")

    @staticmethod
    def prefer_exact_answer(question: str, plan: QueryPlan | None = None) -> bool:
        if question.startswith(("how many ", "show me ", "find ", "quote ", "where does ", "which page ")):
            return True
        exact_kind = AnswerPolicy.exact_attribute_kind(question, plan=plan)
        return exact_kind is not None and question.startswith(("what is ", "what are ", "what model ", "which model "))

    @staticmethod
    def reference_lookup_kind(question: str) -> str | None:
        normalized = " ".join(question.lower().split())
        if normalized.startswith(("what section", "which section")):
            return "section"
        if normalized.startswith(("what page", "which page")):
            return "page"
        return None

    @staticmethod
    def count_exact_items(question: str, hits: list[ExactPageHit]) -> int | None:
        item_phrase = question.removeprefix("how many ").strip(" ?.")
        if item_phrase.endswith("s"):
            item_phrase = item_phrase[:-1]
        if not item_phrase:
            return None
        seen_numbers: set[str] = set()
        pattern = re.compile(rf"{re.escape(item_phrase)}\s+(\d+)", re.IGNORECASE)
        for hit in hits:
            for match in pattern.finditer(hit.page_text):
                seen_numbers.add(match.group(1))
        return len(seen_numbers) if seen_numbers else None

    @classmethod
    def extract_count_value(cls, question: str, hits: list[ExactPageHit]) -> str | None:
        item_phrase = cls.count_item_phrase(question)
        if not item_phrase:
            return None
        patterns = cls.count_value_patterns(item_phrase)
        for hit in hits:
            compact = " ".join(hit.page_text.split())
            for pattern in patterns:
                match = pattern.search(compact)
                if match:
                    return " ".join(match.group(1).split())
        count = cls.count_exact_items(question, hits)
        return str(count) if count is not None else None

    @staticmethod
    def count_item_phrase(question: str) -> str:
        item_phrase = question.removeprefix("how many ").strip(" ?.")
        item_phrase = re.sub(r"\b(do we have|do we use|are there|is there|do we need|do we require)$", "", item_phrase).strip()
        return item_phrase

    @staticmethod
    def count_value_patterns(item_phrase: str) -> list[re.Pattern[str]]:
        phrase = re.escape(item_phrase)
        singular = re.escape(item_phrase[:-1]) if item_phrase.endswith("s") and len(item_phrase) > 4 else phrase
        return [
            re.compile(rf"((?:\w+\s+)?\(?\d+\)?\s*x\s*\d+%[^.]*?{phrase})", re.IGNORECASE),
            re.compile(rf"((?:\w+\s+)?\(?\d+\)?\s*x\s*\d+%[^.]*?{singular})", re.IGNORECASE),
            re.compile(rf"((?:one|two|three|four|five|six|seven|eight|nine|ten|\(?\d+\)?)[^.]*?{phrase})", re.IGNORECASE),
            re.compile(rf"((?:one|two|three|four|five|six|seven|eight|nine|ten|\(?\d+\)?)[^.]*?{singular})", re.IGNORECASE),
            re.compile(rf"({phrase}[^.]*?\(?\d+\)?\s*x\s*\d+%)", re.IGNORECASE),
            re.compile(rf"({singular}[^.]*?\(?\d+\)?\s*x\s*\d+%)", re.IGNORECASE),
        ]

    @classmethod
    def extract_attribute_value(
        cls,
        question: str,
        hits: list[ExactPageHit],
        *,
        plan: QueryPlan | None = None,
    ) -> str | None:
        plan = plan or plan_query(question)
        kind = cls.exact_attribute_kind(question, plan=plan)
        if kind is None:
            return None
        fact_value = cls.extract_attribute_fact_value(question, hits, plan, kind)
        if fact_value is not None:
            return fact_value
        return cls.extract_attribute_evidence_value(question, hits, plan, kind)

    @classmethod
    def exact_attribute_kind(cls, question: str, *, plan: QueryPlan | None = None) -> str | None:
        normalized = " ".join(question.lower().split())
        plan = plan or plan_query(question)
        if plan.attribute_label == "quantity" or re.search(r"\bquantity\b|\bnumber of\b", normalized):
            return "quantity"
        if re.search(r"\bduty\b|\bservice\b", normalized):
            return "duty"
        if re.search(r"\bmodel\b|\bmanufacturer\b|\bvendor\b", normalized):
            return "model"
        if re.search(r"\brating\b|\bhorse power\b|\bhorsepower\b|\bhp\b|\bkw\b", normalized):
            return "rating"
        if plan.attribute_label == "configuration":
            return "configuration"
        if plan.attribute_label == "capacity":
            return "capacity"
        if plan.attribute_label == "service":
            return "duty"
        if plan.attribute_label == "type":
            return "model"
        if plan.attribute_label in {"size", "power"}:
            return "rating"
        return None

    @classmethod
    def extract_attribute_fact_value(
        cls,
        question: str,
        hits: list[ExactPageHit],
        plan: QueryPlan,
        kind: str,
    ) -> str | None:
        subject_terms = cls.attribute_subject_terms(plan, kind)
        if not subject_terms:
            return None
        for hit in hits:
            hit_lower = " ".join(hit.page_text.lower().split())
            for label, value in cls.extract_fact_row_entries(hit.page_text):
                label_matches_subject = cls.matches_attribute_row_label(label, subject_terms)
                label_matches_attribute = cls.matches_requested_attribute_label(label, plan, kind)
                if not label_matches_subject and not (
                    label_matches_attribute and cls.matches_attribute_sentence(hit_lower, subject_terms)
                ):
                    continue
                compact_value = cls.extract_value_from_text(value, plan, kind)
                if compact_value is not None:
                    return compact_value
        return None

    @classmethod
    def extract_attribute_evidence_value(
        cls,
        question: str,
        hits: list[ExactPageHit],
        plan: QueryPlan,
        kind: str,
    ) -> str | None:
        subject_terms = cls.attribute_subject_terms(plan, kind)
        item_phrase = cls.quantity_item_phrase(question) if kind == "quantity" else ""
        candidates: list[tuple[float, str]] = []
        for hit in hits:
            compact = " ".join(hit.page_text.split())
            for sentence in cls.split_sentences(compact):
                lowered = sentence.lower()
                if subject_terms and not cls.matches_attribute_sentence(lowered, subject_terms):
                    continue
                score = cls.score_attribute_sentence(sentence, plan) if plan.attribute_label else cls.score_sentence(sentence, plan)
                if kind == "quantity" and item_phrase and re.search(rf"\b{re.escape(item_phrase)}\b", lowered):
                    score += 1.5
                candidates.append((score, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, sentence in candidates:
            compact_value = cls.extract_value_from_text(sentence, plan, kind, item_phrase=item_phrase)
            if compact_value is not None:
                return compact_value
        return None

    @staticmethod
    def attribute_subject_terms(plan: QueryPlan, kind: str) -> tuple[str, ...]:
        excluded = set(plan.attribute_terms) | {"quantity", "number", "model", "rating"}
        terms: list[str] = []
        for term in plan.system_terms + plan.concept_terms + plan.focus_terms:
            cleaned = term.strip().lower()
            if not cleaned or cleaned in excluded or cleaned in GENERIC_SYSTEM_TERMS:
                continue
            terms.append(cleaned)
        return tuple(dict.fromkeys(terms))

    @classmethod
    def extract_fact_row_entries(cls, page_text: str) -> list[tuple[str, str]]:
        lines = [" ".join(line.split()).strip() for line in str(page_text).splitlines() if line and line.strip()]
        if not lines:
            return []
        entries: list[tuple[str, str]] = []
        for line in lines:
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label and value:
                entries.append((label, value))
        if entries:
            return entries
        for index in range(len(lines) - 1):
            label = lines[index]
            value = lines[index + 1]
            if cls.looks_like_fact_row_label(label) and cls.looks_like_fact_row_value(value):
                entries.append((label, value))
        return entries

    @staticmethod
    def looks_like_fact_row_label(text: str) -> bool:
        lowered = text.lower()
        if re.search(r"\d", text):
            return False
        if lowered in {"characteristic", "specification", "value", "values"}:
            return False
        return len(TOKEN_RE.findall(text)) <= 8

    @staticmethod
    def looks_like_fact_row_value(text: str) -> bool:
        lowered = text.lower()
        if re.search(r"\d", text):
            return True
        return bool(re.search(r"\b(duplex|simplex|duty|standby|lead|lag|indoor|outdoor|n/?a)\b", lowered))

    @staticmethod
    def matches_attribute_row_label(label: str, subject_terms: tuple[str, ...]) -> bool:
        lowered = label.lower()
        hits = sum(1 for term in subject_terms if term in lowered)
        required = max(1, min(len(subject_terms), 2))
        return hits >= required

    @staticmethod
    def matches_attribute_sentence(text: str, subject_terms: tuple[str, ...]) -> bool:
        hits = sum(1 for term in subject_terms if term in text)
        required = max(1, min(len(subject_terms), 2))
        return hits >= required

    @staticmethod
    def matches_requested_attribute_label(label: str, plan: QueryPlan, kind: str) -> bool:
        lowered = label.lower()
        if any(term in lowered for term in plan.attribute_terms):
            return True
        kind_terms = {
            "configuration": ("configuration", "arrangement"),
            "capacity": ("capacity", "output"),
            "duty": ("duty", "service"),
            "model": ("model", "type", "manufacturer", "vendor"),
            "rating": ("rating", "rated", "power", "horsepower", "hp", "kw", "size"),
            "quantity": ("quantity", "number", "count"),
        }.get(kind, ())
        return any(term in lowered for term in kind_terms)

    @classmethod
    def extract_value_from_text(
        cls,
        text: str,
        plan: QueryPlan,
        kind: str,
        *,
        item_phrase: str = "",
    ) -> str | None:
        compact = " ".join(text.split())
        if not compact:
            return None
        if kind == "configuration":
            value = cls.extract_first_pattern_match(compact, cls.configuration_value_patterns())
            return cls.normalize_exact_value(value)
        if kind == "capacity":
            patterns = (
                re.compile(r"\b(\d+(?:\.\d+)?\s*%\s*capacity)\b", re.IGNORECASE),
                re.compile(r"\bcapacity\s+(?:of|is|shall be|=)?\s*(\d+(?:\.\d+)?\s*(?:gpm|lb/hr|scfm|cfm|acfm|mmscfd|mw|kw|hp|tph|tpd|bpd|mbh|mmbtu/hr|kg/hr|lbm/hr|ft3/min|m3/hr|gal/min|gpd|%))\b", re.IGNORECASE),
                re.compile(r"\brated\s+at\s+(\d+(?:\.\d+)?\s*(?:gpm|lb/hr|scfm|cfm|acfm|mmscfd|mw|kw|hp|tph|tpd|bpd|mbh|mmbtu/hr|kg/hr|lbm/hr|ft3/min|m3/hr|gal/min|gpd|%))\b", re.IGNORECASE),
            )
            value = cls.extract_first_pattern_match(compact, patterns)
            return cls.normalize_exact_value(value)
        if kind == "model":
            patterns = (
                re.compile(r"\bmodel(?:\s+\w+){0,3}\s+(?:shall be|is|:)?\s*([A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*)\b"),
                re.compile(r"\bselected\s+\w+\s+model\s+shall\s+be\s+([A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*)\b", re.IGNORECASE),
                re.compile(r"\b([A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*)\b"),
            )
            value = cls.extract_first_pattern_match(compact, patterns)
            return cls.normalize_exact_value(value)
        if kind == "rating":
            patterns = (
                re.compile(r"\brating(?:\s+of|\s+is|:)?\s*(\d+(?:\.\d+)?\s*(?:kv|v|volt(?:s)?|a|amp(?:s)?|ka|kva|mva|mw|kw|hp|psi|psig|psia|inch(?:es)?|mm|ft|%))\b", re.IGNORECASE),
                re.compile(r"\brated\s+at\s+(\d+(?:\.\d+)?\s*(?:kv|v|volt(?:s)?|a|amp(?:s)?|ka|kva|mva|mw|kw|hp|psi|psig|psia|inch(?:es)?|mm|ft|%))\b", re.IGNORECASE),
            )
            value = cls.extract_first_pattern_match(compact, patterns)
            return cls.normalize_exact_value(value)
        if kind == "duty":
            patterns = (
                re.compile(r"\b(?:duty|service)(?:\s+of|\s+is|:)?\s*((?:standby|continuous|intermittent|lead/lag|lead-lag|lead lag|duplex|simplex|primary|backup|spare|fire water|cooling water|startup|shutdown)[A-Za-z0-9 /&()%-]*)\b", re.IGNORECASE),
                re.compile(r"\b((?:standby|continuous|intermittent|lead/lag|lead-lag|lead lag|duplex|simplex|primary|backup|spare)\s+(?:duty|service))\b", re.IGNORECASE),
            )
            value = cls.extract_first_pattern_match(compact, patterns)
            if value is None and re.fullmatch(
                r"(?:standby|continuous|intermittent|lead/lag|lead-lag|lead lag|duplex|simplex|primary|backup|spare|fire water|cooling water|startup|shutdown)(?:\s+[A-Za-z0-9/&()%-]+){0,3}",
                compact,
                re.IGNORECASE,
            ):
                value = compact
            return cls.normalize_exact_value(value)
        if kind == "quantity":
            count_token = r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s*\(\d+\))?|\(?\d+\)?)"
            if item_phrase:
                patterns = cls.quantity_value_patterns(item_phrase)
                value = cls.extract_first_pattern_match(compact, patterns)
                return cls.normalize_exact_value(value)
            value = cls.extract_first_pattern_match(
                compact,
                (
                    re.compile(
                        rf"\b({count_token}(?:\s*[xX]\s*\d+%\s*(?:capacity)?|\s+\d+%\s+capacity|\s+full\s+capacity)?)\b",
                        re.IGNORECASE,
                    ),
                ),
            )
            return cls.normalize_exact_value(value)
        compact_value = cls.extract_compact_attribute_value(compact, plan)
        return cls.normalize_exact_value(compact_value or None)

    @staticmethod
    def extract_first_pattern_match(text: str, patterns: Sequence[re.Pattern[str]]) -> str | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        return None

    @staticmethod
    def normalize_exact_value(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split()).strip(" ,;:.")
        if not cleaned:
            return None
        cleaned = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " x ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def citations_from_exact_hits(cls, question: str, hits: list[ExactPageHit], *, limit: int = 4) -> list[Citation]:
        lowered = " ".join(question.lower().split())
        citations: list[Citation] = []
        seen_pages: set[int] = set()
        for hit in hits:
            if hit.page_num in seen_pages:
                continue
            seen_pages.add(hit.page_num)
            heading = " ".join(str(hit.snippet).split()) or f"Page {hit.page_num}"
            excerpt = cls.trim_page_excerpt(hit.page_text, lowered, limit=SUMMARY_EXCERPT_LIMIT).strip().strip('"')
            citations.append(
                Citation(
                    chunk_id=f"exact_page_{hit.page_num}",
                    section_number=None,
                    heading=heading,
                    attachment=None,
                    page_start=hit.page_num,
                    page_end=hit.page_num,
                    quote=excerpt,
                )
            )
            if len(citations) >= limit:
                break
        return citations

    @staticmethod
    def quantity_item_phrase(question: str) -> str:
        normalized = " ".join(question.lower().split()).strip(" ?.")
        match = re.search(r"\b(?:quantity|number)\s+of\s+(.+)$", normalized)
        if match:
            phrase = match.group(1).strip()
            return re.sub(r"^(?:the|a|an)\s+", "", phrase)
        return ""

    @staticmethod
    def quantity_value_patterns(item_phrase: str) -> list[re.Pattern[str]]:
        phrase = re.escape(item_phrase)
        singular = re.escape(item_phrase[:-1]) if item_phrase.endswith("s") and len(item_phrase) > 4 else phrase
        count_token = r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s*\(\d+\))?|\(?\d+\)?)"
        return [
            re.compile(
                rf"\b({count_token}(?:\s*[xX]\s*\d+%\s*(?:capacity)?|\s+\d+%\s+capacity|\s+full\s+capacity)?)\s+[^.]*?{phrase}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b({count_token}(?:\s*[xX]\s*\d+%\s*(?:capacity)?|\s+\d+%\s+capacity|\s+full\s+capacity)?)\s+[^.]*?{singular}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{phrase}[^.]*?\b({count_token}(?:\s*[xX]\s*\d+%\s*(?:capacity)?|\s+\d+%\s+capacity|\s+full\s+capacity)?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{singular}[^.]*?\b({count_token}(?:\s*[xX]\s*\d+%\s*(?:capacity)?|\s+\d+%\s+capacity|\s+full\s+capacity)?)\b",
                re.IGNORECASE,
            ),
        ]

    @staticmethod
    def trim_page_excerpt(page_text: str, question: str, limit: int = 700) -> str:
        compact = " ".join(page_text.split())
        term = question.removeprefix("how many ").strip(" ?.")
        if term.endswith("s"):
            term = term[:-1]
        needle = term.lower()
        idx = compact.lower().find(needle)
        if idx < 0:
            idx = 0
        start = max(0, idx - 120)
        excerpt = compact[start:start + limit]
        return excerpt if len(excerpt) < len(compact) else compact[:limit]

    @classmethod
    def build_reference_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        lookup_kind = cls.reference_lookup_kind(question)
        if lookup_kind is None or not ranked:
            return None
        plan = plan_query(question)
        for chunk in ranked:
            if "...." in chunk.heading:
                continue
            excerpt = cls.extract_ranked_excerpt(chunk.full_text, plan, limit=520)
            if not excerpt:
                continue
            label = chunk.section_number or "Unnumbered clause"
            pages = cls.format_page_range(chunk.page_start, chunk.page_end)
            heading = " ".join(chunk.heading.split())
            intro = "Most relevant section:" if lookup_kind == "section" else "Most relevant pages:"
            body = f"Section {label} - {heading}\nPages: {pages}\nContract text: {excerpt}"
            return AssistantAnswer(f"{intro}\n\n{body}", citations, False)
        return None

    @staticmethod
    def limit_citations(citations: list[Citation], limit: int = 5) -> list[Citation]:
        limited: list[Citation] = []
        seen_pages: set[int] = set()
        for citation in citations:
            pages = range(citation.page_start, citation.page_end + 1)
            if all(page in seen_pages for page in pages):
                continue
            limited.append(citation)
            seen_pages.update(pages)
            if len(limited) >= limit:
                break
        return limited

    @classmethod
    def build_extractive_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        max_sections: int | None = None,
    ) -> AssistantAnswer | None:
        if not ranked:
            return None
        plan = plan_query(question)
        active_max_sections = cls.default_extractive_sections(plan) if max_sections is None else max_sections
        blocks: list[str] = []
        seen_keys: set[tuple[str | None, int, int]] = set()
        excerpt_limit = 900 if cls.is_value_or_requirement_question(plan) else 650
        for chunk in ranked:
            key = (chunk.section_number, chunk.page_start, chunk.page_end)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            excerpt = cls.extract_ranked_excerpt(chunk.full_text, plan, limit=excerpt_limit)
            if not excerpt or not cls.is_useful_extractive_block(chunk.heading, excerpt, plan):
                continue
            label = chunk.section_number or "Unnumbered clause"
            pages = cls.format_page_range(chunk.page_start, chunk.page_end)
            heading = " ".join(chunk.heading.split())
            blocks.append(f"Section {label} - {heading}\nPages: {pages}\nContract text: {excerpt}")
            if len(blocks) >= active_max_sections:
                break
        if not blocks:
            return None
        return AssistantAnswer("\n\n".join(blocks), citations, False)

    @classmethod
    def build_broad_topic_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        max_sections: int = 4,
    ) -> AssistantAnswer | None:
        if not ranked:
            return None
        plan = plan_query(question)
        if plan.request_shape != REQUEST_SHAPE_BROAD_TOPIC:
            return None
        blocks: list[str] = []
        seen_keys: set[tuple[str | None, int, int]] = set()
        for chunk in ranked:
            key = (chunk.section_number, chunk.page_start, chunk.page_end)
            if key in seen_keys:
                continue
            excerpt = cls.extract_ranked_excerpt(
                chunk.full_text,
                plan,
                limit=SUMMARY_EXCERPT_LIMIT,
                surrounding_sentences=1,
            )
            if not excerpt or not cls.is_useful_summary_block(chunk, excerpt, plan):
                continue
            seen_keys.add(key)
            heading = " ".join(chunk.heading.split())
            cleaned_excerpt = excerpt.strip().strip('"').strip()
            if heading and heading.lower() not in cleaned_excerpt.lower():
                blocks.append(f"- {heading}: {cleaned_excerpt}")
            else:
                blocks.append(f"- {cleaned_excerpt}")
            if len(blocks) >= max_sections:
                break
        if not blocks:
            return None
        return AssistantAnswer(
            f"Here are the strongest contract-supported points about {plan.content_query}:\n" + "\n".join(blocks),
            citations,
            False,
        )

    @staticmethod
    def default_extractive_sections(plan: QueryPlan) -> int:
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            return 2
        if (
            plan.direct_text_question
            or plan.count_question
            or plan.system_phrase
            or plan.attribute_label is not None
            or AnswerPolicy.is_value_or_requirement_question(plan)
        ):
            return 1
        return 2

    @classmethod
    def build_grouped_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        if not ranked:
            return None
        plan = plan_query(question)
        if plan.request_shape != REQUEST_SHAPE_GROUPED_LIST:
            return None
        lines = cls.extract_grouped_lines(plan, ranked, citations)
        if not lines:
            return None
        primary = ranked[0]
        location_citation = next(
            (
                citation
                for citation in citations
                if citation.attachment and re.match(r"^(appendix|attachment|exhibit)\b", citation.heading, re.IGNORECASE)
            ),
            next((citation for citation in citations if citation.attachment), citations[0] if citations else None),
        )
        if location_citation is not None:
            label = location_citation.section_number or "Unnumbered clause"
            heading = " ".join(location_citation.heading.split())
            pages = cls.format_page_range(location_citation.page_start, location_citation.page_end)
        else:
            label = primary.section_number or "Unnumbered clause"
            heading = " ".join(primary.heading.split())
            pages = cls.format_page_range(primary.page_start, primary.page_end)
        line_block = "\n".join(f"- {line}" for line in lines)
        body = f"{line_block}\nSection {label} - {heading}\nPages: {pages}"
        return AssistantAnswer(body, citations, False)

    @classmethod
    def build_compact_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        if not ranked:
            return None
        plan = plan_query(question)
        if not cls.prefers_compact_answer(plan):
            return None
        for chunk in ranked:
            if not cls.is_strong_question_match(plan, chunk):
                continue
            answer_line = cls.extract_attribute_answer(chunk.full_text, plan)
            if not answer_line:
                continue
            label = chunk.section_number or "Unnumbered clause"
            heading = " ".join(chunk.heading.split())
            pages = cls.format_page_range(chunk.page_start, chunk.page_end)
            body = f"{answer_line}\nSection {label} - {heading}\nPages: {pages}"
            return AssistantAnswer(body, citations, False)
        return None

    @staticmethod
    def prefers_generated_answer(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        wrapped = f" {normalized} "
        return normalized.startswith(("summarize ", "summarise ", "explain ")) or any(
            phrase in wrapped for phrase in (" summarize ", " summarise ", " explain ")
        )

    @staticmethod
    def prefers_compact_answer(plan: QueryPlan) -> bool:
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            return False
        return bool(
            plan.attribute_label
            or plan.intent in {"definition", "responsibility"}
            or (plan.system_phrase and plan.attribute_terms)
        )

    @classmethod
    def build_summary_prompt_context(
        cls,
        question: str,
        ranked: list[RankedChunk],
        *,
        max_sections: int = SUMMARY_CONTEXT_MAX_SECTIONS,
    ) -> str:
        if not ranked:
            return ""
        plan = plan_query(question)
        blocks: list[str] = []
        seen: set[tuple[str | None, int, int]] = set()
        for chunk in ranked:
            key = (chunk.section_number, chunk.page_start, chunk.page_end)
            if key in seen:
                continue
            seen.add(key)
            excerpt = cls.extract_ranked_excerpt(
                chunk.full_text,
                plan,
                limit=SUMMARY_EXCERPT_LIMIT,
                surrounding_sentences=1,
            )
            if not excerpt or not cls.is_useful_summary_block(chunk, excerpt, plan):
                continue
            label = html.escape(chunk.section_number or "Unnumbered clause")
            heading = html.escape(" ".join(chunk.heading.split()))
            pages = html.escape(cls.format_page_range(chunk.page_start, chunk.page_end))
            blocks.append(f"Section: {label}\nHeading: {heading}\nPages: {pages}\nExcerpt: {excerpt}")
            if len(blocks) >= max_sections:
                break
        return "\n\n".join(blocks)

    @classmethod
    def build_expand_prompt_context(
        cls,
        question: str,
        ranked: list[RankedChunk],
        previous_answer: str | None,
        *,
        max_sections: int = SUMMARY_CONTEXT_MAX_SECTIONS,
    ) -> str:
        evidence = cls.build_summary_prompt_context(question, ranked, max_sections=max_sections)
        sections: list[str] = []
        prior = str(previous_answer or "").strip()
        if prior:
            sections.append("Previous answer shown to the user:\n" + prior)
        if evidence:
            sections.append("Contract evidence for the expanded answer:\n" + evidence)
        return "\n\n".join(section for section in sections if section).strip()

    def build_deep_prompt_context(
        self,
        question: str,
        ranked: list[RankedChunk],
        exact_hits: list[ExactPageHit],
    ) -> str:
        sections: list[str] = []
        ranked_blocks = self.build_summary_prompt_context(question, ranked, max_sections=DEEP_CONTEXT_MAX_SECTIONS)
        if ranked_blocks:
            sections.append("Ranked contract sections:\n" + ranked_blocks)

        exact_blocks = self.build_exact_hit_prompt_context(question, exact_hits)
        if exact_blocks:
            sections.append("Exact page hits:\n" + exact_blocks)

        page_context = self.build_deep_page_context(ranked)
        if page_context:
            sections.append("Nearby page context:\n" + page_context)

        return "\n\n".join(section for section in sections if section).strip()

    @classmethod
    def build_exact_hit_prompt_context(cls, question: str, exact_hits: list[ExactPageHit]) -> str:
        lowered = " ".join(question.lower().split())
        blocks = [f"Page {hit.page_num}: {cls.quote_excerpt(cls.trim_page_excerpt(hit.page_text, lowered))}" for hit in exact_hits[:DEEP_CONTEXT_EXACT_HITS]]
        return "\n\n".join(blocks)

    def build_deep_page_context(self, ranked: list[RankedChunk]) -> str:
        document_id = self.retriever.resolve_document_id()
        if not self.store or not document_id or not ranked:
            return ""
        seen_pages: set[tuple[int, int]] = set()
        blocks: list[str] = []
        for chunk in ranked:
            page_key = (chunk.page_start, chunk.page_end)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            label = chunk.section_number or "Unnumbered clause"
            heading = " ".join(chunk.heading.split())
            rows = self.store.fetch_page_window(document_id, chunk.page_start, chunk.page_end, padding=0, limit=2)
            if not rows:
                continue
            page_lines = [
                f"Page {int(row['page_num'])}: {self.quote_excerpt(self.truncate_text(str(row['page_text']), limit=500))}"
                for row in rows
            ]
            blocks.append(f"Section {label} - {heading}\n" + "\n".join(page_lines))
            if len(blocks) >= DEEP_PAGE_CONTEXT_SECTIONS:
                break
        return "\n\n".join(blocks)

    @staticmethod
    def format_page_range(page_start: int, page_end: int) -> str:
        return f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"

    @staticmethod
    def truncate_text(text: str, limit: int = 340) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[: limit - 3] + "..."

    @classmethod
    def extract_design_conditions_table_entries(
        cls,
        page_text: str,
        *,
        section_number: str | None = None,
        heading: str = "",
    ) -> list[tuple[str, str]]:
        lines = [" ".join(line.split()) for line in str(page_text).splitlines() if line and line.strip()]
        if not lines:
            return []
        if section_number:
            normalized_section = section_number.strip().strip(".")
            if lines and lines[0].strip().strip(".") == normalized_section:
                lines = lines[1:]
        normalized_heading = " ".join(heading.split()).lower()
        if normalized_heading and lines and lines[0].lower() == normalized_heading:
            lines = lines[1:]
        if len(lines) >= 2 and lines[0].lower() == "characteristic" and lines[1].lower() == "specification":
            lines = lines[2:]
        entries: list[tuple[str, str]] = []
        index = 0
        while index + 1 < len(lines):
            label = lines[index]
            value = lines[index + 1]
            if cls.is_design_conditions_header(label):
                index += 1
                continue
            if cls.is_design_conditions_bullet(value):
                break
            if cls.looks_like_design_conditions_value(value):
                entries.append((label, value))
                index += 2
                continue
            index += 1
        return entries

    @staticmethod
    def is_design_conditions_header(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return normalized in {"characteristic", "specification"}

    @staticmethod
    def is_design_conditions_bullet(text: str) -> bool:
        normalized = text.strip()
        return normalized in {"•", "\uf0b7"} or normalized.startswith(("• ", "\uf0b7 "))

    @classmethod
    def looks_like_design_conditions_value(cls, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if not normalized or cls.is_design_conditions_header(normalized) or cls.is_design_conditions_bullet(text):
            return False
        if normalized.endswith("conditions"):
            return False
        if re.search(r"\d", text):
            return True
        if normalized in {"yes", "no", "indoor", "outdoor", "inside", "outside", "n/a", "na"}:
            return True
        return len(TOKEN_RE.findall(text)) <= 4

    @classmethod
    def extract_ranked_excerpt(
        cls,
        text: str,
        plan: QueryPlan,
        *,
        limit: int = 650,
        surrounding_sentences: int = 0,
    ) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        attribute_excerpt = cls.extract_attribute_excerpt(compact, plan, limit=limit)
        if attribute_excerpt:
            return cls.quote_excerpt(attribute_excerpt)
        term_window = cls.extract_term_window(compact, plan, limit=limit)
        if term_window:
            return cls.quote_excerpt(term_window)
        sentences = cls.split_sentences(compact)
        if not sentences:
            return cls.quote_excerpt(compact[:limit].rstrip())
        scored = [(cls.score_sentence(sentence, plan), index) for index, sentence in enumerate(sentences)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_index = scored[0][1]
        selected_indices = {best_index}
        if cls.is_value_or_requirement_question(plan):
            if best_index + 1 < len(sentences):
                selected_indices.add(best_index + 1)
            elif best_index > 0:
                selected_indices.add(best_index - 1)
        for offset in range(1, surrounding_sentences + 1):
            if best_index - offset >= 0:
                selected_indices.add(best_index - offset)
            if best_index + offset < len(sentences):
                selected_indices.add(best_index + offset)
        selected = [sentences[index] for index in sorted(selected_indices)]
        excerpt = " ".join(selected).strip()
        if len(excerpt) < min(180, limit) and len(sentences) > 1:
            if best_index > 0:
                excerpt = f"{sentences[best_index - 1]} {excerpt}".strip()
            if len(excerpt) < limit and best_index + 1 < len(sentences):
                excerpt = f"{excerpt} {sentences[best_index + 1]}".strip()
        if len(excerpt) > limit:
            excerpt = excerpt[: limit - 3].rstrip() + "..."
        return cls.quote_excerpt(excerpt)

    @classmethod
    def extract_attribute_excerpt(cls, text: str, plan: QueryPlan, *, limit: int) -> str:
        if not plan.attribute_label and not plan.system_phrase:
            return ""
        sentences = cls.split_sentences(text)
        if not sentences:
            return ""
        scored = [(cls.score_attribute_sentence(sentence, plan), index) for index, sentence in enumerate(sentences)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_score, best_index = scored[0]
        if best_score < 1.4:
            return ""
        selected_indices = {best_index}
        label = plan.attribute_label or ""
        if label in {"design_conditions", "function"}:
            selected_indices.update(cls.expand_related_sentences(sentences, best_index, plan, limit=2))
        elif label in {"configuration", "type", "size", "capacity", "pressure", "temperature", "flow"}:
            if best_index + 1 < len(sentences) and cls.score_attribute_sentence(sentences[best_index + 1], plan) >= 1.0:
                selected_indices.add(best_index + 1)
        elif not label and len(sentences[best_index].strip()) < 50:
            selected_indices.update(cls.expand_related_sentences(sentences, best_index, plan, limit=2))
        excerpt = " ".join(sentences[index] for index in sorted(selected_indices)).strip()
        if len(excerpt) > limit:
            excerpt = excerpt[: limit - 3].rstrip() + "..."
        return excerpt

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [piece.strip() for piece in re.split(r"(?<=[.!?;:])\s+|\s{2,}", text) if piece.strip()]

    @classmethod
    def extract_attribute_answer(cls, text: str, plan: QueryPlan, *, limit: int = 320) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        sentences = cls.split_sentences(compact)
        if not sentences:
            return ""
        scored = [(cls.score_attribute_sentence(sentence, plan), index) for index, sentence in enumerate(sentences)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_score, best_index = scored[0]
        if best_score <= 0.0:
            fallback = cls.extract_ranked_excerpt(compact, plan, limit=limit)
            return fallback.strip().strip('"')
        selected = [sentences[best_index].strip()]
        if plan.attribute_label == "design_conditions":
            for direction in (1, -1):
                neighbor_index = best_index + direction
                if 0 <= neighbor_index < len(sentences):
                    neighbor = sentences[neighbor_index].strip()
                    if cls.score_attribute_sentence(neighbor, plan) >= 1.25 and len(" ".join(selected + [neighbor])) <= limit:
                        if direction < 0:
                            selected.insert(0, neighbor)
                        else:
                            selected.append(neighbor)
        excerpt = " ".join(selected).strip()
        compact_value = cls.extract_compact_attribute_value(excerpt, plan)
        if compact_value:
            return cls.quote_excerpt(compact_value)
        if len(excerpt) > limit:
            excerpt = excerpt[: limit - 3].rstrip() + "..."
        return cls.quote_excerpt(excerpt)

    @classmethod
    def extract_compact_attribute_value(cls, text: str, plan: QueryPlan) -> str:
        label = plan.attribute_label or ""
        if label == "configuration":
            for pattern in cls.configuration_value_patterns():
                match = pattern.search(text)
                if match:
                    return " ".join(match.group(0).split())
        if label == "service":
            match = re.search(
                r"\b((?:standby|continuous|intermittent|lead/lag|lead-lag|lead lag|duplex|simplex|primary|backup|spare)\s+(?:duty|service))\b",
                text,
                re.IGNORECASE,
            )
            if match:
                return " ".join(match.group(1).split())
        return ""

    @staticmethod
    def configuration_value_patterns() -> tuple[re.Pattern[str], ...]:
        return (
            re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+\(?\d+\)?\s*[xX]\s*\d+%", re.IGNORECASE),
            re.compile(r"\b\(?\d+\)?\s*[xX]\s*\d+%", re.IGNORECASE),
            re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+\d+%\s+capacity\b", re.IGNORECASE),
            re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+full\s+capacity\b", re.IGNORECASE),
            re.compile(r"\b(?:duty|standby|lead|lag)(?:\s*[/,-]?\s*(?:duty|standby|lead|lag))+\b", re.IGNORECASE),
        )

    @classmethod
    def extract_grouped_lines(
        cls,
        plan: QueryPlan,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        limit: int = 6,
    ) -> list[str]:
        texts = [chunk.full_text for chunk in ranked[:4]]
        texts.extend(citation.quote for citation in citations[:4] if citation.quote)
        lines: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for sentence in cls.split_sentences(" ".join(str(text).split())):
                candidate = cls.normalize_grouped_line(sentence)
                if not candidate or candidate.lower() in seen:
                    continue
                if not cls.is_grouped_line_match(candidate, plan):
                    continue
                seen.add(candidate.lower())
                lines.append(candidate)
                if len(lines) >= limit:
                    return lines
        return lines

    @staticmethod
    def normalize_grouped_line(sentence: str) -> str:
        cleaned = " ".join(sentence.split()).strip().strip('"')
        return cleaned

    @classmethod
    def is_grouped_line_match(cls, sentence: str, plan: QueryPlan) -> bool:
        lowered = sentence.lower()
        if plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
            has_family_marker = bool(
                re.search(r"\bguarantee\b|\bguarantees\b|\bshall not exceed\b|\bnot exceed\b|\blimit\b|\blimits\b", lowered)
            )
            has_value = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:ppmvd|ppmv|lb/hr|mg/nm3|%)\b", lowered))
            concept_hits = sum(1 for term in plan.concept_terms if term and term in lowered)
            return (has_family_marker or has_value) and concept_hits >= 1
        return False

    @classmethod
    def score_sentence(cls, sentence: str, plan: QueryPlan) -> float:
        normalized = sentence.lower()
        score = 0.0
        if plan.content_query and plan.content_query in normalized:
            score += 4.0
        if plan.system_phrase and plan.system_phrase in normalized:
            score += 2.2
        score += 1.2 * sum(1 for term in plan.concept_terms if term in normalized)
        score += 0.8 * sum(1 for term in plan.scope_terms if term in normalized)
        score += 1.0 * sum(1 for term in plan.system_terms if term in normalized)
        score += 1.0 * sum(1 for term in plan.attribute_terms if term in normalized)
        score += 1.5 * sum(1 for term in plan.focus_terms if term in normalized)
        score += 0.8 * sum(1 for term in plan.topic_terms + plan.action_terms if term in normalized)
        score += 0.5 * sum(1 for term in plan.actor_terms if term in normalized)
        if plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT and re.search(
            r"\bguarantee\b|\bguarantees\b|\bshall not exceed\b|\bnot exceed\b|\blimit\b",
            normalized,
        ):
            score += 1.6
        if cls.is_value_or_requirement_question(plan) and re.search(r"\d", sentence):
            score += 1.2
        if cls.is_requirement_question(plan) and re.search(r"\b(shall|must|required|responsible|obligated)\b", normalized):
            score += 1.1
        if sentence.count("....") >= 2:
            score -= 2.0
        return score

    @classmethod
    def score_attribute_sentence(cls, sentence: str, plan: QueryPlan) -> float:
        normalized = sentence.lower()
        score = cls.score_sentence(sentence, plan)
        label = plan.attribute_label or ""
        if label == "design_conditions":
            if re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", normalized):
                score += 2.4
            score += min(2.0, cls.numeric_value_count(sentence) * 0.4)
            if cls.is_generic_attribute_question(plan) and cls.has_attribute_guidance_marker(normalized):
                score += 2.2
        elif label == "configuration":
            if cls.matches_configuration_text(normalized):
                score += 2.1
        elif label == "type":
            if re.search(r"\bmodel\b|\btype\b|\bselected\b|\bvendor\b|\bmanufacturer\b|\buse\b|\busing\b", normalized):
                score += 2.2
            if re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*\b", sentence):
                score += 2.0
        elif label == "size":
            if re.search(r"\bsize\b|\bdiameter\b|\brating\b|\b(?:inch|inches|mm|ft)\b", normalized):
                score += 2.0
            score += min(1.6, cls.numeric_value_count(sentence) * 0.35)
        elif label == "power":
            score += min(1.8, cls.numeric_value_count(sentence) * 0.35)
            if re.search(r"\bhorse\s*power\b|\bhp\b|\bkw\b|\bkilowatt", normalized):
                score += 1.8
        elif label in {"capacity", "pressure", "temperature", "flow"}:
            score += min(1.8, cls.numeric_value_count(sentence) * 0.35)
            if any(term in normalized for term in plan.attribute_terms):
                score += 1.6
            if label in {"pressure", "temperature"} and cls.is_generic_attribute_question(plan) and cls.has_attribute_guidance_marker(normalized):
                score += 1.9
                if cls.numeric_value_count(sentence) >= 3:
                    score -= 0.55
        elif label == "service":
            if cls.matches_duty_text(normalized):
                score += 1.9
        elif label == "responsibility":
            if re.search(r"\b(shall|must|required|responsible|provide|furnish|supply|perform|deliver)\b", normalized):
                score += 2.0
        elif label == "definition" or plan.intent == "definition":
            if re.search(r"\bmeans\b|\bdefined as\b|\bdefinition\b", normalized):
                score += 2.2
        elif label == "function":
            if re.search(r"\b(receives|supplies|provides|distributes|conditions|transports|serves|used to|operates)\b", normalized):
                score += 2.0
        return score

    @classmethod
    def expand_related_sentences(
        cls,
        sentences: list[str],
        anchor_index: int,
        plan: QueryPlan,
        *,
        limit: int = 2,
    ) -> set[int]:
        selected: set[int] = set()
        for offset in range(1, limit + 1):
            next_index = anchor_index + offset
            if next_index < len(sentences) and cls.score_attribute_sentence(sentences[next_index], plan) >= 1.0:
                selected.add(next_index)
            prev_index = anchor_index - offset
            if prev_index >= 0 and cls.score_attribute_sentence(sentences[prev_index], plan) >= 1.0:
                selected.add(prev_index)
        return selected

    @staticmethod
    def numeric_value_count(text: str) -> int:
        pattern = r"\b\d+(?:\.\d+)?\s*(?:psi|psig|psia|degf|deg c|gpm|lb/hr|scfm|mw|mva|hp|%|inch|inches|mm|ft)?\b"
        return len(re.findall(pattern, text, re.IGNORECASE))

    @staticmethod
    def is_value_or_requirement_question(plan: QueryPlan) -> bool:
        return AnswerPolicy.is_value_question(plan) or AnswerPolicy.is_requirement_question(plan)

    @staticmethod
    def is_value_question(plan: QueryPlan) -> bool:
        normalized = plan.normalized_query
        if plan.count_question:
            return True
        if plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
            return True
        if plan.attribute_label in {"design_conditions", "size", "capacity", "power", "pressure", "temperature", "flow", "service"}:
            return True
        markers = (
            "how much",
            "amount",
            "price",
            "cost",
            "payment",
            "payments",
            "damages",
            "rate",
            "percent",
            "percentage",
            "days",
            "hours",
            "deadline",
            "value",
            "values",
            "mw",
            "kw",
            "psi",
            "voltage",
            "temperature",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def is_requirement_question(plan: QueryPlan) -> bool:
        normalized = plan.normalized_query
        return (
            plan.intent == "responsibility"
            or plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT
            or "required" in normalized
            or "requirement" in normalized
            or "requirements" in normalized
            or "shall" in normalized
            or "must" in normalized
            or "responsible" in normalized
            or "obligation" in normalized
        )

    @staticmethod
    def quote_excerpt(text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        return f'"{cleaned}"' if cleaned else ""

    @classmethod
    def requires_strict_grounding(cls, plan: QueryPlan) -> bool:
        return bool(
            plan.system_phrase
            or plan.concept_terms
            or plan.attribute_label
            or plan.count_question
            or plan.direct_text_question
            or cls.is_value_or_requirement_question(plan)
            or plan.intent in {"definition", "responsibility"}
        )

    @classmethod
    def has_strong_grouped_evidence(
        cls,
        plan: QueryPlan,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> bool:
        if plan.request_shape != REQUEST_SHAPE_GROUPED_LIST:
            return False
        if any(cls.is_strong_question_match(plan, chunk) for chunk in ranked[:3]):
            return True
        lines = cls.extract_grouped_lines(plan, ranked[:3], citations[:3])
        return len(lines) >= 1

    @classmethod
    def has_strong_ranked_evidence(cls, plan: QueryPlan, ranked: list[RankedChunk]) -> bool:
        if plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC:
            return cls.has_strong_broad_topic_evidence(plan, ranked)
        if (
            plan.request_shape == "scalar"
            and plan.attribute_label is None
            and not plan.count_question
            and not plan.direct_text_question
            and cls.is_requirement_question(plan)
        ):
            return max((cls.score_requirement_candidate(chunk, plan) for chunk in ranked[:5]), default=0.0) >= 7.0
        return any(cls.is_strong_question_match(plan, chunk) for chunk in ranked[:3])

    @classmethod
    def has_strong_broad_topic_evidence(cls, plan: QueryPlan, ranked: list[RankedChunk]) -> bool:
        useful_blocks = 0
        for chunk in ranked[:5]:
            excerpt = cls.extract_ranked_excerpt(
                chunk.full_text,
                plan,
                limit=SUMMARY_EXCERPT_LIMIT,
                surrounding_sentences=1,
            )
            if excerpt and cls.is_useful_summary_block(chunk, excerpt, plan):
                useful_blocks += 1
            if useful_blocks >= 1:
                return True
        return False

    @classmethod
    def is_strong_question_match(cls, plan: QueryPlan, chunk: RankedChunk) -> bool:
        minimum_score = 0.42
        if plan.attribute_label or plan.count_question or cls.is_value_or_requirement_question(plan):
            minimum_score = 0.56
        elif plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            minimum_score = 0.5
        elif plan.direct_text_question or plan.system_phrase:
            minimum_score = 0.48
        if chunk.total_score < minimum_score and cls.requires_strict_grounding(plan):
            return False
        combined = " ".join([chunk.heading, chunk.full_text])
        focus_hits = sum(1 for term in plan.focus_terms if has_term_overlap(combined, (term,)))
        concept_hits = sum(1 for term in plan.concept_terms if has_term_overlap(combined, (term,)))
        if plan.direct_text_question and len(plan.focus_terms) >= 2 and focus_hits < min(2, len(plan.focus_terms)):
            return False
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST and concept_hits < max(1, min(len(plan.concept_terms), 2)):
            return False
        if plan.system_phrase and not cls.matches_system(plan, combined):
            return False
        if (plan.attribute_label or plan.count_question) and not cls.matches_attribute(plan, combined):
            return False
        if plan.scope_terms and not cls.matches_scope(plan, combined):
            return False
        if plan.intent == "definition" and not re.search(r"\bmeans\b|\bdefined as\b|\bdefinition\b", combined, re.IGNORECASE):
            return False
        if plan.intent == "responsibility" and not re.search(
            r"\b(shall|must|responsible|required|provide|furnish|supply|perform|deliver|submit)\b",
            combined,
            re.IGNORECASE,
        ):
            return False
        return True

    @staticmethod
    def matches_system(plan: QueryPlan, text: str) -> bool:
        if not plan.system_phrase and not plan.system_terms:
            return True
        if plan.system_phrase and has_term_overlap(text, (plan.system_phrase,)):
            return True
        significant_terms = tuple(term for term in plan.system_terms if term not in GENERIC_SYSTEM_TERMS)
        if significant_terms:
            significant_hits = sum(1 for term in significant_terms if has_term_overlap(text, (term,)))
            return significant_hits >= len(significant_terms)
        elif any(has_term_overlap(text, (alias,)) for alias in plan.system_aliases):
            return True
        if plan.system_terms:
            hits = sum(1 for term in plan.system_terms if has_term_overlap(text, (term,)))
            return hits >= max(1, min(len(plan.system_terms), 2))
        return False

    @staticmethod
    def matches_scope(plan: QueryPlan, text: str) -> bool:
        if not plan.scope_terms:
            return True
        specific_scope_terms = AnswerPolicy.specific_scope_terms(plan.scope_terms)
        if specific_scope_terms:
            return any(has_term_overlap(text, (term,)) for term in specific_scope_terms)
        return any(has_term_overlap(text, (term,)) for term in plan.scope_terms)

    @staticmethod
    def specific_scope_terms(scope_terms: tuple[str, ...]) -> tuple[str, ...]:
        generic = {"appendix", "appendices", "attachment", "attachments", "exhibit", "exhibits"}
        return tuple(term for term in scope_terms if term.strip().lower() and term.strip().lower() not in generic)

    @staticmethod
    def matches_attribute(plan: QueryPlan, text: str) -> bool:
        if not plan.attribute_label and not plan.attribute_terms:
            return not plan.count_question
        lowered = text.lower()
        if plan.count_question:
            return bool(re.search(r"\b\d", lowered) or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", lowered))
        if any(has_term_overlap(text, (term,)) for term in plan.attribute_terms):
            return True
        label = plan.attribute_label or ""
        if label == "design_conditions":
            return bool(re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", lowered))
        if label == "configuration":
            return AnswerPolicy.matches_configuration_text(lowered)
        if label == "type":
            return bool(re.search(r"\bmodel\b|\btype\b|\bselected\b|\bmanufacturer\b|\bvendor\b", lowered) or re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*\b", text))
        if label == "size":
            return bool(re.search(r"\bsize\b|\bdiameter\b|\brating\b|\bdimension\b", lowered))
        if label == "power":
            return bool(re.search(r"\bhorse\s*power\b|\bhp\b|\bkw\b|\bkilowatt", lowered))
        if label == "capacity":
            return bool(re.search(r"\bcapacity\b|\boutput\b|\bduty\b|\bthroughput\b", lowered))
        if label == "service":
            return AnswerPolicy.matches_duty_text(lowered)
        if label == "pressure":
            return bool(re.search(r"\bpressure\b|\bpsig\b|\bpsi\b", lowered))
        if label == "temperature":
            return bool(re.search(r"\btemperature\b|\bdegf\b|\bdeg c\b", lowered))
        if label == "flow":
            return bool(re.search(r"\bflow\b|\bflow rate\b|\bflowrate\b|\bthroughput\b", lowered))
        if label == "responsibility":
            return bool(re.search(r"\b(shall|must|responsible|required|provide|furnish|supply)\b", lowered))
        if label == "definition":
            return bool(re.search(r"\bmeans\b|\bdefined as\b|\bdefinition\b", lowered))
        if label == "function":
            return bool(re.search(r"\b(receives|supplies|provides|distributes|used to|serves to|operates|functions)\b", lowered))
        return False

    @classmethod
    def extract_term_window(cls, text: str, plan: QueryPlan, *, limit: int) -> str:
        lowered = text.lower()
        candidates: list[str] = []
        if plan.system_phrase and plan.attribute_terms:
            candidates.append(f"{plan.system_phrase} {plan.attribute_terms[0]}")
        if plan.system_phrase:
            candidates.append(plan.system_phrase)
        if plan.content_query:
            candidates.append(plan.content_query)
        if plan.concept_terms:
            candidates.append(" ".join(plan.concept_terms[:4]))
        candidates.extend(cls.focus_phrases(plan.focus_terms))
        candidates.extend(plan.scope_terms)
        candidates.extend(plan.concept_terms)
        candidates.extend(plan.system_aliases)
        candidates.extend(plan.attribute_terms)
        candidates.extend(plan.focus_terms)
        candidates.extend(plan.topic_terms + plan.action_terms)
        seen: set[str] = set()
        for candidate in candidates:
            needle = candidate.lower().strip()
            if not needle or needle in seen:
                continue
            seen.add(needle)
            index = lowered.find(needle)
            if index < 0:
                continue
            start = max(0, index - 140)
            end = min(len(text), index + max(limit - 140, len(needle) + 220))
            window = text[start:end].strip()
            if start > 0:
                first_space = window.find(" ")
                if first_space > 0:
                    window = window[first_space + 1 :].strip()
            if len(window) > limit:
                window = window[: limit - 3].rstrip() + "..."
            return window
        return ""

    @staticmethod
    def is_useful_extractive_block(heading: str, excerpt: str, plan: QueryPlan) -> bool:
        normalized_heading = " ".join(heading.lower().split())
        cleaned = excerpt.strip().strip('"').strip()
        if not cleaned or len(cleaned) < 35:
            return False
        if "...." in heading:
            return False
        if normalized_heading in {"front matter"}:
            return False
        if re.fullmatch(r"[A-Za-z0-9.()\- ]{1,24}", cleaned):
            return False
        if cleaned.lower().startswith("section ") and "agreement" in cleaned.lower() and len(cleaned) < 90:
            return False
        combined = f"{normalized_heading} {cleaned.lower()}"
        focus_candidates: list[str] = []
        if plan.content_query:
            focus_candidates.append(plan.content_query)
        if plan.concept_terms:
            focus_candidates.append(" ".join(plan.concept_terms[:4]))
        focus_candidates.extend(AnswerPolicy.focus_phrases(plan.focus_terms))
        focus_candidates.extend(plan.scope_terms)
        focus_candidates.extend(plan.concept_terms)
        focus_candidates.extend(plan.focus_terms)
        focus_candidates.extend(plan.topic_terms + plan.action_terms)
        if focus_candidates and not any(candidate and candidate in combined for candidate in focus_candidates):
            return False
        lowered = cleaned.lower()
        if plan.system_phrase and plan.system_phrase not in combined:
            system_hits = sum(1 for term in plan.system_terms if term and term in combined)
            if system_hits < max(1, min(2, len(plan.system_terms))):
                return False
        if ((plan.attribute_label and not plan.direct_text_question) or plan.count_question) and not AnswerPolicy.excerpt_matches_attribute(lowered, plan):
            return False
        if AnswerPolicy.is_requirement_question(plan) and not re.search(
            r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply)\b",
            lowered,
        ) and not re.search(r"\d", cleaned):
            return False
        if AnswerPolicy.is_value_question(plan) and not re.search(r"\d", cleaned):
            return False
        return True

    @staticmethod
    def excerpt_matches_attribute(lowered_excerpt: str, plan: QueryPlan) -> bool:
        label = plan.attribute_label or ""
        if plan.count_question:
            return bool(re.search(r"\b\d", lowered_excerpt) or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", lowered_excerpt))
        if not label:
            return True
        if label == "design_conditions":
            return bool(re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", lowered_excerpt) or re.search(r"\d", lowered_excerpt))
        if label == "configuration":
            return AnswerPolicy.matches_configuration_text(lowered_excerpt)
        if label == "type":
            return bool(re.search(r"\bmodel\b|\btype\b|\bselected\b|\bvendor\b|\bmanufacturer\b", lowered_excerpt) or re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*\b", lowered_excerpt))
        if label == "size":
            return bool(re.search(r"\bsize\b|\bdiameter\b|\brating\b|\b(?:inch|inches|mm|ft)\b", lowered_excerpt) or re.search(r"\d", lowered_excerpt))
        if label == "power":
            return bool(any(term in lowered_excerpt for term in plan.attribute_terms) or re.search(r"\b\d+(?:\.\d+)?\s*(?:hp|kw|kilowatt(?:s)?)\b", lowered_excerpt))
        if label == "service":
            return AnswerPolicy.matches_duty_text(lowered_excerpt)
        if label in {"capacity", "pressure", "temperature", "flow"}:
            return bool(any(term in lowered_excerpt for term in plan.attribute_terms) or re.search(r"\d", lowered_excerpt))
        if label == "responsibility":
            return bool(re.search(r"\b(shall|must|required|responsible|provide|furnish|supply|perform|deliver)\b", lowered_excerpt))
        if label == "function":
            return bool(re.search(r"\b(receives|supplies|provides|distributes|conditions|transports|serves|used to|operates)\b", lowered_excerpt))
        return True

    @classmethod
    def is_useful_summary_block(cls, chunk: RankedChunk, excerpt: str, plan: QueryPlan) -> bool:
        if "...." in chunk.heading:
            return False
        if chunk.total_score < 0.42:
            return False
        cleaned_heading = " ".join(chunk.heading.split()).lower()
        cleaned_excerpt = excerpt.strip().strip('"').strip()
        if cleaned_heading in {"front matter"} or len(cleaned_excerpt) < 60:
            return False
        if re.fullmatch(r"[A-Za-z0-9.()\- ]{1,28}", cleaned_excerpt):
            return False
        if cleaned_excerpt.lower().startswith("section ") and "agreement" in cleaned_excerpt.lower() and len(cleaned_excerpt) < 110:
            return False
        if plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC:
            if cls.is_broad_topic_noise_heading(cleaned_heading):
                return False
            if cls.is_broad_topic_definition_only(cleaned_excerpt):
                return False
            if not cls.has_broad_topic_substantive_signal(cleaned_heading, cleaned_excerpt, plan):
                return False
        return cls.is_useful_extractive_block(chunk.heading, excerpt, plan)

    @staticmethod
    def is_broad_topic_noise_heading(cleaned_heading: str) -> bool:
        return bool(
            re.search(r"\b(common acronyms|acronyms|abbreviations?|glossary)\b", cleaned_heading)
            or re.fullmatch(r"(section\s+)?(appendix|attachment|exhibit)\s+[a-z0-9.\-]+", cleaned_heading.strip())
        )

    @staticmethod
    def is_broad_topic_definition_only(cleaned_excerpt: str) -> bool:
        lowered = cleaned_excerpt.lower()
        if not re.search(
            r"\b(defined as|definition|definitions|means|meaning set forth|interpretation)\b",
            lowered,
        ):
            return False
        return not re.search(
            r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply|permit|permits|compliance|testing|test|demonstrate|monitor|guarantee|guarantees|limit|limits|emissions?)\b",
            lowered,
        )

    @staticmethod
    def has_broad_topic_substantive_signal(cleaned_heading: str, cleaned_excerpt: str, plan: QueryPlan) -> bool:
        combined = f"{cleaned_heading} {cleaned_excerpt.lower()}"
        if re.search(
            r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply|obtain|monitor|demonstrate)\b",
            combined,
        ):
            return True
        if re.search(
            r"\b(permit|permits|permitting|compliance|testing|test|environmental|emissions?|guarantees?|limits?)\b",
            combined,
        ) and (re.search(r"\d", cleaned_excerpt) or len(cleaned_excerpt) >= 110):
            return True
        concept_terms = tuple(term for term in plan.concept_terms if term)
        if concept_terms:
            concept_hits = sum(1 for term in concept_terms if term in combined)
            if concept_hits >= max(1, min(len(concept_terms), 2)) and len(cleaned_excerpt) >= 120:
                return True
        return False

    @staticmethod
    def has_attribute_guidance_marker(text: str) -> bool:
        return bool(
            re.search(
                r"\b(different|var(?:y|ies|ied)|based on|depends on|determined|established|same basis|basis as|all components)\b",
                text,
            )
        )

    @classmethod
    def is_generic_attribute_question(cls, plan: QueryPlan) -> bool:
        if plan.attribute_label not in {"design_conditions", "pressure", "temperature"}:
            return False
        subject_terms = [
            term
            for term in plan.system_terms + plan.concept_terms + plan.scope_terms
            if term
            and term not in GENERIC_SUBJECT_TERMS
            and term not in plan.attribute_terms
        ]
        return not subject_terms

    @staticmethod
    def matches_configuration_text(text: str) -> bool:
        return bool(
            re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b|\bduplex\b|\bredun", text)
            or re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(?:x\s*)?\d+%\s*capacity\b", text)
            or re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+full\s+capacity\b", text)
            or re.search(r"\b(?:duty|standby|lead|lag)\b", text)
        )

    @staticmethod
    def matches_duty_text(text: str) -> bool:
        return bool(
            re.search(r"\b(?:duty|duties|service|services)\b", text)
            or re.search(r"\b(?:standby|continuous|intermittent|lead/lag|lead-lag|lead lag|primary|backup|spare)\b", text)
        )

    @staticmethod
    def focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
