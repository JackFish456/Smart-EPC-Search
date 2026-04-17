from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from epc_smart_search.query_planner import (
    ANSWER_FAMILY_GUARANTEE_OR_LIMIT,
    QueryPlan,
    REQUEST_SHAPE_BROAD_TOPIC,
    REQUEST_SHAPE_GROUPED_LIST,
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
        plan = self._plan_query(effective_question)
        exact_hits = self.retriever.find_exact_page_hits(effective_question)
        exact_answer = None if expand_answer else self.build_exact_answer(effective_question, exact_hits)
        if exact_answer is not None and not deep_think:
            return exact_answer

        broad_topic_request = plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
        summary_request = broad_topic_request or self.prefers_generated_answer(question)
        trace = self._retrieve_trace(effective_question, gemma_client, deep_think=deep_think)
        if broad_topic_request and trace is not None:
            ranked = list(trace.merged_ranked)
            citations = self.citations_from_ranked(ranked, effective_question)
        elif trace is not None and trace.selected_bundle is not None:
            ranked = list(trace.selected_bundle.ranked_chunks)
            citations = list(trace.selected_bundle.citations)
            if self.should_use_merged_ranked_for_requirements(plan, trace):
                ranked = sorted(
                    trace.merged_ranked,
                    key=lambda chunk: self.score_requirement_candidate(chunk, plan),
                    reverse=True,
                )
                citations = self.citations_from_ranked(ranked, effective_question)
        else:
            ranked = (
                self.retriever.retrieve(effective_question, profile="deep")
                if deep_think
                else self.retriever.retrieve(effective_question)
            )
            citations = self.retriever.expand_with_context(ranked) if ranked else []
        citations = self.limit_citations(citations) if citations else []

        if not deep_think and not ranked:
            return AssistantAnswer("I can't verify that from the contract.", [], True)

        strong_ranked_evidence = self.has_strong_ranked_evidence(plan, ranked)
        evidence_is_weak = not citations or not strong_ranked_evidence
        refusal_citations = citations if not evidence_is_weak else []
        reference_answer = self.build_reference_answer(effective_question, ranked, citations)
        grouped_answer = self.build_grouped_answer(effective_question, ranked, citations)
        page_attribute_answer = self.build_page_attribute_answer(effective_question, ranked, citations)
        compact_answer = self.build_compact_answer(effective_question, ranked, citations)
        extractive_answer = self.build_extractive_answer(effective_question, ranked, citations)
        broad_topic_answer = self.build_broad_topic_answer(effective_question, ranked, citations) if broad_topic_request else None
        grounded_page_attribute_answer = None if evidence_is_weak else page_attribute_answer
        grounded_compact_answer = None if evidence_is_weak else compact_answer
        grounded_extractive_answer = None if evidence_is_weak else extractive_answer
        grounded_broad_topic_answer = None if evidence_is_weak else broad_topic_answer

        if reference_answer is not None and not summary_request and not deep_think and not expand_answer:
            return reference_answer
        if grouped_answer is not None and not summary_request and not deep_think and not expand_answer:
            return grouped_answer
        if grounded_page_attribute_answer is not None and not summary_request and not deep_think and not expand_answer:
            return grounded_page_attribute_answer
        if grounded_compact_answer is not None and not summary_request and not deep_think and not expand_answer:
            return grounded_compact_answer
        if grounded_extractive_answer is not None and not summary_request and not deep_think and not expand_answer:
            return grounded_extractive_answer

        if deep_think:
            prompt_context = self.build_deep_prompt_context(effective_question, ranked, exact_hits)
            if not prompt_context and grounded_extractive_answer is not None:
                return grounded_extractive_answer
            if evidence_is_weak and not exact_hits:
                if grounded_extractive_answer is not None:
                    return grounded_extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
        elif expand_answer:
            prompt_context = self.build_expand_prompt_context(effective_question, ranked, previous_answer)
            if not prompt_context:
                if grounded_extractive_answer is not None:
                    return grounded_extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
        else:
            prompt_context = (
                self.build_summary_prompt_context(effective_question, ranked)
                if summary_request
                else self._build_standard_prompt_context(effective_question, ranked, citations, trace)
            )
            if summary_request and not prompt_context:
                if grounded_broad_topic_answer is not None:
                    return grounded_broad_topic_answer
                if grounded_extractive_answer is not None:
                    return grounded_extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", refusal_citations, True)
            if not summary_request and not prompt_context:
                if grounded_extractive_answer is not None:
                    return grounded_extractive_answer
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
            if grounded_extractive_answer is not None:
                return grounded_extractive_answer
            answer_text = "I can't verify that from the contract."

        answer_text = answer_text.strip() or "I can't verify that from the contract."
        refused = answer_text == "I can't verify that from the contract."
        if (summary_request or deep_think or expand_answer) and (refused or not answer_text):
            if grounded_broad_topic_answer is not None:
                return grounded_broad_topic_answer
            if grounded_extractive_answer is not None:
                return grounded_extractive_answer
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
        if not self.prefer_exact_answer(lowered):
            return None
        excerpts = [f"Page {hit.page_num}: {self.trim_page_excerpt(hit.page_text, lowered)}" for hit in hits[:4]]
        if self.is_count_question(lowered):
            quantity = self.extract_count_value(lowered, hits)
            if quantity is not None:
                body = f"Answer: {quantity}\n\nDirect contract text:\n" + "\n\n".join(excerpts)
                return AssistantAnswer(body, [], False)
        return AssistantAnswer("Direct contract text:\n" + "\n\n".join(excerpts), [], False)

    @staticmethod
    def is_count_question(question: str) -> bool:
        return question.startswith("how many ")

    @staticmethod
    def prefer_exact_answer(question: str) -> bool:
        return question.startswith(("how many ", "show me ", "find ", "quote ", "where does ", "which page "))

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
        if label != "configuration":
            return ""
        for pattern in cls.configuration_value_patterns():
            match = pattern.search(text)
            if match:
                return " ".join(match.group(0).split())
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
        if plan.attribute_label in {"design_conditions", "size", "capacity", "power", "pressure", "temperature", "flow"}:
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
        if (plan.attribute_label or plan.count_question) and not AnswerPolicy.excerpt_matches_attribute(lowered, plan):
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
    def focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
