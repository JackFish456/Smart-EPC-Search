from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from epc_smart_search.query_planner import QueryPlan, has_term_overlap, plan_query
from epc_smart_search.retrieval import Citation, ExactPageHit, RankedChunk


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    citations: list[Citation]
    refused: bool


SUMMARY_MAX_NEW_TOKENS = 224
SUMMARY_ENABLE_THINKING = False
SUMMARY_CONTEXT_MAX_SECTIONS = 4
SUMMARY_EXCERPT_LIMIT = 520
EXPAND_MAX_NEW_TOKENS = 320
DEEP_MAX_NEW_TOKENS = 384
DEEP_ENABLE_THINKING = True
DEEP_CONTEXT_MAX_SECTIONS = 5
DEEP_CONTEXT_EXACT_HITS = 2
DEEP_PAGE_CONTEXT_SECTIONS = 1
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
        plan = plan_query(effective_question)
        exact_hits = self.retriever.find_exact_page_hits(effective_question)
        exact_answer = None if expand_answer else self.build_exact_answer(effective_question, exact_hits)
        if exact_answer is not None and not deep_think:
            return exact_answer

        summary_request = self.prefers_generated_answer(question)
        ranked = (
            self.retriever.retrieve(effective_question, profile="deep")
            if deep_think
            else self.retriever.retrieve(effective_question)
        )
        citations = self.retriever.expand_with_context(ranked) if ranked else []
        citations = self.limit_citations(citations) if citations else []

        if not deep_think and not ranked:
            return AssistantAnswer("I can't verify that from the contract.", [], True)
        if not deep_think and not citations:
            return AssistantAnswer("I can't verify that from the contract.", [], True)

        best = ranked[0] if ranked else None
        strict_gate_active = (
            not summary_request
            and not deep_think
            and not expand_answer
            and self.reference_lookup_kind(effective_question) is None
        )
        evidence_is_weak = (
            best is None
            or (best.total_score < 0.16 and best.lexical_score < 0.05)
            or (strict_gate_active and self.requires_strict_grounding(plan) and not self.is_strong_question_match(plan, best))
        )
        if evidence_is_weak and not deep_think:
            return AssistantAnswer("I can't verify that from the contract.", citations, True)

        reference_answer = self.build_reference_answer(effective_question, ranked, citations)
        if reference_answer is not None and not summary_request and not deep_think and not expand_answer:
            return reference_answer

        compact_answer = self.build_compact_answer(effective_question, ranked, citations)
        if compact_answer is not None and not summary_request and not deep_think and not expand_answer:
            return compact_answer

        extractive_answer = self.build_extractive_answer(effective_question, ranked, citations)
        if extractive_answer is not None and not summary_request and not deep_think and not expand_answer:
            return extractive_answer

        if deep_think:
            prompt_context = self.build_deep_prompt_context(effective_question, ranked, exact_hits)
            if evidence_is_weak and not exact_hits:
                if extractive_answer is not None:
                    return extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", citations, True)
            if not prompt_context:
                if extractive_answer is not None:
                    return extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", citations, True)
        elif expand_answer:
            prompt_context = self.build_expand_prompt_context(effective_question, ranked, previous_answer)
            if not prompt_context:
                if extractive_answer is not None:
                    return extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", citations, True)
        else:
            prompt_context = (
                self.build_summary_prompt_context(effective_question, ranked)
                if summary_request
                else self.retriever.build_evidence_pack(effective_question, ranked, citations)
            )
            if summary_request and not prompt_context:
                if extractive_answer is not None:
                    return extractive_answer
                return AssistantAnswer("I can't verify that from the contract.", citations, True)

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
            if extractive_answer is not None:
                return extractive_answer
            answer_text = "I can't verify that from the contract."

        answer_text = answer_text.strip() or "I can't verify that from the contract."
        refused = answer_text == "I can't verify that from the contract."
        if (summary_request or deep_think or expand_answer) and (refused or not answer_text):
            if extractive_answer is not None:
                return extractive_answer
        return AssistantAnswer(answer_text, citations, refused)

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
        if normalized.startswith(("what section", "which section")):
            return f"what section talks about {anchor}"
        if normalized.startswith(("what page", "which page")):
            return f"which page mentions {anchor}"
        if normalized.startswith(("quote that", "show me that")):
            return f"show me {anchor}"
        if normalized.startswith(("where ", "who ", "when ", "how many ", "how much ")):
            return f"{stem} for {anchor}"
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
    def limit_citations(citations: list[Citation], limit: int = 3) -> list[Citation]:
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

    @staticmethod
    def default_extractive_sections(plan: QueryPlan) -> int:
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
        if len(excerpt) > limit:
            excerpt = excerpt[: limit - 3].rstrip() + "..."
        return cls.quote_excerpt(excerpt)

    @classmethod
    def score_sentence(cls, sentence: str, plan: QueryPlan) -> float:
        normalized = sentence.lower()
        score = 0.0
        if plan.content_query and plan.content_query in normalized:
            score += 4.0
        if plan.system_phrase and plan.system_phrase in normalized:
            score += 2.2
        score += 1.0 * sum(1 for term in plan.system_terms if term in normalized)
        score += 1.0 * sum(1 for term in plan.attribute_terms if term in normalized)
        score += 1.5 * sum(1 for term in plan.focus_terms if term in normalized)
        score += 0.8 * sum(1 for term in plan.topic_terms + plan.action_terms if term in normalized)
        score += 0.5 * sum(1 for term in plan.actor_terms if term in normalized)
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
        elif label == "configuration":
            if re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b|\bduplex\b|\bredun", normalized):
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
            or plan.attribute_label
            or plan.count_question
            or plan.direct_text_question
            or cls.is_value_or_requirement_question(plan)
            or plan.intent in {"definition", "responsibility"}
        )

    @classmethod
    def is_strong_question_match(cls, plan: QueryPlan, chunk: RankedChunk) -> bool:
        minimum_score = 0.42
        if plan.attribute_label or plan.count_question or cls.is_value_or_requirement_question(plan):
            minimum_score = 0.56
        elif plan.direct_text_question or plan.system_phrase:
            minimum_score = 0.48
        if chunk.total_score < minimum_score and cls.requires_strict_grounding(plan):
            return False
        combined = " ".join([chunk.heading, chunk.full_text])
        focus_hits = sum(1 for term in plan.focus_terms if has_term_overlap(combined, (term,)))
        if plan.direct_text_question and len(plan.focus_terms) >= 2 and focus_hits < min(2, len(plan.focus_terms)):
            return False
        if plan.system_phrase and not cls.matches_system(plan, combined):
            return False
        if (plan.attribute_label or plan.count_question) and not cls.matches_attribute(plan, combined):
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
            return bool(re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b", lowered))
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
        candidates.extend(cls.focus_phrases(plan.focus_terms))
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
        focus_candidates.extend(AnswerPolicy.focus_phrases(plan.focus_terms))
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
            return bool(re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b|\bduplex\b|\bredun", lowered_excerpt))
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
        return cls.is_useful_extractive_block(chunk.heading, excerpt, plan)

    @staticmethod
    def focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
