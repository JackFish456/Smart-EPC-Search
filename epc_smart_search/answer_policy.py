from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from epc_smart_search.query_planner import QueryPlan, plan_query
from epc_smart_search.retrieval import Citation, ExactPageHit, RankedChunk


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    citations: list[Citation]
    refused: bool


SUMMARY_MAX_NEW_TOKENS = 768
SUMMARY_ENABLE_THINKING = False
SUMMARY_CONTEXT_MAX_SECTIONS = 6
SUMMARY_EXCERPT_LIMIT = 760
EXPAND_MAX_NEW_TOKENS = 896
DEEP_MAX_NEW_TOKENS = 896
DEEP_ENABLE_THINKING = True
DEEP_CONTEXT_MAX_SECTIONS = 8
DEEP_CONTEXT_EXACT_HITS = 3
DEEP_PAGE_CONTEXT_SECTIONS = 2


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
        evidence_is_weak = best is None or (best.total_score < 0.16 and best.lexical_score < 0.05)
        if evidence_is_weak and not deep_think:
            return AssistantAnswer("I can't verify that from the contract.", citations, True)

        reference_answer = self.build_reference_answer(effective_question, ranked, citations)
        if reference_answer is not None and not summary_request and not deep_think and not expand_answer:
            return reference_answer

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
            count = self.count_exact_items(lowered, hits)
            if count is not None:
                body = f"I found {count} matching items in the contract.\n\nDirect contract text:\n" + "\n\n".join(excerpts)
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
        max_sections: int = 3,
    ) -> AssistantAnswer | None:
        if not ranked:
            return None
        plan = plan_query(question)
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
            if len(blocks) >= max_sections:
                break
        if not blocks:
            return None
        return AssistantAnswer("\n\n".join(blocks), citations, False)

    @staticmethod
    def prefers_generated_answer(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        wrapped = f" {normalized} "
        return normalized.startswith(("summarize ", "summarise ", "explain ")) or any(
            phrase in wrapped for phrase in (" summarize ", " summarise ", " explain ")
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

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [piece.strip() for piece in re.split(r"(?<=[.!?;:])\s+|\s{2,}", text) if piece.strip()]

    @classmethod
    def score_sentence(cls, sentence: str, plan: QueryPlan) -> float:
        normalized = sentence.lower()
        score = 0.0
        if plan.content_query and plan.content_query in normalized:
            score += 4.0
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

    @staticmethod
    def is_value_or_requirement_question(plan: QueryPlan) -> bool:
        return AnswerPolicy.is_value_question(plan) or AnswerPolicy.is_requirement_question(plan)

    @staticmethod
    def is_value_question(plan: QueryPlan) -> bool:
        normalized = plan.normalized_query
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
    def extract_term_window(cls, text: str, plan: QueryPlan, *, limit: int) -> str:
        lowered = text.lower()
        candidates: list[str] = []
        if plan.content_query:
            candidates.append(plan.content_query)
        candidates.extend(cls.focus_phrases(plan.focus_terms))
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
        if AnswerPolicy.is_requirement_question(plan) and not re.search(
            r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply)\b",
            lowered,
        ) and not re.search(r"\d", cleaned):
            return False
        if AnswerPolicy.is_value_question(plan) and not re.search(r"\d", cleaned):
            return False
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
