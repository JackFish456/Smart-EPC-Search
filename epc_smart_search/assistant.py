from __future__ import annotations

import atexit
import html
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import requests

from epc_smart_search.answer_policy import (
    AssistantAnswer as PolicyAssistantAnswer,
    AnswerPolicy,
)
from epc_smart_search.app_paths import DB_PATH, GEMMA_TEST_PYTHON, WORKSPACE_ROOT, seed_preloaded_db
from epc_smart_search.config import GEMMA_SERVICE_HOST, GEMMA_SERVICE_PORT, SEARCH_SCHEMA_VERSION
from epc_smart_search.gemma_client import GemmaServiceClient as ManagedGemmaServiceClient
from epc_smart_search.query_planner import QueryPlan, plan_query
from epc_smart_search.retrieval import Citation, ExactPageHit, HybridRetriever, RankedChunk
from epc_smart_search.storage import ContractStore


AssistantAnswer = PolicyAssistantAnswer


@dataclass(slots=True)
class IndexValidationResult:
    ready: bool
    error: str | None
    document_id: str | None
    chunk_count: int = 0
    block_count: int = 0
    feature_count: int = 0
    page_text_count: int = 0
    diagnostic_count: int = 0


SUMMARY_MAX_NEW_TOKENS = 768
SUMMARY_ENABLE_THINKING = False
SUMMARY_CONTEXT_MAX_SECTIONS = 6
SUMMARY_EXCERPT_LIMIT = 760
DEEP_MAX_NEW_TOKENS = 896
DEEP_ENABLE_THINKING = True
DEEP_CONTEXT_MAX_SECTIONS = 8
DEEP_CONTEXT_EXACT_HITS = 3
DEEP_PAGE_CONTEXT_SECTIONS = 2
DEEP_EXCERPT_LIMIT = 920
INTERNAL_REBUILD_ERROR = "Contract rebuild is available only through the internal rebuild tool."


def validate_contract_store(store: ContractStore) -> IndexValidationResult:
    document = store.get_document()
    if document is None:
        return IndexValidationResult(False, "Bundled contract data is missing.", None)
    document_id = str(document["document_id"])
    if store.get_metadata("search_schema_version") != str(SEARCH_SCHEMA_VERSION):
        return IndexValidationResult(False, "Bundled contract data uses an incompatible search schema.", document_id)
    chunk_count = store.get_chunk_count(document_id)
    if chunk_count <= 0:
        return IndexValidationResult(False, "Bundled contract data does not contain searchable chunks.", document_id)
    feature_count = store.get_feature_count(document_id)
    if feature_count <= 0:
        return IndexValidationResult(False, "Bundled contract data is missing search features.", document_id, chunk_count=chunk_count)
    if feature_count != chunk_count:
        return IndexValidationResult(
            False,
            "Bundled contract data is incomplete. Search features do not match the indexed chunks.",
            document_id,
            chunk_count=chunk_count,
            feature_count=feature_count,
        )
    block_count = store.get_block_count(document_id)
    if block_count <= 0:
        return IndexValidationResult(
            False,
            "Bundled contract data is missing block-level search coverage.",
            document_id,
            chunk_count=chunk_count,
            feature_count=feature_count,
            block_count=block_count,
        )
    page_text_count = store.get_page_text_count(document_id)
    if page_text_count <= 0:
        return IndexValidationResult(
            False,
            "Bundled contract data is missing page evidence.",
            document_id,
            chunk_count=chunk_count,
            block_count=block_count,
            feature_count=feature_count,
            page_text_count=page_text_count,
        )
    diagnostic_count = store.get_ingest_diagnostic_count(document_id)
    if diagnostic_count != page_text_count:
        return IndexValidationResult(
            False,
            "Bundled contract data is missing ingest diagnostics.",
            document_id,
            chunk_count=chunk_count,
            block_count=block_count,
            feature_count=feature_count,
            page_text_count=page_text_count,
            diagnostic_count=diagnostic_count,
        )
    return IndexValidationResult(
        True,
        None,
        document_id,
        chunk_count=chunk_count,
        block_count=block_count,
        feature_count=feature_count,
        page_text_count=page_text_count,
        diagnostic_count=diagnostic_count,
    )


class GemmaServiceClient:
    def __init__(self, host: str = GEMMA_SERVICE_HOST, port: int = GEMMA_SERVICE_PORT) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen[str] | None = None
        atexit.register(self.stop)

    def ensure_running(self) -> None:
        if self._is_healthy():
            return
        if not GEMMA_TEST_PYTHON.exists():
            raise RuntimeError(f"Gemma Python not found: {GEMMA_TEST_PYTHON}")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            [str(GEMMA_TEST_PYTHON), str(WORKSPACE_ROOT / "gemma_service.py"), "--port", str(self.port)],
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=creation_flags,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self._is_healthy():
                return
            time.sleep(1.0)
        raise RuntimeError("Gemma helper service did not become ready.")

    def ask(
        self,
        question: str,
        context: str,
        *,
        enable_thinking: bool | None = None,
        max_new_tokens: int | None = None,
        response_style: str | None = None,
        previous_answer: str | None = None,
    ) -> str:
        self.ensure_running()
        payload: dict[str, object] = {"question": question, "context": context}
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if response_style is not None:
            payload["response_style"] = response_style
        if previous_answer is not None:
            payload["previous_answer"] = previous_answer
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("answer", "")).strip()

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def _is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
            return response.ok
        except Exception:
            return False


class ContractAssistant:
    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        seed_preloaded_db()
        self.store = ContractStore(db_path)
        self.retriever = HybridRetriever(self.store)
        self.gemma = ManagedGemmaServiceClient()
        self.answer_policy = AnswerPolicy(self.store, self.retriever)

    def _get_answer_policy(self) -> AnswerPolicy:
        policy = getattr(self, "answer_policy", None)
        if policy is None:
            policy = AnswerPolicy(getattr(self, "store", None), self.retriever)
            self.answer_policy = policy
        return policy

    def get_index_status(self) -> IndexValidationResult:
        return validate_contract_store(self.store)

    def is_index_ready(self) -> bool:
        return self.get_index_status().ready

    def build_index(self, progress_callback=None) -> dict[str, int | str]:
        raise RuntimeError(INTERNAL_REBUILD_ERROR)

    def ask(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None = None,
        *,
        deep_think: bool = False,
        expand_answer: bool = False,
        previous_answer: str | None = None,
    ) -> AssistantAnswer:
        return self._get_answer_policy().answer(
            question,
            history,
            self.gemma,
            deep_think=deep_think,
            expand_answer=expand_answer,
            previous_answer=previous_answer,
        )

    @classmethod
    def _resolve_question(cls, question: str, history: Sequence[dict[str, str]] | None = None) -> str:
        if not history:
            return question
        normalized = " ".join(question.lower().split())
        plan = plan_query(question)
        if not cls._looks_like_follow_up(question, plan):
            return question
        anchor = cls._find_follow_up_anchor(history)
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
    def _find_follow_up_anchor(cls, history: Sequence[dict[str, str]]) -> str:
        for turn in reversed(history):
            if str(turn.get("role", "")).strip() != "user":
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            plan = plan_query(content)
            if cls._looks_like_follow_up(content, plan):
                continue
            anchor = re.sub(r"^(summari[sz]e|explain)\s+", "", plan.content_query.strip(" ?.!"), flags=re.IGNORECASE)
            anchor = re.sub(r"\s+in plain english$", "", anchor, flags=re.IGNORECASE)
            if anchor:
                return anchor
        return ""

    @staticmethod
    def _looks_like_follow_up(question: str, plan: QueryPlan) -> bool:
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
        referential_terms = (
            " it ",
            " that ",
            " this ",
            " these ",
            " those ",
            " they ",
            " them ",
            " same ",
        )
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

    def _build_exact_answer(self, question: str, hits: list[ExactPageHit]) -> AssistantAnswer | None:
        if not hits:
            return None
        lowered = " ".join(question.lower().split())
        if not self._prefer_exact_answer(lowered):
            return None
        excerpts = []
        for hit in hits[:4]:
            excerpts.append(f"Page {hit.page_num}: {self._trim_page_excerpt(hit.page_text, lowered)}")
        if self._is_count_question(lowered):
            count = self._count_exact_items(lowered, hits)
            if count is not None:
                body = (
                    f"I found {count} matching items in the contract.\n\n"
                    "Direct contract text:\n"
                    + "\n\n".join(excerpts)
                )
                return AssistantAnswer(body, [], False)
        body = "Direct contract text:\n" + "\n\n".join(excerpts)
        return AssistantAnswer(body, [], False)

    @staticmethod
    def _is_count_question(question: str) -> bool:
        return question.startswith("how many ")

    @staticmethod
    def _prefer_exact_answer(question: str) -> bool:
        prefixes = (
            "how many ",
            "show me ",
            "find ",
            "quote ",
            "where does ",
            "which page ",
        )
        return question.startswith(prefixes)

    @staticmethod
    def _reference_lookup_kind(question: str) -> str | None:
        normalized = " ".join(question.lower().split())
        if normalized.startswith(("what section", "which section")):
            return "section"
        if normalized.startswith(("what page", "which page")):
            return "page"
        return None

    @staticmethod
    def _count_exact_items(question: str, hits: list[ExactPageHit]) -> int | None:
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
        if seen_numbers:
            return len(seen_numbers)
        return None

    @staticmethod
    def _trim_page_excerpt(page_text: str, question: str, limit: int = 700) -> str:
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
    def _build_reference_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        lookup_kind = cls._reference_lookup_kind(question)
        if lookup_kind is None or not ranked:
            return None
        plan = plan_query(question)
        for chunk in ranked:
            if "...." in chunk.heading:
                continue
            excerpt = cls._extract_ranked_excerpt(chunk.full_text, plan, limit=520)
            if not excerpt:
                continue
            label = chunk.section_number or "Unnumbered clause"
            pages = cls._format_page_range(chunk.page_start, chunk.page_end)
            heading = " ".join(chunk.heading.split())
            intro = "Most relevant section:" if lookup_kind == "section" else "Most relevant pages:"
            body = (
                f"Section {label} - {heading}\n"
                f"Pages: {pages}\n"
                f"Contract text: {excerpt}"
            )
            return AssistantAnswer(f"{intro}\n\n{body}", citations, False)
        return None

    @staticmethod
    def _limit_citations(citations: list[Citation], limit: int = 3) -> list[Citation]:
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
    def _build_extractive_answer(
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
        excerpt_limit = 900 if cls._is_value_or_requirement_question(plan) else 650
        for chunk in ranked:
            key = (chunk.section_number, chunk.page_start, chunk.page_end)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            excerpt = cls._extract_ranked_excerpt(chunk.full_text, plan, limit=excerpt_limit)
            if not excerpt or not cls._is_useful_extractive_block(chunk.heading, excerpt, plan):
                continue
            label = chunk.section_number or "Unnumbered clause"
            pages = cls._format_page_range(chunk.page_start, chunk.page_end)
            heading = " ".join(chunk.heading.split())
            blocks.append(
                f"Section {label} - {heading}\n"
                f"Pages: {pages}\n"
                f"Contract text: {excerpt}"
            )
            if len(blocks) >= max_sections:
                break
        if not blocks:
            return None
        return AssistantAnswer("\n\n".join(blocks), citations, False)

    @staticmethod
    def _prefers_generated_answer(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        explicit_prefixes = (
            "summarize ",
            "summarise ",
            "explain ",
        )
        explicit_phrases = (
            " summarize ",
            " summarise ",
            " explain ",
        )
        wrapped = f" {normalized} "
        return normalized.startswith(explicit_prefixes) or any(phrase in wrapped for phrase in explicit_phrases)

    @classmethod
    def _build_summary_prompt_context(
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
            excerpt = cls._extract_ranked_excerpt(
                chunk.full_text,
                plan,
                limit=SUMMARY_EXCERPT_LIMIT,
                surrounding_sentences=1,
            )
            if not excerpt or not cls._is_useful_summary_block(chunk, excerpt, plan):
                continue
            label = html.escape(chunk.section_number or "Unnumbered clause")
            heading = html.escape(" ".join(chunk.heading.split()))
            pages = html.escape(cls._format_page_range(chunk.page_start, chunk.page_end))
            blocks.append(
                f"Section: {label}\n"
                f"Heading: {heading}\n"
                f"Pages: {pages}\n"
                f"Excerpt: {excerpt}"
            )
            if len(blocks) >= max_sections:
                break
        return "\n\n".join(blocks)

    def _build_deep_prompt_context(
        self,
        question: str,
        ranked: list[RankedChunk],
        exact_hits: list[ExactPageHit],
    ) -> str:
        sections: list[str] = []
        ranked_blocks = self._build_summary_prompt_context(
            question,
            ranked,
            max_sections=DEEP_CONTEXT_MAX_SECTIONS,
        )
        if ranked_blocks:
            sections.append("Ranked contract sections:\n" + ranked_blocks)

        exact_blocks = self._build_exact_hit_prompt_context(question, exact_hits)
        if exact_blocks:
            sections.append("Exact page hits:\n" + exact_blocks)

        page_context = self._build_deep_page_context(ranked)
        if page_context:
            sections.append("Nearby page context:\n" + page_context)

        return "\n\n".join(section for section in sections if section).strip()

    @classmethod
    def _build_exact_hit_prompt_context(cls, question: str, exact_hits: list[ExactPageHit]) -> str:
        lowered = " ".join(question.lower().split())
        blocks: list[str] = []
        for hit in exact_hits[:DEEP_CONTEXT_EXACT_HITS]:
            excerpt = cls._trim_page_excerpt(hit.page_text, lowered)
            blocks.append(f"Page {hit.page_num}: {cls._quote_excerpt(excerpt)}")
        return "\n\n".join(blocks)

    def _build_deep_page_context(self, ranked: list[RankedChunk]) -> str:
        document_id = self.retriever.resolve_document_id()
        if not document_id or not ranked:
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
            rows = self.store.fetch_page_window(
                document_id,
                chunk.page_start,
                chunk.page_end,
                padding=0,
                limit=2,
            )
            if not rows:
                continue
            page_lines = [
                f"Page {int(row['page_num'])}: {self._quote_excerpt(self._truncate_text(str(row['page_text']), limit=500))}"
                for row in rows
            ]
            blocks.append(f"Section {label} - {heading}\n" + "\n".join(page_lines))
            if len(blocks) >= DEEP_PAGE_CONTEXT_SECTIONS:
                break
        return "\n\n".join(blocks)

    @staticmethod
    def _format_page_range(page_start: int, page_end: int) -> str:
        if page_start == page_end:
            return f"{page_start}"
        return f"{page_start}-{page_end}"

    @staticmethod
    def _truncate_text(text: str, limit: int = 340) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[: limit - 3] + "..."

    @classmethod
    def _extract_ranked_excerpt(
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
        term_window = cls._extract_term_window(compact, plan, limit=limit)
        if term_window:
            return cls._quote_excerpt(term_window)
        sentences = cls._split_sentences(compact)
        if not sentences:
            return cls._quote_excerpt(compact[:limit].rstrip())
        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(sentences):
            score = cls._score_sentence(sentence, plan)
            scored.append((score, index))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_index = scored[0][1]
        selected_indices = {best_index}
        if cls._is_value_or_requirement_question(plan):
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
        return cls._quote_excerpt(excerpt)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?;:])\s+|\s{2,}", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    @classmethod
    def _score_sentence(cls, sentence: str, plan: QueryPlan) -> float:
        normalized = sentence.lower()
        score = 0.0
        if plan.content_query and plan.content_query in normalized:
            score += 4.0
        score += 1.5 * sum(1 for term in plan.focus_terms if term in normalized)
        score += 0.8 * sum(1 for term in plan.topic_terms + plan.action_terms if term in normalized)
        score += 0.5 * sum(1 for term in plan.actor_terms if term in normalized)
        if cls._is_value_or_requirement_question(plan) and re.search(r"\d", sentence):
            score += 1.2
        if cls._is_requirement_question(plan) and re.search(r"\b(shall|must|required|responsible|obligated)\b", normalized):
            score += 1.1
        if sentence.count("....") >= 2:
            score -= 2.0
        return score

    @staticmethod
    def _is_value_or_requirement_question(plan: QueryPlan) -> bool:
        return ContractAssistant._is_value_question(plan) or ContractAssistant._is_requirement_question(plan)

    @staticmethod
    def _is_value_question(plan: QueryPlan) -> bool:
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
    def _is_requirement_question(plan: QueryPlan) -> bool:
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
    def _quote_excerpt(text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        return f'"{cleaned}"'

    @classmethod
    def _extract_term_window(cls, text: str, plan: QueryPlan, *, limit: int) -> str:
        lowered = text.lower()
        candidates: list[str] = []
        if plan.content_query:
            candidates.append(plan.content_query)
        candidates.extend(cls._focus_phrases(plan.focus_terms))
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
    def _is_useful_extractive_block(heading: str, excerpt: str, plan: QueryPlan) -> bool:
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
        focus_candidates.extend(ContractAssistant._focus_phrases(plan.focus_terms))
        focus_candidates.extend(plan.focus_terms)
        focus_candidates.extend(plan.topic_terms + plan.action_terms)
        if focus_candidates and not any(candidate and candidate in combined for candidate in focus_candidates):
            return False
        lowered = cleaned.lower()
        if ContractAssistant._is_requirement_question(plan) and not re.search(
            r"\b(shall|must|required|responsible|obligated|provide|perform|submit|deliver|maintain|comply)\b",
            lowered,
        ) and not re.search(r"\d", cleaned):
            return False
        if ContractAssistant._is_value_question(plan) and not re.search(r"\d", cleaned):
            return False
        return True

    @classmethod
    def _is_useful_summary_block(cls, chunk: RankedChunk, excerpt: str, plan: QueryPlan) -> bool:
        heading = chunk.heading
        if "...." in heading:
            return False
        if chunk.total_score < 0.42:
            return False
        cleaned_heading = " ".join(heading.split()).lower()
        if cleaned_heading in {"front matter"}:
            return False
        cleaned_excerpt = excerpt.strip().strip('"').strip()
        if len(cleaned_excerpt) < 60:
            return False
        if re.fullmatch(r"[A-Za-z0-9.()\- ]{1,28}", cleaned_excerpt):
            return False
        if cleaned_excerpt.lower().startswith("section ") and "agreement" in cleaned_excerpt.lower() and len(cleaned_excerpt) < 110:
            return False
        return cls._is_useful_extractive_block(heading, excerpt, plan)

    @staticmethod
    def _focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
