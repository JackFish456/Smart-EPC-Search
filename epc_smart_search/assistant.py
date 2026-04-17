from __future__ import annotations

import atexit
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from epc_smart_search import answer_policy as answer_policy_module
from epc_smart_search.answer_policy import AnswerPolicy
from epc_smart_search.app_paths import DB_PATH, GEMMA_TEST_PYTHON, WORKSPACE_ROOT, seed_preloaded_db
from epc_smart_search.config import GEMMA_SERVICE_HOST, GEMMA_SERVICE_PORT, SEARCH_SCHEMA_VERSION
from epc_smart_search.gemma_client import GemmaServiceClient as ManagedGemmaServiceClient
from epc_smart_search.query_planner import QueryPlan
from epc_smart_search.retrieval import Citation, ExactPageHit, HashingEmbedder, HybridRetriever, RankedChunk
from epc_smart_search.storage import ContractStore


AssistantAnswer = answer_policy_module.AssistantAnswer
SUMMARY_MAX_NEW_TOKENS = answer_policy_module.SUMMARY_MAX_NEW_TOKENS
SUMMARY_ENABLE_THINKING = answer_policy_module.SUMMARY_ENABLE_THINKING
SUMMARY_CONTEXT_MAX_SECTIONS = answer_policy_module.SUMMARY_CONTEXT_MAX_SECTIONS
SUMMARY_EXCERPT_LIMIT = answer_policy_module.SUMMARY_EXCERPT_LIMIT
EXPAND_MAX_NEW_TOKENS = answer_policy_module.EXPAND_MAX_NEW_TOKENS
DEEP_MAX_NEW_TOKENS = answer_policy_module.DEEP_MAX_NEW_TOKENS
DEEP_ENABLE_THINKING = answer_policy_module.DEEP_ENABLE_THINKING
DEEP_CONTEXT_MAX_SECTIONS = answer_policy_module.DEEP_CONTEXT_MAX_SECTIONS
DEEP_CONTEXT_EXACT_HITS = answer_policy_module.DEEP_CONTEXT_EXACT_HITS
DEEP_PAGE_CONTEXT_SECTIONS = answer_policy_module.DEEP_PAGE_CONTEXT_SECTIONS


@dataclass(slots=True)
class IndexValidationResult:
    ready: bool
    error: str | None
    document_id: str | None
    chunk_count: int = 0
    feature_count: int = 0
    fact_count: int = 0


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
        return IndexValidationResult(
            False,
            "Bundled contract data is missing search features.",
            document_id,
            chunk_count=chunk_count,
            feature_count=feature_count,
        )
    if feature_count != chunk_count:
        return IndexValidationResult(
            False,
            "Bundled contract data is incomplete. Search features do not match the indexed chunks.",
            document_id,
            chunk_count=chunk_count,
            feature_count=feature_count,
        )
    fact_count = store.get_fact_count(document_id)
    if fact_count <= 0:
        return IndexValidationResult(
            False,
            "Bundled contract data is missing structured contract facts.",
            document_id,
            chunk_count=chunk_count,
            feature_count=feature_count,
            fact_count=fact_count,
        )
    return IndexValidationResult(
        True,
        None,
        document_id,
        chunk_count=chunk_count,
        feature_count=feature_count,
        fact_count=fact_count,
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
        self.retriever = HybridRetriever(self.store, HashingEmbedder())
        self.gemma = ManagedGemmaServiceClient()
        self.answer_policy = AnswerPolicy(self.store, self.retriever)

    def _get_answer_policy(self) -> AnswerPolicy:
        policy = getattr(self, "answer_policy", None)
        if policy is None:
            policy = AnswerPolicy(getattr(self, "store", None), getattr(self, "retriever", None))
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
        return AnswerPolicy.resolve_question(question, history)

    @classmethod
    def _find_follow_up_anchor(cls, history: Sequence[dict[str, str]]) -> str:
        return AnswerPolicy.find_follow_up_anchor(history)

    @staticmethod
    def _looks_like_follow_up(question: str, plan: QueryPlan) -> bool:
        return AnswerPolicy.looks_like_follow_up(question, plan)

    def _build_exact_answer(self, question: str, hits: list[ExactPageHit]) -> AssistantAnswer | None:
        return AnswerPolicy(getattr(self, "store", None), getattr(self, "retriever", None)).build_exact_answer(question, hits)

    @staticmethod
    def _is_count_question(question: str) -> bool:
        return AnswerPolicy.is_count_question(question)

    @staticmethod
    def _prefer_exact_answer(question: str) -> bool:
        return AnswerPolicy.prefer_exact_answer(question)

    @staticmethod
    def _reference_lookup_kind(question: str) -> str | None:
        return AnswerPolicy.reference_lookup_kind(question)

    @staticmethod
    def _count_exact_items(question: str, hits: list[ExactPageHit]) -> int | None:
        return AnswerPolicy.count_exact_items(question, hits)

    @classmethod
    def _extract_count_value(cls, question: str, hits: list[ExactPageHit]) -> str | None:
        return AnswerPolicy.extract_count_value(question, hits)

    @staticmethod
    def _count_item_phrase(question: str) -> str:
        return AnswerPolicy.count_item_phrase(question)

    @staticmethod
    def _count_value_patterns(item_phrase: str) -> list[re.Pattern[str]]:
        return AnswerPolicy.count_value_patterns(item_phrase)

    @staticmethod
    def _trim_page_excerpt(page_text: str, question: str, limit: int = 700) -> str:
        return AnswerPolicy.trim_page_excerpt(page_text, question, limit=limit)

    @classmethod
    def _build_reference_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        return AnswerPolicy.build_reference_answer(question, ranked, citations)

    @staticmethod
    def _limit_citations(citations: list[Citation], limit: int = 3) -> list[Citation]:
        return AnswerPolicy.limit_citations(citations, limit=limit)

    @classmethod
    def _build_extractive_answer(
        cls,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
        *,
        max_sections: int | None = None,
    ) -> AssistantAnswer | None:
        return AnswerPolicy.build_extractive_answer(question, ranked, citations, max_sections=max_sections)

    @staticmethod
    def _default_extractive_sections(plan: QueryPlan) -> int:
        return AnswerPolicy.default_extractive_sections(plan)

    def _build_compact_answer(
        self,
        question: str,
        ranked: list[RankedChunk],
        citations: list[Citation],
    ) -> AssistantAnswer | None:
        return AnswerPolicy(getattr(self, "store", None), getattr(self, "retriever", None)).build_compact_answer(
            question,
            ranked,
            citations,
        )

    @staticmethod
    def _prefers_generated_answer(question: str) -> bool:
        return AnswerPolicy.prefers_generated_answer(question)

    @classmethod
    def _build_summary_prompt_context(
        cls,
        question: str,
        ranked: list[RankedChunk],
        *,
        max_sections: int = SUMMARY_CONTEXT_MAX_SECTIONS,
    ) -> str:
        return AnswerPolicy.build_summary_prompt_context(question, ranked, max_sections=max_sections)

    @classmethod
    def _build_expand_prompt_context(
        cls,
        question: str,
        ranked: list[RankedChunk],
        previous_answer: str | None,
        *,
        max_sections: int = SUMMARY_CONTEXT_MAX_SECTIONS,
    ) -> str:
        return AnswerPolicy.build_expand_prompt_context(question, ranked, previous_answer, max_sections=max_sections)

    def _build_deep_prompt_context(
        self,
        question: str,
        ranked: list[RankedChunk],
        exact_hits: list[ExactPageHit],
    ) -> str:
        return AnswerPolicy(getattr(self, "store", None), getattr(self, "retriever", None)).build_deep_prompt_context(
            question,
            ranked,
            exact_hits,
        )

    @classmethod
    def _build_exact_hit_prompt_context(cls, question: str, exact_hits: list[ExactPageHit]) -> str:
        return AnswerPolicy.build_exact_hit_prompt_context(question, exact_hits)

    def _build_deep_page_context(self, ranked: list[RankedChunk]) -> str:
        return AnswerPolicy(getattr(self, "store", None), getattr(self, "retriever", None)).build_deep_page_context(ranked)

    @staticmethod
    def _format_page_range(page_start: int, page_end: int) -> str:
        return AnswerPolicy.format_page_range(page_start, page_end)

    @staticmethod
    def _truncate_text(text: str, limit: int = 340) -> str:
        return AnswerPolicy.truncate_text(text, limit=limit)

    @classmethod
    def _extract_ranked_excerpt(
        cls,
        text: str,
        plan: QueryPlan,
        *,
        limit: int = 650,
        surrounding_sentences: int = 0,
    ) -> str:
        return AnswerPolicy.extract_ranked_excerpt(
            text,
            plan,
            limit=limit,
            surrounding_sentences=surrounding_sentences,
        )

    @staticmethod
    def _is_value_or_requirement_question(plan: QueryPlan) -> bool:
        return AnswerPolicy.is_value_or_requirement_question(plan)

    @staticmethod
    def _is_value_question(plan: QueryPlan) -> bool:
        return AnswerPolicy.is_value_question(plan)

    @staticmethod
    def _is_requirement_question(plan: QueryPlan) -> bool:
        return AnswerPolicy.is_requirement_question(plan)

    @staticmethod
    def _quote_excerpt(text: str) -> str:
        return AnswerPolicy.quote_excerpt(text)

    @classmethod
    def _extract_term_window(cls, text: str, plan: QueryPlan, *, limit: int) -> str:
        return AnswerPolicy.extract_term_window(text, plan, limit=limit)

    @staticmethod
    def _is_useful_extractive_block(heading: str, excerpt: str, plan: QueryPlan) -> bool:
        return AnswerPolicy.is_useful_extractive_block(heading, excerpt, plan)

    @staticmethod
    def _excerpt_matches_attribute(lowered_excerpt: str, plan: QueryPlan) -> bool:
        return AnswerPolicy.excerpt_matches_attribute(lowered_excerpt, plan)

    @classmethod
    def _is_useful_summary_block(cls, chunk: RankedChunk, excerpt: str, plan: QueryPlan) -> bool:
        return AnswerPolicy.is_useful_summary_block(chunk, excerpt, plan)

    @staticmethod
    def _focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        return AnswerPolicy.focus_phrases(focus_terms)
