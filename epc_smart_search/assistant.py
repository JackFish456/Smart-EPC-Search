from __future__ import annotations

import atexit
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import re

import requests

from epc_smart_search.app_paths import CONTRACT_PATH, DB_PATH, GEMMA_TEST_PYTHON, WORKSPACE_ROOT, seed_preloaded_db
from epc_smart_search.config import GEMMA_SERVICE_HOST, GEMMA_SERVICE_PORT, SEARCH_SCHEMA_VERSION
from epc_smart_search.indexer import build_index, refresh_query_index
from epc_smart_search.query_planner import QueryPlan, plan_query
from epc_smart_search.retrieval import Citation, ExactPageHit, HashingEmbedder, HybridRetriever, RankedChunk
from epc_smart_search.storage import ContractStore


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    citations: list[Citation]
    refused: bool


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

    def ask(self, question: str, context: str) -> str:
        self.ensure_running()
        response = requests.post(
            f"{self.base_url}/generate",
            json={"question": question, "context": context},
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
        self.gemma = GemmaServiceClient()
        self._expected_contract_hash = self._sha256(CONTRACT_PATH) if CONTRACT_PATH.exists() else ""
        self._ensure_query_index_ready()

    def is_index_ready(self) -> bool:
        document = self.store.get_document()
        if document is None:
            return False
        if str(document["sha256"]) != self._expected_contract_hash:
            return False
        if self.store.get_metadata("search_schema_version") != str(SEARCH_SCHEMA_VERSION):
            return False
        return self.store.get_stats()["chunk_count"] > 0 and self.store.get_feature_count(str(document["document_id"])) > 0

    def build_index(self, progress_callback=None) -> dict[str, int | str]:
        result = build_index(pdf_path=CONTRACT_PATH, db_path=self.store.db_path, progress_callback=progress_callback)
        self._ensure_query_index_ready(progress_callback=progress_callback)
        return result

    def ask(self, question: str) -> AssistantAnswer:
        exact_hits = self.retriever.find_exact_page_hits(question)
        exact_answer = self._build_exact_answer(question, exact_hits)
        if exact_answer is not None:
            return exact_answer
        ranked = self.retriever.retrieve(question)
        if not ranked:
            return AssistantAnswer("I can't verify that from the contract.", [], True)
        citations = self.retriever.expand_with_context(ranked)
        if not citations:
            return AssistantAnswer("I can't verify that from the contract.", [], True)
        citations = self._limit_citations(citations)
        best = ranked[0]
        if best.total_score < 0.16 and best.lexical_score < 0.05:
            return AssistantAnswer("I can't verify that from the contract.", citations, True)
        extractive_answer = self._build_extractive_answer(question, ranked, citations)
        if extractive_answer is not None and not self._prefers_generated_answer(question):
            return extractive_answer
        prompt_context = self.retriever.build_evidence_pack(question, ranked, citations)
        try:
            answer_text = self.gemma.ask(question, prompt_context)
        except Exception:
            if extractive_answer is not None:
                return extractive_answer
            answer_text = "I can't verify that from the contract."
        answer_text = answer_text.strip() or "I can't verify that from the contract."
        refused = answer_text == "I can't verify that from the contract."
        return AssistantAnswer(answer_text, citations, refused)

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

    def _ensure_query_index_ready(self, progress_callback=None) -> None:
        document = self.store.get_document()
        if document is None:
            return
        document_id = str(document["document_id"])
        if self.store.get_feature_count(document_id) > 0 and self.store.get_metadata("search_schema_version") == str(SEARCH_SCHEMA_VERSION):
            return
        refresh_query_index(self.store, document_id, progress_callback=progress_callback)

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

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

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
        intro = "Most relevant contract text:"
        return AssistantAnswer(f"{intro}\n\n" + "\n\n".join(blocks), citations, False)

    @staticmethod
    def _prefers_generated_answer(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        prompts = (
            "summarize",
            "summarise",
            "explain",
            "plain english",
            "plain language",
            "overview",
            "compare",
            "difference between",
        )
        return any(prompt in normalized for prompt in prompts)

    @staticmethod
    def _format_page_range(page_start: int, page_end: int) -> str:
        if page_start == page_end:
            return f"{page_start}"
        return f"{page_start}-{page_end}"

    @classmethod
    def _extract_ranked_excerpt(cls, text: str, plan: QueryPlan, *, limit: int = 650) -> str:
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
        selected = [sentences[best_index]]
        if cls._is_value_or_requirement_question(plan):
            if best_index + 1 < len(sentences):
                selected.append(sentences[best_index + 1])
            elif best_index > 0:
                selected.insert(0, sentences[best_index - 1])
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

    @staticmethod
    def _focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
