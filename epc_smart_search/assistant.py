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
from epc_smart_search.retrieval import Citation, ExactPageHit, HashingEmbedder, HybridRetriever
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
        best = ranked[0]
        if best.total_score < 0.16 and best.lexical_score < 0.05:
            return AssistantAnswer("I can't verify that from the contract.", citations[:3], True)
        prompt_context = self.retriever.build_evidence_pack(question, ranked, citations[:4])
        try:
            answer_text = self.gemma.ask(question, prompt_context)
        except Exception:
            answer_text = "I can't verify that from the contract."
        answer_text = answer_text.strip() or "I can't verify that from the contract."
        refused = answer_text == "I can't verify that from the contract."
        return AssistantAnswer(answer_text, citations[:4], refused)

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
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
