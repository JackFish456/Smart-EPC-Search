from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import uuid

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
import epc_smart_search.rebuild_contract as rebuild_contract_module
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractFactRow, ContractStore


def test_delete_artifacts_removes_sqlite_sidecars() -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"rebuild_contract_{uuid.uuid4().hex[:8]}"
    temp_root.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / "contract_store.db"
    for path in rebuild_contract_module._sqlite_artifact_paths(db_path):  # noqa: SLF001
        path.write_bytes(b"payload")

    removed = rebuild_contract_module._delete_artifacts(  # noqa: SLF001
        rebuild_contract_module._sqlite_artifact_paths(db_path)  # noqa: SLF001
    )

    assert {path.name for path in removed} == {
        "contract_store.db",
        "contract_store.db-journal",
        "contract_store.db-wal",
        "contract_store.db-shm",
    }
    assert all(not path.exists() for path in rebuild_contract_module._sqlite_artifact_paths(db_path))  # noqa: SLF001


def test_validate_rebuilt_database_reports_expected_counts() -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"rebuild_validate_{uuid.uuid4().hex[:8]}"
    temp_root.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / "rebuilt_contract.db"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="7.4.2",
        heading="Dew Point Heaters",
        full_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
        page_start=41,
        page_end=41,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    fact = ContractFactRow(
        document_id="doc1",
        system="Dew Point Heaters",
        system_normalized="dew point heaters",
        attribute="Configuration",
        attribute_normalized="configuration",
        value="4 x 50%",
        evidence_text=chunk.full_text,
        source_chunk_id=chunk.chunk_id,
        page_start=41,
        page_end=41,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc123",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=41, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        facts=[fact],
        embeddings={"chunk1": b"\x00\x00\x00\x00"},
        model_name="test",
        dimension=1,
    )

    summary = rebuild_contract_module.validate_rebuilt_database(
        db_path,
        expected_system="dew point heaters",
        expected_attribute="configuration",
    )

    assert summary.document_id == "doc1"
    assert summary.document_count == 1
    assert summary.page_count == 1
    assert summary.chunk_count == 1
    assert summary.feature_count == 1
    assert summary.fact_count == 1
    assert summary.embedding_count == 1
    assert summary.expected_values == ("4 x 50%",)


def test_run_smoke_queries_requires_fact_and_summary_routes(monkeypatch) -> None:
    exact_trace = SimpleNamespace(
        plan=SimpleNamespace(retrieval_mode="fact_lookup"),
        fact_lookup_attempted=True,
        fact_rows_returned=1,
        fallback_reason=None,
        fact_hit=object(),
        selected_bundle=SimpleNamespace(bundle_id="dew-point"),
    )
    summary_trace = SimpleNamespace(
        plan=SimpleNamespace(retrieval_mode="topic_summary"),
        fact_lookup_attempted=False,
        fact_rows_returned=0,
        fallback_reason="topic_summary_mode",
        fact_hit=None,
        selected_bundle=SimpleNamespace(bundle_id="ccw-summary"),
    )

    class FakeRetriever:
        def retrieve_trace(self, question: str):
            return exact_trace if "dew point" in question.lower() else summary_trace

    class FakeAnswerPolicy:
        def answer(self, question, history, gemma_client):
            return SimpleNamespace(text=f"answer for {question}", refused=False)

    class FakeAssistant:
        def __init__(self, db_path) -> None:
            self.retriever = FakeRetriever()
            self.answer_policy = FakeAnswerPolicy()

        def is_index_ready(self) -> bool:
            return True

        def get_index_status(self):
            return SimpleNamespace(error=None)

    monkeypatch.setattr(rebuild_contract_module, "ContractAssistant", FakeAssistant)

    exact_summary, broad_summary = rebuild_contract_module.run_smoke_queries(
        Path(tempfile.gettempdir()) / f"rebuild_smoke_{uuid.uuid4().hex[:8]}.db",
        exact_query="What is the configuration of the dew point heaters?",
        summary_query="Summarize the closed cooling water system",
    )

    assert exact_summary.retrieval_mode == "fact_lookup"
    assert exact_summary.fact_lookup_attempted is True
    assert exact_summary.fact_rows_returned == 1
    assert exact_summary.fallback_reason is None
    assert exact_summary.used_expected_path is True
    assert broad_summary.retrieval_mode == "topic_summary"
    assert broad_summary.fact_lookup_attempted is False
    assert broad_summary.fact_rows_returned == 0
    assert broad_summary.used_expected_path is True


def test_rebuild_entrypoint_populates_expected_contract_facts(monkeypatch) -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"rebuild_entrypoint_{uuid.uuid4().hex[:8]}"
    temp_root.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_root / "Contract.pdf"
    out_path = temp_root / "rebuilt_contract.db"
    pdf_path.write_text("stub", encoding="utf-8")

    pages = [
        PageText(
            page_num=12,
            text=(
                "1.1\n"
                "Dew Point Heaters\n"
                "Dew Point Heaters: 4 x 50%\n"
                "Configuration: 4 x 50%\n"
            ),
            ocr_used=False,
        ),
        PageText(
            page_num=13,
            text=(
                "1.2\n"
                "Closed Cooling Water Pumps\n"
                "Closed Cooling Water Pumps: 2 x 100%\n"
                "Configuration: 2 x 100%\n"
            ),
            ocr_used=False,
        ),
    ]

    class _FakeDoc:
        page_count = 2

        def __enter__(self) -> _FakeDoc:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("epc_smart_search.indexer.extract_pages", lambda _path: pages)
    monkeypatch.setattr("epc_smart_search.indexer.fitz.open", lambda _path: _FakeDoc())
    monkeypatch.setattr(
        rebuild_contract_module,
        "run_smoke_queries",
        lambda *args, **kwargs: (
            rebuild_contract_module.SmokeQuerySummary(
                label="exact",
                question="What is the configuration of the dew point heaters?",
                retrieval_mode="fact_lookup",
                fact_lookup_attempted=True,
                fact_rows_returned=1,
                fallback_reason=None,
                used_expected_path=True,
                selected_bundle_id="dew-point",
                answer_text="4 x 50%",
            ),
            rebuild_contract_module.SmokeQuerySummary(
                label="summary",
                question="Summarize the closed cooling water system",
                retrieval_mode="topic_summary",
                fact_lookup_attempted=False,
                fact_rows_returned=0,
                fallback_reason="topic_summary_mode",
                used_expected_path=True,
                selected_bundle_id="ccw",
                answer_text="summary",
            ),
        ),
    )

    exit_code = rebuild_contract_module.main(["--pdf", str(pdf_path), "--out", str(out_path)])

    assert exit_code == 0
    store = ContractStore(out_path)
    document = store.get_document()
    assert document is not None
    document_id = str(document["document_id"])
    assert store.get_fact_count(document_id) > 0
    dew_rows = store.lookup_facts_by_system_attribute(document_id, "dew point heaters", "configuration")
    ccw_rows = store.lookup_facts_by_system_attribute(document_id, "closed cooling water pumps", "configuration")
    assert [row.value for row in dew_rows] == ["4 x 50%"]
    assert [row.value for row in ccw_rows] == ["2 x 100%"]
