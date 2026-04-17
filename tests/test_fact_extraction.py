from __future__ import annotations

from pathlib import Path

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.fact_extraction import extract_chunk_facts
from epc_smart_search.indexer import build_index, refresh_query_index
from epc_smart_search.ocr_support import PageText
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore


def test_extract_chunk_facts_handles_configuration_rows_and_buried_prose() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk_fact_1",
        document_id="doc1",
        chunk_type="section",
        section_number="7.4.2",
        heading="Dew Point Heaters",
        full_text=(
            "Configuration: 4 x 50%\n"
            "Dew Point Heaters | Configuration | 2 x 100%\n"
            "Auxiliary Boiler Feed Pump | Capacity | 500 gpm\n"
            "Sampling Pumps | Quantity | 2\n"
            "The selected turbine generator model shall be Siemens SGT6-5000F.\n"
            "Each fire water pump shall be rated at 350 HP for fire water service."
        ),
        page_start=41,
        page_end=41,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )

    facts = extract_chunk_facts(chunk)
    fact_map = {(fact.normalized_system, fact.normalized_attribute, fact.raw_value): fact for fact in facts}

    assert ("dew point heaters", "configuration", "4 x 50%") in fact_map
    assert ("dew point heaters", "configuration", "2 x 100%") in fact_map
    assert ("auxiliary boiler feed pump", "capacity", "500 gpm") in fact_map
    assert ("sampling pumps", "quantity", "2") in fact_map
    assert ("turbine generator", "model", "Siemens SGT6-5000F") in fact_map
    assert ("fire water pump", "rating", "350 HP") in fact_map
    assert ("fire water pump", "service", "fire water") in fact_map
    assert all(fact.page == 41 for fact in facts)
    assert all(fact.page_end == 41 for fact in facts)
    assert all(fact.source_chunk_id == "chunk_fact_1" for fact in facts)


def test_build_index_persists_extracted_contract_facts(monkeypatch) -> None:
    temp_dir = Path(".tmp_test")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / "ContractFactsTest.pdf"
    db_path = "file:contract_facts_test?mode=memory&cache=shared"
    pdf_path.write_text("stub", encoding="utf-8")

    pages = [
        PageText(
            page_num=12,
            text=(
                "1.1\n"
                "Dew Point Heaters\n"
                "Configuration: 4 x 50%\n"
                "Service: Fuel gas dew point control\n"
            ),
            ocr_used=False,
        )
    ]

    class _FakeDoc:
        page_count = 1

        def __enter__(self) -> _FakeDoc:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("epc_smart_search.indexer.extract_pages", lambda _path: pages)
    monkeypatch.setattr("epc_smart_search.indexer.fitz.open", lambda _path: _FakeDoc())

    result = build_index(pdf_path=pdf_path, db_path=db_path, version_label="test")
    store = ContractStore(db_path)

    configuration_rows = store.lookup_facts_by_system_attribute(result["document_id"], "dew point heaters", "configuration")
    service_rows = store.lookup_facts_by_system_attribute(result["document_id"], "dew point heaters", "service")

    assert result["fact_count"] >= 2
    assert len(configuration_rows) == 1
    assert configuration_rows[0].value == "4 x 50%"
    assert len(service_rows) == 1
    assert service_rows[0].value == "Fuel gas dew point control"
    assert service_rows[0].page_start == 12


def test_refresh_query_index_rebuilds_contract_facts() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk_fact_refresh",
        document_id="doc1",
        chunk_type="section",
        section_number="8.4.2",
        heading="Fire Water Pump",
        full_text="Each fire water pump shall be rated at 350 HP for the project fire water service.",
        page_start=412,
        page_end=413,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    store = ContractStore("file:fact_refresh?mode=memory&cache=shared")
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=2,
        chunks=[chunk],
        pages=[PageText(page_num=412, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        facts=[],
        embeddings={"chunk_fact_refresh": b"\x00\x00\x00\x00"},
        model_name="test",
        dimension=1,
    )

    refresh_query_index(store, "doc1")
    power_rows = store.lookup_facts_by_system_attribute("doc1", "fire water pump", "rating")

    assert len(power_rows) == 1
    assert power_rows[0].value == "350 HP"
    assert power_rows[0].page_start == 412
    assert power_rows[0].page_end == 413
