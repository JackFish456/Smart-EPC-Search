from __future__ import annotations

from types import SimpleNamespace

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.fact_extraction import extract_chunk_facts
from epc_smart_search.indexer import build_index
from epc_smart_search.ocr_support import PageText
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

    assert ("dew point heater", "configuration", "4 x 50%") in fact_map
    assert ("dew point heater", "configuration", "2 x 100%") in fact_map
    assert ("auxiliary boiler feed pump", "capacity", "500 gpm") in fact_map
    assert ("sampling pump", "quantity", "2") in fact_map
    assert ("turbine generator", "model", "Siemens SGT6-5000F") in fact_map
    assert ("fire water pump", "rating", "350 HP for fire water service") in fact_map
    assert ("fire water pump", "service", "fire water") in fact_map
    assert all(fact.page == 41 for fact in facts)
    assert all(fact.source_chunk_id == "chunk_fact_1" for fact in facts)


def test_build_index_persists_extracted_contract_facts(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "Contract.pdf"
    db_path = tmp_path / "contract.db"
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

    configuration_rows = store.lookup_facts_by_system_attribute(result["document_id"], "dew point heater", "configuration")
    service_rows = store.lookup_facts_by_system_attribute(result["document_id"], "dew point heater", "service")

    assert result["fact_count"] >= 2
    assert len(configuration_rows) == 1
    assert configuration_rows[0].value == "4 x 50%"
    assert len(service_rows) == 1
    assert service_rows[0].value == "Fuel gas dew point control"
    assert service_rows[0].page_start == 12
