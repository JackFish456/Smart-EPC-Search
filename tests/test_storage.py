from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractFactRow, ContractStore


def _seed_store(db_path: str = "file:storage_contract?mode=memory&cache=shared") -> tuple[ContractStore, ChunkRecord]:
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="14.2.1",
        heading="Liquidated Damages",
        full_text="Liquidated damages apply when delay occurs.",
        page_start=10,
        page_end=10,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=10, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        embeddings={"chunk1": b"\x00\x00\x00\x00"},
        model_name="test",
        dimension=1,
    )
    return store, chunk


def test_fts_triggers_stay_in_sync() -> None:
    store, _chunk = _seed_store()
    rows = store.search_fts("doc1", '"liquidated"', limit=5)
    assert rows
    assert rows[0]["chunk_id"] == "chunk1"

    feature_rows = store.search_chunk_feature_fts("doc1", 'topic_tags : "liquidated damages" OR heading : "liquidated"', limit=5)
    assert feature_rows
    assert feature_rows[0]["chunk_id"] == "chunk1"
    assert store.get_metadata("search_schema_version") is not None


def test_contract_facts_schema_created() -> None:
    store, _chunk = _seed_store("file:storage_contract_schema?mode=memory&cache=shared")

    with store._connect() as connection:
        columns = {
            str(row["name"]): str(row["type"])
            for row in connection.execute("PRAGMA table_info(contract_facts)").fetchall()
        }
        index_names = {
            str(row["name"])
            for row in connection.execute("PRAGMA index_list(contract_facts)").fetchall()
        }

    assert columns["document_id"] == "TEXT"
    assert columns["system_normalized"] == "TEXT"
    assert columns["attribute_normalized"] == "TEXT"
    assert columns["source_chunk_id"] == "TEXT"
    assert "idx_contract_facts_doc_system_attr" in index_names
    assert "idx_contract_facts_doc_system" in index_names


def test_insert_and_fetch_contract_facts() -> None:
    store, chunk = _seed_store("file:storage_contract_facts?mode=memory&cache=shared")
    inserted = store.insert_contract_facts(
        [
            ContractFactRow(
                document_id="doc1",
                system="Dew Point Heaters",
                attribute="Configuration",
                value="4 x 50%",
                evidence_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            ),
            ContractFactRow(
                document_id="doc1",
                system="Dew Point Heaters",
                attribute="Duty",
                value="Standby",
                evidence_text="One heater train shall remain in standby service.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            ),
        ]
    )

    by_attribute = store.lookup_facts_by_system_attribute("doc1", "Dew Point Heaters", "Configuration")
    by_system = store.lookup_facts_by_system("doc1", "Dew Point Heaters")

    assert inserted == 2
    assert len(by_attribute) == 1
    assert by_attribute[0].value == "4 x 50%"
    assert by_attribute[0].evidence_text.startswith("Dew point heaters shall be furnished")
    assert by_attribute[0].source_chunk_id == chunk.chunk_id
    assert [fact.attribute for fact in by_system] == ["Configuration", "Duty"]


def test_contract_fact_lookup_uses_normalized_matching() -> None:
    store, chunk = _seed_store("file:storage_contract_normalized?mode=memory&cache=shared")
    store.insert_contract_facts(
        [
            ContractFactRow(
                document_id="doc1",
                system="Dew-Point Heaters",
                attribute="Configuration / Arrangement",
                value="4 x 50%",
                evidence_text="Dew-point heaters shall be arranged as 4 x 50%.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            )
        ]
    )

    rows = store.lookup_facts_by_system_attribute("doc1", " dew point heaters ", "configuration arrangement")

    assert len(rows) == 1
    assert rows[0].system_normalized == "dew point heater"
    assert rows[0].attribute_normalized == "configuration"


def test_contract_fact_lookup_supports_explicit_system_aliases_without_collisions() -> None:
    store, chunk = _seed_store("file:storage_contract_aliases?mode=memory&cache=shared")
    store.insert_contract_facts(
        [
            ContractFactRow(
                document_id="doc1",
                system="Closed Cooling Water System",
                attribute="Configuration / Arrangement",
                value="2 x 100%",
                evidence_text="The closed cooling water system shall be arranged as 2 x 100%.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            )
        ]
    )

    alias_rows = store.lookup_facts_by_system_attribute("doc1", "CCW", "configuration")
    unrelated_rows = store.lookup_facts_by_system_attribute("doc1", "cooling water", "configuration")

    assert len(alias_rows) == 1
    assert alias_rows[0].system_normalized == "closed cooling water"
    assert alias_rows[0].attribute_normalized == "configuration"
    assert unrelated_rows == []


def test_contract_fact_inserts_are_duplicate_safe() -> None:
    store, chunk = _seed_store("file:storage_contract_dupes?mode=memory&cache=shared")
    first_inserted = store.insert_contract_facts(
        [
            ContractFactRow(
                document_id="doc1",
                system="Dew Point Heaters",
                attribute="Configuration",
                value="4 x 50%",
                evidence_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            )
        ]
    )
    second_inserted = store.insert_contract_facts(
        [
            ContractFactRow(
                document_id="doc1",
                system="dew-point heaters",
                attribute="configuration",
                value="4 x 50%",
                evidence_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
                source_chunk_id=chunk.chunk_id,
                page_start=10,
                page_end=10,
            )
        ]
    )

    rows = store.lookup_facts_by_system_attribute("doc1", "Dew Point Heaters", "Configuration")

    assert first_inserted == 1
    assert second_inserted == 0
    assert len(rows) == 1
