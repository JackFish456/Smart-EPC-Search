from __future__ import annotations

import sqlite3
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.config import SEARCH_SCHEMA_VERSION
from epc_smart_search.ocr_support import PageText, build_page_diagnostics, segment_text_blocks
from epc_smart_search.search_features import ChunkFeatures, alias_terms_for_text, normalize_text


@dataclass(slots=True)
class ContractBlockRecord:
    block_id: str
    document_id: str
    page_num: int
    block_ordinal: int
    block_type: str
    block_text: str
    normalized_text: str
    alias_text: str
    parent_chunk_id: str | None
    noise_flags: str = ""


@dataclass(slots=True)
class PageIngestDiagnosticRecord:
    document_id: str
    page_num: int
    meaningful_chars: int
    word_count: int
    block_count: int
    short_line_count: int
    flags: str = ""

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    version_label TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contract_chunks (
    chunk_rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('article','section','subsection','exhibit','definition')),
    section_number TEXT,
    heading TEXT NOT NULL DEFAULT '',
    full_text TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    parent_chunk_id TEXT,
    ordinal_in_document INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id),
    FOREIGN KEY (parent_chunk_id) REFERENCES contract_chunks(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_parent
ON contract_chunks(parent_chunk_id);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_pages
ON contract_chunks(document_id, page_start, page_end);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_doc_ordinal
ON contract_chunks(document_id, ordinal_in_document);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_doc_section_ordinal
ON contract_chunks(document_id, section_number, ordinal_in_document);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    vector_rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES contract_chunks(chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc
ON chunk_vectors(document_id, chunk_id);

CREATE TABLE IF NOT EXISTS contract_pages (
    page_rowid INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    page_text TEXT NOT NULL,
    ocr_used INTEGER NOT NULL DEFAULT 0,
    UNIQUE(document_id, page_num),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_contract_pages_doc_page
ON contract_pages(document_id, page_num);

CREATE TABLE IF NOT EXISTS contract_blocks (
    block_rowid INTEGER PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    block_ordinal INTEGER NOT NULL,
    block_type TEXT NOT NULL CHECK (block_type IN ('heading','paragraph','list_item','table_row')),
    block_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL DEFAULT '',
    alias_text TEXT NOT NULL DEFAULT '',
    parent_chunk_id TEXT,
    noise_flags TEXT NOT NULL DEFAULT '',
    UNIQUE(document_id, page_num, block_ordinal),
    FOREIGN KEY (document_id) REFERENCES documents(document_id),
    FOREIGN KEY (parent_chunk_id) REFERENCES contract_chunks(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_contract_blocks_doc_page_ordinal
ON contract_blocks(document_id, page_num, block_ordinal);

CREATE INDEX IF NOT EXISTS idx_contract_blocks_parent
ON contract_blocks(parent_chunk_id);

CREATE TABLE IF NOT EXISTS page_ingest_diagnostics (
    diagnostic_rowid INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    meaningful_chars INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    block_count INTEGER NOT NULL,
    short_line_count INTEGER NOT NULL DEFAULT 0,
    flags TEXT NOT NULL DEFAULT '',
    UNIQUE(document_id, page_num),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_page_ingest_diagnostics_doc_page
ON page_ingest_diagnostics(document_id, page_num);

CREATE TABLE IF NOT EXISTS chunk_search_features (
    feature_rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    section_number TEXT,
    heading TEXT NOT NULL DEFAULT '',
    parent_heading TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT '',
    normalized_text TEXT NOT NULL DEFAULT '',
    rescue_text TEXT NOT NULL DEFAULT '',
    clause_type TEXT NOT NULL DEFAULT '',
    actor_tags TEXT NOT NULL DEFAULT '',
    action_tags TEXT NOT NULL DEFAULT '',
    topic_tags TEXT NOT NULL DEFAULT '',
    noise_flags TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (chunk_id) REFERENCES contract_chunks(chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_search_features_doc
ON chunk_search_features(document_id, chunk_id);

CREATE VIRTUAL TABLE IF NOT EXISTS contract_pages_fts USING fts5(
    page_text,
    document_id UNINDEXED,
    page_num UNINDEXED,
    content='contract_pages',
    content_rowid='page_rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search_fts USING fts5(
    section_number,
    heading,
    parent_heading,
    search_text,
    actor_tags,
    action_tags,
    topic_tags,
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content='chunk_search_features',
    content_rowid='feature_rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search_rescue_fts USING fts5(
    section_number,
    heading,
    parent_heading,
    rescue_text,
    actor_tags,
    action_tags,
    topic_tags,
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content='chunk_search_features',
    content_rowid='feature_rowid',
    tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS contract_blocks_fts USING fts5(
    block_text,
    normalized_text,
    alias_text,
    block_type,
    noise_flags,
    parent_chunk_id UNINDEXED,
    document_id UNINDEXED,
    page_num UNINDEXED,
    content='contract_blocks',
    content_rowid='block_rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS contract_blocks_rescue_fts USING fts5(
    block_text,
    normalized_text,
    alias_text,
    block_type,
    noise_flags,
    parent_chunk_id UNINDEXED,
    document_id UNINDEXED,
    page_num UNINDEXED,
    content='contract_blocks',
    content_rowid='block_rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS contract_pages_ai AFTER INSERT ON contract_pages BEGIN
  INSERT INTO contract_pages_fts(rowid, page_text, document_id, page_num)
  VALUES (new.page_rowid, new.page_text, new.document_id, new.page_num);
END;

CREATE TRIGGER IF NOT EXISTS contract_pages_ad AFTER DELETE ON contract_pages BEGIN
  INSERT INTO contract_pages_fts(contract_pages_fts, rowid, page_text, document_id, page_num)
  VALUES ('delete', old.page_rowid, old.page_text, old.document_id, old.page_num);
END;

CREATE TRIGGER IF NOT EXISTS contract_pages_au AFTER UPDATE ON contract_pages BEGIN
  INSERT INTO contract_pages_fts(contract_pages_fts, rowid, page_text, document_id, page_num)
  VALUES ('delete', old.page_rowid, old.page_text, old.document_id, old.page_num);

  INSERT INTO contract_pages_fts(rowid, page_text, document_id, page_num)
  VALUES (new.page_rowid, new.page_text, new.document_id, new.page_num);
END;

CREATE TRIGGER IF NOT EXISTS contract_blocks_ai AFTER INSERT ON contract_blocks BEGIN
  INSERT INTO contract_blocks_fts(
    rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    new.block_rowid, new.block_text, new.normalized_text, new.alias_text, new.block_type, new.noise_flags,
    new.parent_chunk_id, new.document_id, new.page_num
  );

  INSERT INTO contract_blocks_rescue_fts(
    rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    new.block_rowid, new.block_text, new.normalized_text, new.alias_text, new.block_type, new.noise_flags,
    new.parent_chunk_id, new.document_id, new.page_num
  );
END;

CREATE TRIGGER IF NOT EXISTS contract_blocks_ad AFTER DELETE ON contract_blocks BEGIN
  INSERT INTO contract_blocks_fts(
    contract_blocks_fts, rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    'delete', old.block_rowid, old.block_text, old.normalized_text, old.alias_text, old.block_type, old.noise_flags,
    old.parent_chunk_id, old.document_id, old.page_num
  );

  INSERT INTO contract_blocks_rescue_fts(
    contract_blocks_rescue_fts, rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    'delete', old.block_rowid, old.block_text, old.normalized_text, old.alias_text, old.block_type, old.noise_flags,
    old.parent_chunk_id, old.document_id, old.page_num
  );
END;

CREATE TRIGGER IF NOT EXISTS contract_blocks_au AFTER UPDATE ON contract_blocks BEGIN
  INSERT INTO contract_blocks_fts(
    contract_blocks_fts, rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    'delete', old.block_rowid, old.block_text, old.normalized_text, old.alias_text, old.block_type, old.noise_flags,
    old.parent_chunk_id, old.document_id, old.page_num
  );

  INSERT INTO contract_blocks_rescue_fts(
    contract_blocks_rescue_fts, rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    'delete', old.block_rowid, old.block_text, old.normalized_text, old.alias_text, old.block_type, old.noise_flags,
    old.parent_chunk_id, old.document_id, old.page_num
  );

  INSERT INTO contract_blocks_fts(
    rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    new.block_rowid, new.block_text, new.normalized_text, new.alias_text, new.block_type, new.noise_flags,
    new.parent_chunk_id, new.document_id, new.page_num
  );

  INSERT INTO contract_blocks_rescue_fts(
    rowid, block_text, normalized_text, alias_text, block_type, noise_flags, parent_chunk_id, document_id, page_num
  ) VALUES (
    new.block_rowid, new.block_text, new.normalized_text, new.alias_text, new.block_type, new.noise_flags,
    new.parent_chunk_id, new.document_id, new.page_num
  );
END;

CREATE TRIGGER IF NOT EXISTS chunk_search_features_ai AFTER INSERT ON chunk_search_features BEGIN
  INSERT INTO chunk_search_fts(
    rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.search_text,
    new.actor_tags, new.action_tags, new.topic_tags, new.chunk_id, new.document_id
  );

  INSERT INTO chunk_search_rescue_fts(
    rowid, section_number, heading, parent_heading, rescue_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.rescue_text,
    new.actor_tags, new.action_tags, new.topic_tags, new.chunk_id, new.document_id
  );
END;

CREATE TRIGGER IF NOT EXISTS chunk_search_features_ad AFTER DELETE ON chunk_search_features BEGIN
  INSERT INTO chunk_search_fts(
    chunk_search_fts, rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    'delete', old.feature_rowid, old.section_number, old.heading, old.parent_heading, old.search_text,
    old.actor_tags, old.action_tags, old.topic_tags, old.chunk_id, old.document_id
  );

  INSERT INTO chunk_search_rescue_fts(
    chunk_search_rescue_fts, rowid, section_number, heading, parent_heading, rescue_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    'delete', old.feature_rowid, old.section_number, old.heading, old.parent_heading, old.rescue_text,
    old.actor_tags, old.action_tags, old.topic_tags, old.chunk_id, old.document_id
  );
END;

CREATE TRIGGER IF NOT EXISTS chunk_search_features_au AFTER UPDATE ON chunk_search_features BEGIN
  INSERT INTO chunk_search_fts(
    chunk_search_fts, rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    'delete', old.feature_rowid, old.section_number, old.heading, old.parent_heading, old.search_text,
    old.actor_tags, old.action_tags, old.topic_tags, old.chunk_id, old.document_id
  );

  INSERT INTO chunk_search_rescue_fts(
    chunk_search_rescue_fts, rowid, section_number, heading, parent_heading, rescue_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    'delete', old.feature_rowid, old.section_number, old.heading, old.parent_heading, old.rescue_text,
    old.actor_tags, old.action_tags, old.topic_tags, old.chunk_id, old.document_id
  );

  INSERT INTO chunk_search_fts(
    rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.search_text,
    new.actor_tags, new.action_tags, new.topic_tags, new.chunk_id, new.document_id
  );

  INSERT INTO chunk_search_rescue_fts(
    rowid, section_number, heading, parent_heading, rescue_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.rescue_text,
    new.actor_tags, new.action_tags, new.topic_tags, new.chunk_id, new.document_id
  );
END;
"""


class ContractStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._use_uri = self.db_path.startswith("file:")
        self._keepalive_connection: sqlite3.Connection | None = None
        if not self._use_uri and self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        elif "mode=memory" in self.db_path or self.db_path == ":memory:":
            self._keepalive_connection = sqlite3.connect(
                self.db_path,
                uri=self._use_uri,
                check_same_thread=False,
            )
            self._keepalive_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._keepalive_connection is not None:
            if self.db_path == ":memory:":
                return self._keepalive_connection
            connection = sqlite3.connect(
                self.db_path,
                uri=self._use_uri,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            return connection
        connection = sqlite3.connect(self.db_path, uri=self._use_uri)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            _ensure_compat_columns(connection)
            _backfill_block_coverage(connection)
            connection.execute(
                """
                INSERT INTO app_metadata (key, value)
                VALUES ('search_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SEARCH_SCHEMA_VERSION),),
            )
            connection.commit()

    def replace_document(
        self,
        *,
        document_id: str,
        display_name: str,
        version_label: str,
        file_path: str,
        sha256: str,
        page_count: int,
        chunks: list[ChunkRecord],
        pages: list[PageText],
        features: list[ChunkFeatures],
        blocks: list[ContractBlockRecord] | None = None,
        diagnostics: list[PageIngestDiagnosticRecord] | None = None,
        embeddings: dict[str, bytes] | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        blocks = blocks or build_block_records(document_id, pages, chunks)
        diagnostics = diagnostics or build_diagnostic_records(document_id, pages)
        with self._connect() as connection:
            connection.execute("DELETE FROM page_ingest_diagnostics WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM contract_blocks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM chunk_vectors WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM chunk_search_features WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM contract_chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM contract_pages WHERE document_id = ?", (document_id,))
            connection.execute(
                """
                INSERT INTO documents (document_id, display_name, version_label, file_path, sha256, page_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    version_label = excluded.version_label,
                    file_path = excluded.file_path,
                    sha256 = excluded.sha256,
                    page_count = excluded.page_count
                """,
                (document_id, display_name, version_label, file_path, sha256, page_count),
            )
            connection.executemany(
                """
                INSERT INTO contract_chunks (
                    chunk_id, document_id, chunk_type, section_number, heading,
                    full_text, page_start, page_end, parent_chunk_id, ordinal_in_document
                ) VALUES (
                    :chunk_id, :document_id, :chunk_type, :section_number, :heading,
                    :full_text, :page_start, :page_end, :parent_chunk_id, :ordinal_in_document
                )
                """,
                [asdict(chunk) for chunk in chunks],
            )
            connection.executemany(
                """
                INSERT INTO contract_pages (document_id, page_num, page_text, ocr_used)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (document_id, page.page_num, page.text, int(page.ocr_used))
                    for page in pages
                ],
            )
            self._insert_blocks(connection, blocks)
            self._insert_diagnostics(connection, diagnostics)
            self._insert_features(connection, features)
            self._insert_embeddings(connection, document_id, embeddings, model_name, dimension)
            connection.execute(
                """
                INSERT INTO app_metadata (key, value)
                VALUES ('search_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SEARCH_SCHEMA_VERSION),),
            )
            connection.commit()

    @staticmethod
    def _insert_blocks(connection: sqlite3.Connection, blocks: list[ContractBlockRecord]) -> None:
        if not blocks:
            return
        connection.executemany(
            """
            INSERT INTO contract_blocks (
                block_id, document_id, page_num, block_ordinal, block_type,
                block_text, normalized_text, alias_text, parent_chunk_id, noise_flags
            ) VALUES (
                :block_id, :document_id, :page_num, :block_ordinal, :block_type,
                :block_text, :normalized_text, :alias_text, :parent_chunk_id, :noise_flags
            )
            """,
            [asdict(block) for block in blocks],
        )

    @staticmethod
    def _insert_diagnostics(connection: sqlite3.Connection, diagnostics: list[PageIngestDiagnosticRecord]) -> None:
        if not diagnostics:
            return
        connection.executemany(
            """
            INSERT INTO page_ingest_diagnostics (
                document_id, page_num, meaningful_chars, word_count, block_count,
                short_line_count, flags
            ) VALUES (
                :document_id, :page_num, :meaningful_chars, :word_count, :block_count,
                :short_line_count, :flags
            )
            """,
            [asdict(diagnostic) for diagnostic in diagnostics],
        )

    @staticmethod
    def _insert_features(connection: sqlite3.Connection, features: list[ChunkFeatures]) -> None:
        connection.executemany(
            """
            INSERT INTO chunk_search_features (
                chunk_id, document_id, section_number, heading, parent_heading,
                search_text, normalized_text, rescue_text, clause_type, actor_tags,
                action_tags, topic_tags, noise_flags
            ) VALUES (
                :chunk_id, :document_id, :section_number, :heading, :parent_heading,
                :search_text, :normalized_text, :rescue_text, :clause_type, :actor_tags,
                :action_tags, :topic_tags, :noise_flags
            )
            """,
            [
                {
                    **asdict(feature),
                    "normalized_text": feature.rescue_text,
                }
                for feature in features
            ],
        )

    @staticmethod
    def _insert_embeddings(
        connection: sqlite3.Connection,
        document_id: str,
        embeddings: dict[str, bytes] | None,
        model_name: str | None,
        dimension: int | None,
    ) -> None:
        if not embeddings or not model_name or not dimension:
            return
        connection.executemany(
            """
            INSERT INTO chunk_vectors (
                chunk_id, document_id, model_name, dimension, vector
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (chunk_id, document_id, model_name, dimension, vector)
                for chunk_id, vector in embeddings.items()
                if vector
            ],
        )

    def replace_search_features(self, document_id: str, features: list[ChunkFeatures]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunk_search_features WHERE document_id = ?", (document_id,))
            self._insert_features(connection, features)
            connection.execute(
                """
                INSERT INTO app_metadata (key, value)
                VALUES ('search_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SEARCH_SCHEMA_VERSION),),
            )
            connection.commit()

    def replace_chunk_embeddings(
        self,
        document_id: str,
        embeddings: dict[str, bytes] | None,
        *,
        model_name: str | None,
        dimension: int | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunk_vectors WHERE document_id = ?", (document_id,))
            self._insert_embeddings(connection, document_id, embeddings, model_name, dimension)
            connection.execute(
                """
                INSERT INTO app_metadata (key, value)
                VALUES ('search_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SEARCH_SCHEMA_VERSION),),
            )
            connection.commit()

    def get_metadata(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM app_metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def get_document(self) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT document_id, display_name, version_label, file_path, sha256, page_count FROM documents ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

    def get_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM contract_chunks) AS chunk_count,
                    (SELECT COUNT(*) FROM contract_blocks) AS block_count,
                    COUNT(DISTINCT document_id) AS document_count
                FROM contract_chunks
                """
            ).fetchone()
        return {
            "chunk_count": int(row["chunk_count"]) if row else 0,
            "block_count": int(row["block_count"]) if row else 0,
            "document_count": int(row["document_count"]) if row else 0,
        }

    def get_feature_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS feature_count FROM chunk_search_features WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["feature_count"]) if row else 0

    def get_chunk_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS chunk_count FROM contract_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["chunk_count"]) if row else 0

    def get_block_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS block_count FROM contract_blocks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["block_count"]) if row else 0

    def get_page_text_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS page_count FROM contract_pages WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["page_count"]) if row else 0

    def get_embedding_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS embedding_count FROM chunk_vectors WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["embedding_count"]) if row else 0

    def get_embedding_metadata(self, document_id: str) -> tuple[str | None, int | None]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_name, dimension
                FROM chunk_vectors
                WHERE document_id = ?
                ORDER BY vector_rowid
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None, None
        return str(row["model_name"]), int(row["dimension"])

    def get_ingest_diagnostic_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS diagnostic_count FROM page_ingest_diagnostics WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["diagnostic_count"]) if row else 0

    def get_ingest_diagnostics(self, document_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT page_num, meaningful_chars, word_count, block_count, short_line_count, flags
                FROM page_ingest_diagnostics
                WHERE document_id = ?
                ORDER BY page_num
                """,
                (document_id,),
            ).fetchall()

    def get_ingest_diagnostic_summary(self, document_id: str) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS page_count,
                    SUM(CASE WHEN instr(flags, 'ocr_used') > 0 THEN 1 ELSE 0 END) AS ocr_pages,
                    SUM(CASE WHEN instr(flags, 'low_text_density') > 0 THEN 1 ELSE 0 END) AS low_text_density_pages,
                    SUM(CASE WHEN instr(flags, 'schedule_like') > 0 THEN 1 ELSE 0 END) AS schedule_like_pages,
                    SUM(CASE WHEN instr(flags, 'table_like') > 0 THEN 1 ELSE 0 END) AS table_like_pages,
                    SUM(CASE WHEN instr(flags, 'many_short_fragments') > 0 THEN 1 ELSE 0 END) AS fragmented_pages
                FROM page_ingest_diagnostics
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return {
                "page_count": 0,
                "ocr_pages": 0,
                "low_text_density_pages": 0,
                "schedule_like_pages": 0,
                "table_like_pages": 0,
                "fragmented_pages": 0,
            }
        return {key: int(row[key]) for key in row.keys()}

    def section_lookup(self, document_id: str, section_number: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ? AND c.section_number = ?
                ORDER BY c.ordinal_in_document
                """,
                (document_id, section_number),
            ).fetchall()

    def heading_lookup(self, document_id: str, heading_query: str, limit: int = 8) -> list[sqlite3.Row]:
        normalized = " ".join(heading_query.lower().split()).strip()
        if not normalized:
            return []
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    f.rescue_text
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ?
                  AND (
                    lower(c.heading) = ?
                    OR lower(f.parent_heading) = ?
                  )
                ORDER BY c.ordinal_in_document
                LIMIT ?
                """,
                (document_id, normalized, normalized, limit),
            ).fetchall()

    def search_fts(self, document_id: str, match_query: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.search_chunk_feature_fts(document_id, match_query, limit=limit)

    def search_chunk_feature_fts(self, document_id: str, match_query: str, limit: int = 24) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    f.search_text,
                    f.rescue_text,
                    rank AS rank_score,
                    snippet(chunk_search_fts, 3, '[', ']', '...', 18) AS hit_snippet
                FROM chunk_search_fts
                JOIN chunk_search_features f ON f.feature_rowid = chunk_search_fts.rowid
                JOIN contract_chunks c ON c.chunk_id = f.chunk_id
                WHERE f.document_id = ?
                  AND chunk_search_fts MATCH ?
                  AND rank MATCH 'bm25(10.0, 7.0, 4.0, 1.2, 4.0, 4.0, 4.0)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

    def search_chunk_feature_rescue_fts(self, document_id: str, match_query: str, limit: int = 24) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    f.search_text,
                    f.rescue_text,
                    rank AS rank_score
                FROM chunk_search_rescue_fts
                JOIN chunk_search_features f ON f.feature_rowid = chunk_search_rescue_fts.rowid
                JOIN contract_chunks c ON c.chunk_id = f.chunk_id
                WHERE f.document_id = ?
                  AND chunk_search_rescue_fts MATCH ?
                  AND rank MATCH 'bm25(6.0, 5.0, 3.0, 2.5, 3.0, 3.0, 3.0)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

    def search_block_fts(self, document_id: str, match_query: str, limit: int = 24) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    b.block_id,
                    b.document_id,
                    b.page_num,
                    b.block_ordinal,
                    b.block_type,
                    b.block_text,
                    b.normalized_text,
                    b.alias_text,
                    b.parent_chunk_id,
                    b.noise_flags AS block_noise_flags,
                    c.chunk_id,
                    c.section_number,
                    c.heading,
                    c.full_text,
                    c.page_start,
                    c.page_end,
                    c.ordinal_in_document,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    rank AS rank_score,
                    snippet(contract_blocks_fts, 0, '[', ']', '...', 18) AS hit_snippet
                FROM contract_blocks_fts
                JOIN contract_blocks b ON b.block_rowid = contract_blocks_fts.rowid
                LEFT JOIN contract_chunks c ON c.chunk_id = b.parent_chunk_id
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE b.document_id = ?
                  AND contract_blocks_fts MATCH ?
                  AND rank MATCH 'bm25(8.0, 6.5, 4.5, 1.5, 1.0)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

    def search_block_rescue_fts(self, document_id: str, match_query: str, limit: int = 24) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    b.block_id,
                    b.document_id,
                    b.page_num,
                    b.block_ordinal,
                    b.block_type,
                    b.block_text,
                    b.normalized_text,
                    b.alias_text,
                    b.parent_chunk_id,
                    b.noise_flags AS block_noise_flags,
                    c.chunk_id,
                    c.section_number,
                    c.heading,
                    c.full_text,
                    c.page_start,
                    c.page_end,
                    c.ordinal_in_document,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    rank AS rank_score
                FROM contract_blocks_rescue_fts
                JOIN contract_blocks b ON b.block_rowid = contract_blocks_rescue_fts.rowid
                LEFT JOIN contract_chunks c ON c.chunk_id = b.parent_chunk_id
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE b.document_id = ?
                  AND contract_blocks_rescue_fts MATCH ?
                  AND rank MATCH 'bm25(6.0, 5.0, 4.0, 1.5, 1.0)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

    def keyword_like_search(self, document_id: str, query: str, limit: int = 20) -> list[sqlite3.Row]:
        like = f"%{query}%"
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags,
                    9999.0 AS rank_score,
                    substr(c.full_text, 1, 240) AS hit_snippet
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ?
                  AND (
                    c.heading LIKE ? OR c.full_text LIKE ? OR c.section_number LIKE ?
                    OR f.parent_heading LIKE ? OR f.search_text LIKE ? OR f.topic_tags LIKE ?
                  )
                ORDER BY c.ordinal_in_document
                LIMIT ?
                """,
                (document_id, like, like, like, like, like, like, limit),
            ).fetchall()

    def fetch_context_neighbors(self, document_id: str, ordinal: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ?
                  AND c.ordinal_in_document BETWEEN ? AND ?
                ORDER BY c.ordinal_in_document
                """,
                (document_id, ordinal - 1, ordinal + 1),
            ).fetchall()

    def fetch_parent(self, parent_chunk_id: str | None) -> sqlite3.Row | None:
        if not parent_chunk_id:
            return None
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.chunk_id = ?
                """,
                (parent_chunk_id,),
            ).fetchone()

    def fetch_chunk(self, chunk_id: str | None) -> sqlite3.Row | None:
        if not chunk_id:
            return None
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

    def fetch_document_chunks(self, document_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM contract_chunks
                WHERE document_id = ?
                ORDER BY ordinal_in_document
                """,
                (document_id,),
            ).fetchall()

    def fetch_page_window(
        self,
        document_id: str,
        page_start: int,
        page_end: int,
        *,
        padding: int = 0,
        limit: int = 4,
    ) -> list[sqlite3.Row]:
        lower = max(1, page_start - padding)
        upper = max(page_end, page_end + padding)
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT page_num, page_text, ocr_used
                FROM contract_pages
                WHERE document_id = ?
                  AND page_num BETWEEN ? AND ?
                ORDER BY page_num
                LIMIT ?
                """,
                (document_id, lower, upper, limit),
            ).fetchall()

    def fetch_chunks_on_pages(
        self,
        document_id: str,
        page_start: int,
        page_end: int,
        *,
        limit: int = 6,
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    f.parent_heading,
                    f.clause_type,
                    f.actor_tags,
                    f.action_tags,
                    f.topic_tags,
                    f.noise_flags
                FROM contract_chunks c
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ?
                  AND NOT (c.page_end < ? OR c.page_start > ?)
                ORDER BY c.ordinal_in_document
                LIMIT ?
                """,
                (document_id, page_start, page_end, limit),
            ).fetchall()

    def fetch_chunk_vectors(self, document_id: str, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, model_name, dimension, vector
                FROM chunk_vectors
                WHERE document_id = ?
                  AND chunk_id IN ({placeholders})
                """,
                (document_id, *chunk_ids),
            ).fetchall()
        return {
            str(row["chunk_id"]): {
                "chunk_id": str(row["chunk_id"]),
                "model_name": str(row["model_name"]),
                "dimension": int(row["dimension"]),
                "vector": unpack_vector(bytes(row["vector"])),
            }
            for row in rows
        }

    def fetch_blocks_on_pages(
        self,
        document_id: str,
        page_start: int,
        page_end: int,
        *,
        limit: int = 24,
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM contract_blocks
                WHERE document_id = ?
                  AND page_num BETWEEN ? AND ?
                ORDER BY page_num, block_ordinal
                LIMIT ?
                """,
                (document_id, page_start, page_end, limit),
            ).fetchall()

    def rebuild_fts(self) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO contract_pages_fts(contract_pages_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO chunk_search_fts(chunk_search_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO chunk_search_rescue_fts(chunk_search_rescue_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO contract_blocks_fts(contract_blocks_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO contract_blocks_rescue_fts(contract_blocks_rescue_fts) VALUES('rebuild')")
            connection.commit()

    def search_pages_fts(self, document_id: str, match_query: str, limit: int = 12) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    p.page_num,
                    p.page_text,
                    snippet(contract_pages_fts, 0, '[', ']', '...', 28) AS hit_snippet,
                    rank AS rank_score
                FROM contract_pages_fts
                JOIN contract_pages p ON p.page_rowid = contract_pages_fts.rowid
                WHERE p.document_id = ?
                  AND contract_pages_fts MATCH ?
                  AND rank MATCH 'bm25(1.0)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

    def search_blocks_exact(self, document_id: str, match_query: str, limit: int = 12) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    b.page_num,
                    b.block_text,
                    snippet(contract_blocks_fts, 0, '[', ']', '...', 18) AS hit_snippet,
                    rank AS rank_score
                FROM contract_blocks_fts
                JOIN contract_blocks b ON b.block_rowid = contract_blocks_fts.rowid
                WHERE b.document_id = ?
                  AND contract_blocks_fts MATCH ?
                  AND rank MATCH 'bm25(1.0, 0.8, 0.8, 0.5, 0.2)'
                ORDER BY rank
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()


def build_block_records(
    document_id: str,
    pages: list[PageText],
    chunks: list[ChunkRecord],
) -> list[ContractBlockRecord]:
    chunks_by_page: dict[int, list[ChunkRecord]] = {}
    for chunk in chunks:
        for page_num in range(chunk.page_start, chunk.page_end + 1):
            chunks_by_page.setdefault(page_num, []).append(chunk)

    records: list[ContractBlockRecord] = []
    for page in pages:
        page_blocks = page.blocks or segment_text_blocks(page.text)
        for block in page_blocks:
            normalized_text = normalize_text(block.text)
            alias_text = " ".join(alias_terms_for_text(block.text))
            parent_chunk_id = _resolve_block_parent_chunk(chunks_by_page.get(page.page_num, []), normalized_text)
            records.append(
                ContractBlockRecord(
                    block_id=f"block_{document_id}_{page.page_num}_{block.block_ordinal}",
                    document_id=document_id,
                    page_num=page.page_num,
                    block_ordinal=block.block_ordinal,
                    block_type=block.block_type,
                    block_text=block.text,
                    normalized_text=normalized_text,
                    alias_text=alias_text,
                    parent_chunk_id=parent_chunk_id,
                    noise_flags=" ".join(block.noise_flags),
                )
            )
    return records


def build_diagnostic_records(document_id: str, pages: list[PageText]) -> list[PageIngestDiagnosticRecord]:
    records: list[PageIngestDiagnosticRecord] = []
    for page in pages:
        page_blocks = page.blocks or segment_text_blocks(page.text)
        diagnostics = page.diagnostics or build_page_diagnostics(page.page_num, page.text, page.ocr_used, page_blocks)
        records.append(
            PageIngestDiagnosticRecord(
                document_id=document_id,
                page_num=page.page_num,
                meaningful_chars=diagnostics.meaningful_chars,
                word_count=diagnostics.word_count,
                block_count=diagnostics.block_count,
                short_line_count=diagnostics.short_line_count,
                flags=" ".join(diagnostics.flags),
            )
        )
    return records


def _resolve_block_parent_chunk(chunks: list[ChunkRecord], normalized_block_text: str) -> str | None:
    if not chunks:
        return None
    if not normalized_block_text:
        return chunks[0].chunk_id
    heading_matches = [
        chunk for chunk in chunks
        if normalize_text(chunk.heading) == normalized_block_text
    ]
    if heading_matches:
        return min(heading_matches, key=lambda chunk: len(chunk.full_text)).chunk_id
    body_matches = [
        chunk for chunk in chunks
        if normalized_block_text in normalize_text(chunk.full_text)
    ]
    if body_matches:
        return min(body_matches, key=lambda chunk: len(chunk.full_text)).chunk_id
    token_set = set(normalized_block_text.split())
    scored = []
    for chunk in chunks:
        chunk_tokens = set(normalize_text(f"{chunk.heading} {chunk.full_text}").split())
        overlap = len(token_set & chunk_tokens)
        scored.append((overlap, -chunk.ordinal_in_document, chunk))
    best_overlap, _, best_chunk = max(scored, key=lambda item: item[0])
    return best_chunk.chunk_id if best_overlap > 0 else chunks[0].chunk_id


def pack_vector(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _ensure_compat_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(chunk_search_features)").fetchall()
    }
    if "normalized_text" not in columns:
        connection.execute(
            "ALTER TABLE chunk_search_features ADD COLUMN normalized_text TEXT NOT NULL DEFAULT ''"
        )
    connection.execute(
        """
        UPDATE chunk_search_features
        SET normalized_text = rescue_text
        WHERE normalized_text = ''
        """
    )


def _backfill_block_coverage(connection: sqlite3.Connection) -> None:
    document_rows = connection.execute(
        """
        SELECT
            d.document_id,
            COUNT(p.page_rowid) AS page_count,
            (
                SELECT COUNT(*)
                FROM contract_blocks b
                WHERE b.document_id = d.document_id
            ) AS block_count,
            (
                SELECT COUNT(*)
                FROM page_ingest_diagnostics pid
                WHERE pid.document_id = d.document_id
            ) AS diagnostic_count
        FROM documents d
        LEFT JOIN contract_pages p ON p.document_id = d.document_id
        GROUP BY d.document_id
        """
    ).fetchall()
    for row in document_rows:
        document_id = str(row["document_id"])
        page_count = int(row["page_count"] or 0)
        block_count = int(row["block_count"] or 0)
        diagnostic_count = int(row["diagnostic_count"] or 0)
        if page_count <= 0:
            continue
        if block_count > 0 and diagnostic_count == page_count:
            continue

        chunk_rows = connection.execute(
            """
            SELECT
                chunk_id,
                document_id,
                chunk_type,
                section_number,
                heading,
                full_text,
                page_start,
                page_end,
                parent_chunk_id,
                ordinal_in_document
            FROM contract_chunks
            WHERE document_id = ?
            ORDER BY ordinal_in_document
            """,
            (document_id,),
        ).fetchall()
        page_rows = connection.execute(
            """
            SELECT page_num, page_text, ocr_used
            FROM contract_pages
            WHERE document_id = ?
            ORDER BY page_num
            """,
            (document_id,),
        ).fetchall()
        if not page_rows:
            continue

        chunks = [
            ChunkRecord(
                chunk_id=str(chunk_row["chunk_id"]),
                document_id=str(chunk_row["document_id"]),
                chunk_type=str(chunk_row["chunk_type"]),
                section_number=str(chunk_row["section_number"] or ""),
                heading=str(chunk_row["heading"] or ""),
                full_text=str(chunk_row["full_text"] or ""),
                page_start=int(chunk_row["page_start"]),
                page_end=int(chunk_row["page_end"]),
                parent_chunk_id=str(chunk_row["parent_chunk_id"]) if chunk_row["parent_chunk_id"] else None,
                ordinal_in_document=int(chunk_row["ordinal_in_document"]),
            )
            for chunk_row in chunk_rows
        ]
        pages = [
            PageText(
                page_num=int(page_row["page_num"]),
                text=str(page_row["page_text"] or ""),
                ocr_used=bool(page_row["ocr_used"]),
            )
            for page_row in page_rows
        ]
        blocks = build_block_records(document_id, pages, chunks)
        diagnostics = build_diagnostic_records(document_id, pages)

        connection.execute("DELETE FROM contract_blocks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM page_ingest_diagnostics WHERE document_id = ?", (document_id,))
        ContractStore._insert_blocks(connection, blocks)
        ContractStore._insert_diagnostics(connection, diagnostics)
