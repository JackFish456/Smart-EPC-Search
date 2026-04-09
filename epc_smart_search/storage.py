from __future__ import annotations

import sqlite3
from array import array
from dataclasses import asdict
from pathlib import Path
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.config import SEARCH_SCHEMA_VERSION
from epc_smart_search.ocr_support import PageText
from epc_smart_search.search_features import ChunkFeatures

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

CREATE INDEX IF NOT EXISTS idx_contract_chunks_doc_section
ON contract_chunks(document_id, section_number);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_parent
ON contract_chunks(parent_chunk_id);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_pages
ON contract_chunks(document_id, page_start, page_end);

CREATE INDEX IF NOT EXISTS idx_contract_chunks_ordinal
ON contract_chunks(document_id, ordinal_in_document);

CREATE VIRTUAL TABLE IF NOT EXISTS contract_chunks_fts USING fts5(
    section_number,
    heading,
    full_text,
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content='contract_chunks',
    content_rowid='chunk_rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES contract_chunks(chunk_id)
);

CREATE TRIGGER IF NOT EXISTS contract_chunks_ai AFTER INSERT ON contract_chunks BEGIN
  INSERT INTO contract_chunks_fts(rowid, section_number, heading, full_text, chunk_id, document_id)
  VALUES (new.chunk_rowid, new.section_number, new.heading, new.full_text, new.chunk_id, new.document_id);
END;

CREATE TRIGGER IF NOT EXISTS contract_chunks_ad AFTER DELETE ON contract_chunks BEGIN
  INSERT INTO contract_chunks_fts(contract_chunks_fts, rowid, section_number, heading, full_text, chunk_id, document_id)
  VALUES ('delete', old.chunk_rowid, old.section_number, old.heading, old.full_text, old.chunk_id, old.document_id);
END;

CREATE TRIGGER IF NOT EXISTS contract_chunks_au AFTER UPDATE ON contract_chunks BEGIN
  INSERT INTO contract_chunks_fts(contract_chunks_fts, rowid, section_number, heading, full_text, chunk_id, document_id)
  VALUES ('delete', old.chunk_rowid, old.section_number, old.heading, old.full_text, old.chunk_id, old.document_id);

  INSERT INTO contract_chunks_fts(rowid, section_number, heading, full_text, chunk_id, document_id)
  VALUES (new.chunk_rowid, new.section_number, new.heading, new.full_text, new.chunk_id, new.document_id);
END;

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

CREATE TABLE IF NOT EXISTS chunk_search_features (
    feature_rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    section_number TEXT,
    heading TEXT NOT NULL DEFAULT '',
    parent_heading TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT '',
    normalized_text TEXT NOT NULL DEFAULT '',
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
    tokenize='unicode61 remove_diacritics 2'
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

CREATE TRIGGER IF NOT EXISTS chunk_search_features_ai AFTER INSERT ON chunk_search_features BEGIN
  INSERT INTO chunk_search_fts(
    rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.search_text,
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
END;

CREATE TRIGGER IF NOT EXISTS chunk_search_features_au AFTER UPDATE ON chunk_search_features BEGIN
  INSERT INTO chunk_search_fts(
    chunk_search_fts, rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    'delete', old.feature_rowid, old.section_number, old.heading, old.parent_heading, old.search_text,
    old.actor_tags, old.action_tags, old.topic_tags, old.chunk_id, old.document_id
  );

  INSERT INTO chunk_search_fts(
    rowid, section_number, heading, parent_heading, search_text,
    actor_tags, action_tags, topic_tags, chunk_id, document_id
  ) VALUES (
    new.feature_rowid, new.section_number, new.heading, new.parent_heading, new.search_text,
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
        embeddings: dict[str, bytes],
        model_name: str,
        dimension: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM contract_chunks WHERE document_id = ?)",
                (document_id,),
            )
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
            self._insert_features(connection, features)
            connection.executemany(
                """
                INSERT INTO chunk_embeddings (chunk_id, model_name, vector_blob, dimension)
                VALUES (?, ?, ?, ?)
                """,
                [(chunk_id, model_name, blob, dimension) for chunk_id, blob in embeddings.items()],
            )
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
    def _insert_features(connection: sqlite3.Connection, features: list[ChunkFeatures]) -> None:
        connection.executemany(
            """
            INSERT INTO chunk_search_features (
                chunk_id, document_id, section_number, heading, parent_heading,
                search_text, normalized_text, clause_type, actor_tags,
                action_tags, topic_tags, noise_flags
            ) VALUES (
                :chunk_id, :document_id, :section_number, :heading, :parent_heading,
                :search_text, :normalized_text, :clause_type, :actor_tags,
                :action_tags, :topic_tags, :noise_flags
            )
            """,
            [asdict(feature) for feature in features],
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
                "SELECT COUNT(*) AS chunk_count, COUNT(DISTINCT document_id) AS document_count FROM contract_chunks"
            ).fetchone()
        return {
            "chunk_count": int(row["chunk_count"]) if row else 0,
            "document_count": int(row["document_count"]) if row else 0,
        }

    def get_feature_count(self, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS feature_count FROM chunk_search_features WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["feature_count"]) if row else 0

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

    def search_fts(self, document_id: str, match_query: str, limit: int = 20) -> list[sqlite3.Row]:
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
                    bm25(contract_chunks_fts, 5.0, 3.0, 1.0) AS bm25_score,
                    snippet(contract_chunks_fts, 2, '[', ']', '...', 18) AS hit_snippet
                FROM contract_chunks_fts
                JOIN contract_chunks c ON c.chunk_rowid = contract_chunks_fts.rowid
                LEFT JOIN chunk_search_features f ON f.chunk_id = c.chunk_id
                WHERE c.document_id = ? AND contract_chunks_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()

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
                    f.normalized_text,
                    bm25(chunk_search_fts, 10.0, 7.0, 4.0, 1.2, 4.0, 4.0, 4.0) AS bm25_score,
                    snippet(chunk_search_fts, 3, '[', ']', '...', 18) AS hit_snippet
                FROM chunk_search_fts
                JOIN chunk_search_features f ON f.feature_rowid = chunk_search_fts.rowid
                JOIN contract_chunks c ON c.chunk_id = f.chunk_id
                WHERE f.document_id = ? AND chunk_search_fts MATCH ?
                ORDER BY bm25_score
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
                    9999.0 AS bm25_score,
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

    def iter_embeddings(self, document_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT c.chunk_id, c.heading, c.section_number, c.page_start, c.page_end, c.full_text, c.ordinal_in_document,
                       e.vector_blob, e.dimension
                FROM chunk_embeddings e
                JOIN contract_chunks c ON c.chunk_id = e.chunk_id
                WHERE c.document_id = ?
                ORDER BY c.ordinal_in_document
                """,
                (document_id,),
            ).fetchall()

    def rebuild_fts(self) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO contract_chunks_fts(contract_chunks_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO contract_pages_fts(contract_pages_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO chunk_search_fts(chunk_search_fts) VALUES('rebuild')")
            connection.commit()

    def search_pages_fts(self, document_id: str, match_query: str, limit: int = 12) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    p.page_num,
                    p.page_text,
                    snippet(contract_pages_fts, 0, '[', ']', '...', 28) AS hit_snippet,
                    bm25(contract_pages_fts, 1.0) AS bm25_score
                FROM contract_pages_fts
                JOIN contract_pages p ON p.page_rowid = contract_pages_fts.rowid
                WHERE p.document_id = ? AND contract_pages_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (document_id, match_query, limit),
            ).fetchall()


def pack_vector(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)
