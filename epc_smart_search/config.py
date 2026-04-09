from __future__ import annotations

APP_NAME = "EPC Smart Search"
GREETING = (
    "Hi, I'm Kiewey. Ask me about the EPC contract. "
    "I'll answer only from cited contract text."
)

GEMMA_SERVICE_HOST = "127.0.0.1"
GEMMA_SERVICE_PORT = 8051

MIN_NATIVE_TEXT_CHARS = 120
MIN_NATIVE_WORDS = 20
MIN_HEADING_LENGTH = 3
MAX_HEADING_LENGTH = 140
MAX_EMBEDDING_DIM = 512
MAX_SEARCH_RESULTS = 6
MAX_SEMANTIC_SCAN = 18
SEARCH_SCHEMA_VERSION = 2

STRICT_SYSTEM_PROMPT = (
    "You are EPC Smart Search, a local contract assistant for a single EPC agreement. "
    "Use only the provided contract excerpts. Do not rely on outside knowledge. "
    "If the excerpts do not fully support an answer, respond exactly with: "
    "\"I can't verify that from the contract.\" "
    "Keep answers concise and factual."
)
