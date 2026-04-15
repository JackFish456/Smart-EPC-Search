from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from epc_smart_search.app_paths import ASSETS_DIR
from epc_smart_search.config import MAX_EMBEDDING_DIM
from epc_smart_search.query_planner import QueryPlan
from epc_smart_search.search_features import (
    ACTION_LEXICON,
    ACTOR_LEXICON,
    DOMAIN_ALIASES,
    QUERY_EXPANSIONS,
    TOPIC_LEXICON,
    is_numericish_token,
    normalize_text,
    tokenize,
)

SEMANTIC_MODEL_ENV_VAR = "EPC_SMART_SEARCH_SEMANTIC_MODEL_DIR"
DEFAULT_SEMANTIC_MODEL_FILENAME = "semantic_model.json"


@dataclass(slots=True, frozen=True)
class SemanticModelConfig:
    model_name: str
    dimension: int
    stopwords: tuple[str, ...] = ()
    synonyms: dict[str, tuple[str, ...]] | None = None


def resolve_semantic_model_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        raw = str(explicit_path).strip()
    else:
        raw = os.environ.get(SEMANTIC_MODEL_ENV_VAR, "").strip()
    base = Path(raw).expanduser() if raw else (ASSETS_DIR / DEFAULT_SEMANTIC_MODEL_FILENAME)
    if base.is_dir():
        return base / DEFAULT_SEMANTIC_MODEL_FILENAME
    return base


def load_semantic_model_config(explicit_path: str | Path | None = None) -> SemanticModelConfig | None:
    path = resolve_semantic_model_path(explicit_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    dimension = int(payload.get("dimension", 0) or 0)
    if dimension <= 0 or dimension > MAX_EMBEDDING_DIM:
        raise ValueError(f"Invalid semantic embedding dimension in {path}: {dimension}")
    synonyms_payload = payload.get("synonyms") or {}
    synonyms = {
        normalize_text(str(canonical)): tuple(normalize_text(str(variant)) for variant in variants if str(variant).strip())
        for canonical, variants in synonyms_payload.items()
    }
    return SemanticModelConfig(
        model_name=str(payload.get("model_name", path.stem)).strip() or path.stem,
        dimension=dimension,
        stopwords=tuple(normalize_text(str(word)) for word in payload.get("stopwords", ()) if str(word).strip()),
        synonyms=synonyms,
    )


def _compose_semantic_text(
    *,
    identifier: str = "",
    phrase: str = "",
    context: str = "",
    hierarchy: str = "",
    priority: str = "",
    numeric: str = "",
    actor: str = "",
    action: str = "",
    topic: str = "",
    body: str = "",
) -> str:
    parts = [
        identifier,
        phrase,
        context,
        hierarchy,
        priority,
        numeric,
        actor,
        action,
        topic,
        body,
    ]
    return " ".join(part for part in parts if part)


def build_chunk_semantic_text(chunk, feature, *, max_body_tokens: int = 96) -> str:
    normalized_body = normalize_text(chunk.full_text)
    body_tokens = tokenize(normalized_body)[:max_body_tokens]
    return _compose_semantic_text(
        identifier=chunk.section_number or "",
        phrase=normalize_text(chunk.heading),
        context=normalize_text(feature.parent_heading),
        hierarchy=normalize_text(feature.hierarchy_path),
        priority=normalize_text(feature.priority_flags),
        numeric=normalize_text(feature.numeric_text),
        actor=normalize_text(feature.actor_tags),
        action=normalize_text(feature.action_tags),
        topic=normalize_text(feature.topic_tags),
        body=" ".join(body_tokens),
    )


def build_query_semantic_text(query: str, plan: QueryPlan) -> str:
    normalized_query = normalize_text(query)
    numeric_terms = " ".join(
        term for term in (*plan.focus_terms, *plan.expansion_terms) if term and is_numericish_token(term)
    )
    return _compose_semantic_text(
        identifier=plan.section_number or "",
        phrase=plan.content_query or normalized_query,
        context=" ".join(plan.focus_terms),
        hierarchy=" ".join(plan.expansion_terms),
        numeric=numeric_terms,
        actor=" ".join(plan.actor_terms),
        action=" ".join(plan.action_terms),
        topic=" ".join(plan.topic_terms),
        body=normalized_query,
    )


class LocalEmbedder:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = model_path
        self._config: SemanticModelConfig | None | object = _MISSING
        self._canonical_map: dict[str, str] | None = None
        self._canonical_phrases: tuple[str, ...] = ()
        self._stopwords: set[str] = set()

    @property
    def config(self) -> SemanticModelConfig | None:
        if self._config is _MISSING:
            config = load_semantic_model_config(self._model_path)
            self._config = config
            if config is not None:
                self._stopwords = set(config.stopwords)
                self._canonical_map = self._build_canonical_map(config)
                self._canonical_phrases = tuple(sorted({*self._canonical_map.values()}, key=len, reverse=True))
            else:
                self._canonical_map = {}
                self._canonical_phrases = ()
        return self._config if self._config is not _MISSING else None

    @property
    def model_name(self) -> str | None:
        config = self.config
        return config.model_name if config is not None else None

    @property
    def dimension(self) -> int | None:
        config = self.config
        return config.dimension if config is not None else None

    def is_available(self) -> bool:
        return self.config is not None

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        config = self.config
        if config is None:
            raise RuntimeError("Semantic model is unavailable.")
        return [self._encode_one(text, config.dimension) for text in texts]

    def _encode_one(self, text: str, dimension: int) -> list[float]:
        normalized = self._canonicalize_text(text)
        tokens = [token for token in tokenize(normalized) if token and token not in self._stopwords]
        if not tokens:
            return [0.0] * dimension
        features: list[tuple[str, float]] = []
        features.extend((f"tok:{token}", 1.0) for token in tokens)
        features.extend((f"bigram:{tokens[index]}_{tokens[index + 1]}", 0.85) for index in range(len(tokens) - 1))
        collapsed = "".join(ch for ch in normalized if ch.isalnum())
        features.extend((f"tri:{collapsed[index:index + 3]}", 0.32) for index in range(max(0, len(collapsed) - 2)))
        features.extend((f"phrase:{phrase}", 1.25) for phrase in self._canonical_phrases if f" {phrase} " in f" {normalized} ")
        vector = [0.0] * dimension
        for feature, weight in features:
            index, sign = self._feature_slot(feature, dimension)
            vector[index] += weight * sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return [0.0] * dimension
        return [value / norm for value in vector]

    def _canonicalize_text(self, text: str) -> str:
        normalized = normalize_text(text)
        canonical_map = self._canonical_map or {}
        if not canonical_map:
            return normalized
        wrapped = f" {normalized} "
        for variant, canonical in sorted(canonical_map.items(), key=lambda item: len(item[0]), reverse=True):
            wrapped = wrapped.replace(f" {variant} ", f" {canonical} ")
        return " ".join(wrapped.split())

    @staticmethod
    def _feature_slot(feature: str, dimension: int) -> tuple[int, float]:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = -1.0 if digest[8] & 1 else 1.0
        return index, sign

    @staticmethod
    def _build_canonical_map(config: SemanticModelConfig) -> dict[str, str]:
        groups: dict[str, set[str]] = {}
        for canonical, variants in DOMAIN_ALIASES.items():
            groups.setdefault(normalize_text(canonical), set()).update(normalize_text(variant) for variant in variants)
        for canonical, variants in QUERY_EXPANSIONS.items():
            groups.setdefault(normalize_text(canonical), set()).update(normalize_text(variant) for variant in variants)
        for lexicon in (ACTOR_LEXICON, ACTION_LEXICON, TOPIC_LEXICON):
            for canonical, variants in lexicon.items():
                groups.setdefault(normalize_text(canonical), set()).update(normalize_text(variant) for variant in variants)
        if config.synonyms:
            for canonical, variants in config.synonyms.items():
                groups.setdefault(canonical, set()).update(variants)
        canonical_map: dict[str, str] = {}
        for canonical, variants in groups.items():
            canonical_map[canonical] = canonical
            for variant in variants:
                canonical_map[variant] = canonical
        return canonical_map


class SemanticReranker:
    def __init__(self, store, embedder: LocalEmbedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or LocalEmbedder()

    def rerank(self, document_id: str, query_text: str, candidates: list[object]) -> bool:
        if not candidates or not self.embedder.is_available():
            return False
        model_name = self.embedder.model_name
        dimension = self.embedder.dimension
        if not model_name or not dimension:
            return False
        query_vector = self.embedder.encode([query_text])[0]
        if not any(query_vector):
            return False
        rows = self.store.fetch_chunk_vectors(document_id, [str(candidate.row["chunk_id"]) for candidate in candidates])
        if not rows:
            return False
        reranked = False
        for candidate in candidates:
            row = rows.get(str(candidate.row["chunk_id"]))
            candidate.semantic_score = 0.0
            if row is None:
                continue
            if str(row["model_name"]) != model_name or int(row["dimension"]) != dimension:
                continue
            chunk_vector = row["vector"]
            if not chunk_vector:
                continue
            score = cosine_similarity(query_vector, chunk_vector)
            candidate.semantic_score = max(0.0, min(1.0, score))
            reranked = reranked or candidate.semantic_score > 0.0
        return reranked


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


_MISSING = object()
