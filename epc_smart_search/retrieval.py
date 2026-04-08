from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from epc_smart_search.config import MAX_EMBEDDING_DIM, MAX_SEARCH_RESULTS, MAX_SEMANTIC_SCAN
from epc_smart_search.storage import ContractStore, unpack_vector

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{1,}")
SECTION_QUERY_RE = re.compile(r"\b(?:section|sec\.?)\s*(\d+(?:\.\d+){0,5})\b", re.IGNORECASE)
BARE_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s*$")


@dataclass(slots=True)
class Citation:
    chunk_id: str
    section_number: str | None
    heading: str
    attachment: str | None
    page_start: int
    page_end: int
    quote: str


@dataclass(slots=True)
class ExactPageHit:
    page_num: int
    snippet: str
    page_text: str


@dataclass(slots=True)
class RankedChunk:
    chunk_id: str
    section_number: str | None
    heading: str
    full_text: str
    page_start: int
    page_end: int
    ordinal_in_document: int
    total_score: float
    lexical_score: float
    semantic_score: float


class HashingEmbedder:
    def __init__(self, dimension: int = MAX_EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.model_name = f"hashing-embedder-{dimension}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = " ".join(text.lower().split())
        tokens = TOKEN_RE.findall(normalized)
        for token in tokens:
            self._add_feature(vector, f"tok:{token}", 1.0)
        collapsed = normalized.replace(" ", "_")
        for index in range(max(0, len(collapsed) - 2)):
            self._add_feature(vector, f"tri:{collapsed[index:index + 3]}", 0.35)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.sha1(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def build_match_query(query: str) -> str:
    tokens = [token for token in TOKEN_RE.findall(query.lower()) if len(token) > 2]
    if not tokens:
        clean = query.strip().replace('"', "")
        return f'"{clean}"' if clean else ""
    if len(tokens) >= 2:
        phrase = " ".join(tokens[:6])
        parts = [f'"{phrase}"']
        parts.extend(f'"{token}"' for token in tokens[:12])
        return " OR ".join(parts)
    return " OR ".join(f'"{token}"' for token in tokens[:12])


class HybridRetriever:
    def __init__(self, store: ContractStore, embedder: HashingEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def resolve_document_id(self) -> str | None:
        document = self.store.get_document()
        return str(document["document_id"]) if document else None

    def retrieve(self, query: str, *, limit: int = MAX_SEARCH_RESULTS) -> list[RankedChunk]:
        document_id = self.resolve_document_id()
        if not document_id:
            return []
        section_number = self._extract_section_number(query)
        direct_rows = self.store.section_lookup(document_id, section_number) if section_number else []
        lexical_rows = []
        match_query = build_match_query(query)
        if match_query:
            lexical_rows = self.store.search_fts(document_id, match_query, limit=24)
        if not lexical_rows:
            lexical_rows = self.store.keyword_like_search(document_id, query.strip(), limit=18)
        query_vector = self.embedder.embed(query)
        semantic_rows = self._semantic_candidates(document_id, query_vector)

        combined: dict[str, RankedChunk] = {}
        for row in direct_rows:
            ranked = self._rank_row(row, lexical_score=1.0, semantic_score=0.85, bonus=0.35)
            combined[ranked.chunk_id] = ranked
        for row in lexical_rows:
            lexical_score = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
            semantic_score = cosine_similarity(query_vector, self._row_vector(row))
            ranked = self._rank_row(row, lexical_score=lexical_score, semantic_score=semantic_score)
            existing = combined.get(ranked.chunk_id)
            if existing is None or ranked.total_score > existing.total_score:
                combined[ranked.chunk_id] = ranked
        for row, semantic_score in semantic_rows:
            ranked = self._rank_row(row, lexical_score=0.0, semantic_score=semantic_score)
            existing = combined.get(ranked.chunk_id)
            if existing is None or ranked.total_score > existing.total_score:
                combined[ranked.chunk_id] = ranked
        return sorted(combined.values(), key=lambda item: item.total_score, reverse=True)[:limit]

    def expand_with_context(self, ranked: list[RankedChunk]) -> list[Citation]:
        document_id = self.resolve_document_id()
        if not document_id or not ranked:
            return []
        chosen_ids: set[str] = set()
        citations: list[Citation] = []
        for item in ranked[:3]:
            for row in self.store.fetch_context_neighbors(document_id, item.ordinal_in_document):
                if row["chunk_id"] in chosen_ids:
                    continue
                heading = self._compact_heading(str(row["heading"]))
                chosen_ids.add(str(row["chunk_id"]))
                citations.append(
                    Citation(
                        chunk_id=str(row["chunk_id"]),
                        section_number=row["section_number"],
                        heading=heading,
                        attachment=self._attachment_label(str(row["chunk_id"])),
                        page_start=int(row["page_start"]),
                        page_end=int(row["page_end"]),
                        quote=self._short_quote(str(row["full_text"])),
                    )
                )
        return citations[:6]

    def build_prompt_context(self, citations: list[Citation]) -> str:
        blocks: list[str] = []
        for index, citation in enumerate(citations, start=1):
            label = citation.section_number or "Unnumbered clause"
            blocks.append(
                f"[{index}] Section: {label}\n"
                f"Heading: {citation.heading}\n"
                f"Pages: {citation.page_start}-{citation.page_end}\n"
                f"Excerpt: {citation.quote}"
            )
        return "\n\n".join(blocks)

    def find_exact_page_hits(self, query: str, limit: int = 8) -> list[ExactPageHit]:
        document_id = self.resolve_document_id()
        if not document_id:
            return []
        search_terms = self._candidate_exact_terms(query)
        seen_pages: set[int] = set()
        hits: list[ExactPageHit] = []
        for term in search_terms:
            match_query = f'"{term}"'
            rows = self.store.search_pages_fts(document_id, match_query, limit=limit)
            for row in rows:
                page_num = int(row["page_num"])
                if page_num in seen_pages:
                    continue
                seen_pages.add(page_num)
                hits.append(
                    ExactPageHit(
                        page_num=page_num,
                        snippet=self._clean_snippet(str(row["hit_snippet"])),
                        page_text=str(row["page_text"]),
                    )
                )
                if len(hits) >= limit:
                    return hits
        return hits

    def _semantic_candidates(self, document_id: str, query_vector: list[float]) -> list[tuple[dict, float]]:
        rows = self.store.iter_embeddings(document_id)
        scored: list[tuple[dict, float]] = []
        for row in rows:
            semantic_score = cosine_similarity(query_vector, unpack_vector(row["vector_blob"]))
            if semantic_score > 0:
                scored.append((row, semantic_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:MAX_SEMANTIC_SCAN]

    @staticmethod
    def _row_vector(row: dict) -> list[float]:
        blob = row["vector_blob"] if "vector_blob" in row.keys() else None
        return unpack_vector(blob) if blob else []

    @staticmethod
    def _extract_section_number(query: str) -> str | None:
        match = SECTION_QUERY_RE.search(query) or BARE_SECTION_RE.match(query.strip())
        return match.group(1) if match else None

    @staticmethod
    def _short_quote(text: str, limit: int = 340) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[:limit - 3] + "..."

    @staticmethod
    def _clean_snippet(text: str) -> str:
        return " ".join(text.replace("[", "").replace("]", "").split())

    @staticmethod
    def _candidate_exact_terms(query: str) -> list[str]:
        raw = " ".join(query.lower().split())
        prefixes = [
            "how many ",
            "what does the contract say about ",
            "what does it say about ",
            "what is ",
            "define ",
            "tell me about ",
            "show me ",
        ]
        for prefix in prefixes:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        raw = raw.strip(" ?.")
        variants: list[str] = []
        if raw:
            variants.append(raw)
            if raw.endswith("s") and len(raw) > 4:
                variants.append(raw[:-1])
        compact_tokens = [token for token in TOKEN_RE.findall(raw) if len(token) > 2]
        if compact_tokens and " ".join(compact_tokens) not in variants:
            variants.append(" ".join(compact_tokens))
        seen: set[str] = set()
        return [term for term in variants if term and not (term in seen or seen.add(term))]

    @staticmethod
    def _compact_heading(heading: str, limit: int = 110) -> str:
        compact = " ".join(heading.replace("....", " ").split())
        compact = compact.strip(" -|:")
        return compact if len(compact) <= limit else compact[:limit - 3] + "..."

    def _attachment_label(self, chunk_id: str) -> str | None:
        current = self.store.fetch_chunk(chunk_id)
        while current is not None:
            chunk_type = str(current["chunk_type"])
            heading = self._compact_heading(str(current["heading"]))
            section_number = str(current["section_number"] or "").strip()
            if chunk_type == "exhibit":
                return self._normalize_attachment_heading(heading, section_number)
            parent_id = current["parent_chunk_id"]
            current = self.store.fetch_parent(parent_id)
        return None

    @staticmethod
    def _normalize_attachment_heading(heading: str, section_number: str) -> str:
        match = re.match(r"^(attachment|appendix|exhibit)\s+([A-Z0-9.\-]+)\b", heading, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1).title()} {match.group(2)}"
        if section_number and len(section_number) <= 4 and section_number.isalpha():
            return f"Appendix {section_number.upper()}"
        return heading

    def _rank_row(self, row: dict, *, lexical_score: float, semantic_score: float, bonus: float = 0.0) -> RankedChunk:
        heading_bonus = 0.1 if row["heading"] else 0.0
        total_score = lexical_score * 0.65 + semantic_score * 0.35 + bonus + heading_bonus - self._toc_penalty(row)
        return RankedChunk(
            chunk_id=str(row["chunk_id"]),
            section_number=row["section_number"],
            heading=str(row["heading"]),
            full_text=str(row["full_text"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            ordinal_in_document=int(row["ordinal_in_document"]),
            total_score=total_score,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
        )

    @staticmethod
    def _toc_penalty(row: dict) -> float:
        heading = str(row["heading"])
        full_text = str(row["full_text"])
        if "...." in heading:
            return 0.45
        if full_text.count("....") >= 3 and int(row["page_start"]) <= 20:
            return 0.45
        return 0.0
