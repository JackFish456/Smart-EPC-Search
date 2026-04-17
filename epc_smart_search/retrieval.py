from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from epc_smart_search.config import MAX_EMBEDDING_DIM, MAX_SEARCH_RESULTS, MAX_SEMANTIC_SCAN
from epc_smart_search.query_planner import (
    ANSWER_FAMILY_GUARANTEE_OR_LIMIT,
    QueryPlan,
    REQUEST_SHAPE_GROUPED_LIST,
    build_like_fallback,
    build_match_queries,
    has_term_overlap,
    plan_query,
)
from epc_smart_search.storage import ContractStore, unpack_vector
from epc_smart_search.system_vocabulary import SystemVocabulary, build_contract_system_vocabulary

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{1,}")


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


@dataclass(slots=True)
class SearchCandidate:
    row: dict
    total_score: float
    lexical_score: float
    semantic_score: float


@dataclass(slots=True, frozen=True)
class SearchPass:
    name: str
    query: str
    bonus: float


@dataclass(slots=True, frozen=True)
class RetrievalProfile:
    name: str
    result_limit: int
    fts_limit: int
    keyword_limit: int
    semantic_limit: int
    use_query_expansion: bool = False


DEFAULT_RETRIEVAL_PROFILE = RetrievalProfile(
    name="normal",
    result_limit=MAX_SEARCH_RESULTS,
    fts_limit=24,
    keyword_limit=18,
    semantic_limit=MAX_SEMANTIC_SCAN,
)
DEEP_RETRIEVAL_PROFILE = RetrievalProfile(
    name="deep",
    result_limit=10,
    fts_limit=40,
    keyword_limit=30,
    semantic_limit=36,
    use_query_expansion=True,
)


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
        self._system_vocabulary_document_id: str | None = None
        self._system_vocabulary_cache = SystemVocabulary(())

    def resolve_document_id(self) -> str | None:
        document = self.store.get_document()
        return str(document["document_id"]) if document else None

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        profile: str = "normal",
    ) -> list[RankedChunk]:
        document_id = self.resolve_document_id()
        if not document_id:
            return []
        plan = self.plan_query(query)
        active_profile = self._resolve_profile(profile, limit)
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            active_profile = RetrievalProfile(
                name=active_profile.name,
                result_limit=max(active_profile.result_limit, 6),
                fts_limit=max(active_profile.fts_limit, 36),
                keyword_limit=max(active_profile.keyword_limit, 24),
                semantic_limit=max(active_profile.semantic_limit, 24),
                use_query_expansion=active_profile.use_query_expansion,
            )
        semantic_query = plan.content_query or query
        combined: dict[str, SearchCandidate] = {}

        if plan.section_number:
            for row in self.store.section_lookup(document_id, plan.section_number):
                candidate = self._score_row(plan, row, lexical_score=1.0, semantic_score=0.85, bonus=0.9)
                combined[str(row["chunk_id"])] = candidate

        for search_pass in self._search_passes_for_profile(plan, active_profile):
            for row in self.store.search_chunk_feature_fts(document_id, search_pass.query, limit=active_profile.fts_limit):
                lexical_score = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
                semantic_score = self._semantic_for_row(semantic_query, row)
                candidate = self._score_row(
                    plan,
                    row,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    bonus=search_pass.bonus,
                    search_pass_name=search_pass.name,
                )
                self._merge_candidate(combined, candidate)

        if not combined:
            for fallback_query in self._fallback_queries_for_profile(plan, active_profile):
                for row in self.store.keyword_like_search(document_id, fallback_query, limit=active_profile.keyword_limit):
                    lexical_score = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
                    semantic_score = self._semantic_for_row(semantic_query, row)
                    candidate = self._score_row(plan, row, lexical_score=lexical_score, semantic_score=semantic_score)
                    self._merge_candidate(combined, candidate)

        if not combined:
            query_vector = self.embedder.embed(semantic_query)
            for row, semantic_score in self._semantic_candidates(
                document_id,
                query_vector,
                limit=active_profile.semantic_limit,
            ):
                candidate = self._score_row(plan, row, lexical_score=0.0, semantic_score=semantic_score)
                self._merge_candidate(combined, candidate)

        ranked = [
            RankedChunk(
                chunk_id=str(candidate.row["chunk_id"]),
                section_number=candidate.row["section_number"],
                heading=str(candidate.row["heading"]),
                full_text=str(candidate.row["full_text"]),
                page_start=int(candidate.row["page_start"]),
                page_end=int(candidate.row["page_end"]),
                ordinal_in_document=int(candidate.row["ordinal_in_document"]),
                total_score=candidate.total_score,
                lexical_score=candidate.lexical_score,
                semantic_score=candidate.semantic_score,
            )
            for candidate in sorted(combined.values(), key=lambda item: item.total_score, reverse=True)[: active_profile.result_limit]
        ]
        return ranked

    def expand_with_context(self, ranked: list[RankedChunk]) -> list[Citation]:
        document_id = self.resolve_document_id()
        if not document_id or not ranked:
            return []
        chosen_ids: set[str] = set()
        citations: list[Citation] = []
        primary = ranked[0]
        candidate_rows = list(self.store.fetch_context_neighbors(document_id, primary.ordinal_in_document))
        primary_row = self.store.fetch_chunk(primary.chunk_id)
        parent = self.store.fetch_parent(primary_row["parent_chunk_id"]) if primary_row is not None else None
        if parent is not None:
            candidate_rows.insert(0, parent)
        candidate_rows.extend(self.store.fetch_chunks_on_pages(document_id, primary.page_start, primary.page_end, limit=4))
        for row in candidate_rows:
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

    def build_prompt_context(self, citations: list[Citation], page_context: list[str] | None = None) -> str:
        blocks: list[str] = []
        for index, citation in enumerate(citations, start=1):
            label = citation.section_number or "Unnumbered clause"
            blocks.append(
                f"[{index}] Section: {label}\n"
                f"Heading: {citation.heading}\n"
                f"Pages: {citation.page_start}-{citation.page_end}\n"
                f"Excerpt: {citation.quote}"
            )
        if page_context:
            blocks.append("Page context:\n" + "\n\n".join(page_context))
        return "\n\n".join(blocks)

    def build_evidence_pack(self, question: str, ranked: list[RankedChunk], citations: list[Citation]) -> str:
        document_id = self.resolve_document_id()
        if not document_id or not ranked:
            return self.build_prompt_context(citations)
        primary = ranked[0]
        pages = self.store.fetch_page_window(document_id, primary.page_start, primary.page_end, padding=0, limit=2)
        page_blocks = [
            f"Page {int(row['page_num'])}: {self._short_quote(str(row['page_text']), limit=500)}"
            for row in pages
        ]
        return self.build_prompt_context(citations, page_blocks)

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

    def _semantic_candidates(
        self,
        document_id: str,
        query_vector: list[float],
        *,
        limit: int = MAX_SEMANTIC_SCAN,
    ) -> list[tuple[dict, float]]:
        rows = self.store.iter_embeddings(document_id)
        scored: list[tuple[dict, float]] = []
        for row in rows:
            semantic_score = cosine_similarity(query_vector, unpack_vector(row["vector_blob"]))
            if semantic_score > 0:
                scored.append((row, semantic_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _resolve_profile(profile: str, limit: int | None) -> RetrievalProfile:
        selected = DEEP_RETRIEVAL_PROFILE if profile == "deep" else DEFAULT_RETRIEVAL_PROFILE
        if limit is None:
            return selected
        return RetrievalProfile(
            name=selected.name,
            result_limit=limit,
            fts_limit=selected.fts_limit,
            keyword_limit=selected.keyword_limit,
            semantic_limit=selected.semantic_limit,
            use_query_expansion=selected.use_query_expansion,
        )

    def plan_query(self, query: str) -> QueryPlan:
        return plan_query(query, system_vocabulary=self.system_vocabulary())

    def system_vocabulary(self) -> SystemVocabulary:
        document_id = self.resolve_document_id()
        if not document_id:
            return SystemVocabulary(())
        if document_id == self._system_vocabulary_document_id:
            return self._system_vocabulary_cache
        rows = self.store.fetch_document_chunks(document_id)
        self._system_vocabulary_cache = build_contract_system_vocabulary(rows)
        self._system_vocabulary_document_id = document_id
        return self._system_vocabulary_cache

    def _match_queries_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[str]:
        queries = list(build_match_queries(plan))
        if not profile.use_query_expansion:
            return queries
        for variant in self._query_variants(plan):
            queries.extend(build_match_queries(self.plan_query(variant)))
        return self._dedupe_queries(queries)

    def _search_passes_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[SearchPass]:
        passes: list[SearchPass] = []

        attribute_block = self._fts_term_block(plan.attribute_terms[:4])
        concept_block = self._fts_term_block(plan.concept_terms[:5])
        scope_block = self._fts_term_block(plan.scope_terms[:3])

        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST and concept_block:
            heading_concept_phrase = " ".join(plan.concept_terms[:4])
            passes.append(
                SearchPass(
                    "grouped_heading_concept",
                    self._fts_column_block(("heading", "parent_heading"), heading_concept_phrase),
                    1.55,
                )
            )
            passes.append(
                SearchPass(
                    "grouped_concept",
                    concept_block,
                    1.1,
                )
            )
            if scope_block:
                passes.append(
                    SearchPass(
                        "grouped_scope_concept",
                        self._fts_and([concept_block, scope_block]),
                        1.65,
                    )
                )
                passes.append(
                    SearchPass(
                        "grouped_heading_scope",
                        self._fts_and(
                            [
                                self._fts_column_block(("heading", "parent_heading"), heading_concept_phrase),
                                scope_block,
                            ]
                        ),
                        1.95,
                    )
                )

        if plan.system_phrase and attribute_block:
            passes.append(
                SearchPass(
                    "exact_system_attribute",
                    self._fts_and([self._fts_phrase(plan.system_phrase), attribute_block]),
                    1.55,
                )
            )
            passes.append(
                SearchPass(
                    "heading_system_attribute",
                    self._fts_and([self._fts_column_block(("heading", "parent_heading"), plan.system_phrase), attribute_block]),
                    1.8,
                )
            )

        if plan.system_phrase:
            passes.append(
                SearchPass(
                    "exact_system_heading",
                    self._fts_column_block(("heading", "parent_heading"), plan.system_phrase),
                    1.25,
                )
            )
            passes.append(
                SearchPass(
                    "exact_system_text",
                    self._fts_phrase(plan.system_phrase),
                    0.95,
                )
            )

        if attribute_block:
            passes.append(
                SearchPass(
                    "attribute_only",
                    attribute_block,
                    0.45,
                )
            )

        for alias in plan.system_aliases[:2]:
            if not alias:
                continue
            alias_phrase = self._fts_phrase(alias)
            if attribute_block:
                passes.append(
                    SearchPass(
                        "alias_system_attribute",
                        self._fts_and([alias_phrase, attribute_block]),
                        0.7,
                    )
                )
            passes.append(
                SearchPass(
                    "alias_system",
                    alias_phrase,
                    0.35,
                )
            )

        passes.extend(SearchPass("general", query, 0.0) for query in self._match_queries_for_profile(plan, profile))
        return self._dedupe_search_passes(passes)

    def _fallback_queries_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[str]:
        queries = [build_like_fallback(plan)]
        if not profile.use_query_expansion:
            return [query for query in queries if query]
        for variant in self._query_variants(plan):
            queries.append(build_like_fallback(self.plan_query(variant)))
        return self._dedupe_queries(queries)

    def _query_variants(self, plan: QueryPlan) -> list[str]:
        candidates = [plan.raw_query, plan.content_query, plan.system_phrase]
        candidates.extend(plan.scope_terms)
        if plan.concept_terms:
            candidates.append(" ".join(plan.concept_terms))
        candidates.extend(self._focus_phrases(plan.focus_terms))
        return self._dedupe_queries(candidates)

    @staticmethod
    def _dedupe_queries(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for query in queries:
            normalized = " ".join(str(query).split())
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _dedupe_search_passes(search_passes: list[SearchPass]) -> list[SearchPass]:
        seen: set[str] = set()
        deduped: list[SearchPass] = []
        for search_pass in search_passes:
            normalized = " ".join(search_pass.query.split())
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            deduped.append(search_pass)
        return deduped

    @staticmethod
    def _fts_escape(text: str) -> str:
        return " ".join(str(text).replace('"', " ").split())

    @classmethod
    def _fts_phrase(cls, text: str) -> str:
        escaped = cls._fts_escape(text)
        return f'"{escaped}"' if escaped else ""

    @classmethod
    def _fts_term_block(cls, terms: tuple[str, ...]) -> str:
        phrases = [cls._fts_phrase(term) for term in terms if term]
        phrases = [phrase for phrase in phrases if phrase]
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        return "(" + " OR ".join(phrases) + ")"

    @classmethod
    def _fts_column_block(cls, columns: tuple[str, ...], phrase: str) -> str:
        quoted = cls._fts_phrase(phrase)
        if not quoted:
            return ""
        parts = [f"{column} : {quoted}" for column in columns]
        if len(parts) == 1:
            return parts[0]
        return "(" + " OR ".join(parts) + ")"

    @staticmethod
    def _fts_and(parts: list[str]) -> str:
        filtered = [part for part in parts if part]
        if not filtered:
            return ""
        if len(filtered) == 1:
            return filtered[0]
        return "(" + " AND ".join(filtered) + ")"

    @staticmethod
    def _row_vector(row: dict) -> list[float]:
        blob = row["vector_blob"] if "vector_blob" in row.keys() else None
        return unpack_vector(blob) if blob else []

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
            "what are ",
            "define ",
            "tell me about ",
            "show me ",
        ]
        for prefix in prefixes:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        raw = raw.strip(" ?.")
        raw = re.sub(r"\b(my|our|the)\b\s+", "", raw).strip()
        raw = re.sub(r"\b(do we have|do we use|are we using|are there|is there|do we need|do we require)$", "", raw).strip()
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

    def _semantic_for_row(self, query: str, row: dict) -> float:
        vector = self._row_vector(row)
        if vector:
            return cosine_similarity(self.embedder.embed(query), vector)
        return 0.0

    @staticmethod
    def _merge_candidate(combined: dict[str, SearchCandidate], candidate: SearchCandidate) -> None:
        chunk_id = str(candidate.row["chunk_id"])
        existing = combined.get(chunk_id)
        if existing is None or candidate.total_score > existing.total_score:
            combined[chunk_id] = candidate

    def _score_row(
        self,
        plan: QueryPlan,
        row: dict,
        *,
        lexical_score: float,
        semantic_score: float,
        bonus: float = 0.0,
        search_pass_name: str = "",
    ) -> SearchCandidate:
        heading = str(row["heading"])
        parent_heading = str(row["parent_heading"] or "")
        full_text = str(row["full_text"])
        actor_tags = str(row["actor_tags"] or "")
        action_tags = str(row["action_tags"] or "")
        topic_tags = str(row["topic_tags"] or "")
        clause_type = str(row["clause_type"] or row["chunk_type"])
        combined_text = " ".join([heading, parent_heading, full_text, actor_tags, action_tags, topic_tags])
        focus_phrases = self._focus_phrases(plan.focus_terms)
        system_score, exact_system_match = self._system_match_score(plan, heading, parent_heading, combined_text)
        attribute_score, attribute_match = self._attribute_match_score(plan, heading, parent_heading, full_text, combined_text)
        concept_score, concept_match = self._concept_match_score(plan, heading, parent_heading, combined_text)
        scope_score, scope_match = self._scope_match_score(plan, heading, parent_heading, clause_type)
        family_score = self._answer_family_score(plan, full_text, heading, parent_heading)

        score = lexical_score * 0.55 + semantic_score * 0.15 + bonus
        score += system_score
        score += attribute_score
        score += concept_score
        score += scope_score
        score += family_score
        if row["section_number"] and row["section_number"] == plan.section_number:
            score += 1.2
        if plan.content_query:
            if has_term_overlap(heading, (plan.content_query,)):
                score += 0.7
            elif has_term_overlap(parent_heading, (plan.content_query,)):
                score += 0.35
            elif has_term_overlap(full_text, (plan.content_query,)):
                score += 0.45
        for phrase in focus_phrases:
            if has_term_overlap(heading, (phrase,)):
                score += 0.55
            elif has_term_overlap(parent_heading, (phrase,)):
                score += 0.35
        if has_term_overlap(heading, plan.focus_terms):
            score += 0.75
        if has_term_overlap(parent_heading, plan.focus_terms):
            score += 0.25
        if has_term_overlap(heading, plan.topic_terms + plan.action_terms):
            score += 0.8
        if has_term_overlap(parent_heading, plan.topic_terms):
            score += 0.4
        if has_term_overlap(actor_tags, plan.actor_terms):
            score += 0.75
        if has_term_overlap(action_tags, plan.action_terms):
            score += 0.65
        if has_term_overlap(topic_tags, plan.topic_terms):
            score += 0.8
        if has_term_overlap(topic_tags, plan.concept_terms):
            score += 0.45
        if has_term_overlap(combined_text, plan.expansion_terms):
            score += 0.9
        score += 0.16 * self._term_hit_count(combined_text, plan.focus_terms)
        score += 0.18 * self._term_hit_count(combined_text, plan.concept_terms + plan.scope_terms)
        score += 0.12 * self._term_hit_count(combined_text, plan.actor_terms + plan.action_terms + plan.topic_terms + plan.expansion_terms)
        if exact_system_match and attribute_match:
            score += 1.25
        elif plan.system_phrase and plan.attribute_terms:
            if exact_system_match or attribute_match:
                score -= 0.95
            else:
                score -= 1.35
        elif plan.system_phrase and not exact_system_match:
            score -= 1.05
        elif plan.attribute_terms and not attribute_match:
            score -= 0.55
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            if not concept_match:
                score -= 1.2
            if plan.scope_terms and not scope_match:
                score -= 0.45
        if search_pass_name in {"exact_system_attribute", "heading_system_attribute"} and not (exact_system_match and attribute_match):
            score -= 1.25
        if search_pass_name in {"grouped_heading_concept", "grouped_concept"} and not concept_match:
            score -= 0.9
        if search_pass_name in {"grouped_scope_concept", "grouped_heading_scope"} and not (concept_match and (scope_match or not plan.scope_terms)):
            score -= 1.1
        if search_pass_name == "exact_system_heading" and not exact_system_match:
            score -= 0.75
        if plan.intent == "definition" and clause_type == "definition":
            score += 1.0
        if plan.intent == "responsibility" and re.search(r"\b(shall|must|responsible|required)\b", full_text, re.IGNORECASE):
            score += 0.55
            if ("permit" in plan.action_terms or "permitting" in plan.topic_terms) and not has_term_overlap(
                combined_text,
                ("permit", "permits", "permitting", "approval", "approvals"),
            ):
                score -= 0.8
        if plan.intent == "payment_liability" and re.search(r"\b(pay|payment|cost|compensation|damages)\b", full_text, re.IGNORECASE):
            score += 0.5
            if plan.expansion_terms and not has_term_overlap(combined_text, plan.expansion_terms):
                score -= 0.45
            if "late" in plan.normalized_query and not has_term_overlap(
                combined_text,
                ("late", "delay", "completion", "liquidated damages", "substantial completion"),
            ):
                score -= 0.75
        if plan.intent == "termination" and re.search(r"\b(terminate|termination|convenience|default)\b", full_text, re.IGNORECASE):
            score += 0.5
            if "owner" in plan.actor_terms and not has_term_overlap(combined_text, ("owner",)):
                score -= 0.4
            if "convenience" in plan.normalized_query and not has_term_overlap(combined_text, ("convenience",)):
                score -= 0.55
        if plan.intent == "delay_schedule" and re.search(r"\b(delay|weather|completion|schedule|liquidated)\b", full_text, re.IGNORECASE):
            score += 0.5
            if "weather" in plan.normalized_query and not has_term_overlap(combined_text, ("weather",)):
                score -= 0.6
        score -= self._noise_penalty(row)
        return SearchCandidate(
            row=row,
            total_score=score,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
        )

    @staticmethod
    def _system_match_score(plan: QueryPlan, heading: str, parent_heading: str, combined_text: str) -> tuple[float, bool]:
        if not plan.system_phrase and not plan.system_terms:
            return 0.0, False
        score = 0.0
        exact_match = False
        if plan.system_phrase:
            if has_term_overlap(heading, (plan.system_phrase,)):
                score += 1.8
                exact_match = True
            elif has_term_overlap(parent_heading, (plan.system_phrase,)):
                score += 1.15
                exact_match = True
            elif has_term_overlap(combined_text, (plan.system_phrase,)):
                score += 0.8
                exact_match = True
        for alias in plan.system_aliases:
            if has_term_overlap(heading, (alias,)) or has_term_overlap(parent_heading, (alias,)):
                score += 0.55
                break
        if plan.system_terms:
            hit_count = HybridRetriever._term_hit_count(combined_text, plan.system_terms)
            if hit_count:
                score += 0.22 * hit_count
            significant_hits = HybridRetriever._term_hit_count(combined_text, plan.system_significant_terms)
            if significant_hits:
                score += 0.34 * significant_hits
            if plan.system_significant_terms:
                exact_match = exact_match or significant_hits >= len(plan.system_significant_terms)
            else:
                exact_match = exact_match or hit_count >= max(1, min(len(plan.system_terms), 2))
        return score, exact_match

    @staticmethod
    def _attribute_match_score(
        plan: QueryPlan,
        heading: str,
        parent_heading: str,
        full_text: str,
        combined_text: str,
    ) -> tuple[float, bool]:
        if not plan.attribute_terms and not plan.attribute_label:
            return 0.0, False
        score = 0.0
        matched = False
        for phrase in plan.attribute_terms:
            if has_term_overlap(heading, (phrase,)) or has_term_overlap(parent_heading, (phrase,)):
                score += 0.95
                matched = True
            elif has_term_overlap(combined_text, (phrase,)):
                score += 0.45
                matched = True
        label = plan.attribute_label or ""
        lowered_full_text = full_text.lower()
        if label == "design_conditions":
            if re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", lowered_full_text):
                score += 1.1
                matched = True
            value_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:psi|psig|psia|degf|deg c|gpm|lb/hr|scfm|mw|mva|hp)\b", lowered_full_text, re.IGNORECASE))
            score += min(0.75, value_hits * 0.18)
        elif label == "configuration":
            if re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b", lowered_full_text):
                score += 1.0
                matched = True
        elif label == "type":
            if re.search(r"\bmodel\b|\btype\b|\bselected\b|\bmanufacturer\b|\bvendor\b", lowered_full_text):
                score += 1.0
                matched = True
            if re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*\b", full_text):
                score += 0.75
                matched = True
        elif label == "size":
            if re.search(r"\bsize\b|\bdiameter\b|\brating\b|\b(?:inch|inches|mm|ft)\b", lowered_full_text):
                score += 0.95
                matched = True
        elif label == "power":
            if re.search(r"\bhorse\s*power\b|\bhp\b|\bkw\b|\bkilowatt", lowered_full_text):
                score += 1.2
                matched = True
            value_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:hp|kw|kilowatt(?:s)?)\b", full_text, re.IGNORECASE))
            score += min(0.9, value_hits * 0.25)
        elif label == "function":
            if re.search(r"\b(receives|supplies|provides|distributes|used to|serves to|operates)\b", lowered_full_text):
                score += 0.8
                matched = True
        elif label == "responsibility":
            if re.search(r"\b(shall|must|responsible|provide|furnish|supply|required)\b", lowered_full_text):
                score += 0.65
                matched = True
        return score, matched

    @staticmethod
    def _concept_match_score(plan: QueryPlan, heading: str, parent_heading: str, combined_text: str) -> tuple[float, bool]:
        if not plan.concept_terms:
            return 0.0, False
        score = 0.0
        if has_term_overlap(heading, plan.concept_terms):
            score += 1.1
        if has_term_overlap(parent_heading, plan.concept_terms):
            score += 0.8
        hits = HybridRetriever._term_hit_count(combined_text, plan.concept_terms)
        if hits:
            score += min(1.4, hits * 0.28)
        matched = hits >= max(1, min(len(plan.concept_terms), 2))
        return score, matched

    @staticmethod
    def _scope_match_score(plan: QueryPlan, heading: str, parent_heading: str, clause_type: str) -> tuple[float, bool]:
        if not plan.scope_terms:
            return 0.0, False
        score = 0.0
        matched = False
        if has_term_overlap(heading, plan.scope_terms):
            score += 1.2
            matched = True
        if has_term_overlap(parent_heading, plan.scope_terms):
            score += 0.9
            matched = True
        if clause_type == "exhibit":
            score += 0.35
            matched = True
        return score, matched

    @staticmethod
    def _answer_family_score(plan: QueryPlan, full_text: str, heading: str, parent_heading: str) -> float:
        if plan.answer_family != ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
            return 0.0
        normalized = " ".join([heading, parent_heading, full_text]).lower()
        score = 0.0
        if re.search(r"\bguarantee\b|\bguarantees\b|\bguaranteed\b", normalized):
            score += 0.9
        if re.search(r"\bshall not exceed\b|\bnot exceed\b|\blimit\b|\blimits\b", normalized):
            score += 1.1
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:ppmvd|ppmv|lb/hr|mg/nm3|%)\b", normalized):
            score += 0.95
        return score

    @staticmethod
    def _term_hit_count(text: str, terms: tuple[str, ...]) -> int:
        normalized = text.lower()
        return sum(1 for term in terms if term and term.lower() in normalized)

    @staticmethod
    def _noise_penalty(row: dict) -> float:
        penalty = 0.0
        heading = str(row["heading"])
        full_text = str(row["full_text"])
        if "...." in heading:
            penalty += 0.45
        if full_text.count("....") >= 3 and int(row["page_start"]) <= 20:
            penalty += 0.45
        noise_flags = set(str(row["noise_flags"] or "").split())
        if "date_heading" in noise_flags:
            penalty += 0.75
        if "thin" in noise_flags:
            penalty += 0.2
        if str(row["section_number"] or "").strip() == "0":
            penalty += 0.55
        return penalty

    @staticmethod
    def _focus_phrases(focus_terms: tuple[str, ...]) -> tuple[str, ...]:
        if len(focus_terms) < 2:
            return ()
        phrases = [" ".join(focus_terms[index:index + 2]) for index in range(len(focus_terms) - 1)]
        if len(focus_terms) >= 3:
            phrases.append(" ".join(focus_terms[:3]))
        seen: set[str] = set()
        return tuple(phrase for phrase in phrases if phrase and not (phrase in seen or seen.add(phrase)))
