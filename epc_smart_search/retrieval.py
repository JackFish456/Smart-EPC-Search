from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

from epc_smart_search.config import MAX_EMBEDDING_DIM, MAX_SEARCH_RESULTS, MAX_SEMANTIC_SCAN
from epc_smart_search.name_normalization import build_system_aliases, normalize_attribute_name, normalize_system_name
from epc_smart_search.query_planner import (
    ANSWER_FAMILY_GUARANTEE_OR_LIMIT,
    EXACT_VALUE_ATTRIBUTE_LABELS,
    QueryPlan,
    REQUEST_SHAPE_BROAD_TOPIC,
    REQUEST_SHAPE_GROUPED_LIST,
    RETRIEVAL_MODE_FACT_LOOKUP,
    build_like_fallback,
    build_match_queries,
    has_term_overlap,
    plan_query,
)
from epc_smart_search.storage import ContractFactRow, ContractStore, unpack_vector
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
    source_names: tuple[str, ...] = ()


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


@dataclass(slots=True)
class EvidenceBundle:
    bundle_id: str
    primary_chunk_id: str
    ranked_chunks: tuple[RankedChunk, ...]
    citations: tuple[Citation, ...]
    bundle_score: float
    confidence: float
    source_names: tuple[str, ...]
    gemma_selected: bool = False
    supporting_quote: str = ""


@dataclass(slots=True)
class RetrievalTrace:
    query: str
    plan: QueryPlan
    recall_sources: dict[str, list[RankedChunk]]
    merged_ranked: list[RankedChunk]
    bundles: list[EvidenceBundle]
    selected_bundle: EvidenceBundle | None
    used_gemma_disambiguation: bool = False


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
SEMANTIC_RECALL_KEEP = 20
LEXICAL_RECALL_KEEP = 20
MERGED_RECALL_KEEP = 24
GROUPED_MERGED_RECALL_KEEP = 40
BROAD_TOPIC_MERGED_RECALL_KEEP = 40
DEFAULT_FINAL_RANKED_KEEP = 8
DEEP_FINAL_RANKED_KEEP = 10
GEMMA_DISAMBIGUATION_TOP_K = 3


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
        trace = self.retrieve_trace(query, limit=limit, profile=profile)
        return trace.merged_ranked

    def retrieve_trace(
        self,
        query: str,
        *,
        limit: int | None = None,
        profile: str = "normal",
        gemma_client=None,
    ) -> RetrievalTrace:
        document_id = self.resolve_document_id()
        if not document_id:
            return RetrievalTrace(query, self.plan_query(query), {}, [], [], None, False)
        plan = self.plan_query(query)
        active_profile = self._resolve_profile(profile, limit)
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            active_profile = RetrievalProfile(
                name=active_profile.name,
                result_limit=max(active_profile.result_limit, DEFAULT_FINAL_RANKED_KEEP),
                fts_limit=max(active_profile.fts_limit, 36),
                keyword_limit=max(active_profile.keyword_limit, 24),
                semantic_limit=max(active_profile.semantic_limit, MAX_SEMANTIC_SCAN),
                use_query_expansion=active_profile.use_query_expansion,
            )
        elif plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC:
            active_profile = RetrievalProfile(
                name=active_profile.name,
                result_limit=max(active_profile.result_limit, DEFAULT_FINAL_RANKED_KEEP),
                fts_limit=max(active_profile.fts_limit, 40),
                keyword_limit=max(active_profile.keyword_limit, 24),
                semantic_limit=max(active_profile.semantic_limit, MAX_SEMANTIC_SCAN),
                use_query_expansion=True,
            )
        query_vector = self.embedder.embed(query)
        recall_sources: dict[str, list[RankedChunk]] = {}
        combined: dict[str, SearchCandidate] = {}
        if plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP:
            fact_sources, fact_candidates, fact_confident = self._collect_fact_lookup_candidates(
                document_id,
                query,
                query_vector,
                plan,
            )
            recall_sources.update(fact_sources)
            combined.update(fact_candidates)
            if not fact_confident:
                fallback_sources, fallback_candidates = self._collect_recall_candidates(
                    document_id,
                    query,
                    query_vector,
                    plan,
                    active_profile,
                )
                recall_sources.update(fallback_sources)
                for candidate in fallback_candidates.values():
                    self._merge_recall_candidate(combined, candidate)
        else:
            recall_sources, combined = self._collect_recall_candidates(document_id, query, query_vector, plan, active_profile)
        if plan.request_shape == REQUEST_SHAPE_GROUPED_LIST:
            merged_limit = GROUPED_MERGED_RECALL_KEEP
        elif plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC:
            merged_limit = BROAD_TOPIC_MERGED_RECALL_KEEP
        else:
            merged_limit = MERGED_RECALL_KEEP
        merged = sorted(combined.values(), key=lambda item: item.total_score, reverse=True)[:merged_limit]
        bundles = self._build_ranked_bundles(document_id, query_vector, plan, merged, active_profile)
        selected_bundle = bundles[0] if bundles else None
        used_gemma_disambiguation = False
        if gemma_client is not None and bundles:
            gemma_bundle = self._select_bundle_with_gemma(query, bundles, gemma_client)
            if gemma_bundle is not None:
                selected_bundle = gemma_bundle
                used_gemma_disambiguation = True
                selected_bundle.gemma_selected = True
        merged_ranked: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        max_depth = max((len(bundle.ranked_chunks) for bundle in bundles), default=0)
        for depth in range(max_depth):
            for bundle in bundles:
                if depth >= len(bundle.ranked_chunks):
                    continue
                chunk = bundle.ranked_chunks[depth]
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                merged_ranked.append(chunk)
                if len(merged_ranked) >= active_profile.result_limit:
                    break
            if len(merged_ranked) >= active_profile.result_limit:
                break
        return RetrievalTrace(
            query=query,
            plan=plan,
            recall_sources=recall_sources,
            merged_ranked=merged_ranked,
            bundles=bundles,
            selected_bundle=selected_bundle,
            used_gemma_disambiguation=used_gemma_disambiguation,
        )

    def _collect_recall_candidates(
        self,
        document_id: str,
        query: str,
        query_vector: list[float],
        plan: QueryPlan,
        profile: RetrievalProfile,
    ) -> tuple[dict[str, list[RankedChunk]], dict[str, SearchCandidate]]:
        combined: dict[str, SearchCandidate] = {}
        recall_sources: dict[str, list[RankedChunk]] = {}

        if plan.section_number:
            rows = self.store.section_lookup(document_id, plan.section_number)
            recall_sources["section_lookup"] = []
            for row in rows:
                candidate = self._make_recall_candidate(
                    query,
                    plan,
                    row,
                    lexical_score=1.0,
                    semantic_score=0.85,
                    bonus=0.9,
                    source_name="section_lookup",
                )
                recall_sources["section_lookup"].append(self._ranked_from_candidate(candidate))
                self._merge_recall_candidate(combined, candidate)

        raw_match_query = build_match_query(query)
        if raw_match_query:
            rows = self.store.search_chunk_feature_fts(document_id, raw_match_query, limit=LEXICAL_RECALL_KEEP)
            recall_sources["raw_fts"] = []
            for row in rows:
                lexical_score = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
                semantic_score = self._semantic_for_row(query_vector, row)
                candidate = self._make_recall_candidate(
                    query,
                    plan,
                    row,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    source_name="raw_fts",
                )
                recall_sources["raw_fts"].append(self._ranked_from_candidate(candidate))
                self._merge_recall_candidate(combined, candidate)

        recall_sources["planner_hints"] = []
        for search_pass in self._search_passes_for_profile(plan, profile):
            rows = self.store.search_chunk_feature_fts(document_id, search_pass.query, limit=profile.fts_limit)
            for row in rows:
                lexical_score = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
                semantic_score = self._semantic_for_row(query_vector, row)
                candidate = self._make_recall_candidate(
                    query,
                    plan,
                    row,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    bonus=min(search_pass.bonus, 0.8) * 0.2,
                    source_name=f"hint:{search_pass.name}",
                    search_pass_name=search_pass.name,
                )
                recall_sources["planner_hints"].append(self._ranked_from_candidate(candidate))
                self._merge_recall_candidate(combined, candidate)

        recall_sources["keyword"] = []
        for fallback_query in self._fallback_queries_for_profile(plan, profile):
            rows = self.store.keyword_like_search(document_id, fallback_query, limit=profile.keyword_limit)
            for row in rows:
                semantic_score = self._semantic_for_row(query_vector, row)
                candidate = self._make_recall_candidate(
                    query,
                    plan,
                    row,
                    lexical_score=0.2,
                    semantic_score=semantic_score,
                    source_name="keyword",
                )
                recall_sources["keyword"].append(self._ranked_from_candidate(candidate))
                self._merge_recall_candidate(combined, candidate)

        recall_sources["semantic"] = []
        for row, semantic_score in self._semantic_candidates(
            document_id,
            query_vector,
            limit=max(profile.semantic_limit, MAX_SEMANTIC_SCAN),
        ):
            candidate = self._make_recall_candidate(
                query,
                plan,
                row,
                lexical_score=0.0,
                semantic_score=semantic_score,
                source_name="semantic",
            )
            recall_sources["semantic"].append(self._ranked_from_candidate(candidate))
            self._merge_recall_candidate(combined, candidate)

        for source_name, rows in list(recall_sources.items()):
            recall_sources[source_name] = sorted(rows, key=lambda item: item.total_score, reverse=True)[:LEXICAL_RECALL_KEEP]
        return recall_sources, combined

    def _collect_fact_lookup_candidates(
        self,
        document_id: str,
        query: str,
        query_vector: list[float],
        plan: QueryPlan,
    ) -> tuple[dict[str, list[RankedChunk]], dict[str, SearchCandidate], bool]:
        combined: dict[str, SearchCandidate] = {}
        recall_sources: dict[str, list[RankedChunk]] = {"fact_lookup": []}
        best_match_score = 0.0
        for fact in self._candidate_facts_for_plan(document_id, plan):
            match_score = self._fact_match_score(plan, fact)
            if match_score <= 0.0:
                continue
            row = self.store.fetch_chunk(fact.source_chunk_id)
            if row is None:
                continue
            candidate = self._make_recall_candidate(
                query,
                plan,
                row,
                lexical_score=0.0,
                semantic_score=self._semantic_for_row(query_vector, row),
                bonus=min(2.4, match_score),
                source_name="fact_lookup",
            )
            recall_sources["fact_lookup"].append(self._ranked_from_candidate(candidate))
            self._merge_recall_candidate(combined, candidate)
            best_match_score = max(best_match_score, match_score)
        recall_sources["fact_lookup"] = sorted(
            recall_sources["fact_lookup"],
            key=lambda item: item.total_score,
            reverse=True,
        )[:LEXICAL_RECALL_KEEP]
        return recall_sources, combined, best_match_score >= 2.0

    def _candidate_facts_for_plan(self, document_id: str, plan: QueryPlan) -> list[ContractFactRow]:
        if plan.attribute_label not in EXACT_VALUE_ATTRIBUTE_LABELS or not plan.system_phrase:
            return []
        for system_name in (plan.system_phrase, *plan.system_aliases):
            if not system_name:
                continue
            direct = self.store.lookup_facts_by_system_attribute(document_id, system_name, plan.attribute_label)
            if direct:
                return direct
        return self.store.lookup_facts_by_attribute(document_id, plan.attribute_label)

    @staticmethod
    def _fact_match_score(plan: QueryPlan, fact: ContractFactRow) -> float:
        if not plan.system_phrase or not fact.attribute_normalized:
            return 0.0
        plan_system = normalize_system_name(plan.system_phrase)
        fact_system = normalize_system_name(fact.system_normalized or fact.system)
        if not fact_system:
            return 0.0

        plan_attribute = normalize_attribute_name(plan.attribute_label or "")
        fact_attribute = normalize_attribute_name(fact.attribute_normalized or fact.attribute)
        if plan_attribute and fact_attribute != plan_attribute:
            return 0.0

        score = 1.0
        plan_systems = {plan_system}
        plan_systems.update(
            normalize_system_name(alias)
            for alias in build_system_aliases(plan.system_phrase) + plan.system_aliases
            if alias
        )
        plan_systems.discard("")

        if fact_system in plan_systems:
            score += 1.5
        elif len(plan.system_significant_terms) <= 1:
            system_hits = sum(1 for term in plan.system_terms if term and f" {term} " in f" {fact_system} ")
            significant_hits = sum(1 for term in plan.system_significant_terms if term and f" {term} " in f" {fact_system} ")
            if not system_hits and not significant_hits:
                return 0.0
            score += 0.3 * system_hits
            score += 0.45 * significant_hits
        else:
            return 0.0
        if plan.content_query:
            normalized_evidence = f"{fact.evidence_text} {fact.value}".lower()
            if plan.content_query in normalized_evidence:
                score += 0.3
        if plan.attribute_label == "type" and re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\d[A-Z0-9\-]*\b", fact.value):
            score += 0.35
        if plan.attribute_label in {"capacity", "size", "power", "pressure", "temperature", "flow"} and re.search(r"\d", fact.value):
            score += 0.25
        return score

    def _build_ranked_bundles(
        self,
        document_id: str,
        query_vector: list[float],
        plan: QueryPlan,
        merged: Sequence[SearchCandidate],
        profile: RetrievalProfile,
    ) -> list[EvidenceBundle]:
        bundles = [
            self._build_bundle(document_id, query_vector, plan, candidate)
            for candidate in merged
        ]
        bundles = [bundle for bundle in bundles if bundle.ranked_chunks]
        bundles.sort(key=lambda item: item.bundle_score, reverse=True)
        limit = DEEP_FINAL_RANKED_KEEP if profile.name == "deep" else DEFAULT_FINAL_RANKED_KEEP
        return bundles[:limit]

    def _build_bundle(
        self,
        document_id: str,
        query_vector: list[float],
        plan: QueryPlan,
        candidate: SearchCandidate,
    ) -> EvidenceBundle:
        rows = self._bundle_rows_for_candidate(document_id, candidate.row)
        ranked_chunks = self._rank_bundle_rows(query_vector, plan, rows, candidate)
        citations = self._citations_from_rows(rows)
        bundle_text = "\n".join(" ".join(str(row["full_text"]).split()) for row in rows if str(row["full_text"]).strip())
        bundle_semantic = cosine_similarity(query_vector, self.embedder.embed(bundle_text)) if bundle_text else 0.0
        primary_chunk_score = ranked_chunks[0].total_score if ranked_chunks else 0.0
        bundle_lower = (
            " "
            + " ".join(f"{row['heading']} {row['full_text']}" for row in rows)
            + " "
        ).lower()
        system_binding = self._bundle_system_binding(plan, bundle_lower)
        attribute_binding = self._bundle_attribute_binding(plan, bundle_lower)
        support_binding = self._bundle_support_binding(plan, rows)
        attachment_bonus = 0.15 if any(str(row["chunk_type"]).lower() == "exhibit" for row in rows) else 0.0
        thin_penalty = 0.4 if self._is_low_content_body(str(candidate.row["full_text"]), str(candidate.row["section_number"] or "")) else 0.0
        bundle_score = (
            primary_chunk_score * 0.22
            + candidate.total_score * 0.18
            + bundle_semantic * 1.35
            + system_binding
            + attribute_binding
            + support_binding
            + attachment_bonus
            - thin_penalty
        )
        confidence = max(0.0, min(1.0, 0.46 + bundle_semantic * 0.35 + system_binding * 0.08 + attribute_binding * 0.08 + support_binding * 0.05))
        return EvidenceBundle(
            bundle_id=str(candidate.row["chunk_id"]),
            primary_chunk_id=str(candidate.row["chunk_id"]),
            ranked_chunks=tuple(ranked_chunks),
            citations=tuple(citations),
            bundle_score=bundle_score,
            confidence=confidence,
            source_names=candidate.source_names,
        )

    def _bundle_rows_for_candidate(self, document_id: str, row: dict) -> list[dict]:
        rows: list[dict] = [row]
        parent_chunk_id = self._row_value(row, "parent_chunk_id")
        parent = self.store.fetch_parent(parent_chunk_id)
        if parent is not None:
            rows.append(parent)
        page_start = int(self._row_value(row, "page_start", 0) or 0)
        page_end = int(self._row_value(row, "page_end", 0) or 0)
        is_exhibit_anchor = str(self._row_value(row, "chunk_type", "")).lower() == "exhibit" or bool(
            re.match(r"^(appendix|attachment|exhibit)\b", str(self._row_value(row, "heading", "")), flags=re.IGNORECASE)
        )
        rows.extend(self.store.fetch_children(str(self._row_value(row, "chunk_id", "")), limit=4))
        rows.extend(self.store.fetch_context_neighbors(document_id, int(self._row_value(row, "ordinal_in_document", 0) or 0)))
        rows.extend(
            self.store.fetch_chunks_on_pages(
                document_id,
                page_start,
                page_end + (4 if is_exhibit_anchor else 0),
                limit=12 if is_exhibit_anchor else 6,
            )
        )
        deduped: list[dict] = []
        seen_ids: set[str] = set()
        for candidate_row in rows:
            chunk_id = str(candidate_row["chunk_id"])
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            deduped.append(candidate_row)
        return deduped

    def _rank_bundle_rows(
        self,
        query_vector: list[float],
        plan: QueryPlan,
        rows: Sequence[dict],
        primary_candidate: SearchCandidate,
    ) -> list[RankedChunk]:
        ranked: list[RankedChunk] = []
        primary_id = str(primary_candidate.row["chunk_id"])
        for row in rows:
            chunk_id = str(row["chunk_id"])
            lexical_score = primary_candidate.lexical_score if chunk_id == primary_id else 0.0
            semantic_score = self._semantic_for_row(query_vector, row)
            scored = self._score_row(plan, row, lexical_score=lexical_score, semantic_score=semantic_score)
            ranked.append(self._ranked_from_candidate(scored))
        ranked.sort(
            key=lambda item: (
                item.total_score,
                item.semantic_score,
                1 if item.chunk_id == primary_id else 0,
            ),
            reverse=True,
        )
        return ranked

    def _select_bundle_with_gemma(
        self,
        query: str,
        bundles: Sequence[EvidenceBundle],
        gemma_client,
    ) -> EvidenceBundle | None:
        if not self._should_use_gemma_disambiguation(bundles):
            return None
        prompt_context = self._build_disambiguation_context(query, bundles[:GEMMA_DISAMBIGUATION_TOP_K])
        try:
            raw = gemma_client.ask(query, prompt_context, response_style="candidate_select", max_new_tokens=192)
        except Exception:
            return None
        bundle_id, supporting_quote = self._parse_gemma_bundle_selection(raw)
        if not bundle_id:
            return None
        for bundle in bundles[:GEMMA_DISAMBIGUATION_TOP_K]:
            if bundle.bundle_id == bundle_id:
                bundle.supporting_quote = supporting_quote
                return bundle
        return None

    @staticmethod
    def _should_use_gemma_disambiguation(bundles: Sequence[EvidenceBundle]) -> bool:
        if len(bundles) < 2:
            return False
        top = bundles[0]
        second = bundles[1]
        return (top.bundle_score - second.bundle_score) < 0.35 or top.confidence < 0.62

    def _build_disambiguation_context(self, query: str, bundles: Sequence[EvidenceBundle]) -> str:
        blocks: list[str] = []
        for bundle in bundles:
            section = bundle.ranked_chunks[0].section_number if bundle.ranked_chunks else ""
            heading = bundle.ranked_chunks[0].heading if bundle.ranked_chunks else ""
            pages = self._format_page_range(bundle.ranked_chunks[0].page_start, bundle.ranked_chunks[0].page_end) if bundle.ranked_chunks else ""
            excerpts = [f"- {self._short_quote(chunk.full_text, limit=220)}" for chunk in bundle.ranked_chunks[:3] if chunk.full_text]
            blocks.append(
                f"Candidate ID: {bundle.bundle_id}\n"
                f"Section: {section or 'Unnumbered clause'}\n"
                f"Heading: {heading}\n"
                f"Pages: {pages}\n"
                f"Evidence:\n" + "\n".join(excerpts)
            )
        return (
            "Select the single best-supported candidate that directly answers the user question.\n"
            "Return JSON only in this format: "
            '{"candidate_id":"...", "supporting_quote":"...", "insufficient_support":false}\n\n'
            f"User question:\n{query}\n\n"
            "Candidates:\n" + "\n\n".join(blocks)
        )

    @staticmethod
    def _parse_gemma_bundle_selection(raw: str) -> tuple[str, str]:
        compact = str(raw or "").strip()
        match = re.search(r'"candidate_id"\s*:\s*"([^"]+)"', compact)
        quote_match = re.search(r'"supporting_quote"\s*:\s*"([^"]*)"', compact)
        insufficient_match = re.search(r'"insufficient_support"\s*:\s*(true|false)', compact, re.IGNORECASE)
        if insufficient_match and insufficient_match.group(1).lower() == "true":
            return "", ""
        return (match.group(1).strip() if match else "", quote_match.group(1).strip() if quote_match else "")

    def _make_recall_candidate(
        self,
        query: str,
        plan: QueryPlan,
        row: dict,
        *,
        lexical_score: float,
        semantic_score: float,
        bonus: float = 0.0,
        source_name: str,
        search_pass_name: str = "",
    ) -> SearchCandidate:
        recall_score = self._recall_score(plan, row, lexical_score=lexical_score, semantic_score=semantic_score, bonus=bonus)
        candidate = self._score_row(
            plan,
            row,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            bonus=bonus,
            search_pass_name=search_pass_name,
        )
        candidate.total_score = recall_score
        candidate.source_names = (source_name,)
        return candidate

    @staticmethod
    def _recall_score(
        plan: QueryPlan,
        row: dict,
        *,
        lexical_score: float,
        semantic_score: float,
        bonus: float = 0.0,
    ) -> float:
        score = semantic_score * 0.62 + lexical_score * 0.38 + bonus
        section_number = str(HybridRetriever._row_value(row, "section_number", "") or "")
        full_text = str(HybridRetriever._row_value(row, "full_text", "") or "")
        if plan.section_number and section_number == plan.section_number:
            score += 0.75
        if HybridRetriever._is_low_content_body(full_text, section_number):
            score -= 0.15
        return score

    @staticmethod
    def _merge_recall_candidate(combined: dict[str, SearchCandidate], candidate: SearchCandidate) -> None:
        chunk_id = str(candidate.row["chunk_id"])
        existing = combined.get(chunk_id)
        if existing is None:
            combined[chunk_id] = candidate
            return
        source_names = tuple(sorted(set(existing.source_names + candidate.source_names)))
        if candidate.total_score > existing.total_score:
            candidate.source_names = source_names
            combined[chunk_id] = candidate
            return
        existing.source_names = source_names

    def _citations_from_rows(self, rows: Sequence[dict], *, limit: int = 8) -> list[Citation]:
        citations: list[Citation] = []
        seen_ids: set[str] = set()
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    section_number=row["section_number"],
                    heading=self._compact_heading(str(row["heading"])),
                    attachment=self._attachment_label(chunk_id),
                    page_start=int(row["page_start"]),
                    page_end=int(row["page_end"]),
                    quote=self._short_quote(str(row["full_text"])),
                )
            )
            if len(citations) >= limit:
                break
        return citations

    @staticmethod
    def _bundle_system_binding(plan: QueryPlan, bundle_text: str) -> float:
        if not plan.system_phrase and not plan.system_terms:
            return 0.0
        score = 0.0
        if plan.system_phrase and plan.system_phrase in bundle_text:
            score += 0.55
        score += 0.12 * sum(1 for term in plan.system_terms if term and term in bundle_text)
        score += 0.18 * sum(1 for term in plan.system_significant_terms if term and term in bundle_text)
        return score

    @staticmethod
    def _bundle_attribute_binding(plan: QueryPlan, bundle_text: str) -> float:
        if not plan.attribute_terms and not plan.attribute_label:
            return 0.0
        score = 0.1 * sum(1 for term in plan.attribute_terms if term and term in bundle_text)
        label = plan.attribute_label or ""
        if label == "design_conditions" and re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", bundle_text):
            score += 0.45
        elif label == "configuration" and re.search(r"\bconfiguration\b|\bconfigured\b|\barrangement\b|\bduplex\b|\bredun|\bstandby\b|\blead\b|\blag\b", bundle_text):
            score += 0.42
        elif label == "type" and re.search(r"\bmodel\b|\btype\b|\bselected\b|\bmanufacturer\b|\bvendor\b", bundle_text):
            score += 0.42
        elif label in {"pressure", "temperature", "flow", "size", "capacity", "power"} and re.search(r"\b\d", bundle_text):
            score += 0.3
        elif label == "function" and re.search(r"\b(receives|supplies|provides|distributes|conditions|transports|serves|operates)\b", bundle_text):
            score += 0.35
        return score

    @staticmethod
    def _bundle_support_binding(plan: QueryPlan, rows: Sequence[dict]) -> float:
        if not rows:
            return 0.0
        support = 0.0
        if any(HybridRetriever._row_value(row, "parent_chunk_id") for row in rows):
            support += 0.08
        primary_chunk_id = str(HybridRetriever._row_value(rows[0], "chunk_id", ""))
        if any(str(HybridRetriever._row_value(row, "parent_chunk_id", "") or "") == primary_chunk_id for row in rows[1:]):
            support += 0.15
        if len({int(HybridRetriever._row_value(row, "page_start", 0) or 0) for row in rows}) == 1 and len(rows) > 1:
            support += 0.12
        if plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT:
            normalized = " ".join(str(HybridRetriever._row_value(row, "full_text", "")) for row in rows).lower()
            if re.search(r"\bguarantee\b|\bshall not exceed\b|\bnot exceed\b|\blimits?\b", normalized):
                support += 0.18
        return support

    @staticmethod
    def _format_page_range(page_start: int, page_end: int) -> str:
        return f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"

    @staticmethod
    def _ranked_from_candidate(candidate: SearchCandidate) -> RankedChunk:
        row = candidate.row
        return RankedChunk(
            chunk_id=str(row["chunk_id"]),
            section_number=row["section_number"],
            heading=str(row["heading"]),
            full_text=str(row["full_text"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            ordinal_in_document=int(row["ordinal_in_document"]),
            total_score=candidate.total_score,
            lexical_score=candidate.lexical_score,
            semantic_score=candidate.semantic_score,
        )

    def expand_with_context(self, ranked: list[RankedChunk]) -> list[Citation]:
        document_id = self.resolve_document_id()
        if not document_id or not ranked:
            return []
        chosen_ids: set[str] = set()
        citations: list[Citation] = []
        primary = self._select_primary_ranked_chunk(ranked)
        primary_row = self.store.fetch_chunk(primary.chunk_id)
        candidate_rows = [primary_row] if primary_row is not None else []
        parent = self.store.fetch_parent(primary_row["parent_chunk_id"]) if primary_row is not None else None
        if parent is not None:
            candidate_rows.append(parent)
        candidate_rows.extend(self.store.fetch_context_neighbors(document_id, primary.ordinal_in_document))
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
        primary = self._select_primary_ranked_chunk(ranked)
        pages = self.store.fetch_page_window(document_id, primary.page_start, primary.page_end, padding=0, limit=2)
        page_blocks = [
            f"Page {int(row['page_num'])}: {self._short_quote(str(row['page_text']), limit=500)}"
            for row in pages
        ]
        return self.build_prompt_context(citations, page_blocks)

    def build_bundle_evidence_pack(self, bundle: EvidenceBundle | None) -> str:
        document_id = self.resolve_document_id()
        if bundle is None:
            return ""
        citations = list(bundle.citations)
        if not document_id or not bundle.ranked_chunks:
            return self.build_prompt_context(citations)
        primary = self._select_primary_ranked_chunk(list(bundle.ranked_chunks))
        pages = self.store.fetch_page_window(document_id, primary.page_start, primary.page_end, padding=0, limit=3)
        page_blocks = [
            f"Page {int(row['page_num'])}: {self._short_quote(str(row['page_text']), limit=500)}"
            for row in pages
        ]
        if bundle.supporting_quote:
            page_blocks.insert(0, f"Selected support: {bundle.supporting_quote}")
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
        guarantee_family_terms = tuple(
            term for term in plan.expansion_terms if term in {"guarantee", "guarantees", "shall not exceed", "nox", "co", "ppmvd", "emissions"}
        )
        guarantee_family_block = self._fts_term_block(guarantee_family_terms[:6])
        broad_topic_terms = tuple(
            term
            for term in (
                plan.concept_terms
                + plan.focus_terms
                + ("requirements", "required", "shall", "must", "permit", "permits", "compliance", "test", "testing", "demonstrate")
            )
            if term
        )
        broad_topic_block = self._fts_term_block(self._dedupe_queries(list(broad_topic_terms))[:8])

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
            if plan.answer_family == ANSWER_FAMILY_GUARANTEE_OR_LIMIT and guarantee_family_block:
                passes.append(
                    SearchPass(
                        "grouped_guarantee_family",
                        guarantee_family_block,
                        1.45,
                    )
                )
                if scope_block:
                    passes.append(
                        SearchPass(
                            "grouped_scope_guarantee_family",
                            self._fts_and([guarantee_family_block, scope_block]),
                            1.9,
                    )
                )

        if plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC and concept_block:
            heading_concept_phrase = " ".join(plan.concept_terms[:4] or plan.focus_terms[:4])
            if heading_concept_phrase:
                passes.append(
                    SearchPass(
                        "broad_topic_heading_concept",
                        self._fts_column_block(("heading", "parent_heading"), heading_concept_phrase),
                        1.25,
                    )
                )
            passes.append(
                SearchPass(
                    "broad_topic_concept",
                    concept_block,
                    0.95,
                )
            )
            if broad_topic_block:
                passes.append(
                    SearchPass(
                        "broad_topic_requirement_signal",
                        self._fts_and([concept_block, broad_topic_block]),
                        1.45,
                    )
                )
            if scope_block:
                passes.append(
                    SearchPass(
                        "broad_topic_scope_concept",
                        self._fts_and([concept_block, scope_block]),
                        1.55,
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

    def _semantic_for_row(self, query_vector: list[float], row: dict) -> float:
        vector = self._row_vector(row)
        if vector:
            return cosine_similarity(query_vector, vector)
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
        heading = str(self._row_value(row, "heading", ""))
        parent_heading = str(self._row_value(row, "parent_heading", "") or "")
        full_text = str(self._row_value(row, "full_text", ""))
        actor_tags = str(self._row_value(row, "actor_tags", "") or "")
        action_tags = str(self._row_value(row, "action_tags", "") or "")
        topic_tags = str(self._row_value(row, "topic_tags", "") or "")
        clause_type = str(self._row_value(row, "clause_type", self._row_value(row, "chunk_type", "")) or "")
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
        if self._row_value(row, "section_number") and self._row_value(row, "section_number") == plan.section_number:
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
        if plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC:
            score += self._broad_topic_support_score(heading, parent_heading, full_text, clause_type)
            score -= self._broad_topic_noise_penalty(heading, parent_heading, full_text, clause_type)
            if not concept_match:
                score -= 0.45
            if plan.scope_terms and not scope_match:
                score -= 0.6
        if search_pass_name in {"exact_system_attribute", "heading_system_attribute"} and not (exact_system_match and attribute_match):
            score -= 1.25
        if search_pass_name in {"grouped_heading_concept", "grouped_concept"} and not concept_match:
            score -= 0.9
        if search_pass_name in {"grouped_scope_concept", "grouped_heading_scope"} and not (concept_match and (scope_match or not plan.scope_terms)):
            score -= 1.1
        if search_pass_name in {"broad_topic_heading_concept", "broad_topic_concept"} and not concept_match:
            score -= 0.85
        if search_pass_name in {"broad_topic_scope_concept", "broad_topic_requirement_signal"} and not (concept_match and (scope_match or not plan.scope_terms)):
            score -= 0.95
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
        if self._requires_meaningful_body(plan) and self._is_low_content_body(full_text, str(self._row_value(row, "section_number", "") or "")):
            score -= 2.5
        return SearchCandidate(
            row=row,
            total_score=score,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
        )

    @staticmethod
    def _row_value(row: dict, key: str, default=None):
        if row is None:
            return default
        if isinstance(row, dict):
            return row.get(key, default)
        keys = getattr(row, "keys", None)
        if callable(keys):
            row_keys = keys()
            if key in row_keys:
                return row[key]
            return default
        try:
            return row[key]
        except Exception:
            return default

    @staticmethod
    def _requires_meaningful_body(plan: QueryPlan) -> bool:
        return bool(
            plan.system_phrase
            or plan.attribute_label
            or plan.count_question
            or plan.direct_text_question
            or plan.request_shape == REQUEST_SHAPE_GROUPED_LIST
        )

    @staticmethod
    def _is_low_content_body(full_text: str, section_number: str) -> bool:
        compact = " ".join(full_text.split()).strip().strip(".")
        if not compact:
            return True
        if section_number and compact == section_number.strip().strip("."):
            return True
        return len(TOKEN_RE.findall(compact)) <= 2

    @classmethod
    def _select_primary_ranked_chunk(cls, ranked: list[RankedChunk]) -> RankedChunk:
        for chunk in ranked:
            if not cls._is_low_content_body(chunk.full_text, chunk.section_number or ""):
                return chunk
        return ranked[0]

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
        lowered_heading = f"{heading} {parent_heading}".lower()
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
            if re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", lowered_heading):
                score += 1.0
                matched = True
            if re.search(r"\bdesign conditions?\b|\bdesign basis\b|\boperating conditions?\b", lowered_full_text):
                score += 1.1
                matched = True
            value_hits = len(
                re.findall(
                    r"\b\d+(?:\.\d+)?\s*(?:psi|psig|psia|degf|deg c|°f|°c|gpm|lb/hr|scfm|mw|mva|hp)\b",
                    lowered_full_text,
                    re.IGNORECASE,
                )
            )
            score += min(1.1, value_hits * 0.22)
            if value_hits >= 3:
                score += 0.45
                matched = True
            if "site" in plan.focus_terms and has_term_overlap(heading, ("site",)):
                score += 0.45
                matched = True
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
        specific_scope_terms = HybridRetriever._specific_scope_terms(plan.scope_terms)
        score = 0.0
        matched = False
        if specific_scope_terms:
            if has_term_overlap(heading, specific_scope_terms):
                score += 1.25
                matched = True
            if has_term_overlap(parent_heading, specific_scope_terms):
                score += 1.0
                matched = True
            if not matched and clause_type == "exhibit" and (has_term_overlap(heading, plan.scope_terms) or has_term_overlap(parent_heading, plan.scope_terms)):
                score += 0.15
        else:
            if has_term_overlap(heading, plan.scope_terms):
                score += 1.2
                matched = True
            if has_term_overlap(parent_heading, plan.scope_terms):
                score += 0.9
                matched = True
        if clause_type == "exhibit":
            score += 0.35
            matched = matched or not specific_scope_terms
        return score, matched

    @staticmethod
    def _specific_scope_terms(scope_terms: tuple[str, ...]) -> tuple[str, ...]:
        generic = {"appendix", "appendices", "attachment", "attachments", "exhibit", "exhibits"}
        specific: list[str] = []
        for term in scope_terms:
            normalized = term.strip().lower()
            if normalized and normalized not in generic:
                specific.append(term)
        return tuple(specific)

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
    def _broad_topic_support_score(heading: str, parent_heading: str, full_text: str, clause_type: str) -> float:
        normalized = " ".join([heading, parent_heading, full_text]).lower()
        score = 0.0
        score += 0.2 * sum(
            1
            for pattern in (
                r"\bshall\b",
                r"\bmust\b",
                r"\brequired\b",
                r"\bpermit\b",
                r"\bpermits\b",
                r"\bcompliance\b",
                r"\btesting\b",
                r"\btest\b",
                r"\bdemonstrate\b",
                r"\benvironmental\b",
                r"\bemissions?\b",
                r"\bguarantees?\b",
                r"\blimits?\b",
            )
            if re.search(pattern, normalized)
        )
        if clause_type == "exhibit":
            score += 0.1
        return min(score, 1.6)

    @classmethod
    def _broad_topic_noise_penalty(cls, heading: str, parent_heading: str, full_text: str, clause_type: str) -> float:
        normalized_heading = " ".join([heading, parent_heading]).lower()
        normalized_text = " ".join(full_text.split()).lower()
        penalty = 0.0
        if re.search(r"\b(common acronyms|acronyms|abbreviations?)\b", normalized_heading):
            penalty += 1.45
        if re.search(r"\b(defined as|definition|definitions?)\b", normalized_text) and not re.search(
            r"\b(shall|must|required|permit|compliance|testing|guarantee|limit)\b",
            normalized_text,
        ):
            penalty += 0.8
        if clause_type == "exhibit" and cls._is_low_content_body(full_text, ""):
            penalty += 0.9
        if re.fullmatch(r"(appendix|attachment|exhibit)\s+[a-z0-9.\-]+", normalized_heading.strip()):
            penalty += 0.75
        return penalty

    @staticmethod
    def _term_hit_count(text: str, terms: tuple[str, ...]) -> int:
        normalized = text.lower()
        return sum(1 for term in terms if term and term.lower() in normalized)

    @staticmethod
    def _noise_penalty(row: dict) -> float:
        penalty = 0.0
        heading = str(HybridRetriever._row_value(row, "heading", "") or "")
        full_text = str(HybridRetriever._row_value(row, "full_text", "") or "")
        if "...." in heading:
            penalty += 0.45
        if full_text.count("....") >= 3 and int(HybridRetriever._row_value(row, "page_start", 0) or 0) <= 20:
            penalty += 0.45
        noise_flags = set(str(HybridRetriever._row_value(row, "noise_flags", "") or "").split())
        if "date_heading" in noise_flags:
            penalty += 0.75
        if "thin" in noise_flags:
            penalty += 0.2
        if str(HybridRetriever._row_value(row, "section_number", "") or "").strip() == "0":
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
