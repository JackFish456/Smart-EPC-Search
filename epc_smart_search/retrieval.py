from __future__ import annotations

import re
from dataclasses import dataclass, field

from epc_smart_search.config import MAX_SEARCH_RESULTS, MAX_SEMANTIC_SCAN
from epc_smart_search.query_planner import (
    QueryPlan,
    build_match_queries,
    has_term_overlap,
    plan_query,
)
from epc_smart_search.semantic import LocalEmbedder, SemanticReranker, build_query_semantic_text
from epc_smart_search.storage import ContractStore

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
    retrieval_stage: str = "unknown"
    matched_block_count: int = 0
    matched_terms: int = 0


@dataclass(slots=True)
class SearchCandidate:
    row: dict
    total_score: float
    lexical_score: float
    semantic_score: float
    retrieval_stage: str = "unknown"
    matched_block_ids: set[str] = field(default_factory=set)
    matched_terms: set[str] = field(default_factory=set)
    supporting_stages: set[str] = field(default_factory=set)


@dataclass(slots=True, frozen=True)
class RetrievalProfile:
    name: str
    result_limit: int
    fts_limit: int
    rescue_limit: int
    use_query_expansion: bool = False
    semantic_enabled: bool = True
    semantic_on_weak_only: bool = True
    semantic_scan_limit: int = MAX_SEMANTIC_SCAN


@dataclass(slots=True, frozen=True)
class SearchCoverageCase:
    label: str
    query: str
    expected_chunk_id: str


@dataclass(slots=True)
class SearchCoverageResult:
    label: str
    query: str
    expected_chunk_id: str
    found: bool
    top_chunk_id: str | None
    retrieval_stage: str | None


DEFAULT_RETRIEVAL_PROFILE = RetrievalProfile(
    name="normal",
    result_limit=MAX_SEARCH_RESULTS,
    fts_limit=24,
    rescue_limit=18,
)
DEEP_RETRIEVAL_PROFILE = RetrievalProfile(
    name="deep",
    result_limit=10,
    fts_limit=40,
    rescue_limit=30,
    use_query_expansion=True,
    semantic_on_weak_only=False,
)


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
    def __init__(
        self,
        store: ContractStore,
        embedder: LocalEmbedder | None = None,
        semantic_reranker: SemanticReranker | None = None,
    ) -> None:
        self.store = store
        self.semantic_reranker = semantic_reranker or SemanticReranker(store, embedder=embedder)

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
        plan = plan_query(query)
        active_profile = self._resolve_profile(profile, limit)
        combined: dict[str, SearchCandidate] = {}

        if plan.section_number:
            for row in self.store.section_lookup(document_id, plan.section_number):
                candidate = self._score_row(plan, row, lexical_score=1.0, bonus=1.15, retrieval_stage="exact_lookup")
                combined[str(row["chunk_id"])] = candidate

        if plan.content_query:
            for row in self.store.heading_lookup(document_id, plan.content_query):
                candidate = self._score_row(plan, row, lexical_score=1.0, bonus=0.85, retrieval_stage="exact_lookup")
                self._merge_candidate(combined, candidate)

        for match_query in self._match_queries_for_profile(plan, active_profile):
            for row in self.store.search_chunk_feature_fts(document_id, match_query, limit=active_profile.fts_limit):
                lexical_score = self._lexical_score_from_rank(row["rank_score"])
                candidate = self._score_row(plan, row, lexical_score=lexical_score, retrieval_stage="chunk_fts")
                self._merge_candidate(combined, candidate)

        if self._primary_evidence_is_weak(plan, combined):
            for block_query in self._block_queries_for_profile(plan, active_profile):
                for row in self.store.search_block_fts(document_id, block_query, limit=active_profile.fts_limit):
                    lexical_score = self._lexical_score_from_rank(row["rank_score"])
                    candidate = self._score_block_row(plan, row, lexical_score=lexical_score, retrieval_stage="block_fts")
                    if candidate is None:
                        continue
                    self._merge_candidate(combined, candidate)

        if self._primary_evidence_is_weak(plan, combined):
            for rescue_query in self._rescue_queries_for_profile(plan, active_profile):
                for row in self.store.search_chunk_feature_rescue_fts(
                    document_id,
                    rescue_query,
                    limit=active_profile.rescue_limit,
                ):
                    lexical_score = self._rescue_score(plan, row)
                    if lexical_score <= 0.0:
                        continue
                    candidate = self._score_row(
                        plan,
                        row,
                        lexical_score=lexical_score,
                        bonus=-0.05,
                        retrieval_stage="trigram_rescue",
                    )
                    self._merge_candidate(combined, candidate)
                for row in self.store.search_block_rescue_fts(document_id, rescue_query, limit=active_profile.rescue_limit):
                    lexical_score = self._rescue_score(plan, row)
                    if lexical_score <= 0.0:
                        continue
                    candidate = self._score_block_row(
                        plan,
                        row,
                        lexical_score=lexical_score,
                        retrieval_stage="trigram_rescue",
                    )
                    if candidate is None:
                        continue
                    self._merge_candidate(combined, candidate)

        lexical_ranked = sorted(
            combined.values(),
            key=lambda item: self._lexical_final_score(plan, item),
            reverse=True,
        )
        evidence_is_weak = self._primary_evidence_is_weak(plan, combined)
        reranked = self._apply_semantic_reranking(
            document_id,
            query,
            plan,
            lexical_ranked,
            active_profile,
            evidence_is_weak=evidence_is_weak,
        )
        ranked = [
            RankedChunk(
                chunk_id=str(candidate.row["chunk_id"]),
                section_number=candidate.row["section_number"],
                heading=str(candidate.row["heading"]),
                full_text=str(candidate.row["full_text"]),
                page_start=int(candidate.row["page_start"]),
                page_end=int(candidate.row["page_end"]),
                ordinal_in_document=int(candidate.row["ordinal_in_document"]),
                total_score=self._final_score(plan, candidate),
                lexical_score=candidate.lexical_score,
                semantic_score=candidate.semantic_score,
                retrieval_stage=candidate.retrieval_stage,
                matched_block_count=len(candidate.matched_block_ids),
                matched_terms=len(candidate.matched_terms),
            )
            for candidate in reranked[: active_profile.result_limit]
        ]
        return ranked

    def evaluate_coverage_cases(self, cases: list[SearchCoverageCase]) -> list[SearchCoverageResult]:
        results: list[SearchCoverageResult] = []
        for case in cases:
            ranked = self.retrieve(case.query)
            top = ranked[0] if ranked else None
            results.append(
                SearchCoverageResult(
                    label=case.label,
                    query=case.query,
                    expected_chunk_id=case.expected_chunk_id,
                    found=bool(top and top.chunk_id == case.expected_chunk_id),
                    top_chunk_id=top.chunk_id if top else None,
                    retrieval_stage=top.retrieval_stage if top else None,
                )
            )
        return results

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
            block_rows = self.store.search_blocks_exact(document_id, match_query, limit=limit)
            for row in block_rows:
                page_num = int(row["page_num"])
                if page_num in seen_pages:
                    continue
                seen_pages.add(page_num)
                hits.append(
                    ExactPageHit(
                        page_num=page_num,
                        snippet=self._clean_snippet(str(row["hit_snippet"])),
                        page_text=str(row["block_text"]),
                    )
                )
                if len(hits) >= limit:
                    return hits
        return hits

    @staticmethod
    def _resolve_profile(profile: str, limit: int | None) -> RetrievalProfile:
        selected = DEEP_RETRIEVAL_PROFILE if profile == "deep" else DEFAULT_RETRIEVAL_PROFILE
        if limit is None:
            return selected
        return RetrievalProfile(
            name=selected.name,
            result_limit=limit,
            fts_limit=selected.fts_limit,
            rescue_limit=selected.rescue_limit,
            use_query_expansion=selected.use_query_expansion,
            semantic_enabled=selected.semantic_enabled,
            semantic_on_weak_only=selected.semantic_on_weak_only,
            semantic_scan_limit=selected.semantic_scan_limit,
        )

    def _match_queries_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[str]:
        queries = list(build_match_queries(plan))
        if not profile.use_query_expansion:
            return queries
        for variant in self._query_variants(plan):
            queries.extend(build_match_queries(plan_query(variant)))
        return self._dedupe_queries(queries)

    def _block_queries_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[str]:
        queries: list[str] = []
        candidates = [plan.content_query, *plan.focus_terms, *self._focus_phrases(plan.focus_terms)]
        candidates.extend(plan.actor_terms)
        candidates.extend(plan.action_terms)
        candidates.extend(plan.topic_terms)
        candidates.extend(plan.expansion_terms)
        if profile.use_query_expansion:
            candidates.extend(self._query_variants(plan))
        queries.extend(
            query
            for query in (build_match_query(candidate) for candidate in candidates if candidate)
            if query
        )
        return self._dedupe_queries(queries)

    def _rescue_queries_for_profile(self, plan: QueryPlan, profile: RetrievalProfile) -> list[str]:
        rescue_sources = [plan.content_query]
        rescue_sources.extend(plan.focus_terms)
        rescue_sources.extend(self._focus_phrases(plan.focus_terms))
        if profile.use_query_expansion:
            rescue_sources.extend(self._query_variants(plan))
        queries = [query for query in (self._build_trigram_query(source) for source in rescue_sources) if query]
        return self._dedupe_queries(queries)

    def _query_variants(self, plan: QueryPlan) -> list[str]:
        candidates = [plan.raw_query, plan.content_query]
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
        match = re.match(r"^(attachment|appendix|exhibit|annex|schedule)\s+([A-Z0-9.\-]+)\b", heading, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1).title()} {match.group(2)}"
        if section_number and len(section_number) <= 4 and section_number.isalpha():
            return f"Appendix {section_number.upper()}"
        return heading

    @staticmethod
    def _lexical_score_from_rank(rank_score: object) -> float:
        return 1.0 / (1.0 + max(float(rank_score), 0.0))

    def _primary_evidence_is_weak(self, plan: QueryPlan, combined: dict[str, SearchCandidate]) -> bool:
        if not combined:
            return True
        top_candidates = sorted(combined.values(), key=lambda item: self._lexical_final_score(plan, item), reverse=True)[:3]
        for candidate in top_candidates:
            if self._is_strong_match(plan, candidate):
                return False
        return True

    def _is_strong_match(self, plan: QueryPlan, candidate: SearchCandidate) -> bool:
        row = candidate.row
        if row["section_number"] and row["section_number"] == plan.section_number:
            return True
        heading = str(row["heading"])
        full_text = str(row["full_text"])
        combined_text = " ".join([heading, full_text, str(row["parent_heading"] or "")])
        if plan.content_query and (
            has_term_overlap(heading, (plan.content_query,))
            or has_term_overlap(full_text, (plan.content_query,))
        ):
            return True
        if plan.focus_terms and self._term_hit_count(combined_text, plan.focus_terms) >= max(2, len(plan.focus_terms) - 1):
            return True
        return self._lexical_final_score(plan, candidate) >= 2.9

    @staticmethod
    def _build_trigram_query(text: str) -> str:
        normalized = " ".join(text.lower().split())
        cleaned = "".join(ch for ch in normalized if ch.isalnum() or ch.isspace()).strip()
        if len(cleaned.replace(" ", "")) < 4:
            return ""
        trigrams: list[str] = []
        collapsed = cleaned.replace(" ", "")
        for index in range(len(collapsed) - 2):
            trigram = collapsed[index:index + 3]
            if trigram not in trigrams:
                trigrams.append(trigram)
        return " OR ".join(f'"{trigram}"' for trigram in trigrams[:24])

    def _rescue_score(self, plan: QueryPlan, row: dict) -> float:
        row_keys = set(row.keys()) if hasattr(row, "keys") else set()
        rescue_targets = [
            str(row["heading"]),
            str(row["parent_heading"] or ""),
            str(
                row["rescue_text"] if "rescue_text" in row_keys else
                row["normalized_text"] if "normalized_text" in row_keys else
                row["block_text"] if "block_text" in row_keys else
                ""
            ),
            str(row["topic_tags"] or ""),
        ]
        query_variants = [plan.content_query, *plan.focus_terms, *self._focus_phrases(plan.focus_terms)]
        best_similarity = max(
            (
                self._trigram_similarity(query, target)
                for query in query_variants
                if query
                for target in rescue_targets
                if target
            ),
            default=0.0,
        )
        if best_similarity < 0.34:
            return 0.0
        return min(0.92, 0.34 + (best_similarity * 0.58))

    @staticmethod
    def _trigram_similarity(left: str, right: str) -> float:
        left_set = HybridRetriever._trigram_set(left)
        right_set = HybridRetriever._trigram_set(right)
        if not left_set or not right_set:
            return 0.0
        overlap = len(left_set & right_set)
        return (2.0 * overlap) / (len(left_set) + len(right_set))

    @staticmethod
    def _trigram_set(text: str) -> set[str]:
        normalized = "".join(ch for ch in " ".join(text.lower().split()) if ch.isalnum())
        if len(normalized) < 3:
            return set()
        return {normalized[index:index + 3] for index in range(len(normalized) - 2)}

    @staticmethod
    def _merge_candidate(combined: dict[str, SearchCandidate], candidate: SearchCandidate) -> None:
        chunk_id = str(candidate.row["chunk_id"])
        existing = combined.get(chunk_id)
        if existing is None:
            combined[chunk_id] = candidate
            return
        existing.total_score = max(existing.total_score, candidate.total_score)
        existing.lexical_score = max(existing.lexical_score, candidate.lexical_score)
        existing.semantic_score = max(existing.semantic_score, candidate.semantic_score)
        existing.matched_block_ids.update(candidate.matched_block_ids)
        existing.matched_terms.update(candidate.matched_terms)
        existing.supporting_stages.update(candidate.supporting_stages)
        if HybridRetriever._stage_priority(candidate.retrieval_stage) > HybridRetriever._stage_priority(existing.retrieval_stage):
            existing.retrieval_stage = candidate.retrieval_stage

    def _score_row(
        self,
        plan: QueryPlan,
        row: dict,
        *,
        lexical_score: float,
        bonus: float = 0.0,
        retrieval_stage: str,
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

        score = lexical_score * 0.85 + bonus
        if row["section_number"] and row["section_number"] == plan.section_number:
            score += 1.2
        if plan.content_query:
            if has_term_overlap(heading, (plan.content_query,)):
                score += 0.7
            elif has_term_overlap(parent_heading, (plan.content_query,)):
                score += 0.35
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
        if has_term_overlap(combined_text, plan.expansion_terms):
            score += 0.9
        score += 0.16 * self._term_hit_count(combined_text, plan.focus_terms)
        score += 0.12 * self._term_hit_count(combined_text, plan.actor_terms + plan.action_terms + plan.topic_terms + plan.expansion_terms)
        if plan.intent == "definition" and clause_type == "definition":
            score += 1.35
            if plan.content_query and has_term_overlap(full_text, (plan.content_query,)):
                score += 0.65
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
            semantic_score=0.0,
            retrieval_stage=retrieval_stage,
            matched_terms=self._matched_terms_in_text(plan, combined_text),
            supporting_stages={retrieval_stage},
        )

    def _score_block_row(
        self,
        plan: QueryPlan,
        row: dict,
        *,
        lexical_score: float,
        retrieval_stage: str,
    ) -> SearchCandidate | None:
        if not row["heading"] or not row["full_text"]:
            return None
        parent_candidate = self._score_row(
            plan,
            row,
            lexical_score=max(0.08, lexical_score * 0.75),
            bonus=-0.05,
            retrieval_stage=retrieval_stage,
        )
        block_text = str(row["block_text"] or "")
        block_type = str(row["block_type"] or "")
        noise_flags = set(str(row["noise_flags"] or "").split()) | set(str(row["block_noise_flags"] or row["noise_flags"] or "").split())
        block_bonus = 0.0
        if plan.content_query and has_term_overlap(block_text, (plan.content_query,)):
            block_bonus += 0.55
        if has_term_overlap(block_text, plan.focus_terms):
            block_bonus += 0.35
        if block_type == "heading":
            block_bonus += 0.28
        if block_type == "table_row":
            block_bonus += 0.18
        if "table_like" in noise_flags:
            block_bonus += 0.1
        if "schedule_like" in noise_flags and plan.intent not in {"delay_schedule", "direct_text"}:
            block_bonus -= 0.18
        if "date_heading" in noise_flags:
            block_bonus -= 0.12
        parent_candidate.total_score += block_bonus
        parent_candidate.matched_block_ids.add(str(row["block_id"]))
        parent_candidate.matched_terms.update(self._matched_terms_in_text(plan, block_text))
        return parent_candidate

    def _lexical_final_score(self, plan: QueryPlan, candidate: SearchCandidate) -> float:
        score = candidate.total_score
        block_count = len(candidate.matched_block_ids)
        if block_count:
            score += min(0.9, 0.22 * block_count)
        score += min(0.9, 0.14 * len(candidate.matched_terms))
        if {"chunk_fts", "block_fts"} <= candidate.supporting_stages:
            score += 0.18
        if candidate.retrieval_stage == "exact_lookup":
            score += 0.12
        elif candidate.retrieval_stage == "block_fts":
            score += 0.08
        elif candidate.retrieval_stage == "trigram_rescue" and block_count == 0:
            score -= 0.05
        if candidate.row["parent_heading"] and has_term_overlap(str(candidate.row["parent_heading"]), plan.focus_terms):
            score += 0.12
        return score

    def _final_score(self, plan: QueryPlan, candidate: SearchCandidate) -> float:
        return self._lexical_final_score(plan, candidate) + self._semantic_bonus(plan, candidate)

    def _apply_semantic_reranking(
        self,
        document_id: str,
        query: str,
        plan: QueryPlan,
        ranked: list[SearchCandidate],
        profile: RetrievalProfile,
        *,
        evidence_is_weak: bool,
    ) -> list[SearchCandidate]:
        if not ranked or not profile.semantic_enabled:
            return ranked
        if profile.semantic_on_weak_only and not evidence_is_weak:
            return ranked
        semantic_window = ranked[: profile.semantic_scan_limit]
        if not semantic_window:
            return ranked
        semantic_query = build_query_semantic_text(query, plan)
        if not semantic_query:
            return ranked
        reranked = list(semantic_window)
        if not self.semantic_reranker.rerank(document_id, semantic_query, reranked):
            return ranked
        reranked.sort(key=lambda item: self._final_score(plan, item), reverse=True)
        return reranked + ranked[profile.semantic_scan_limit :]

    def _semantic_bonus(self, plan: QueryPlan, candidate: SearchCandidate) -> float:
        semantic_score = max(0.0, min(1.0, candidate.semantic_score))
        if semantic_score <= 0.18:
            return 0.0
        row = candidate.row
        if row["section_number"] and row["section_number"] == plan.section_number:
            return 0.0
        lexical_score = self._lexical_final_score(plan, candidate)
        bonus = (semantic_score - 0.18) * 1.1
        if candidate.retrieval_stage == "exact_lookup":
            bonus *= 0.3
        elif candidate.retrieval_stage == "trigram_rescue":
            bonus *= 1.1
        if lexical_score >= 3.8:
            bonus *= 0.45
        return min(0.82, bonus)

    @staticmethod
    def _stage_priority(stage: str) -> int:
        priorities = {
            "trigram_rescue": 1,
            "block_fts": 2,
            "chunk_fts": 3,
            "exact_lookup": 4,
        }
        return priorities.get(stage, 0)

    def _matched_terms_in_text(self, plan: QueryPlan, text: str) -> set[str]:
        normalized = text.lower()
        terms = plan.focus_terms + plan.actor_terms + plan.action_terms + plan.topic_terms + plan.expansion_terms
        return {term for term in terms if term and term.lower() in normalized}

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
