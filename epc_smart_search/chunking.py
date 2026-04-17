from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from epc_smart_search.config import MAX_HEADING_LENGTH, MIN_HEADING_LENGTH
from epc_smart_search.ocr_support import PageText

ARTICLE_RE = re.compile(r"^ARTICLE\s+([A-Z0-9IVXLC\-]+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
EXHIBIT_RE = re.compile(r"^(EXHIBIT|APPENDIX)\s+([A-Z0-9.\-]+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^(?:SECTION\s+)?(\d+(?:\.\d+){0,5})[.)]?\s+(.+)$", re.IGNORECASE)
SECTION_ONLY_RE = re.compile(r"^(?:SECTION\s+)?(\d+(?:\.\d+){0,5})[.)]?$", re.IGNORECASE)
DEFINITION_RE = re.compile(
    r"(?P<term>[\"“][^\"”]{2,90}[\"”]|[A-Z][A-Za-z0-9/&(),\- ]{2,90})\s+"
    r"(?P<verb>means|shall mean|is defined as)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/&\-]{0,}")
BULLET_RE = re.compile(r"^(?:[-*•]|[A-Za-z]\)|\([A-Za-z0-9]+\)|\d+[.)])\s+\S")
VALUE_SIGNAL_RE = re.compile(
    r"\d|[%$]|°|[A-Za-z]{1,5}/[A-Za-z]{1,5}|"
    r"\b(?:psia|psig|hp|kw|mw|mva|amps?|volts?|hz|gpm|ppm)\b",
    re.IGNORECASE,
)
GENERIC_TABLE_HEADERS = {
    "characteristic",
    "description",
    "equipment",
    "item",
    "items",
    "parameter",
    "qty",
    "quantity",
    "remarks",
    "service",
    "specification",
    "unit",
    "units",
    "value",
    "values",
}


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_type: str
    section_number: str | None
    heading: str
    full_text: str
    page_start: int
    page_end: int
    parent_chunk_id: str | None
    ordinal_in_document: int


@dataclass(slots=True)
class _ChunkBuilder:
    chunk_type: str
    section_number: str | None
    heading: str
    page_start: int
    rows: list[tuple[int, str]]
    article_context: str | None = None

    def add_line(self, page_num: int, text: str) -> None:
        if text:
            self.rows.append((page_num, text))


@dataclass(slots=True)
class _ChunkDraft:
    record: ChunkRecord
    rows: list[tuple[int, str]]


def build_document_id(display_name: str, version_label: str) -> str:
    digest = hashlib.sha1(f"{display_name}|{version_label}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:12]}"


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_heading_text(text: str) -> bool:
    cleaned = _clean_line(text)
    if len(cleaned) < MIN_HEADING_LENGTH or len(cleaned) > MAX_HEADING_LENGTH:
        return False
    if sum(ch.isalpha() for ch in cleaned) < 3:
        return False
    if cleaned.lower() in {"confidential", "execution version"}:
        return False
    return True


def _is_plausible_section_number(section_number: str) -> bool:
    try:
        parts = [int(part) for part in section_number.split(".")]
    except ValueError:
        return False
    if not parts or parts[0] > 40:
        return False
    return all(part <= 99 for part in parts[1:])


def _iter_lines(pages: list[PageText]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for page in pages:
        for raw_line in page.text.splitlines():
            line = _clean_line(raw_line)
            if line:
                rows.append((page.page_num, line))
    return rows


def _flush_chunk(chunks: list[_ChunkDraft], builder: _ChunkBuilder | None, document_id: str, ordinal: int) -> int:
    if builder is None:
        return ordinal
    full_text = "\n".join(text for _, text in builder.rows).strip()
    if not full_text:
        return ordinal
    seed = f"{document_id}|{builder.chunk_type}|{builder.section_number}|{builder.heading}|{builder.page_start}|{ordinal}"
    chunk_id = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    chunks.append(
        _ChunkDraft(
            record=ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_type=builder.chunk_type,
                section_number=builder.section_number,
                heading=builder.heading[:MAX_HEADING_LENGTH],
                full_text=full_text,
                page_start=builder.page_start,
                page_end=builder.page_start,
                parent_chunk_id=builder.article_context,
                ordinal_in_document=ordinal,
            ),
            rows=list(builder.rows),
        )
    )
    return ordinal + 1


def _match_article(lines: list[tuple[int, str]], index: int) -> tuple[str, str, int] | None:
    _, line = lines[index]
    match = ARTICLE_RE.match(line)
    if not match:
        return None
    number = match.group(1).strip()
    inline_title = _clean_line(match.group(2))
    if _is_heading_text(inline_title):
        return number, inline_title, index + 1
    for scan in range(index + 1, min(index + 4, len(lines))):
        _, candidate = lines[scan]
        if _is_heading_text(candidate):
            return number, candidate, scan + 1
    return number, f"Article {number}", index + 1


def _match_exhibit(lines: list[tuple[int, str]], index: int) -> tuple[str, str, int] | None:
    _, line = lines[index]
    match = EXHIBIT_RE.match(line)
    if not match:
        return None
    prefix = match.group(1).upper()
    number = match.group(2).strip()
    inline_title = _clean_line(match.group(3))
    heading = inline_title if _is_heading_text(inline_title) else f"{prefix} {number}"
    return number, heading, index + 1


def _match_section(lines: list[tuple[int, str]], index: int) -> tuple[str, str, int] | None:
    _, line = lines[index]
    match = SECTION_RE.match(line)
    if match and _is_plausible_section_number(match.group(1)) and _is_heading_text(match.group(2)):
        section_number = match.group(1)
        return section_number, _clean_line(match.group(2)), index + 1
    match = SECTION_ONLY_RE.match(line)
    if not match:
        return None
    section_number = match.group(1)
    if not _is_plausible_section_number(section_number):
        return None
    for scan in range(index + 1, min(index + 4, len(lines))):
        _, candidate = lines[scan]
        if _is_heading_text(candidate):
            return section_number, candidate, scan + 1
    return None


def _set_page_end(chunks: list[_ChunkDraft], lines: list[tuple[int, str]]) -> list[_ChunkDraft]:
    out: list[_ChunkDraft] = []
    for idx, chunk in enumerate(chunks):
        next_page = chunks[idx + 1].record.page_start - 1 if idx + 1 < len(chunks) else lines[-1][0]
        out.append(replace(chunk, record=replace(chunk.record, page_end=max(chunk.record.page_start, next_page))))
    return out


def _assign_parents(chunks: list[_ChunkDraft]) -> list[_ChunkDraft]:
    latest_by_section: dict[str, str] = {}
    latest_major: str | None = None
    resolved: list[_ChunkDraft] = []
    for chunk in chunks:
        record = chunk.record
        parent_chunk_id = record.parent_chunk_id
        if record.chunk_type in {"article", "exhibit"}:
            latest_major = record.chunk_id
            parent_chunk_id = None
        elif record.section_number:
            if "." in record.section_number:
                parent_key = record.section_number.rsplit(".", 1)[0]
                parent_chunk_id = latest_by_section.get(parent_key, latest_major)
            else:
                parent_chunk_id = latest_major
            latest_by_section[record.section_number] = record.chunk_id
        elif parent_chunk_id is None:
            parent_chunk_id = latest_major
        resolved.append(replace(chunk, record=replace(record, parent_chunk_id=parent_chunk_id)))
    return resolved


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _has_value_signal(text: str) -> bool:
    return bool(VALUE_SIGNAL_RE.search(text))


def _is_prose_like(text: str) -> bool:
    cleaned = _clean_line(text)
    tokens = _token_count(cleaned)
    if tokens >= 14:
        return True
    if tokens >= 8 and re.search(r"[.;]$", cleaned):
        return True
    return tokens >= 11 and not _has_value_signal(cleaned) and not BULLET_RE.match(cleaned)


def _is_table_header_like(text: str) -> bool:
    return _clean_line(text).strip(":").lower() in GENERIC_TABLE_HEADERS


def _is_compact_line(text: str) -> bool:
    cleaned = _clean_line(text)
    if not cleaned or BULLET_RE.match(cleaned):
        return False
    if _is_prose_like(cleaned):
        return False
    return _token_count(cleaned) <= 10


def _matches_parent_marker(parent: ChunkRecord, text: str) -> bool:
    cleaned = _clean_line(text)
    if cleaned == _clean_line(parent.heading):
        return True
    if not parent.section_number:
        return False
    return bool(
        re.match(
            rf"^(?:section\s+)?{re.escape(parent.section_number)}(?:[.)]?\s+.*)?$",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def _structured_rows(parent: ChunkRecord, rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    trimmed = list(rows)
    while trimmed and _matches_parent_marker(parent, trimmed[0][1]):
        trimmed = trimmed[1:]
    return trimmed


def _is_structured_run(rows: list[tuple[int, str]]) -> bool:
    if len(rows) < 2:
        return False
    bullet_count = sum(1 for _, text in rows if BULLET_RE.match(text))
    compact_count = sum(1 for _, text in rows if _is_compact_line(text))
    value_count = sum(1 for _, text in rows if _has_value_signal(text))
    if bullet_count >= 2:
        return True
    if len(rows) >= 3 and compact_count >= len(rows) - 1:
        return True
    return len(rows) >= 4 and value_count >= 2 and compact_count >= 3


def _collect_structured_runs(parent: ChunkRecord, rows: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for row in _structured_rows(parent, rows):
        _, text = row
        if _is_prose_like(text):
            if _is_structured_run(current):
                runs.append(current)
            current = []
            continue
        current.append(row)
    if _is_structured_run(current):
        runs.append(current)
    return runs


def _strip_bullet_marker(text: str) -> str:
    return re.sub(r"^(?:[-*•]|[A-Za-z]\)|\([A-Za-z0-9]+\)|\d+[.)])\s+", "", _clean_line(text))


def _build_bullet_items(rows: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    items: list[list[tuple[int, str]]] = []
    for row in rows:
        if BULLET_RE.match(row[1]) or not items:
            items.append([row])
        else:
            items[-1].append(row)
    return [item for item in items if item]


def _can_pair_compact_rows(current: str, following: str, *, allow_generic_pairs: bool) -> bool:
    current_clean = _clean_line(current)
    following_clean = _clean_line(following)
    if not current_clean or not following_clean:
        return False
    if _is_table_header_like(current_clean) or _is_table_header_like(following_clean):
        return False
    if BULLET_RE.match(current_clean) or BULLET_RE.match(following_clean):
        return False
    if _is_prose_like(current_clean) or _is_prose_like(following_clean):
        return False
    if _token_count(current_clean) > 8 or _token_count(following_clean) > 8:
        return False
    if _has_value_signal(current_clean):
        return False
    return _has_value_signal(following_clean) or allow_generic_pairs


def _extract_compact_items(
    rows: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[list[tuple[int, str]]], int]:
    header_rows: list[tuple[int, str]] = []
    index = 0
    while index < len(rows) and len(header_rows) < 2 and _is_table_header_like(rows[index][1]):
        header_rows.append(rows[index])
        index += 1
    items: list[list[tuple[int, str]]] = []
    pair_count = 0
    while index < len(rows):
        row = rows[index]
        if index + 1 < len(rows) and _can_pair_compact_rows(
            row[1],
            rows[index + 1][1],
            allow_generic_pairs=bool(header_rows),
        ):
            items.append([row, rows[index + 1]])
            pair_count += 1
            index += 2
            continue
        items.append([row])
        index += 1
    return header_rows, items, pair_count


def _group_items(items: list[list[tuple[int, str]]], group_size: int) -> list[list[list[tuple[int, str]]]]:
    if len(items) < 2:
        return []
    groups: list[list[list[tuple[int, str]]]] = []
    index = 0
    while index < len(items):
        remaining = len(items) - index
        take = group_size
        if remaining == 4:
            take = 2
        elif remaining == 1 and groups:
            groups[-1].append(items[index])
            break
        groups.append(items[index:index + take])
        index += take
    if len(groups) >= 2 and len(groups[-1]) == 1:
        groups[-2].append(groups[-1][0])
        groups.pop()
    return groups


def _detail_from_items(items: list[list[tuple[int, str]]], *, bullet: bool) -> str:
    if not items or not items[0]:
        return ""
    text = items[0][0][1]
    detail = _strip_bullet_marker(text) if bullet else _clean_line(text)
    if _is_table_header_like(detail):
        return ""
    return detail[:60]


def _compose_structured_heading(parent: ChunkRecord, chunk_type: str, detail: str) -> str:
    kind = {
        "bullet_group": "Bullets",
        "schedule_block": "Schedule",
        "line_item_group": "Line Items",
    }[chunk_type]
    if detail:
        return f"{parent.heading} - {kind}: {detail}"[:MAX_HEADING_LENGTH]
    return f"{parent.heading} - {kind}"[:MAX_HEADING_LENGTH]


def _new_structured_chunk(
    parent: ChunkRecord,
    chunk_type: str,
    heading: str,
    rows: list[tuple[int, str]],
    ordinal_seed: str,
) -> ChunkRecord:
    full_text = "\n".join(text for _, text in rows).strip()
    seed = f"{parent.chunk_id}|{chunk_type}|{ordinal_seed}|{heading}|{rows[0][0]}|{rows[-1][0]}"
    chunk_id = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=parent.document_id,
        chunk_type=chunk_type,
        section_number=parent.section_number,
        heading=heading[:MAX_HEADING_LENGTH],
        full_text=full_text,
        page_start=rows[0][0],
        page_end=rows[-1][0],
        parent_chunk_id=parent.chunk_id,
        ordinal_in_document=0,
    )


def _extract_structured_chunks(base_chunks: list[_ChunkDraft]) -> list[ChunkRecord]:
    derived: list[ChunkRecord] = []
    for draft in base_chunks:
        parent = draft.record
        if parent.chunk_type == "definition":
            continue
        for run_index, run_rows in enumerate(_collect_structured_runs(parent, draft.rows), start=1):
            if sum(1 for _, text in run_rows if BULLET_RE.match(text)) >= 2:
                items = _build_bullet_items(run_rows)
                for group_index, group in enumerate(_group_items(items, group_size=3), start=1):
                    rows = [row for item in group for row in item]
                    heading = _compose_structured_heading(parent, "bullet_group", _detail_from_items(group, bullet=True))
                    derived.append(
                        _new_structured_chunk(
                            parent,
                            "bullet_group",
                            heading,
                            rows,
                            ordinal_seed=f"{run_index}:{group_index}",
                        )
                    )
                continue
            header_rows, items, pair_count = _extract_compact_items(run_rows)
            chunk_type = "schedule_block" if header_rows or pair_count >= 2 else "line_item_group"
            group_size = 2 if chunk_type == "schedule_block" else 3
            for group_index, group in enumerate(_group_items(items, group_size=group_size), start=1):
                rows = [*header_rows, *(row for item in group for row in item)]
                heading = _compose_structured_heading(parent, chunk_type, _detail_from_items(group, bullet=False))
                derived.append(
                    _new_structured_chunk(
                        parent,
                        chunk_type,
                        heading,
                        rows,
                        ordinal_seed=f"{run_index}:{group_index}",
                    )
                )
    return derived


def _definition_sentence(text: str, start: int) -> str:
    left = text.rfind("\n", 0, start)
    left = max(left, text.rfind(". ", 0, start))
    left = 0 if left < 0 else left + 1
    right_candidates = [text.find(". ", start), text.find("\n", start)]
    valid = [value for value in right_candidates if value >= 0]
    right = min(valid) if valid else len(text)
    return _clean_line(text[left:right + 1])


def _extract_definition_chunks(base_chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    derived: list[ChunkRecord] = []
    ordinal = len(base_chunks) + 1
    for chunk in base_chunks:
        allow = "definition" in chunk.heading.lower() or bool(DEFINITION_RE.search(chunk.full_text))
        if not allow:
            continue
        seen_terms: set[str] = set()
        for match in DEFINITION_RE.finditer(chunk.full_text):
            term = _clean_line(match.group("term").strip("\"“”"))
            if term.casefold() in seen_terms:
                continue
            seen_terms.add(term.casefold())
            sentence = _definition_sentence(chunk.full_text, match.start())
            if len(sentence) < 20:
                continue
            seed = f"{chunk.chunk_id}|definition|{term}|{ordinal}"
            definition_id = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
            derived.append(
                ChunkRecord(
                    chunk_id=definition_id,
                    document_id=chunk.document_id,
                    chunk_type="definition",
                    section_number=chunk.section_number,
                    heading=term[:MAX_HEADING_LENGTH],
                    full_text=sentence,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    parent_chunk_id=chunk.chunk_id,
                    ordinal_in_document=ordinal,
                )
            )
            ordinal += 1
    return derived


def _renumber_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    return [replace(chunk, ordinal_in_document=index) for index, chunk in enumerate(chunks, start=1)]


def parse_chunks(pages: list[PageText], document_id: str) -> list[ChunkRecord]:
    lines = _iter_lines(pages)
    if not lines:
        return []
    drafts: list[_ChunkDraft] = []
    current: _ChunkBuilder | None = None
    ordinal = 1
    article_context: str | None = None
    index = 0
    while index < len(lines):
        page_num, line = lines[index]
        article_match = _match_article(lines, index)
        if article_match is not None:
            ordinal = _flush_chunk(drafts, current, document_id, ordinal)
            number, heading, next_index = article_match
            current = _ChunkBuilder("article", number, heading, page_num, [(page_num, line)])
            article_context = None
            index = next_index
            continue
        exhibit_match = _match_exhibit(lines, index)
        if exhibit_match is not None:
            ordinal = _flush_chunk(drafts, current, document_id, ordinal)
            number, heading, next_index = exhibit_match
            current = _ChunkBuilder("exhibit", number, heading, page_num, [(page_num, line)])
            article_context = None
            index = next_index
            continue
        section_match = _match_section(lines, index)
        if section_match is not None:
            ordinal = _flush_chunk(drafts, current, document_id, ordinal)
            number, heading, next_index = section_match
            chunk_type = "section" if number.count(".") == 0 else "subsection"
            current = _ChunkBuilder(chunk_type, number, heading, page_num, [(page_num, line)], article_context=article_context)
            index = next_index
            continue
        if current is None:
            current = _ChunkBuilder("section", None, "Front Matter", page_num, [], None)
        current.add_line(page_num, line)
        index += 1
        if current.chunk_type in {"article", "exhibit"} and not article_context:
            seed = f"{document_id}|{current.chunk_type}|{current.section_number}|{current.heading}|{current.page_start}|{ordinal}"
            article_context = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
            current.article_context = None
    _flush_chunk(drafts, current, document_id, ordinal)
    drafts = _set_page_end(drafts, lines)
    drafts = _assign_parents(drafts)
    base_chunks = [draft.record for draft in drafts]
    structured_chunks = _extract_structured_chunks(drafts)
    definition_chunks = _extract_definition_chunks(base_chunks)
    structured_by_parent: dict[str, list[ChunkRecord]] = {}
    for chunk in structured_chunks:
        structured_by_parent.setdefault(chunk.parent_chunk_id or "", []).append(chunk)
    ordered: list[ChunkRecord] = []
    for chunk in base_chunks:
        ordered.append(chunk)
        ordered.extend(structured_by_parent.get(chunk.chunk_id, []))
    ordered.extend(definition_chunks)
    return _renumber_chunks(ordered)
