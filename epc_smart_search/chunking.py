from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from epc_smart_search.config import MAX_HEADING_LENGTH, MIN_HEADING_LENGTH
from epc_smart_search.ocr_support import PageText

ARTICLE_RE = re.compile(r"^ARTICLE\s+([A-Z0-9IVXLC\-]+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
EXHIBIT_RE = re.compile(r"^(EXHIBIT|APPENDIX|ATTACHMENT|ANNEX|SCHEDULE)\s+([A-Z0-9.\-]+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^(?:SECTION\s+)?(\d+(?:\.\d+){0,5})[.)]?\s+(.+)$", re.IGNORECASE)
SECTION_ONLY_RE = re.compile(r"^(?:SECTION\s+)?(\d+(?:\.\d+){0,5})[.)]?$", re.IGNORECASE)
DEFINITION_RE = re.compile(
    r"(?P<term>[\"“][^\"”]{2,90}[\"”]|[A-Z][A-Za-z0-9/&(),\- ]{2,90})\s+"
    r"(?P<verb>means|shall mean|is defined as)\b",
    re.IGNORECASE,
)


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
    lines: list[str]
    article_context: str | None = None

    def add_line(self, text: str) -> None:
        if text:
            self.lines.append(text)


def build_document_id(display_name: str, version_label: str) -> str:
    digest = hashlib.sha1(f"{display_name}|{version_label}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:12]}"


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_heading_text(text: str) -> bool:
    cleaned = _clean_line(text)
    if len(cleaned) < MIN_HEADING_LENGTH or len(cleaned) > MAX_HEADING_LENGTH:
        return False
    if cleaned.endswith((".", ";", ",")):
        return False
    if sum(ch.isalpha() for ch in cleaned) < 3:
        return False
    if cleaned.lower() in {"confidential", "execution version"}:
        return False
    return True


def _iter_lines(pages: list[PageText]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for page in pages:
        for raw_line in page.text.splitlines():
            line = _clean_line(raw_line)
            if line:
                rows.append((page.page_num, line))
    return rows


def _flush_chunk(chunks: list[ChunkRecord], builder: _ChunkBuilder | None, document_id: str, ordinal: int) -> int:
    if builder is None:
        return ordinal
    full_text = "\n".join(builder.lines).strip()
    if not full_text:
        return ordinal
    seed = f"{document_id}|{builder.chunk_type}|{builder.section_number}|{builder.heading}|{builder.page_start}|{ordinal}"
    chunk_id = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    chunks.append(
        ChunkRecord(
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
    if match and _is_heading_text(match.group(2)):
        section_number = match.group(1)
        return section_number, _clean_line(match.group(2)), index + 1
    match = SECTION_ONLY_RE.match(line)
    if not match:
        return None
    section_number = match.group(1)
    for scan in range(index + 1, min(index + 4, len(lines))):
        _, candidate = lines[scan]
        if _is_heading_text(candidate):
            return section_number, candidate, scan + 1
    return None


def _match_standalone_heading(
    lines: list[tuple[int, str]],
    index: int,
    current: _ChunkBuilder | None,
) -> tuple[str, int] | None:
    if current is None or len(current.lines) < 1:
        return None
    _, line = lines[index]
    if not _is_heading_text(line):
        return None
    if ARTICLE_RE.match(line) or EXHIBIT_RE.match(line) or SECTION_RE.match(line) or SECTION_ONLY_RE.match(line):
        return None
    if index + 1 >= len(lines):
        return None
    _, next_line = lines[index + 1]
    if _is_heading_text(next_line):
        return None
    return _clean_line(line), index + 1


def _set_page_end(chunks: list[ChunkRecord], lines: list[tuple[int, str]]) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    for idx, chunk in enumerate(chunks):
        next_page = chunks[idx + 1].page_start - 1 if idx + 1 < len(chunks) else lines[-1][0]
        out.append(replace(chunk, page_end=max(chunk.page_start, next_page)))
    return out


def _assign_parents(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    latest_by_section: dict[str, str] = {}
    latest_major: str | None = None
    resolved: list[ChunkRecord] = []
    for chunk in chunks:
        parent_chunk_id = chunk.parent_chunk_id
        if chunk.chunk_type in {"article", "exhibit"}:
            latest_major = chunk.chunk_id
            parent_chunk_id = None
        elif chunk.section_number:
            if "." in chunk.section_number:
                parent_key = chunk.section_number.rsplit(".", 1)[0]
                parent_chunk_id = latest_by_section.get(parent_key, latest_major)
            else:
                parent_chunk_id = latest_major
            latest_by_section[chunk.section_number] = chunk.chunk_id
        elif parent_chunk_id is None:
            parent_chunk_id = latest_major
        resolved.append(replace(chunk, parent_chunk_id=parent_chunk_id))
    return resolved


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


def parse_chunks(pages: list[PageText], document_id: str) -> list[ChunkRecord]:
    lines = _iter_lines(pages)
    if not lines:
        return []
    chunks: list[ChunkRecord] = []
    current: _ChunkBuilder | None = None
    ordinal = 1
    article_context: str | None = None
    index = 0
    while index < len(lines):
        page_num, line = lines[index]
        article_match = _match_article(lines, index)
        if article_match is not None:
            ordinal = _flush_chunk(chunks, current, document_id, ordinal)
            number, heading, next_index = article_match
            current = _ChunkBuilder("article", number, heading, page_num, [line])
            article_context = None
            index = next_index
            continue
        exhibit_match = _match_exhibit(lines, index)
        if exhibit_match is not None:
            ordinal = _flush_chunk(chunks, current, document_id, ordinal)
            number, heading, next_index = exhibit_match
            current = _ChunkBuilder("exhibit", number, heading, page_num, [line])
            article_context = None
            index = next_index
            continue
        section_match = _match_section(lines, index)
        if section_match is not None:
            ordinal = _flush_chunk(chunks, current, document_id, ordinal)
            number, heading, next_index = section_match
            chunk_type = "section" if number.count(".") == 0 else "subsection"
            current = _ChunkBuilder(chunk_type, number, heading, page_num, [line], article_context=article_context)
            index = next_index
            continue
        standalone_heading = _match_standalone_heading(lines, index, current)
        if standalone_heading is not None:
            ordinal = _flush_chunk(chunks, current, document_id, ordinal)
            heading, next_index = standalone_heading
            current = _ChunkBuilder("section", None, heading, page_num, [line], article_context=article_context)
            index = next_index
            continue
        if current is None:
            current = _ChunkBuilder("section", None, "Front Matter", page_num, [], None)
        current.add_line(line)
        index += 1
        if current.chunk_type in {"article", "exhibit"} and not article_context:
            seed = f"{document_id}|{current.chunk_type}|{current.section_number}|{current.heading}|{current.page_start}|{ordinal}"
            article_context = "chunk_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
            current.article_context = None
    _flush_chunk(chunks, current, document_id, ordinal)
    chunks = _set_page_end(chunks, lines)
    chunks = _assign_parents(chunks)
    chunks.extend(_extract_definition_chunks(chunks))
    chunks.sort(key=lambda chunk: chunk.ordinal_in_document)
    return chunks
