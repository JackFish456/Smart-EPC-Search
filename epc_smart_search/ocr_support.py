from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from PIL import Image

from epc_smart_search.config import MIN_NATIVE_TEXT_CHARS, MIN_NATIVE_WORDS

LOGGER = logging.getLogger(__name__)
DATE_HEADING_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}\b")
BULLET_RE = re.compile(r"^(?:[-*•]|[a-z0-9]{1,3}[.)]|\([a-z0-9]{1,3}\))\s+", re.IGNORECASE)
LISTISH_RE = re.compile(r"^(?:note|item|step)\s+\d+\b", re.IGNORECASE)
TABLE_DIVIDER_RE = re.compile(r"(?:\s{2,}|\t+|\|)")


@dataclass(slots=True)
class ExtractedBlock:
    block_ordinal: int
    block_type: str
    text: str
    line_count: int
    noise_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class PageDiagnostics:
    page_num: int
    meaningful_chars: int
    word_count: int
    block_count: int
    short_line_count: int
    flags: tuple[str, ...] = ()


@dataclass(slots=True)
class PageText:
    page_num: int
    text: str
    ocr_used: bool
    blocks: tuple[ExtractedBlock, ...] = ()
    diagnostics: PageDiagnostics | None = None


def _text_quality(text: str) -> tuple[int, int]:
    words = [token for token in text.split() if any(ch.isalpha() for ch in token)]
    meaningful_chars = sum(1 for ch in text if ch.isalnum())
    return meaningful_chars, len(words)


def _render_page(page: object, dpi: int = 150) -> Image.Image:
    import fitz

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGBA")


def _image_to_software_bitmap(image: Image.Image) -> object:
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    import winrt.windows.storage.streams as streams

    rgba = image.convert("RGBA")
    writer = streams.DataWriter()
    writer.write_bytes(rgba.tobytes())
    bitmap = SoftwareBitmap(BitmapPixelFormat.RGBA8, rgba.width, rgba.height)
    bitmap.copy_from_buffer(writer.detach_buffer())
    return bitmap


def _ocr_image(image: Image.Image) -> str:
    import asyncio
    from winrt.windows.media.ocr import OcrEngine

    async def _recognize() -> str:
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise RuntimeError("WinRT OCR engine is unavailable.")
        result = await engine.recognize_async(_image_to_software_bitmap(image))
        return (getattr(result, "text", None) or "").strip()

    return asyncio.run(_recognize())


def _should_ocr(native_text: str) -> bool:
    meaningful_chars, word_count = _text_quality(native_text)
    return meaningful_chars < MIN_NATIVE_TEXT_CHARS or word_count < MIN_NATIVE_WORDS


def _is_heading_like(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact or len(compact) > 140:
        return False
    if compact.endswith((".", ";", ",")):
        return False
    words = compact.split()
    if len(words) > 14:
        return False
    alpha_chars = [char for char in compact if char.isalpha()]
    if len(alpha_chars) < 3:
        return False
    titleish = all(word[:1].isupper() or word.isupper() for word in words if any(ch.isalpha() for ch in word))
    return titleish or compact.isupper() or DATE_HEADING_RE.match(compact) is not None


def _looks_like_table_row(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    if TABLE_DIVIDER_RE.search(text):
        return True
    if compact.count("....") >= 2:
        return True
    if DATE_HEADING_RE.match(compact):
        return True
    token_count = len(compact.split())
    numeric_count = sum(1 for token in compact.split() if any(ch.isdigit() for ch in token))
    return token_count >= 4 and numeric_count >= 2 and ("-" in compact or "/" in compact or ":" in compact)


def _classify_block(text: str) -> tuple[str, tuple[str, ...]]:
    flags: list[str] = []
    if _looks_like_table_row(text):
        flags.append("table_like")
        if DATE_HEADING_RE.match(" ".join(text.split())):
            flags.append("schedule_like")
        return "table_row", tuple(flags)
    if _is_heading_like(text):
        if DATE_HEADING_RE.match(" ".join(text.split())):
            flags.append("date_heading")
        return "heading", tuple(flags)
    if BULLET_RE.match(text) or LISTISH_RE.match(text):
        return "list_item", tuple(flags)
    return "paragraph", tuple(flags)


def _flush_block(blocks: list[ExtractedBlock], parts: list[str], ordinal: int) -> int:
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        return ordinal
    block_type, flags = _classify_block(text)
    blocks.append(
        ExtractedBlock(
            block_ordinal=ordinal,
            block_type=block_type,
            text=text,
            line_count=max(1, len(parts)),
            noise_flags=flags,
        )
    )
    return ordinal + 1


def segment_text_blocks(text: str) -> tuple[ExtractedBlock, ...]:
    lines = [line.rstrip() for line in text.splitlines()]
    blocks: list[ExtractedBlock] = []
    buffer: list[str] = []
    ordinal = 1
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            ordinal = _flush_block(blocks, buffer, ordinal)
            buffer = []
            continue
        block_type, _ = _classify_block(line)
        if block_type in {"heading", "list_item", "table_row"}:
            ordinal = _flush_block(blocks, buffer, ordinal)
            buffer = [line]
            ordinal = _flush_block(blocks, buffer, ordinal)
            buffer = []
            continue
        if buffer and len(" ".join(buffer)) > 420:
            ordinal = _flush_block(blocks, buffer, ordinal)
            buffer = []
        buffer.append(line)
    _flush_block(blocks, buffer, ordinal)
    return tuple(blocks)


def build_page_diagnostics(page_num: int, text: str, ocr_used: bool, blocks: tuple[ExtractedBlock, ...]) -> PageDiagnostics:
    meaningful_chars, word_count = _text_quality(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_line_count = sum(1 for line in lines if len(line.split()) <= 3)
    flags: list[str] = []
    if ocr_used:
        flags.append("ocr_used")
    if meaningful_chars < MIN_NATIVE_TEXT_CHARS or word_count < MIN_NATIVE_WORDS:
        flags.append("low_text_density")
    if sum(1 for block in blocks if block.block_type == "table_row") >= 1:
        flags.append("table_like")
    if sum(1 for block in blocks if "schedule_like" in block.noise_flags) >= 1:
        flags.append("schedule_like")
    if lines and short_line_count >= max(4, len(lines) // 2):
        flags.append("many_short_fragments")
    return PageDiagnostics(
        page_num=page_num,
        meaningful_chars=meaningful_chars,
        word_count=word_count,
        block_count=len(blocks),
        short_line_count=short_line_count,
        flags=tuple(flags),
    )


def extract_pages(pdf_path: str) -> list[PageText]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pages: list[PageText] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            native_text = page.get_text("text")
            use_ocr = _should_ocr(native_text)
            final_text = native_text.strip()
            if use_ocr:
                try:
                    ocr_text = _ocr_image(_render_page(page))
                    if len(ocr_text.strip()) > len(final_text):
                        final_text = ocr_text.strip()
                except Exception as exc:
                    LOGGER.debug("OCR fallback failed for page %s: %s", page_index + 1, exc)
            blocks = segment_text_blocks(final_text)
            diagnostics = build_page_diagnostics(page_index + 1, final_text, use_ocr, blocks)
            pages.append(
                PageText(
                    page_num=page_index + 1,
                    text=final_text,
                    ocr_used=use_ocr,
                    blocks=blocks,
                    diagnostics=diagnostics,
                )
            )
        return pages
    finally:
        doc.close()
