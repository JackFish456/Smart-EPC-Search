from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import fitz
from PIL import Image

from epc_smart_search.config import MIN_NATIVE_TEXT_CHARS, MIN_NATIVE_WORDS

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PageText:
    page_num: int
    text: str
    ocr_used: bool


def _text_quality(text: str) -> tuple[int, int]:
    words = [token for token in text.split() if any(ch.isalpha() for ch in token)]
    meaningful_chars = sum(1 for ch in text if ch.isalnum())
    return meaningful_chars, len(words)


def _render_page(page: fitz.Page, dpi: int = 150) -> Image.Image:
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


def extract_pages(pdf_path: str) -> list[PageText]:
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
            pages.append(PageText(page_num=page_index + 1, text=final_text, ocr_used=use_ocr))
        return pages
    finally:
        doc.close()
