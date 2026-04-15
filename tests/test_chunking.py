from epc_smart_search.chunking import parse_chunks
from epc_smart_search.ocr_support import PageText


def test_parse_chunks_builds_section_hierarchy() -> None:
    pages = [
        PageText(
            page_num=1,
            text="ARTICLE 1\nGENERAL\n1 Purpose\nThe project scope starts here.\n1.1 Definitions\n\"Contract\" means this agreement.",
            ocr_used=False,
        )
    ]
    chunks = parse_chunks(pages, "doc_test")
    section_numbers = [chunk.section_number for chunk in chunks if chunk.section_number]
    assert "1" in section_numbers
    assert "1.1" in section_numbers
    assert any(chunk.chunk_type == "definition" for chunk in chunks)


def test_parse_chunks_recognizes_attachment_headings_and_standalone_titles() -> None:
    pages = [
        PageText(
            page_num=5,
            text=(
                "ATTACHMENT B\n"
                "Fuel Gas Data\n"
                "General Notes\n"
                "The attachment summarizes the fuel gas design basis.\n"
            ),
            ocr_used=False,
        )
    ]

    chunks = parse_chunks(pages, "doc_attachment")

    assert any(chunk.chunk_type == "exhibit" and chunk.section_number == "B" for chunk in chunks)
    assert any(chunk.heading == "General Notes" for chunk in chunks)
