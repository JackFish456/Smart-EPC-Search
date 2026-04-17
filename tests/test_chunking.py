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


def test_parse_chunks_does_not_split_design_conditions_table_on_decimal_values() -> None:
    pages = [
        PageText(
            page_num=1591,
            text=(
                "2.1\n"
                "Site Design Conditions\n"
                "Characteristic\n"
                "Specification\n"
                "Elevation\n"
                "455 ft\n"
                "Ambient Pressure\n"
                "14.457 psia\n"
                "Minimum Outdoor Ambient Temperature\n"
                "-5°F\n"
                "Maximum Outdoor Ambient Temperature\n"
                "110°F\n"
            ),
            ocr_used=False,
        )
    ]

    chunks = parse_chunks(pages, "doc_test")
    site_chunks = [chunk for chunk in chunks if chunk.section_number == "2.1"]

    assert len(site_chunks) == 1
    assert "14.457 psia" in site_chunks[0].full_text
    assert all(chunk.section_number != "14.457" for chunk in chunks)
