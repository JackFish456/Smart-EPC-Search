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


def test_parse_chunks_emits_compact_line_item_groups_without_merging_adjacent_prose() -> None:
    pages = [
        PageText(
            page_num=22,
            text=(
                "4.3\n"
                "Temporary Facilities\n"
                "Contractor shall furnish the following field support equipment.\n"
                "Temporary power panels\n"
                "Portable lighting towers\n"
                "Scaffolding systems\n"
                "Forklifts\n"
                "Contractor shall maintain all temporary facilities in safe condition.\n"
            ),
            ocr_used=False,
        )
    ]

    chunks = parse_chunks(pages, "doc_test")
    base_chunk = next(chunk for chunk in chunks if chunk.section_number == "4.3" and chunk.chunk_type == "subsection")
    item_chunks = [chunk for chunk in chunks if chunk.chunk_type == "line_item_group" and chunk.section_number == "4.3"]

    assert "Temporary power panels" in base_chunk.full_text
    assert len(item_chunks) == 2
    assert all("Contractor shall furnish the following field support equipment." not in chunk.full_text for chunk in item_chunks)
    assert all("Contractor shall maintain all temporary facilities in safe condition." not in chunk.full_text for chunk in item_chunks)
    assert all("Temporary Facilities - Line Items" in chunk.heading for chunk in item_chunks)


def test_parse_chunks_emits_bullet_groups_as_smaller_children() -> None:
    pages = [
        PageText(
            page_num=44,
            text=(
                "5.1\n"
                "Safety Requirements\n"
                "- Maintain site fencing\n"
                "- Provide fire extinguishers\n"
                "- Submit weekly safety reports\n"
                "The Owner may inspect the site at any time.\n"
            ),
            ocr_used=False,
        )
    ]

    chunks = parse_chunks(pages, "doc_test")
    bullet_chunks = [chunk for chunk in chunks if chunk.chunk_type == "bullet_group" and chunk.section_number == "5.1"]

    assert len(bullet_chunks) == 1
    assert "Maintain site fencing" in bullet_chunks[0].full_text
    assert "The Owner may inspect the site at any time." not in bullet_chunks[0].full_text
    assert "Safety Requirements - Bullets" in bullet_chunks[0].heading


def test_parse_chunks_keeps_exact_value_rows_intact_in_schedule_blocks() -> None:
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
    site_chunk = next(chunk for chunk in chunks if chunk.section_number == "2.1" and chunk.chunk_type == "subsection")
    schedule_chunks = [chunk for chunk in chunks if chunk.section_number == "2.1" and chunk.chunk_type == "schedule_block"]

    assert "14.457 psia" in site_chunk.full_text
    assert schedule_chunks
    assert any("Ambient Pressure\n14.457 psia" in chunk.full_text for chunk in schedule_chunks)
    assert all(chunk.section_number != "14.457" for chunk in chunks)
