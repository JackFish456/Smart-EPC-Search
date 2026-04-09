from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.retrieval import Citation


def test_limit_citations_prefers_distinct_pages() -> None:
    citations = [
        Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote"),
        Citation("chunk2", "5.23", "Fuel Gas Supply", None, 271, 271, "quote"),
        Citation("chunk3", "11", "Permit", None, 3131, 3131, "quote"),
    ]

    limited = ContractAssistant._limit_citations(citations, limit=2)

    assert [(citation.page_start, citation.page_end) for citation in limited] == [(270, 271), (3131, 3131)]
