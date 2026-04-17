from __future__ import annotations

from collections import Counter

import pytest

from epc_smart_search.regression_benchmark import (
    RegressionBenchmarkHarness,
    assert_answer_contains_exact_value,
    assert_expected_citations_exist,
    assert_expected_page_hit,
    assert_expected_refusal,
    assert_expected_section,
    assert_expected_top_hit,
    cases_for_category,
    load_regression_benchmark_cases,
)


BENCHMARK_CASES = load_regression_benchmark_cases()


@pytest.fixture(scope="module")
def regression_benchmark_harness() -> RegressionBenchmarkHarness:
    return RegressionBenchmarkHarness.from_defaults()


def test_regression_benchmark_dataset_shape() -> None:
    counts = Counter(case.category for case in BENCHMARK_CASES)

    assert 40 <= len(BENCHMARK_CASES) <= 60
    assert set(counts) == {"exact_value", "system_summary", "section_lookup", "page_lookup", "no_answer"}
    assert all(count >= 5 for count in counts.values())


@pytest.mark.parametrize("case", cases_for_category(BENCHMARK_CASES, "exact_value"), ids=lambda case: case.name)
def test_regression_benchmark_exact_value(
    regression_benchmark_harness: RegressionBenchmarkHarness,
    case,
) -> None:
    result = regression_benchmark_harness.evaluate_case(case)

    assert_expected_refusal(case, result)
    assert_expected_top_hit(case, result)
    assert_answer_contains_exact_value(case, result)
    assert_expected_citations_exist(case, result)


@pytest.mark.parametrize("case", cases_for_category(BENCHMARK_CASES, "system_summary"), ids=lambda case: case.name)
def test_regression_benchmark_system_summary(
    regression_benchmark_harness: RegressionBenchmarkHarness,
    case,
) -> None:
    result = regression_benchmark_harness.evaluate_case(case)

    assert_expected_refusal(case, result)
    assert_expected_top_hit(case, result)
    assert_answer_contains_exact_value(case, result)
    assert_expected_citations_exist(case, result)


@pytest.mark.parametrize("case", cases_for_category(BENCHMARK_CASES, "section_lookup"), ids=lambda case: case.name)
def test_regression_benchmark_section_lookup(
    regression_benchmark_harness: RegressionBenchmarkHarness,
    case,
) -> None:
    result = regression_benchmark_harness.evaluate_case(case)

    assert_expected_refusal(case, result)
    assert_expected_top_hit(case, result)
    assert_expected_section(case, result)
    assert_expected_citations_exist(case, result)


@pytest.mark.parametrize("case", cases_for_category(BENCHMARK_CASES, "page_lookup"), ids=lambda case: case.name)
def test_regression_benchmark_page_lookup(
    regression_benchmark_harness: RegressionBenchmarkHarness,
    case,
) -> None:
    result = regression_benchmark_harness.evaluate_case(case)

    assert_expected_refusal(case, result)
    assert_expected_top_hit(case, result)
    assert_expected_page_hit(case, result)


@pytest.mark.parametrize("case", cases_for_category(BENCHMARK_CASES, "no_answer"), ids=lambda case: case.name)
def test_regression_benchmark_no_answer(
    regression_benchmark_harness: RegressionBenchmarkHarness,
    case,
) -> None:
    result = regression_benchmark_harness.evaluate_case(case)

    assert_expected_refusal(case, result)
