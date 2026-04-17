from epc_smart_search.name_normalization import (
    build_system_aliases,
    normalize_attribute_name,
    normalize_system_name,
)


def test_system_normalization_is_stable_and_explicit() -> None:
    normalized = normalize_system_name("CCW")

    assert normalized == "closed cooling water"
    assert normalize_system_name(normalized) == normalized
    assert "ccw" in build_system_aliases(normalized)
    assert "closed cooling water system" in build_system_aliases(normalized)


def test_attribute_normalization_is_stable_and_maps_to_planner_labels() -> None:
    normalized = normalize_attribute_name("Configuration / Arrangement")

    assert normalized == "configuration"
    assert normalize_attribute_name(normalized) == normalized


def test_system_normalization_does_not_collapse_unrelated_phrases() -> None:
    assert normalize_system_name("cooling water") == "cooling water"
    assert normalize_system_name("closed cooling water") == "closed cooling water"
    assert normalize_system_name("cooling water") != normalize_system_name("closed cooling water")
