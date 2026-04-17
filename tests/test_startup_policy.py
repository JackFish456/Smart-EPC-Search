from pathlib import Path

from epc_smart_search.app_paths import HardwareCapability
from epc_smart_search.app_paths import resolve_model_selection


def test_resolve_model_selection_prefers_high_tier_for_8gb_when_high_assets_exist() -> None:
    capability = HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected.")

    selection = resolve_model_selection(
        capability,
        model_dir_min=Path("models/gemma_min"),
        model_dir_high=Path("models/gemma_high"),
    )

    assert selection.available is True
    assert selection.tier == "ai_high"
    assert selection.model_dir == Path("models/gemma_high")


def test_resolve_model_selection_falls_back_to_min_for_8gb_when_high_assets_are_missing() -> None:
    capability = HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected.")

    selection = resolve_model_selection(
        capability,
        model_dir_min=Path("models/gemma_min"),
        model_dir_high=None,
    )

    assert selection.available is True
    assert selection.tier == "ai_min"
    assert selection.model_dir == Path("models/gemma_min")


def test_resolve_model_selection_falls_back_to_lite_when_4gb_machine_only_has_high_assets() -> None:
    capability = HardwareCapability("cuda_4gb_to_7gb", 4.0, "4 GB CUDA GPU detected.")

    selection = resolve_model_selection(
        capability,
        model_dir_min=None,
        model_dir_high=Path("models/gemma_high"),
    )

    assert selection.available is False
    assert selection.tier == "lite"
    assert "only ai-high assets" in (selection.reason or "").lower()


def test_resolve_model_selection_precedence_is_disable_then_override_then_auto() -> None:
    capability = HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected.")
    override_model = Path("support/override_model")

    disabled = resolve_model_selection(
        capability,
        disabled=True,
        override_model_dir=override_model,
        model_dir_min=Path("models/gemma_min"),
        model_dir_high=Path("models/gemma_high"),
    )
    overridden = resolve_model_selection(
        capability,
        override_model_dir=override_model,
        model_dir_min=Path("models/gemma_min"),
        model_dir_high=Path("models/gemma_high"),
    )
    automatic = resolve_model_selection(
        capability,
        model_dir_min=Path("models/gemma_min"),
        model_dir_high=Path("models/gemma_high"),
    )

    assert disabled.tier == "lite"
    assert disabled.available is False
    assert overridden.model_dir == override_model
    assert overridden.tier == "ai_high"
    assert automatic.model_dir == Path("models/gemma_high")
