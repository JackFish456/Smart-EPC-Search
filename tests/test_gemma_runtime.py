import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

import gemma_runtime


def test_resolve_model_spec_prefers_text_only_env(monkeypatch) -> None:
    base = _test_dir("env_preference")
    multimodal = _write_model_dir(base / "gemma-4-e2b-it" / "1", "Gemma4ForConditionalGeneration", include_vision=True)
    text_only = _write_model_dir(base / "gemma-4-e2b-it-text-only" / "1", "Gemma4ForCausalLM")

    monkeypatch.setenv("GEMMA_MODEL_PATH", str(multimodal))
    monkeypatch.setenv("GEMMA_TEXT_ONLY_MODEL_PATH", str(text_only))

    resolved = gemma_runtime.resolve_model_spec()

    assert resolved.model_path == text_only.resolve()
    assert resolved.model_mode == "text_only"


def test_resolve_model_spec_prefers_derived_text_only_checkpoint(monkeypatch) -> None:
    base = _test_dir("derived_preference")
    multimodal = _write_model_dir(base / "gemma-4-e2b-it" / "1", "Gemma4ForConditionalGeneration", include_vision=True)
    text_only = _write_model_dir(base / "gemma-4-e2b-it-text-only" / "1", "Gemma4ForCausalLM")

    monkeypatch.delenv("GEMMA_MODEL_PATH", raising=False)
    monkeypatch.delenv("GEMMA_QUANTIZED_MODEL_PATH", raising=False)
    monkeypatch.delenv("GEMMA_TEXT_ONLY_MODEL_PATH", raising=False)
    monkeypatch.setattr(gemma_runtime, "DEFAULT_MULTIMODAL_MODEL_PATH", multimodal)
    monkeypatch.setattr(gemma_runtime, "DEFAULT_TEXT_ONLY_MODEL_PATH", text_only)

    resolved = gemma_runtime.resolve_model_spec()

    assert resolved.model_path == text_only.resolve()
    assert resolved.model_mode == "text_only"


def test_build_messages_text_only_uses_string_content() -> None:
    messages = gemma_runtime.build_messages(
        history=[{"role": "assistant", "content": "Earlier answer."}],
        user_text="Summarize this clause.",
        system_prompt="You are concise.",
        supports_images=False,
    )

    assert messages[0]["content"] == "You are concise."
    assert messages[1]["content"] == "Earlier answer."
    assert messages[-1]["content"] == "Summarize this clause."


def test_build_messages_text_only_rejects_images() -> None:
    with pytest.raises(ValueError, match="text-only"):
        gemma_runtime.build_messages(
            history=None,
            user_text="Describe this image.",
            image_path="example.png",
            supports_images=False,
        )


def _write_model_dir(path: Path, architecture: str, *, include_vision: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {"architectures": [architecture], "model_type": "gemma4"}
    if include_vision:
        config["vision_config"] = {"model_type": "gemma4_vision"}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def _test_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"gemma_runtime_{label}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base
