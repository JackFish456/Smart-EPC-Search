from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from gemma_runtime import DEFAULT_MULTIMODAL_MODEL_PATH, derive_text_only_model_path


def export_text_only_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    try:
        import torch
        from accelerate import init_empty_weights
        from transformers import (
            AutoModelForImageTextToText,
            AutoTokenizer,
            Gemma4ForCausalLM,
            Gemma4TextConfig,
            GenerationConfig,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Missing export dependencies. Run this from the CUDA-enabled Gemma venv with transformers and accelerate installed."
        ) from exc

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source model path does not exist: {source}")
    if not (source / "config.json").exists():
        raise SystemExit(f"Source model path is missing config.json: {source}")
    if destination.exists():
        if not force:
            raise SystemExit(f"Destination already exists: {destination}\nUse --force to replace it.")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Loading multimodal checkpoint from {source} ...")
    source_model = AutoModelForImageTextToText.from_pretrained(
        str(source),
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source), padding_side="left")

    source_state = source_model.state_dict()
    mapped_state = {
        key.replace("model.language_model.", "model.", 1): value
        for key, value in source_state.items()
        if key.startswith("model.language_model.")
    }
    mapped_state["lm_head.weight"] = source_state["model.language_model.embed_tokens.weight"]

    text_config = Gemma4TextConfig(**source_model.config.text_config.to_dict())
    with init_empty_weights():
        target_model = Gemma4ForCausalLM(text_config)
    expected_keys = set(target_model.state_dict().keys())
    actual_keys = set(mapped_state.keys())
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise SystemExit(
            "Could not build a complete text-only Gemma checkpoint.\n"
            f"Missing keys: {missing[:8]}\n"
            f"Extra keys: {extra[:8]}"
        )

    print(f"Saving text-only checkpoint to {destination} ...")
    target_model.save_pretrained(
        str(destination),
        state_dict=mapped_state,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(str(destination))

    try:
        generation_config = GenerationConfig.from_pretrained(str(source))
        generation_config.save_pretrained(str(destination))
    except Exception:
        pass

    for filename in ("chat_template.jinja",):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, destination / filename)

    manifest = {
        "source_model_path": str(source),
        "destination_model_path": str(destination),
        "export_type": "gemma4_text_only",
        "notes": "Derived from the multimodal Gemma checkpoint by exporting only language-model weights.",
    }
    (destination / "epc_text_only_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    del target_model
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a text-only Gemma 4 checkpoint from the current multimodal checkpoint.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_MULTIMODAL_MODEL_PATH),
        help="Path to the existing multimodal Gemma checkpoint.",
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Output path for the derived text-only checkpoint. Defaults to a sibling *-text-only folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination folder if it already exists.",
    )
    args = parser.parse_args()

    destination = Path(args.destination).expanduser() if args.destination else derive_text_only_model_path(args.source)
    output = export_text_only_checkpoint(args.source, destination, force=args.force)
    print(f"Text-only Gemma checkpoint exported to: {output}")
    print("The runtime will prefer this checkpoint automatically unless GEMMA_PREFER_TEXT_ONLY=0 is set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
