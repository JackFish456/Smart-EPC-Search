from __future__ import annotations

import json
import os
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

DEFAULT_MULTIMODAL_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "models"
    / "google"
    / "gemma-4"
    / "transformers"
    / "gemma-4-e2b-it"
    / "1"
)
DEFAULT_TEXT_ONLY_MODEL_PATH = (
    DEFAULT_MULTIMODAL_MODEL_PATH.parent.parent
    / f"{DEFAULT_MULTIMODAL_MODEL_PATH.parent.name}-text-only"
    / DEFAULT_MULTIMODAL_MODEL_PATH.name
)
DEFAULT_MODEL_PATH = DEFAULT_MULTIMODAL_MODEL_PATH
DEFAULT_SYSTEM_PROMPT = "You are a helpful local assistant."
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.65
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_IMAGE_TOKEN_ESTIMATE = 280
DEFAULT_ENABLE_THINKING = True
DEFAULT_REASONING_PRESET = "normal"
DEFAULT_TEMPERATURE_PRESET = "balanced"
DEFAULT_PERSONALITY_PRESET = "default"
DEFAULT_CONFIDENCE_PRESET = "medium"
DEFAULT_PREFER_TEXT_ONLY = True
MODEL_DIR_OVERRIDE_ENV_VAR = "EPC_SMART_SEARCH_MODEL_DIR"
CONFIDENCE_RESPONSE_HEADER_MARKER = "[[END_HEADER]]"
CONFIDENCE_REFUSAL_MESSAGE = (
    "I can't answer that within the selected confidence setting. "
    "Try lowering the confidence level or providing more context."
)
TEMPERATURE_PRESET_VALUES = {
    "predictable": 0.2,
    "balanced": 0.65,
    "creative": 1.1,
}
REASONING_PRESET_VALUES = {
    "quick": {"enable_thinking": False, "max_new_tokens": 320},
    "normal": {"enable_thinking": True, "max_new_tokens": 512},
    "thinking": {"enable_thinking": True, "max_new_tokens": 768},
}
PERSONALITY_PROMPT_OVERLAYS = {
    "default": "",
    "smart": (
        "Answer with the shortest complete response that fits the user's question. "
        "Avoid unnecessary exposition, and expand only when the user asks for more detail "
        "or the task genuinely requires it."
    ),
    "reasoning": (
        "Answer in a more thorough, analytical, and explanatory style. "
        "Show your reasoning clearly in the final response when it helps the user."
    ),
}
CONFIDENCE_PRESET_VALUES = {
    "low": {"min_confidence": 0, "max_confidence": 49},
    "medium": {"min_confidence": 50, "max_confidence": 84},
    "high": {"min_confidence": 85, "max_confidence": 100},
}


class GemmaRuntimeError(RuntimeError):
    """Base error for the local Gemma runtime."""


class DependencyError(GemmaRuntimeError):
    """Raised when required Python packages are missing."""


class ModelLoadError(GemmaRuntimeError):
    """Raised when the model cannot be loaded from disk."""


class GenerationError(GemmaRuntimeError):
    """Raised when inference fails."""


@dataclass(slots=True)
class RuntimeInfo:
    model_path: Path
    model_mode: str
    device_label: str
    dtype_label: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChatResult:
    text: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ResolvedGenerationPresets:
    reasoning_preset: str
    temperature_preset: str
    enable_thinking: bool
    max_new_tokens: int
    temperature: float


@dataclass(slots=True, frozen=True)
class ResolvedConfidencePreset:
    confidence_preset: str
    min_confidence: int
    max_confidence: int


@dataclass(slots=True, frozen=True)
class ParsedConfidenceResponse:
    confidence: int
    decision: str
    body: str


@dataclass(slots=True, frozen=True)
class ResolvedModelSpec:
    model_path: Path
    model_mode: str


CONFIDENCE_RESPONSE_PATTERN = re.compile(
    r"^\s*\[\[CONFIDENCE:(\d{1,3})\]\]\r?\n"
    r"\[\[DECISION:(answer|refuse)\]\]\r?\n"
    r"\[\[END_HEADER\]\]\r?\n?(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def resolve_generation_presets(
    reasoning_preset: str | None = None,
    temperature_preset: str | None = None,
) -> ResolvedGenerationPresets:
    resolved_reasoning = str(reasoning_preset or DEFAULT_REASONING_PRESET).strip().lower()
    resolved_temperature = str(temperature_preset or DEFAULT_TEMPERATURE_PRESET).strip().lower()

    if resolved_reasoning not in REASONING_PRESET_VALUES:
        resolved_reasoning = DEFAULT_REASONING_PRESET
    if resolved_temperature not in TEMPERATURE_PRESET_VALUES:
        resolved_temperature = DEFAULT_TEMPERATURE_PRESET

    reasoning = REASONING_PRESET_VALUES[resolved_reasoning]
    return ResolvedGenerationPresets(
        reasoning_preset=resolved_reasoning,
        temperature_preset=resolved_temperature,
        enable_thinking=bool(reasoning["enable_thinking"]),
        max_new_tokens=int(reasoning["max_new_tokens"]),
        temperature=float(TEMPERATURE_PRESET_VALUES[resolved_temperature]),
    )


def resolve_personality_preset(personality_preset: str | None = None) -> str:
    resolved = str(personality_preset or DEFAULT_PERSONALITY_PRESET).strip().lower()
    if resolved not in PERSONALITY_PROMPT_OVERLAYS:
        return DEFAULT_PERSONALITY_PRESET
    return resolved


def resolve_confidence_preset(confidence_preset: str | None = None) -> ResolvedConfidencePreset:
    resolved = str(confidence_preset or DEFAULT_CONFIDENCE_PRESET).strip().lower()
    if resolved not in CONFIDENCE_PRESET_VALUES:
        resolved = DEFAULT_CONFIDENCE_PRESET
    bounds = CONFIDENCE_PRESET_VALUES[resolved]
    return ResolvedConfidencePreset(
        confidence_preset=resolved,
        min_confidence=int(bounds["min_confidence"]),
        max_confidence=int(bounds["max_confidence"]),
    )


def compose_confidence_prompt_overlay(confidence_preset: str | None = None) -> str:
    resolved = resolve_confidence_preset(confidence_preset)
    return (
        "Confidence guidance: Start every response with exactly these three lines:\n"
        "[[CONFIDENCE:<integer 0-100>]]\n"
        "[[DECISION:answer]] or [[DECISION:refuse]]\n"
        f"{CONFIDENCE_RESPONSE_HEADER_MARKER}\n"
        f"Privately choose one confidence score from 0 to 100. "
        f"Only use [[DECISION:answer]] when your score is between "
        f"{resolved.min_confidence} and {resolved.max_confidence}, inclusive. "
        "If your score falls outside that range, use [[DECISION:refuse]]. "
        "After the header, provide only the final user-facing answer text when answering. "
        "Do not mention the confidence score, confidence label, or the header instructions in the answer body. "
        "If refusing, you may leave the body empty."
    )


def compose_system_prompt(
    system_prompt: str | None = None,
    personality_preset: str | None = None,
    confidence_preset: str | None = None,
) -> str:
    base_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    resolved_personality = resolve_personality_preset(personality_preset)
    overlay = PERSONALITY_PROMPT_OVERLAYS[resolved_personality].strip()
    sections = [base_prompt]
    if overlay:
        sections.append(f"Personality guidance: {overlay}")
    sections.append(compose_confidence_prompt_overlay(confidence_preset))
    return "\n\n".join(section for section in sections if section).strip()


def parse_confidence_response(text: str) -> ParsedConfidenceResponse | None:
    match = CONFIDENCE_RESPONSE_PATTERN.match(text)
    if match is None:
        return None

    confidence = int(match.group(1))
    if confidence < 0 or confidence > 100:
        return None

    return ParsedConfidenceResponse(
        confidence=confidence,
        decision=match.group(2).strip().lower(),
        body=match.group(3),
    )


def is_confidence_response_allowed(
    response: ParsedConfidenceResponse,
    confidence_preset: str | None = None,
) -> bool:
    resolved = resolve_confidence_preset(confidence_preset)
    return (
        response.decision == "answer"
        and resolved.min_confidence <= response.confidence <= resolved.max_confidence
    )


def derive_text_only_model_path(model_path: str | os.PathLike[str]) -> Path:
    candidate = Path(model_path).expanduser()
    return candidate.parent.parent / f"{candidate.parent.name}-text-only" / candidate.name


def infer_model_mode_from_config(config_payload: dict[str, Any]) -> str:
    architectures = [str(item).lower() for item in config_payload.get("architectures", [])]
    if any("causallm" in architecture for architecture in architectures):
        return "text_only"
    if "vision_config" not in config_payload and "audio_config" not in config_payload:
        return "text_only"
    return "multimodal"


def resolve_model_spec(
    model_path: str | os.PathLike[str] | None = None,
    *,
    prefer_text_only: bool | None = None,
) -> ResolvedModelSpec:
    if model_path is not None:
        return _resolve_model_candidate(Path(model_path).expanduser())

    explicit_override = os.environ.get(MODEL_DIR_OVERRIDE_ENV_VAR, "").strip()
    if explicit_override:
        return _resolve_model_candidate(Path(explicit_override).expanduser())

    explicit_text_only = os.environ.get("GEMMA_TEXT_ONLY_MODEL_PATH", "").strip()
    if explicit_text_only:
        return _resolve_model_candidate(Path(explicit_text_only).expanduser())

    prefer_text = _resolve_prefer_text_only(prefer_text_only)
    explicit_base = os.environ.get("GEMMA_QUANTIZED_MODEL_PATH") or os.environ.get("GEMMA_MODEL_PATH")
    if explicit_base:
        base_candidate = Path(explicit_base).expanduser()
        if prefer_text:
            derived_candidate = derive_text_only_model_path(base_candidate)
            if derived_candidate.exists():
                return _resolve_model_candidate(derived_candidate)
        return _resolve_model_candidate(base_candidate)

    if prefer_text and DEFAULT_TEXT_ONLY_MODEL_PATH.exists():
        return _resolve_model_candidate(DEFAULT_TEXT_ONLY_MODEL_PATH)
    return _resolve_model_candidate(DEFAULT_MULTIMODAL_MODEL_PATH)


def resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path:
    return resolve_model_spec(model_path).model_path


def _resolve_prefer_text_only(prefer_text_only: bool | None) -> bool:
    if prefer_text_only is not None:
        return prefer_text_only
    raw = os.environ.get("GEMMA_PREFER_TEXT_ONLY", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return DEFAULT_PREFER_TEXT_ONLY


def _resolve_model_candidate(candidate: Path) -> ResolvedModelSpec:
    if not candidate.exists():
        raise ModelLoadError(
            f"Model path does not exist: {candidate}\n"
            "Set GEMMA_MODEL_PATH or GEMMA_TEXT_ONLY_MODEL_PATH to your Gemma folder."
        )

    config_path = candidate / "config.json"
    if not config_path.exists():
        raise ModelLoadError(
            f"Model path is missing config.json: {candidate}\n"
            "Point the Gemma model path at the final checkpoint folder."
        )

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelLoadError(f"Could not parse Gemma config.json at {candidate}.\n{exc}") from exc

    return ResolvedModelSpec(
        model_path=candidate.resolve(),
        model_mode=infer_model_mode_from_config(config_payload),
    )


def build_messages(
    history: Sequence[dict[str, str]] | None,
    user_text: str,
    image_path: str | os.PathLike[str] | None = None,
    system_prompt: str | None = None,
    *,
    supports_images: bool = True,
) -> list[dict[str, Any]]:
    cleaned_user_text = user_text.strip()
    if not cleaned_user_text and not image_path:
        raise ValueError("A prompt or image is required.")
    if image_path and not supports_images:
        raise ValueError("This Gemma checkpoint is text-only and does not support image input.")

    if supports_images:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip(),
                    }
                ],
            }
        ]
    else:
        messages = [{"role": "system", "content": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()}]

    for turn in history or []:
        role = turn.get("role", "").strip()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if supports_images:
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            messages.append({"role": role, "content": content})

    if supports_images:
        user_content: list[dict[str, str]] = []
        if image_path:
            user_content.append({"type": "image", "path": str(Path(image_path).expanduser())})
        if cleaned_user_text:
            user_content.append({"type": "text", "text": cleaned_user_text})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": cleaned_user_text})
    return messages


def build_history_messages(
    history: Sequence[dict[str, str]] | None,
    system_prompt: str | None = None,
    *,
    supports_images: bool = True,
) -> list[dict[str, Any]]:
    if supports_images:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip(),
                    }
                ],
            }
        ]
    else:
        messages = [{"role": "system", "content": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()}]

    for turn in history or []:
        role = turn.get("role", "").strip()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if supports_images:
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            messages.append({"role": role, "content": content})

    return messages


def extract_generated_token_ids(output_ids: Any, input_length: int) -> Any:
    if input_length < 0:
        raise ValueError("input_length must be non-negative")

    sequence = output_ids[0] if hasattr(output_ids, "__getitem__") else output_ids
    return sequence[input_length:]


def clean_response_text(text: str) -> str:
    cleaned = re.sub(r"<\|channel\>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<|channel>", "")
    cleaned = cleaned.replace("<channel|>", "")
    cleaned = cleaned.replace("<turn|>", "")
    cleaned = cleaned.replace("<|turn>", "")
    cleaned = cleaned.replace("<bos>", "")
    cleaned = re.sub(r"<\|[^>]+?\|>", "", cleaned)
    cleaned = re.sub(
        r"^\s*thought(?:\s*[:\-]\s*|\s+)(?:\r?\n+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def decode_generated_response(decoder: Any, output_ids: Any, input_length: int) -> str:
    generated_ids = extract_generated_token_ids(output_ids, input_length)
    text = decoder.decode(generated_ids, skip_special_tokens=False)
    return clean_response_text(text)


def _import_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
            Gemma4ForCausalLM,
        )
    except ImportError as exc:
        raise DependencyError(
            "Missing Gemma runtime dependencies. Run .\\setup_venv.ps1 and start the app from .venv."
        ) from exc

    return torch, AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, Gemma4ForCausalLM


def _yield_text_chunks(text: str, chunk_size: int = 48) -> Iterator[str]:
    words = text.split()
    if not words:
        return

    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > chunk_size:
            yield f"{current} "
            current = word
        else:
            current = candidate

    if current:
        yield current


def _resolve_model_load_options(torch_module: Any) -> tuple[Any | None, Any | None, str, list[str]]:
    """Return (dtype, quantization_config, dtype_label, warnings).

    When ``quantization_config`` is set (4-bit NF4), ``dtype`` is ``None`` and must not be
    passed to ``from_pretrained`` alongside it.
    """
    warnings: list[str] = []

    if torch_module.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401

            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_module.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return None, quant_config, "4-bit NF4", warnings
        except ImportError:
            warnings.append(
                "bitsandbytes is not installed; loading full bfloat16 weights (high VRAM use). "
                "Install bitsandbytes for 4-bit NF4 quantization."
            )
            warnings.append(
                "The 8 GB GPU may still require CPU offload for this unquantized checkpoint, so responses can be slow."
            )
            return torch_module.bfloat16, None, "bfloat16", warnings

    dtype = torch_module.float32
    if getattr(torch_module.version, "cuda", None) is None:
        warnings.append(
            "CPU-only PyTorch is installed, so Gemma will run on the CPU until you install the CUDA wheel in .venv."
        )
    else:
        warnings.append(
            "CUDA build is installed but torch cannot access the GPU. Gemma may fall back to CPU/offload mode."
        )
    warnings.append(
        "CPU inference with full-precision weights is slow; use a CUDA GPU and bitsandbytes for 4-bit loading when possible."
    )

    return dtype, None, "float32", warnings


class GemmaChatRuntime:
    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        prefer_text_only: bool | None = None,
    ) -> None:
        self.model_spec = resolve_model_spec(model_path, prefer_text_only=prefer_text_only)
        self.model_path = self.model_spec.model_path
        self.model_mode = self.model_spec.model_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking

        self.model: Any | None = None
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self.runtime_info: RuntimeInfo | None = None
        self._torch: Any | None = None

    @property
    def runtime_warnings(self) -> list[str]:
        return list(self.runtime_info.warnings if self.runtime_info else [])

    def load(self) -> RuntimeInfo:
        if self.model is not None and self.runtime_info is not None and (self.processor is not None or self.tokenizer is not None):
            return self.runtime_info

        torch, auto_processor_cls, auto_model_cls, auto_tokenizer_cls, causal_model_cls = _import_runtime_dependencies()
        dtype, quantization_config, dtype_label, warnings = _resolve_model_load_options(torch)

        try:
            load_kwargs: dict[str, Any] = {"device_map": "auto"}
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = {"": 0}
            else:
                load_kwargs["dtype"] = dtype
            if self.model_mode == "text_only":
                tokenizer = auto_tokenizer_cls.from_pretrained(
                    str(self.model_path),
                    padding_side="left",
                )
                processor = None
                model = causal_model_cls.from_pretrained(
                    str(self.model_path),
                    **load_kwargs,
                )
            else:
                processor = auto_processor_cls.from_pretrained(
                    str(self.model_path),
                    padding_side="left",
                )
                tokenizer = getattr(processor, "tokenizer", None)
                model = auto_model_cls.from_pretrained(
                    str(self.model_path),
                    **load_kwargs,
                )
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:
            raise ModelLoadError(self._format_model_load_error(exc)) from exc

        device_label = self._describe_device(model, torch)
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self._torch = torch
        self.runtime_info = RuntimeInfo(
            model_path=self.model_path,
            model_mode=self.model_mode,
            device_label=device_label,
            dtype_label=dtype_label,
            warnings=warnings,
        )
        return self.runtime_info

    def generate(
        self,
        *,
        user_text: str,
        history: Sequence[dict[str, str]] | None = None,
        image_path: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatResult:
        runtime_info = self.load()
        assert self.model is not None
        assert self.tokenizer is not None or self.processor is not None
        assert self._torch is not None

        if image_path:
            image_candidate = Path(image_path).expanduser()
            if not image_candidate.exists():
                raise GenerationError(f"Image not found: {image_candidate}")
            if self.model_mode != "multimodal":
                raise GenerationError("This Gemma checkpoint is text-only and cannot accept image input.")

        messages = build_messages(
            history=history,
            user_text=user_text,
            image_path=image_path,
            system_prompt=system_prompt,
            supports_images=self.model_mode == "multimodal",
        )

        try:
            with self._inference_context():
                inputs = self._chat_template_target().apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=(
                        self.enable_thinking if enable_thinking is None else enable_thinking
                    ),
                )
                inputs = inputs.to(self._input_device())
                input_length = int(inputs["input_ids"].shape[-1])

                outputs = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    ),
                )
        except Exception as exc:
            raise GenerationError(self._format_generation_error(exc)) from exc

        response_text = decode_generated_response(
            self._chat_template_target(),
            outputs,
            input_length,
        )
        if not response_text:
            raise GenerationError("Gemma returned an empty response.")

        return ChatResult(text=response_text, warnings=list(runtime_info.warnings))

    def stream_generate(
        self,
        *,
        user_text: str,
        history: Sequence[dict[str, str]] | None = None,
        image_path: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = None,
    ) -> Iterator[str]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None or self.processor is not None
        assert self._torch is not None

        if image_path:
            image_candidate = Path(image_path).expanduser()
            if not image_candidate.exists():
                raise GenerationError(f"Image not found: {image_candidate}")
            if self.model_mode != "multimodal":
                raise GenerationError("This Gemma checkpoint is text-only and cannot accept image input.")

        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            result = self.generate(
                user_text=user_text,
                history=history,
                image_path=image_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            yield from _yield_text_chunks(result.text)
            return

        messages = build_messages(
            history=history,
            user_text=user_text,
            image_path=image_path,
            system_prompt=system_prompt,
            supports_images=self.model_mode == "multimodal",
        )

        try:
            inputs = self._chat_template_target().apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=(
                    self.enable_thinking if enable_thinking is None else enable_thinking
                ),
            )
            inputs = inputs.to(self._input_device())
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )
        except Exception:
            result = self.generate(
                user_text=user_text,
                history=history,
                image_path=image_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            yield from _yield_text_chunks(result.text)
            return

        generation_error: list[Exception] = []

        def _run_generate() -> None:
            try:
                with self._inference_context():
                    self.model.generate(
                        **inputs,
                        streamer=streamer,
                        **self._generation_kwargs(
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        ),
                    )
            except Exception as exc:
                generation_error.append(exc)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        raw_text = ""
        emitted_text = ""
        for piece in streamer:
            raw_text += piece
            cleaned = clean_response_text(raw_text)
            if not cleaned.startswith(emitted_text):
                delta = cleaned
            else:
                delta = cleaned[len(emitted_text):]
            if delta:
                emitted_text = cleaned
                yield delta

        thread.join()
        if generation_error:
            raise GenerationError(self._format_generation_error(generation_error[0]))
        if not emitted_text.strip():
            raise GenerationError("Gemma returned an empty response.")

    def _generation_kwargs(
        self,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        return {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "top_k": self.top_k if top_k is None else top_k,
            "do_sample": True,
            "use_cache": True,
        }

    def _inference_context(self) -> Any:
        assert self._torch is not None

        inference_mode = getattr(self._torch, "inference_mode", None)
        if callable(inference_mode):
            return inference_mode()

        no_grad = getattr(self._torch, "no_grad", None)
        if callable(no_grad):
            return no_grad()

        return nullcontext()

    def estimate_text_tokens(self, text: str) -> int:
        self.load()
        assert self.tokenizer is not None

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def estimate_conversation_tokens(
        self,
        history: Sequence[dict[str, str]] | None = None,
        *,
        pending_user_text: str = "",
        pending_image: bool = False,
        system_prompt: str | None = None,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        add_generation_prompt: bool = False,
    ) -> int:
        self.load()
        assert self.tokenizer is not None or self.processor is not None

        messages = build_history_messages(
            history,
            system_prompt=system_prompt,
            supports_images=self.model_mode == "multimodal",
        )
        if pending_user_text.strip() or pending_image:
            if pending_image and self.model_mode != "multimodal":
                raise GenerationError("This Gemma checkpoint is text-only and cannot accept image input.")
            pending_content: list[dict[str, str]] = []
            if pending_user_text.strip() and self.model_mode == "multimodal":
                pending_content.append({"type": "text", "text": pending_user_text.strip()})
            elif pending_user_text.strip():
                messages.append({"role": "user", "content": pending_user_text.strip()})
            elif pending_image:
                pending_content.append({"type": "text", "text": "[Pending image]"})
            if pending_content:
                messages.append({"role": "user", "content": pending_content})

        tokens = self._chat_template_target().apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        count = len(tokens)
        if pending_image:
            count += self.image_token_estimate
        return count

    @property
    def context_window(self) -> int:
        self.load()
        assert self.model is not None

        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "max_position_embeddings"):
            return int(text_config.max_position_embeddings)
        if hasattr(self.model.config, "max_position_embeddings"):
            return int(self.model.config.max_position_embeddings)
        return 0

    @property
    def image_token_estimate(self) -> int:
        self.load()
        if self.model_mode != "multimodal":
            return 0
        assert self.processor is not None

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "image_seq_length"):
                return int(image_processor.image_seq_length)
            if hasattr(image_processor, "max_soft_tokens"):
                return int(image_processor.max_soft_tokens)
        return DEFAULT_IMAGE_TOKEN_ESTIMATE

    def _input_device(self) -> Any:
        assert self.model is not None
        assert self._torch is not None

        if hasattr(self.model, "device"):
            return self.model.device

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                    return self._torch.device(mapped_device)
            if "cpu" in device_map.values():
                return self._torch.device("cpu")

        return self._torch.device("cpu")

    @staticmethod
    def _describe_device(model: Any, torch_module: Any) -> str:
        if hasattr(model, "hf_device_map"):
            unique_targets = sorted({str(value) for value in model.hf_device_map.values()})
            return ", ".join(unique_targets)

        if hasattr(model, "device"):
            device = str(model.device)
            if device.startswith("cuda") and torch_module.cuda.is_available():
                return torch_module.cuda.get_device_name(0)
            return device

        return "unknown"

    def _chat_template_target(self) -> Any:
        target = self.processor if self.processor is not None else self.tokenizer
        if target is None:
            raise GenerationError("Gemma tokenizer/processor is unavailable.")
        return target

    def _format_model_load_error(self, exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()

        if "out of memory" in lowered:
            return (
                "Gemma ran out of memory while loading.\n"
                "Try closing other GPU apps, shorten context, or confirm bitsandbytes 4-bit NF4 is active on CUDA."
            )
        if "no module named" in lowered:
            return (
                "Gemma dependencies are missing from this environment.\n"
                "Run .\\setup_venv.ps1 and then launch the app from .venv."
            )
        return f"Failed to load Gemma from {self.model_path}.\n{message}"

    @staticmethod
    def _format_generation_error(exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()

        if "out of memory" in lowered:
            return (
                "Gemma ran out of memory while generating a response.\n"
                "Try a shorter prompt or conversation, close GPU-heavy apps, or use CUDA with bitsandbytes 4-bit NF4."
            )
        if "cuda" in lowered:
            return (
                "Gemma hit a CUDA/runtime issue while generating.\n"
                "If the GPU path stays unstable, rerun from the CUDA-enabled .venv or allow CPU/offload mode."
            )
        return f"Gemma could not generate a response.\n{message}"
