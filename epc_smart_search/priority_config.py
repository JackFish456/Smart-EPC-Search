from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class PrioritySectionRule:
    label: str
    section_numbers: tuple[str, ...] = ()
    heading_terms: tuple[str, ...] = ()
    focus_terms: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PriorityConfig:
    priority_sections: tuple[PrioritySectionRule, ...] = ()


def load_priority_config(path: str | Path | None) -> PriorityConfig | None:
    if path is None:
        return None
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    sections = payload.get("priority_sections") or ()
    rules: list[PrioritySectionRule] = []
    for raw_rule in sections:
        if not isinstance(raw_rule, dict):
            continue
        label = str(raw_rule.get("label", "")).strip()
        if not label:
            continue
        rules.append(
            PrioritySectionRule(
                label=label,
                section_numbers=_clean_values(raw_rule.get("section_numbers")),
                heading_terms=_clean_values(raw_rule.get("heading_terms")),
                focus_terms=_clean_values(raw_rule.get("focus_terms")),
            )
        )
    return PriorityConfig(priority_sections=tuple(rules))


def _clean_values(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        cleaned.append(text)
    return tuple(cleaned)
