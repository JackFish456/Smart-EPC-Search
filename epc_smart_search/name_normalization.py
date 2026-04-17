from __future__ import annotations

import re

from epc_smart_search.search_features import normalize_text

LOOKUP_TOKEN_RE = re.compile(r"[a-z0-9]+")

SYSTEM_SUFFIX_TERMS = {"system", "systems"}
SINGULARIZABLE_SYSTEM_HEADS = {
    "system",
    "pump",
    "compressor",
    "turbine",
    "generator",
    "motor",
    "valve",
    "blower",
    "fan",
    "heater",
    "cooler",
    "analyzer",
    "filter",
    "tank",
    "vessel",
    "line",
    "header",
    "train",
    "skid",
    "module",
    "package",
    "unit",
}

SYSTEM_ALIAS_FAMILIES: dict[str, tuple[str, ...]] = {
    "closed cooling water": (
        "closed cooling water",
        "closed cooling water system",
        "closed cooling water systems",
        "ccw",
    ),
    "dew point heater": (
        "dew point heater",
        "dew point heaters",
        "dew point heater system",
        "dew point heater systems",
    ),
}

ATTRIBUTE_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "design_conditions": (
        "design_conditions",
        "design conditions",
        "design condition",
        "operating conditions",
        "operating condition",
        "design basis",
    ),
    "configuration": (
        "configuration",
        "configured",
        "arrangement",
        "arranged",
        "configuration arrangement",
        "configuration / arrangement",
    ),
    "size": (
        "size",
        "sizes",
        "diameter",
        "rating",
        "dimension",
        "dimensions",
    ),
    "capacity": (
        "capacity",
        "capacities",
        "output",
        "duty",
        "throughput",
    ),
    "power": (
        "power",
        "horsepower",
        "horse power",
        "hp",
        "kw",
        "kilowatt",
        "kilowatts",
        "motor power",
    ),
    "pressure": (
        "pressure",
        "pressures",
        "psi",
        "psig",
    ),
    "temperature": (
        "temperature",
        "temperatures",
    ),
    "flow": (
        "flow",
        "flows",
        "flowrate",
        "flow rate",
        "throughput",
    ),
    "type": (
        "type",
        "kind",
        "model",
        "models",
        "selected",
        "selection",
        "manufacturer",
        "vendor",
    ),
    "definition": (
        "definition",
        "defined",
        "means",
        "meaning",
    ),
    "responsibility": (
        "responsibility",
        "responsibilities",
        "responsible",
        "provide",
        "provided",
        "supply",
        "furnish",
    ),
    "function": (
        "function",
        "functions",
        "work",
        "works",
        "operate",
        "operation",
        "process",
    ),
}

ATTRIBUTE_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "configuration": ("configuration", "configured", "arrangement"),
    "design_conditions": ("design conditions", "design condition", "design basis", "operating conditions"),
    "type": ("type", "kind", "model", "selected", "manufacturer"),
    "size": ("size", "diameter", "rating", "dimension"),
    "capacity": ("capacity", "output", "duty", "throughput"),
    "power": ("horsepower", "horse power", "hp", "kw", "kilowatt", "motor power"),
    "pressure": ("pressure", "pressures", "psi", "psig"),
    "temperature": ("temperature", "temperatures"),
    "flow": ("flow", "flow rate", "flowrate", "throughput"),
    "responsibility": ("responsible", "provide", "provided", "supply", "furnish"),
    "definition": ("means", "defined", "definition"),
    "function": ("work", "works", "operate", "operation", "process"),
}


def normalize_lookup_text(text: str) -> str:
    return " ".join(LOOKUP_TOKEN_RE.findall(normalize_text(text)))


_SYSTEM_ALIAS_TO_CANONICAL = {
    normalize_lookup_text(alias): canonical
    for canonical, aliases in SYSTEM_ALIAS_FAMILIES.items()
    for alias in aliases
}

_ATTRIBUTE_ALIAS_TO_LABEL = {
    normalize_lookup_text(alias): label
    for label, aliases in ATTRIBUTE_LABEL_ALIASES.items()
    for alias in aliases
}


def normalize_system_name(text: str) -> str:
    normalized = normalize_lookup_text(text)
    if not normalized:
        return ""
    canonical = _SYSTEM_ALIAS_TO_CANONICAL.get(normalized)
    if canonical:
        return canonical

    tokens = normalized.split()
    if len(tokens) >= 4 and tokens[-1] in SYSTEM_SUFFIX_TERMS:
        normalized = " ".join(tokens[:-1])
        canonical = _SYSTEM_ALIAS_TO_CANONICAL.get(normalized)
        if canonical:
            return canonical
        tokens = normalized.split()

    if len(tokens) >= 2:
        singular_tail = _singularize_system_tail(tokens[-1])
        if singular_tail != tokens[-1]:
            normalized = " ".join([*tokens[:-1], singular_tail])
            canonical = _SYSTEM_ALIAS_TO_CANONICAL.get(normalized)
            if canonical:
                return canonical

    return _SYSTEM_ALIAS_TO_CANONICAL.get(normalized, normalized)


def build_system_aliases(text: str) -> tuple[str, ...]:
    canonical = normalize_system_name(text)
    if not canonical:
        return ()

    aliases = list(SYSTEM_ALIAS_FAMILIES.get(canonical, (canonical,)))
    tokens = canonical.split()
    if len(tokens) >= 2:
        plural_tail = _pluralize_system_tail(tokens[-1])
        if plural_tail and plural_tail != tokens[-1]:
            aliases.append(" ".join([*tokens[:-1], plural_tail]))
    if len(tokens) >= 3 and tokens[-1] not in SYSTEM_SUFFIX_TERMS:
        aliases.append(f"{canonical} system")
    return _dedupe_normalized(aliases)


def normalize_attribute_name(text: str) -> str:
    lowered = normalize_text(text).strip()
    if lowered in ATTRIBUTE_QUERY_TERMS:
        return lowered

    normalized = normalize_lookup_text(text)
    if not normalized:
        return ""

    label = _ATTRIBUTE_ALIAS_TO_LABEL.get(normalized)
    if label:
        return label

    singularized = _singularize_lookup_tail(normalized)
    label = _ATTRIBUTE_ALIAS_TO_LABEL.get(singularized)
    if label:
        return label
    return singularized


def attribute_terms_for_label(label: str) -> tuple[str, ...]:
    normalized = normalize_attribute_name(label)
    return ATTRIBUTE_QUERY_TERMS.get(normalized, (normalized,) if normalized else ())


def _singularize_lookup_tail(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    tail = tokens[-1]
    if len(tail) > 3 and tail.endswith("s") and not tail.endswith("ss"):
        tokens[-1] = tail[:-1]
    return " ".join(tokens)


def _singularize_system_tail(token: str) -> str:
    if len(token) <= 3 or not token.endswith("s") or token.endswith("ss"):
        return token
    singular = token[:-1]
    if singular in SINGULARIZABLE_SYSTEM_HEADS:
        return singular
    return token


def _pluralize_system_tail(token: str) -> str:
    if token in SYSTEM_SUFFIX_TERMS:
        return "systems" if token == "system" else token
    if token in SINGULARIZABLE_SYSTEM_HEADS and not token.endswith("s"):
        return f"{token}s"
    return token


def _dedupe_normalized(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = normalize_lookup_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out)
