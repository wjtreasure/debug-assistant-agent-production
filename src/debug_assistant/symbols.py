"""Representation-only resolution for explicitly declared symbolic domains."""
from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ValidationInfo, model_validator


class CanonicalSymbolError(ValueError):
    def __init__(self, domain: str, reason: str):
        self.domain, self.reason_code = domain, reason
        super().__init__(f"{reason}: {domain}")


def resolve_symbol(value: str, symbols: Iterable[str], *, domain: str) -> str:
    matches = [symbol for symbol in symbols if symbol.casefold() == value.strip().casefold()]
    if len(matches) != 1:
        raise CanonicalSymbolError(domain, "canonical_symbol_ambiguous" if matches else "canonical_symbol_unknown")
    return matches[0]


def canonicalize_output(value: Any, schema: dict, *, domains: dict | None = None,
                        events: list | None = None) -> Any:
    """Only schema enums and explicitly supplied domains authorize rewriting.

    Exact references, prose, argument values and dictionary keys are untouched.
    The input is never mutated. Unknown/ambiguous symbols fail before state writes.
    """
    domains, events = domains or {}, events if events is not None else []

    def visit(item, node, path):
        if "$ref" in node:
            node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        variants = node.get("anyOf", ())
        if variants:
            node = next((part for part in variants if part.get("type") != "null"), node)
            if "$ref" in node:
                node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        domain = node.get("title") or (str(path[-1]) if path else "output")
        symbols = node.get("enum")
        if path and path[-1] in domains:
            domain, symbols = domains[path[-1]]
        if isinstance(item, str) and symbols is not None and all(isinstance(s, str) for s in symbols):
            try:
                result = resolve_symbol(item, symbols, domain=domain)
            except CanonicalSymbolError as exc:
                if exc.reason_code == "canonical_symbol_ambiguous":
                    raise
                # Preserve unknown input for the strict enum/domain validator.
                # This retains nested field locations and all schema failures.
                return item
            if result != item:
                events.append({"field": list(path), "domain": domain, "from": item, "to": result})
            return result
        if isinstance(item, dict):
            properties = node.get("properties", {})
            return {key: visit(val, properties[key], (*path, key)) if key in properties else val
                    for key, val in item.items()}
        if isinstance(item, (list, tuple)) and "items" in node:
            return [visit(val, node["items"], (*path, index)) for index, val in enumerate(item)]
        return item

    return visit(value, schema, ())


class CanonicalOutputModel(BaseModel):
    """Strict role output with one schema-directed representation boundary."""

    @model_validator(mode="before")
    @classmethod
    def canonical_symbols(cls, value, info: ValidationInfo):
        context = info.context or {}
        return canonicalize_output(value, cls.model_json_schema(),
                                   domains=context.get("symbol_domains"),
                                   events=context.get("normalization_events"))
