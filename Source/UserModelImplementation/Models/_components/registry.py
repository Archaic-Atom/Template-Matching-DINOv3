# -*- coding: utf-8 -*-
"""Component registry for YAML-driven model composition.

Each component module (encoders, matching heads, losses) registers its
public class/function under a string name via the @register decorator.
The builder looks names up at config-parse time to instantiate components.
"""
from __future__ import annotations

from typing import Any, Callable


COMPONENT_REGISTRY: dict[str, Any] = {}


def register(name: str) -> Callable[[Any], Any]:
    """Register a class or function under `name` for config-driven lookup."""
    def decorator(obj: Any) -> Any:
        if name in COMPONENT_REGISTRY:
            existing = COMPONENT_REGISTRY[name]
            if existing is not obj:
                raise ValueError(
                    f"Duplicate component registration for '{name}'. "
                    f"Existing: {existing!r}, new: {obj!r}"
                )
        COMPONENT_REGISTRY[name] = obj
        return obj
    return decorator


def get_component(name: str) -> Any:
    if name not in COMPONENT_REGISTRY:
        raise KeyError(
            f"Unknown component type '{name}'. "
            f"Registered components: {sorted(COMPONENT_REGISTRY)}"
        )
    return COMPONENT_REGISTRY[name]


def list_components() -> list[str]:
    return sorted(COMPONENT_REGISTRY)
