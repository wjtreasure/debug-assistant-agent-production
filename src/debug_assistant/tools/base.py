from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from debug_assistant.models import ToolObservation

@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    required_args: tuple[str, ...] = ()
    read_only: bool = True

class Tool:
    spec: ToolSpec
    def execute(self, **kwargs: Any) -> ToolObservation: raise NotImplementedError
