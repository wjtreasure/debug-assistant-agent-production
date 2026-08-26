from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Type
from pydantic import BaseModel, ConfigDict
from debug_assistant.models import ToolObservation


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    capability: str = "repository_read"
    cost_class: str = "light"
    side_effect: str = "none"
    output_limit: int | None = None

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


class Tool:
    spec: ToolSpec
    def execute(self, **kwargs: Any) -> ToolObservation:
        raise NotImplementedError
