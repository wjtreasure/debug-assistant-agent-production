from __future__ import annotations

"""Small provider-boundary contracts used by the CloudOps runtime.

SWE reporting/reflection contracts live in the sibling ``debug-assistant-swe``
project. Keeping this module narrow prevents an old generic schema from
silently re-entering the CloudOps active path.
"""

import json
from typing import Any, Literal, Type

from pydantic import BaseModel, ConfigDict, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


QuestionType = Literal[
    "location", "behavior", "causality", "caller", "test", "history", "contradiction"
]


class PlannerIntent(StrictModel):
    """Optional semantic metadata accompanying a native CloudOps tool call."""

    information_need: str | None = None
    target: str | None = None
    question_type: QuestionType | None = None
    evidence_goal: str | None = None
    reason: str | None = None


def render_contract(model: Type[BaseModel], title: str) -> str:
    schema = model.model_json_schema()
    return (
        f"{title}: Return exactly one JSON object valid under this JSON Schema. "
        "Do not add fields not present in the schema.\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )


def compact_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "location": ".".join(str(x) for x in err.get("loc", ())),
            "message": err.get("msg", "validation error"),
            "type": err.get("type", "validation_error"),
        }
        for err in exc.errors(include_url=False)
    ]
