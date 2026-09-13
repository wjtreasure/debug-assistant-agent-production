from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from .contracts import KnowledgeSource, KnowledgeQuery
from .router import KnowledgeCoordinator
from debug_assistant.models import ToolObservation
from debug_assistant.tools.base import Tool, ToolArgs, ToolSpec


class KnowledgeRetrievalArgs(ToolArgs):
    query_text: str = Field(min_length=1, max_length=4000)
    service: str | None = None
    module: str | None = None
    software_version: str | None = None
    repo_commit: str | None = None
    requested_sources: tuple[KnowledgeSource, ...] = ()
    top_k: int = Field(default=5, ge=1, le=20)
    token_budget: int = Field(default=1200, ge=1, le=100_000)


class KnowledgeRetrievalTool(Tool):
    """Planner-visible high-level Knowledge tool returning Prior only."""

    spec = ToolSpec(
        "knowledge_retrieval",
        "Retrieve historical/domain/static knowledge as prior context. Results cannot be cited as current ev-* evidence and cannot finalize a diagnosis.",
        KnowledgeRetrievalArgs,
        capability="knowledge_read",
        cost_class="medium",
        side_effect="none",
        output_limit=16000,
    )

    def __init__(self, coordinator: KnowledgeCoordinator, incident_id: str, *, max_token_budget: int | None = None) -> None:
        self.coordinator = coordinator
        self.incident_id = incident_id
        self.max_token_budget = None if max_token_budget is None else max(1, int(max_token_budget))
        self.last_prior_context = None
        self.last_result = None

    def execute(self, **kwargs: Any) -> ToolObservation:
        if self.max_token_budget is not None:
            requested = int(kwargs.get("token_budget", self.max_token_budget))
            kwargs["token_budget"] = max(1, min(requested, self.max_token_budget))
        query = KnowledgeQuery(incident_id=self.incident_id, **kwargs)
        result = self.coordinator.retrieve(query)
        context = self.coordinator.prior_context(query, result=result)
        self.last_result = result
        self.last_prior_context = context
        payload = {
            "prior_only": True,
            "prior_context": context.model_dump(mode="json"),
            "diagnostics": result.diagnostics.model_dump(mode="json"),
        }
        return ToolObservation(
            tool=self.spec.name, ok=True,
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata={
                "information_source": "prior",
                "context_kind": "PRIOR",
                "prior_only": True,
                "knowledge_diagnostics": result.diagnostics.model_dump(mode="json"),
                "semantic_negative": not bool(result.candidates),
                "retryable": False,
            },
        )
