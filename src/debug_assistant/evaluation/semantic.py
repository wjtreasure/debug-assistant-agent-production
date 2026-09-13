"""Post-run semantic grading ports used by the evaluator.

The evaluator owns the rubric and the call boundary.  Runtime code never sees
the rubric, and a semantic grader is not allowed to mutate a run artifact.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SemanticRubric(BaseModel):
    """Generic concept groups for a semantic field.

    Each group is an OR group; all required groups must be represented.  The
    rubric is deliberately case-agnostic and contains no case-id dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_concept_groups: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    supporting_concept_groups: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    forbidden_concept_groups: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return bool(
            self.required_concept_groups
            or self.supporting_concept_groups
            or self.forbidden_concept_groups
        )


class SemanticGrade(BaseModel):
    """Immutable, auditable semantic-grader output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: int | None = Field(default=None, ge=0, le=2)
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    rationale: str = ""
    grader: str
    provider: str = "local"
    model: str = ""
    prompt_version: str = ""


class SemanticGrader(Protocol):
    """Port for deterministic or optional post-run semantic judges."""

    def grade(
        self,
        candidate: str,
        reference: str,
        rubric: SemanticRubric,
        *,
        field: str = "",
    ) -> SemanticGrade:
        ...


class DeterministicSemanticGrader:
    """Small rubric grader with deterministic-only semantics.

    It scores concept-group coverage, not phrase aliases.  A future LLM Judge
    can implement the same port after a run; it must return metadata and cannot
    alter the candidate or the evaluator inputs.
    """

    grader_name = "deterministic_rubric_v1"
    prompt_version = "rubric-v1"

    def grade(
        self,
        candidate: str,
        reference: str,
        rubric: SemanticRubric,
        *,
        field: str = "",
    ) -> SemanticGrade:
        del reference  # The rubric is the auditable source of semantic criteria.
        text = _normalized_text(candidate)
        forbidden = [group for group in rubric.forbidden_concept_groups if _group_matches(text, group)]
        if forbidden:
            return self._grade(0, f"forbidden concept matched ({len(forbidden)} group(s))", field)

        required_hits = [
            _group_matches(text, group) for group in rubric.required_concept_groups
        ]
        supporting_hits = [
            _group_matches(text, group) for group in rubric.supporting_concept_groups
        ]
        required_total = len(required_hits)
        required_count = sum(required_hits)
        supporting_total = len(supporting_hits)
        supporting_count = sum(supporting_hits)

        if required_total:
            ratio = required_count / required_total
            if required_count == required_total:
                score = 2
            elif ratio >= 0.5:
                score = 1
            else:
                score = 0
        elif supporting_total:
            score = 2 if supporting_count == supporting_total else 1 if supporting_count else 0
        else:
            score = 0

        rationale = (
            f"required={required_count}/{required_total}; "
            f"supporting={supporting_count}/{supporting_total}"
        )
        return self._grade(score, rationale, field)

    def _grade(self, score: int, rationale: str, field: str) -> SemanticGrade:
        label = f" for {field}" if field else ""
        return SemanticGrade(
            score=score,
            status="AVAILABLE",
            rationale=f"{rationale}{label}",
            grader=self.grader_name,
            provider="local",
            prompt_version=self.prompt_version,
        )


class UnavailableSemanticGrader:
    """Explicit deterministic-only result when no semantic rubric is present."""

    def grade(
        self,
        candidate: str,
        reference: str,
        rubric: SemanticRubric,
        *,
        field: str = "",
    ) -> SemanticGrade:
        del candidate, reference, rubric
        return SemanticGrade(
            score=None,
            status="UNAVAILABLE",
            rationale=f"no evaluator rubric available{(' for ' + field) if field else ''}",
            grader="semantic_grader_unavailable",
            provider="none",
            prompt_version="",
        )


def _normalized_text(value: str) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _group_matches(text: str, concepts: tuple[str, ...]) -> bool:
    text_tokens = tuple(text.split())
    return any(_phrase_in_tokens(text_tokens, _normalized_text(concept).split()) for concept in concepts)


def _phrase_in_tokens(text_tokens: tuple[str, ...], phrase_tokens: list[str]) -> bool:
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index:index + width] == tuple(phrase_tokens)
        for index in range(len(text_tokens) - width + 1)
    )
