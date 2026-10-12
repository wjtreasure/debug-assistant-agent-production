"""Finite local selectors and operational results; no bodies or acquisition state."""
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LocalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FullStoredBody(LocalModel):
    kind: Literal["FULL_STORED_BODY"]


class LineRange(LocalModel):
    kind: Literal["LINE_RANGE"]
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_line < self.start_line:
            raise ValueError("context_range_invalid")
        return self


class FieldPath(LocalModel):
    kind: Literal["FIELD_PATH"]
    path: tuple[str, ...] = Field(min_length=1)


ContextSelector = Annotated[FullStoredBody | LineRange | FieldPath, Field(discriminator="kind")]


class ContextRequest(LocalModel):
    observation_id: str | None = Field(default=None, min_length=1)
    evidence_ref: str | None = Field(default=None, min_length=1)
    selector: ContextSelector

    @model_validator(mode="after")
    def one_reference(self):
        if (self.observation_id is None) == (self.evidence_ref is None):
            raise ValueError("context_reference_requires_exactly_one_target")
        return self


class ContextRequestBatch(LocalModel):
    requests: tuple[ContextRequest, ...] = Field(min_length=1, max_length=8)


class ContextResolutionReason(str, Enum):
    UNKNOWN_OBSERVATION = 'context_unknown_observation_id'
    UNKNOWN_EVIDENCE = 'context_unknown_evidence_ref'
    PROVENANCE_INVALID = 'context_provenance_invalid'
    NOT_RETAINED = 'context_content_not_retained'
    UNSUPPORTED_SELECTOR = 'context_selector_unsupported'
    ALREADY_VISIBLE = 'context_already_visible'
    PACKING_INSUFFICIENT = 'context_packing_insufficient'
    REPEATED_REQUEST = 'context_request_repeated'
    ROUND_LIMIT = 'context_resolution_limit'
    UNAVAILABLE = 'context_resolution_unavailable'


class LocalResolutionFeedback(LocalModel):
    reason: Literal['context_packing_insufficient']
    requests: tuple[dict, ...]


class ContextResolutionError(RuntimeError):
    """Controlled local insufficiency, never a Provider contract retry."""
    def __init__(self, reason_code: str, *, feedback=None):
        self.feedback = feedback
        self.reason = ContextResolutionReason(reason_code)
        self.reason_code = self.reason.value
        super().__init__(self.reason_code)
