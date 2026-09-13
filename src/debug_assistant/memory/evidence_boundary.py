from __future__ import annotations

"""The single source-read -> CODE Evidence boundary.

Search and symbol tools can locate candidates, but only a successful bounded
``read_file`` observation with explicit source metadata can satisfy a source
mechanism obligation.  Keeping this predicate here prevents the Incident
runtime and EvidenceMemory from drifting apart again.
"""

from typing import Any

from debug_assistant.models import Evidence, ToolObservation


_MAX_SOURCE_LINES = 200


def is_canonical_source_evidence(
    observation: ToolObservation | None,
    evidence: Evidence | None,
    *,
    evidence_memory: Any | None = None,
) -> bool:
    """Return whether ``evidence`` is canonical, bounded source Evidence.

    When an observation is supplied, all source-read claims are checked against
    that immutable observation.  The record-only path is used when iterating the
    canonical ledger and additionally requires the provenance fields stamped by
    ``EvidenceMemory``.  Missing information fails closed.
    """

    if observation is not None:
        metadata = observation.metadata or {}
        if not observation.ok or observation.tool != "read_file":
            return False
        if metadata.get("information_source") != "source_read":
            return False
        if str(metadata.get("context_kind") or "").upper() != "CODE":
            return False
        path = str(metadata.get("path") or "").strip()
        start = metadata.get("start_line")
        end = metadata.get("end_line")
        requested_count = metadata.get("requested_line_count")
        if not path or not isinstance(start, int) or not isinstance(end, int):
            return False
        if requested_count is None:
            requested_count = end - start + 1
        if not isinstance(requested_count, int):
            return False
        if start < 1 or end < start or requested_count < 1:
            return False
        if end - start + 1 > _MAX_SOURCE_LINES or requested_count > _MAX_SOURCE_LINES:
            return False
        if evidence_memory is not None and evidence is not None:
            canonical = evidence_memory.evidence_for_observation(observation.observation_id)
            if canonical is None or canonical.evidence_id != evidence.evidence_id:
                return False
    if evidence is None:
        return False
    # Incident projection labels source reads as ``CODE`` while generic
    # callers historically used ``read_file``.  ``source`` plus structured
    # provenance is the canonical boundary; the display kind is not.
    if evidence.source != "read_file":
        return False
    if not evidence.file or not evidence.raw_observation_id:
        return False
    if not isinstance(evidence.source_start_line, int) or not isinstance(evidence.source_end_line, int):
        return False
    if evidence.source_start_line < 1 or evidence.source_end_line < evidence.source_start_line:
        return False
    if evidence.source_end_line - evidence.source_start_line + 1 > _MAX_SOURCE_LINES:
        return False
    provenance = evidence.provenance or {}
    if provenance.get("information_source") != "source_read":
        return False
    if evidence_memory is not None:
        canonical = evidence_memory.evidence_for_observation(evidence.raw_observation_id)
        if canonical is None or canonical.evidence_id != evidence.evidence_id:
            return False
    return True
