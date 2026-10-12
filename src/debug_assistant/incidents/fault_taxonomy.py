"""Public Cloud-OpsBench fault vocabulary used by every experiment arm.

This is domain documentation published by the benchmark, not case-level Gold.
Case labels and process milestones remain evaluator-only.
"""

from __future__ import annotations

import re
from typing import Annotated
from pydantic import AfterValidator, Field
from debug_assistant.symbols import CanonicalSymbolError


PUBLIC_FAULT_TYPES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "admission": (
        "namespace_cpu_quota_exceeded", "namespace_memory_quota_exceeded",
        "namespace_pod_quota_exceeded", "namespace_service_quota_exceeded",
        "namespace_storage_quota_exceeded", "missing_service_account",
    ),
    "scheduling": (
        "node_cordon_mismatch", "node_affinity_mismatch", "node_selector_mismatch",
        "pod_anti_affinity_conflict", "taint_toleration_mismatch",
        "cpu_capacity_mismatch", "memory_capacity_mismatch", "pv_binding_occupied",
        "pvc_selector_mismatch", "pvc_storage_class_mismatch",
        "pvc_capacity_mismatch", "pvc_access_mode_mismatch",
    ),
    "startup": (
        "volume_mount_permission_denied", "missing_secret_binding",
        "incorrect_image_reference", "image_registry_dns_failure",
        "missing_image_pull_secret",
    ),
    "runtime": (
        "container_memory_limit_too_low", "liveness_probe_incorrect_protocol",
        "liveness_probe_incorrect_port", "liveness_probe_incorrect_timing",
        "readiness_probe_incorrect_protocol", "readiness_probe_incorrect_port",
        "service_sidecar_port_conflict", "mysql_invalid_credentials",
        "mysql_invalid_port", "db_read_only", "db_connection_exhaustion",
        "deployment_zero_replicas",
    ),
    "service": (
        "service_selector_mismatch", "service_port_mapping_mismatch",
        "service_protocol_mismatch", "service_env_var_address_mismatch",
        "gateway_misroute", "service_dns_resolution_failure",
    ),
    "performance": (
        "pod_cpu_overload", "pod_network_delay", "node_network_delay",
        "node_network_packet_loss",
    ),
    "infrastructure": (
        "containerd_unavailable", "kubelet_unavailable",
        "kube_proxy_unavailable", "kube_scheduler_unavailable",
    ),
    "codedefect": (
        "code_missing_parameter", "code_busy_loop", "code_excessive_file_reads",
        "code_wrong_return", "code_excessive_file_writes",
        "code_wrong_argument_order", "code_artificial_delay", "code_memory_leak",
    ),
}

CANONICAL_FAULT_TYPES = frozenset(
    fault_type
    for values in PUBLIC_FAULT_TYPES_BY_CATEGORY.values()
    for fault_type in values
)


def _validate_fault_code(value: str) -> str:
    if value and value not in CANONICAL_FAULT_TYPES:
        raise CanonicalSymbolError("fault taxonomy", "canonical_symbol_unknown")
    return value


# Empty is the explicit unknown diagnosis, not an invented taxonomy member.
ProposedFaultCode = Annotated[
    str, AfterValidator(_validate_fault_code),
    Field(json_schema_extra={"enum": ["", *sorted(CANONICAL_FAULT_TYPES)]}),
]

ONTOLOGY_SOURCE = (
    "Cloud-OpsBench README Fault Taxonomy at pinned commit "
    "d79f1526a4c3315004c09457fbec588eb257dd1c"
)
ONTOLOGY_STATUS = "PUBLIC_COMPLETE_57_TYPE_TAXONOMY"

# These are contradiction checks, not answer mappings.  Runtime uses them to
# prevent a Provider from submitting a public code whose own explanation says
# that the defining condition is absent.  They intentionally do not infer a
# replacement code; the Planner must revise the hypothesis or acquire evidence.
_FAULT_CODE_CONTRADICTION_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "service_selector_mismatch": (
        (
            r"\bno\s+(?:service\s+)?selector(?:\s+or\s+targetport)?\s+mismatch\b",
            "the explanation explicitly denies a selector mismatch",
        ),
        (
            r"\bselector\b[^.\n]{0,100}\bmatch(?:es|ed|ing)?\b",
            "the explanation says the Service selector matches",
        ),
    ),
    "service_port_mapping_mismatch": (
        (
            r"\bno\s+(?:service\s+)?(?:targetport|port)\s+mismatch\b",
            "the explanation explicitly denies a port mismatch",
        ),
        (
            r"\btargetport\b[^.\n]{0,100}\bmatch(?:es|ed|ing)?\b[^.\n]{0,100}\b(?:containerport|listener|port)\b",
            "the explanation says targetPort matches the backend port",
        ),
    ),
}


def is_canonical_fault_code(value: str) -> bool:
    return str(value or "").strip() in CANONICAL_FAULT_TYPES


def fault_code_contradiction(code: str, *explanations: str) -> str:
    """Return a bounded ontology contradiction, without choosing a diagnosis."""
    normalized_code = str(code or "").strip()
    patterns = _FAULT_CODE_CONTRADICTION_PATTERNS.get(normalized_code, ())
    text = " ".join(str(value or "") for value in explanations).casefold()
    # An explicit positive mismatch is evidence for the selector-mismatch
    # code, not a contradiction.  Check it before the broader ``selector ...
    # match`` guard, which would otherwise match the tail of "does not match".
    if normalized_code == "service_selector_mismatch" and re.search(
        r"\bselector\b[^.\n]{0,100}\bdoes\s+not\s+match\b", text,
        flags=re.IGNORECASE,
    ):
        return ""
    if normalized_code == "service_port_mapping_mismatch" and re.search(
        r"\b(?:targetport|port)\b[^.\n]{0,100}\bdoes\s+not\s+match\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ""
    for pattern, reason in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return ""


def render_public_fault_taxonomy() -> str:
    """Render the shared case-independent ontology once in the Planner prompt."""
    rows = [
        f"- {category}: {', '.join(values)}"
        for category, values in PUBLIC_FAULT_TYPES_BY_CATEGORY.items()
    ]
    return "PUBLIC CLOUD-OPSBENCH ROOT-CAUSE TAXONOMY (57 types):\n" + "\n".join(rows)
