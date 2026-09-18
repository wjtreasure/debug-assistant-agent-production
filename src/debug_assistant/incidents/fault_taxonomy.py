"""Domain-wide canonical fault vocabulary.

This is an explicit runtime contract, not evaluator Gold and not a
case-specific mapping.  Gold remains loaded only by the post-run evaluator.
"""

from __future__ import annotations


CANONICAL_FAULT_TYPES = frozenset({
    "artificial_delay",
    "code_artificial_delay",
    "code_excessive_file_reads",
    "code_memory_leak",
    "code_wrong_argument_order",
    "code_wrong_return",
    "container_memory_limit_too_low",
    "cpu_overload",
    "excessive_file_reads",
    "liveness_probe_incorrect_port",
    "liveness_probe_incorrect_protocol",
    "memory_leak",
    "network_delay",
    "network_packet_loss",
    "node_network_delay",
    "node_network_packet_loss",
    "pod_cpu_overload",
    "pod_network_delay",
    "readiness_probe_incorrect_port",
    "readiness_probe_incorrect_protocol",
    "resource_saturation",
    "runtime_failure",
    "service_env_var_address_mismatch",
    "service_port_mapping_mismatch",
    "service_protocol_mismatch",
    "service_selector_mismatch",
    "wrong_argument_order",
    "wrong_return",
})


def is_canonical_fault_code(value: str) -> bool:
    return str(value or "").strip() in CANONICAL_FAULT_TYPES
