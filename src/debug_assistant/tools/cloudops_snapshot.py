from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from debug_assistant.models import ToolObservation
from debug_assistant.incidents.contracts import CausalChainLink, ClaimEvidenceMapping, EvidenceId, SourceClaim
from debug_assistant.tools.base import Tool, ToolArgs, ToolSpec
from debug_assistant.tools.cloudops_source import CloudOpsSourceBinding, CloudOpsSourceToolRegistry


class ServiceNameArgs(ToolArgs):
    service_name: str = Field(min_length=1)


class ConnectivityArgs(ServiceNameArgs):
    namespace: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


_KUBERNETES_RESOURCE_TYPES = Literal[
    "configmaps", "daemonsets", "deployments", "endpoints", "events",
    "ingresses", "jobs", "namespaces", "networkpolicies", "nodes",
    "persistentvolumeclaims", "persistentvolumes", "pods", "replicasets",
    "resourcequota", "rolebindings", "secrets", "serviceaccounts", "services",
    "statefulsets", "storageclasses", "horizontalpodautoscalers",
]


class EmptyArgs(ToolArgs):
    pass


class GetResourcesArgs(ToolArgs):
    resource_type: _KUBERNETES_RESOURCE_TYPES
    namespace: str = Field(min_length=1)
    name: str = Field(
        default="",
        description=(
            "Exact captured Kubernetes resource name. This is not a logical "
            "service/app name; leave empty to list resources and obtain an exact name first."
        ),
    )


class GetAppYAMLArgs(ToolArgs):
    app_name: str = Field(min_length=1)


class DescribeResourceArgs(ToolArgs):
    resource_type: _KUBERNETES_RESOURCE_TYPES
    namespace: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        description="Exact captured Kubernetes resource name returned by a list/query result; logical service/app names are not expanded.",
    )


class ErrorLogsArgs(ToolArgs):
    namespace: str = Field(min_length=1)
    service_name: str = Field(min_length=1)


class ListCodeFilesArgs(ToolArgs):
    app_name: str = Field(min_length=1)


class NodeServiceStatusArgs(ToolArgs):
    node_name: str = Field(min_length=1)
    service_name: str = Field(min_length=1)


class FinalizeDiagnosisArgs(ToolArgs):
    component: str = Field(
        min_length=1,
        description=(
            "Compatibility field. Runtime ignores this value and copies the "
            "component from the finalized Incident Hypothesis."
        ),
    )
    fault: str = Field(
        min_length=1,
        description=(
            "Compatibility field. Runtime ignores this value and copies the "
            "fault from the finalized Incident Hypothesis."
        ),
    )
    fault_code: str = Field(
        default="",
        description=(
            "Structured lowercase snake_case fault taxonomy projection. Runtime "
            "uses the already validated Incident Hypothesis as the authority."
        ),
    )
    fault_explanation: str = Field(
        default="",
        description=(
            "Evidence-grounded explanation of the fault. Runtime uses the "
            "already validated Incident Hypothesis as the authority."
        ),
    )
    mechanism: str = Field(
        min_length=1,
        description=(
            "Compatibility field. Runtime ignores this value and copies the "
            "mechanism from the finalized Incident Hypothesis."
        ),
    )
    evidence_ids: list[EvidenceId] = Field(
        min_length=2,
        description=(
            "Compatibility Evidence projection. Runtime validates the IDs but "
            "uses the finalized Incident Hypothesis supporting Evidence as the "
            "Candidate authority. Final Review cannot see uncited Evidence."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    claim_evidence_mapping: list[ClaimEvidenceMapping] = Field(
        default_factory=list,
        description="Optional structured mapping from each material claim to supporting ev-* IDs.",
    )
    causal_chain_summary: list[CausalChainLink] = Field(
        default_factory=list,
        description="Optional concise cause/effect chain; never include raw chain-of-thought.",
    )
    source_claims: list[SourceClaim] = Field(
        default_factory=list,
        description=(
            "Optional source claims. Each claim must declare the canonical file "
            "and exact line range covered by cited read_file CODE Evidence."
        ),
    )


class _SnapshotTool(Tool):
    def __init__(self, spec: ToolSpec, records: list[dict[str, Any]]):
        self.spec = spec
        self._records = records

    def execute(self, **kwargs: Any) -> ToolObservation:
        context_metadata = _incident_context_metadata(self.spec.name, kwargs)
        for record in self._records:
            if record.get("arguments") == kwargs:
                output = record.get("output", "")
                if self.spec.name == "list_code_files":
                    output = _normalize_code_index_output(output)
                content = _serialize_snapshot_output(output)
                empty = _is_semantically_empty(output)
                return ToolObservation(tool=self.spec.name, ok=True, content=content,
                                       metadata={
                                           "arguments": kwargs, "snapshot": True,
                                           "status": "AVAILABLE", "semantic_negative": empty,
                                           **context_metadata,
                                       })
        return ToolObservation(
            tool=self.spec.name,
            ok=False,
            content=json.dumps({
                "status": "UNAVAILABLE",
                "reason": "observation_not_present_in_snapshot",
                "semantic_negative": False,
            }),
            metadata={
                "arguments": kwargs, "snapshot": True, "status": "UNAVAILABLE",
                "reason": "observation_not_present_in_snapshot", "semantic_negative": False,
                **context_metadata,
            },
            error_type="snapshot_unavailable",
        )


class _TopologyTool(Tool):
    spec = ToolSpec(
        "get_service_topology",
        "Read structured Online Boutique responsibility, upstream, downstream, dependencies and typical signals.",
        ServiceNameArgs,
        capability="incident_read", parallel_safe=True,
    )

    def __init__(self, topology_path: Path):
        self._topology = json.loads(topology_path.read_text(encoding="utf-8"))["services"]

    def execute(self, **kwargs: Any) -> ToolObservation:
        name = kwargs["service_name"]
        value = self._topology.get(name)
        if value is None:
            return ToolObservation(tool=self.spec.name, ok=False, content=f"Unknown service: {name}",
                                   metadata={"arguments": kwargs}, error_type="not_found")
        return ToolObservation(tool=self.spec.name, ok=True, content=json.dumps(value, ensure_ascii=False, sort_keys=True),
                               metadata={
                                   "arguments": kwargs, "structured_topology": True,
                                   "context_kind": "SERVICE", "context_target": name,
                               })


class CloudOpsSnapshotToolRegistry:
    """Read-only replay registry for a runtime-visible Cloud-OpsBench snapshot."""

    # These primitives remain registered and executable for compatibility with
    # shared repository/SWE callers.  ``code_search`` is a first-class Incident
    # Planner capability only when a bound source workspace exists; the source
    # adapter is added below only in that case.
    PLANNER_HIDDEN_TOOLS = frozenset({
        "list_code_files", "inspect_symbol_context",
    })
    PLANNER_CODE_TOOLS = ("repo_tree", "grep", "read_file", "symbol_search", "code_search")

    def __init__(self, runtime_data_dir: str | Path, topology_path: str | Path, *, search_engine=None):
        cache = json.loads((Path(runtime_data_dir) / "tool_cache.json").read_text(encoding="utf-8"))
        specs = {
            "check_service_connectivity": ToolSpec(
                "check_service_connectivity", "Probe a service and port from the captured incident environment.",
                ConnectivityArgs, capability="incident_read",
            ),
            "get_resources": ToolSpec(
                "get_resources", "Read captured Kubernetes state. Resource name is exact; omit it to list and discover a captured resource name.",
                GetResourcesArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_app_yaml": ToolSpec(
                "get_app_yaml", "Read captured workload and Service manifests for one application.",
                GetAppYAMLArgs, capability="incident_read", parallel_safe=True,
            ),
            "describe_resource": ToolSpec(
                "describe_resource", "Read a captured Kubernetes resource by its exact captured name, including state, configuration and events.",
                DescribeResourceArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_alerts": ToolSpec(
                "get_alerts", "Read captured anomaly and alert summaries for the incident environment.",
                EmptyArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_error_logs": ToolSpec(
                "get_error_logs", "Read captured error-log summaries for one service in a namespace.",
                ErrorLogsArgs, capability="incident_read", parallel_safe=True,
            ),
            "list_code_files": ToolSpec(
                "list_code_files", "Read the captured application source-file inventory and file descriptions.",
                ListCodeFilesArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_service_dependencies": ToolSpec(
                "get_service_dependencies", "Read captured upstream and downstream service dependencies.",
                ServiceNameArgs, capability="incident_read", parallel_safe=True,
            ),
            "check_node_service_status": ToolSpec(
                "check_node_service_status", "Read captured node-level system service status.",
                NodeServiceStatusArgs, capability="incident_read", parallel_safe=True,
            ),
            "get_cluster_configuration": ToolSpec(
                "get_cluster_configuration", "Read captured cluster node readiness and allocatable resources.",
                EmptyArgs, capability="incident_read", parallel_safe=True,
            ),
        }
        records = _snapshot_records(cache)
        self._tools = {name: _SnapshotTool(spec, records[name]) for name, spec in specs.items()}
        topology = _TopologyTool(Path(topology_path))
        self._tools[topology.spec.name] = topology
        self.source_binding = CloudOpsSourceBinding(runtime_data_dir)
        if self.source_binding.available:
            self._tools.update(CloudOpsSourceToolRegistry(
                self.source_binding, search_engine=search_engine,
            ).tools())
        self._finalize_spec = ToolSpec(
            "finalize_diagnosis", "Submit the evidence-backed final root-cause candidate for review. Final Review sees only the cited evidence_ids, so include every Evidence item required to support the component and mechanism.",
            FinalizeDiagnosisArgs, capability="diagnosis_control",
        )

    def get(self, name: str):
        return self._tools.get(name)

    def register(self, tool: Tool) -> None:
        """Register an optional capability-layer tool on this existing registry."""
        name = str(tool.spec.name)
        if name == self._finalize_spec.name or name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def specs(self):
        return [tool.spec for tool in self._tools.values()] + [self._finalize_spec]

    def function_schemas(self, visible_tools=None):
        visible = None if visible_tools is None else {str(name) for name in visible_tools}
        schemas = []
        for spec in self.specs():
            if spec.name in self.PLANNER_HIDDEN_TOOLS:
                continue
            if visible is not None and spec.name not in visible:
                continue
            schema = _compact_provider_schema(
                spec.function_schema(),
                keep_descriptions=spec.name == "finalize_diagnosis",
            )
            if spec.name == "code_search":
                # The incident provider supplies a query and result bound only.
                # Retrieval mode is an implementation policy selected by the
                # bound source tool, not a Planner knob.
                properties = schema["function"]["parameters"].get("properties", {})
                properties.pop("mode", None)
                required = schema["function"]["parameters"].get("required", [])
                schema["function"]["parameters"]["required"] = [
                    item for item in required if item != "mode"
                ]
            schemas.append(schema)
        return schemas

    def planner_visible_code_tools(self) -> tuple[str, ...]:
        """Return the code-investigation tools actually exposed to the Planner."""
        return tuple(
            name for name in self.PLANNER_CODE_TOOLS
            if self.get(name) is not None and name not in self.PLANNER_HIDDEN_TOOLS
        )

    def validate_arguments(self, name: str, arguments: dict[str, Any]):
        spec = next((item for item in self.specs() if item.name == name), None)
        if spec is None:
            return None, {"error_type": "unknown_tool", "message": f"unknown tool: {name}"}
        try:
            normalized = spec.args_model.model_validate(arguments).model_dump()
            if name == "code_search" and self.get(name) is not None:
                # Keep the Planner-facing contract high level.  The bound
                # CodeSearchTool runs hybrid_ast, then reports lexical_ast or
                # lexical when semantic retrieval is unavailable.
                normalized["mode"] = "hybrid_ast"
            return normalized, None
        except Exception as exc:
            return None, {"error_type": "schema_validation", "message": str(exc)}


_OFFICIAL_TOOL_NAMES = {
    "CheckServiceConnectivity": "check_service_connectivity",
    "GetResources": "get_resources",
    "GetAppYAML": "get_app_yaml",
    "DescribeResource": "describe_resource",
    "GetAlerts": "get_alerts",
    "GetErrorLogs": "get_error_logs",
    "ListCodeFiles": "list_code_files",
    "GetServiceDependencies": "get_service_dependencies",
    "CheckNodeServiceStatus": "check_node_service_status",
    "GetClusterConfiguration": "get_cluster_configuration",
}


def _incident_context_metadata(tool_name: str, arguments: dict[str, Any]) -> dict[str, str]:
    resource_type = str(arguments.get("resource_type") or "").lower()
    resource_kinds = {
        "pods": "POD", "services": "SERVICE", "endpoints": "ENDPOINT",
        "deployments": "DEPLOYMENT", "events": "EVENT", "nodes": "NODE",
        "configmaps": "CONFIGMAP", "daemonsets": "DAEMONSET", "ingresses": "INGRESS",
        "jobs": "JOB", "namespaces": "NAMESPACE", "networkpolicies": "NETWORKPOLICY",
        "persistentvolumeclaims": "PVC", "persistentvolumes": "PV", "replicasets": "REPLICASET",
        "resourcequota": "RESOURCEQUOTA", "rolebindings": "ROLEBINDING",
        "secrets": "SECRET", "serviceaccounts": "SERVICEACCOUNT", "statefulsets": "STATEFULSET",
        "storageclasses": "STORAGECLASS", "horizontalpodautoscalers": "HPA",
    }
    if tool_name == "get_app_yaml":
        kind = "CONFIG"
    elif tool_name == "check_service_connectivity":
        kind = "ENDPOINT"
    elif tool_name == "get_alerts":
        kind = "ALERT"
    elif tool_name == "get_error_logs":
        kind = "LOG"
    elif tool_name == "list_code_files":
        kind = "CODE_INDEX"
    elif tool_name == "get_service_dependencies":
        kind = "SERVICE"
    elif tool_name == "check_node_service_status":
        kind = "NODE"
    elif tool_name == "get_cluster_configuration":
        kind = "CLUSTER"
    else:
        kind = resource_kinds.get(resource_type, "OBSERVATION")
    target = str(arguments.get("name") or arguments.get("app_name")
                  or arguments.get("service_name") or arguments.get("node_name")
                  or ("alerts" if tool_name == "get_alerts" else resource_type))
    return {"context_kind": kind, "context_target": target}


def _serialize_snapshot_output(output: Any) -> str:
    """Keep structured official outputs JSON-shaped in model-visible evidence."""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def _compact_provider_schema(schema: dict[str, Any], *, keep_descriptions: bool = False) -> dict[str, Any]:
    """Remove non-semantic JSON-Schema decoration before provider transport.

    Pydantic remains the validation source of truth.  Provider-facing tool
    schemas only need names, types, enums, constraints and required fields;
    titles/defaults/examples and long field descriptions are repeated on every
    Planner call.  The finalization schema keeps descriptions because its
    Evidence projection is a high-risk compatibility boundary.
    """
    def compact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in {"title", "default", "examples", "deprecated", "$comment"}:
                    continue
                if key == "description" and not keep_descriptions:
                    continue
                result[key] = compact(item)
            return result
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    compacted = compact(copy.deepcopy(schema))
    if not keep_descriptions:
        # A few high-risk argument descriptions are part of the executable
        # contract from a Planner's perspective. Preserve those short cues
        # while still removing the large repeated prose from ordinary fields.
        original_properties = (
            schema.get("function", {}).get("parameters", {}).get("properties", {})
        )
        compacted_properties = (
            compacted.get("function", {}).get("parameters", {}).get("properties", {})
        )
        for field_name in ("name",):
            description = original_properties.get(field_name, {}).get("description")
            if description and field_name in compacted_properties:
                compacted_properties[field_name]["description"] = description
    return compacted


def _normalize_code_index_output(output: Any) -> Any:
    """Make cache paths directly usable with the case-local ``code/`` binding."""
    if not isinstance(output, dict):
        return output
    normalized = dict(output)
    root = str(normalized.get("root") or "").replace("\\", "/").strip("/")
    if root == "code":
        source_root = ""
    elif root.startswith("code/"):
        source_root = root[len("code/"):]
    else:
        source_root = root
    if source_root:
        normalized["root"] = source_root
    normalized["source_root"] = "code"
    normalized["source_relative"] = True
    files = []
    for item in normalized.get("files") or []:
        if not isinstance(item, dict):
            files.append(item)
            continue
        file_item = dict(item)
        path = str(file_item.get("path") or "").replace("\\", "/").lstrip("/")
        if source_root and path != source_root and not path.startswith(source_root + "/"):
            path = f"{source_root}/{path}" if path else source_root
        file_item["path"] = path
        files.append(file_item)
    normalized["files"] = files
    return normalized


def _is_semantically_empty(output: Any) -> bool:
    """An available empty query is a negative observation, not causal evidence."""
    return output is None or output == {} or output == [] or output == ""


def _snapshot_records(cache: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read curated fixtures or the official Cloud-OpsBench flat cache unchanged."""
    records = {name: [] for name in _OFFICIAL_TOOL_NAMES.values()}
    if any(":" in key for key in cache):
        for cache_key, output in cache.items():
            tool_name, separator, raw_arguments = cache_key.partition(":")
            internal_name = _OFFICIAL_TOOL_NAMES.get(tool_name)
            if not separator or internal_name is None:
                continue
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                records[internal_name].append({"arguments": arguments, "output": output})
        return records
    for name in records:
        records[name] = list(cache.get(name) or [])
    return records
