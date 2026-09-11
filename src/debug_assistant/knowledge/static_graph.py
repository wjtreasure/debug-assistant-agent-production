from __future__ import annotations

"""Small deterministic Static Knowledge Graph.

This module models deploy-time and repository structure only.  It is a
Knowledge/Prior capability: graph candidates never enter the run-scoped
Observation/Evidence ledger and there is intentionally no Cypher/Gremlin or
runtime evidence edge.
"""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .contracts import KnowledgeCandidate, KnowledgeProvenance


EntityType = Literal[
    "Service", "Deployment", "API", "Module", "Repository", "File",
    "Symbol", "FailureMode",
]
RelationType = Literal[
    "DEPENDS_ON", "CALLS", "OWNS_API", "DEPLOYED_AS", "IMPLEMENTED_BY",
    "CONTAINS_SYMBOL", "KNOWN_FAILURE_MODE",
]


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    entity_type: EntityType
    name: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    version: str | None = None
    repo_commit: str | None = None
    timestamp: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: KnowledgeProvenance


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relation: RelationType
    version: str | None = None
    repo_commit: str | None = None
    timestamp: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: KnowledgeProvenance


class StaticGraphQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[KnowledgeCandidate, ...] = ()
    query: str
    version: str | None = None
    degraded: bool = False
    reason: str = ""


class GraphStore(Protocol):
    def add_node(self, node: GraphNode) -> GraphNode:
        ...

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        ...

    def get_node(self, node_id: str, *, version: str | None = None) -> GraphNode | None:
        ...


def _stable_id(*parts: object) -> str:
    import hashlib

    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


class InMemoryStaticGraph:
    """Version-aware graph store with deterministic traversal order."""

    def __init__(self) -> None:
        self._nodes: dict[tuple[str | None, str], GraphNode] = {}
        self._edges: dict[tuple[str | None, str], GraphEdge] = {}

    def add_node(self, node: GraphNode) -> GraphNode:
        key = (node.version, node.node_id)
        self._nodes[key] = node
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        key = (edge.version, edge.edge_id)
        self._edges[key] = edge
        return edge

    def get_node(self, node_id: str, *, version: str | None = None) -> GraphNode | None:
        exact = self._nodes.get((version, node_id))
        if exact is not None:
            return exact
        if version is not None:
            return None
        matches = [node for (node_version, item_id), node in self._nodes.items() if item_id == node_id]
        return sorted(matches, key=lambda item: item.version or "")[0] if len(matches) == 1 else None

    def nodes(self, *, version: str | None = None) -> tuple[GraphNode, ...]:
        return tuple(sorted(
            (node for (node_version, _), node in self._nodes.items() if version is None or node_version == version),
            key=lambda item: item.node_id,
        ))

    def edges(self, *, version: str | None = None) -> tuple[GraphEdge, ...]:
        return tuple(sorted(
            (edge for (edge_version, _), edge in self._edges.items() if version is None or edge_version == version),
            key=lambda item: (item.source_id, item.target_id, item.relation, item.edge_id),
        ))

    def _edge_candidates(self, edges: Iterable[GraphEdge], *, query: str, version: str | None) -> StaticGraphQueryResult:
        candidates = tuple(_edge_candidate(edge, self._nodes.get((edge.version, edge.source_id)), self._nodes.get((edge.version, edge.target_id))) for edge in edges)
        return StaticGraphQueryResult(candidates=candidates, query=query, version=version)

    def get_neighbors(self, node_id: str, *, relation: RelationType | None = None, version: str | None = None) -> StaticGraphQueryResult:
        node = self.get_node(node_id, version=version)
        if node is None:
            return StaticGraphQueryResult(query=f"neighbors:{node_id}", version=version, degraded=True, reason="unknown_node")
        edges = [
            edge for edge in self.edges(version=version)
            if edge.source_id == node_id or edge.target_id == node_id
            if relation is None or edge.relation == relation
        ]
        return self._edge_candidates(edges, query=f"neighbors:{node_id}", version=version)

    def find_dependency_path(self, source_id: str, target_id: str, *, version: str | None = None, max_hops: int = 12) -> StaticGraphQueryResult:
        if self.get_node(source_id, version=version) is None or self.get_node(target_id, version=version) is None:
            return StaticGraphQueryResult(query=f"path:{source_id}->{target_id}", version=version, degraded=True, reason="unknown_node")
        outgoing: dict[str, list[GraphEdge]] = {}
        for edge in self.edges(version=version):
            if edge.relation == "DEPENDS_ON":
                outgoing.setdefault(edge.source_id, []).append(edge)
        queue: deque[tuple[str, tuple[GraphEdge, ...]]] = deque([(source_id, ())])
        visited = {source_id}
        while queue:
            current, path = queue.popleft()
            if current == target_id:
                return self._edge_candidates(path, query=f"path:{source_id}->{target_id}", version=version)
            if len(path) >= max(1, int(max_hops)):
                continue
            for edge in sorted(outgoing.get(current, ()), key=lambda item: (item.target_id, item.edge_id)):
                if edge.target_id in visited:
                    continue
                visited.add(edge.target_id)
                queue.append((edge.target_id, (*path, edge)))
        return StaticGraphQueryResult(query=f"path:{source_id}->{target_id}", version=version, degraded=True, reason="path_not_found")

    def get_failure_modes(self, owner_id: str, *, version: str | None = None) -> StaticGraphQueryResult:
        edges = [edge for edge in self.edges(version=version) if edge.source_id == owner_id and edge.relation == "KNOWN_FAILURE_MODE"]
        return self._edge_candidates(edges, query=f"failure_modes:{owner_id}", version=version)

    def get_related_code(self, node_id: str, *, version: str | None = None) -> StaticGraphQueryResult:
        relations = {"IMPLEMENTED_BY", "CONTAINS_SYMBOL", "CALLS"}
        edges = [edge for edge in self.edges(version=version) if edge.relation in relations and (edge.source_id == node_id or edge.target_id == node_id)]
        return self._edge_candidates(edges, query=f"related_code:{node_id}", version=version)


def _edge_candidate(edge: GraphEdge, source: GraphNode | None, target: GraphNode | None) -> KnowledgeCandidate:
    source_name = source.name if source else edge.source_id
    target_name = target.name if target else edge.target_id
    content = f"{source_name} --{edge.relation}--> {target_name}"
    service = source.name if source and source.entity_type == "Service" else None
    provenance = edge.provenance
    return KnowledgeCandidate(
        candidate_id=f"kg-{edge.edge_id}",
        source_type="static_kg",
        content=content,
        service=service,
        software_version=edge.version,
        repo_commit=edge.repo_commit,
        timestamp=edge.timestamp,
        retrieval_score=edge.confidence,
        confidence=edge.confidence,
        provenance=provenance,
    )


def _provenance(source: str, source_id: str, *, path: str | None = None, version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeProvenance:
    return KnowledgeProvenance(
        source=source, source_id=source_id, path=path, version=version,
        repo_commit=repo_commit, timestamp=timestamp, metadata=metadata or {},
    )


class StaticKnowledgeGraphBuilder:
    """Build graph facts from explicit structural assets only."""

    def __init__(self, store: InMemoryStaticGraph | None = None) -> None:
        self.store = store or InMemoryStaticGraph()

    def _node(self, *, node_id: str, entity_type: EntityType, name: str, source: str, source_id: str, version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None, properties: dict[str, Any] | None = None, confidence: float = 1.0) -> GraphNode:
        return self.store.add_node(GraphNode(
            node_id=node_id, entity_type=entity_type, name=name,
            properties=properties or {}, version=version, repo_commit=repo_commit,
            timestamp=timestamp, confidence=confidence,
            provenance=_provenance(source, source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, metadata=properties),
        ))

    def _edge(self, *, source_id: str, target_id: str, relation: RelationType, source: str, source_id_value: str, version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None, confidence: float = 1.0) -> GraphEdge:
        edge = GraphEdge(
            edge_id=f"edge-{_stable_id(version, source_id, relation, target_id)}",
            source_id=source_id, target_id=target_id, relation=relation,
            version=version, repo_commit=repo_commit, timestamp=timestamp,
            confidence=confidence,
            provenance=_provenance(source, source_id_value, version=version, repo_commit=repo_commit, timestamp=timestamp),
        )
        return self.store.add_edge(edge)

    def from_topology(self, topology: str | Path | dict[str, Any], *, version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None) -> InMemoryStaticGraph:
        if isinstance(topology, (str, Path)):
            path = Path(topology)
            data = json.loads(path.read_text(encoding="utf-8"))
            source_id = str(path)
        else:
            data = topology
            source_id = "service_topology"
        services = data.get("services", data) if isinstance(data, dict) else {}
        if not isinstance(services, dict):
            return self.store
        for name, raw in sorted(services.items()):
            props = dict(raw) if isinstance(raw, dict) else {}
            self._node(node_id=f"service:{name}", entity_type="Service", name=str(name), source="service_topology", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties=props)
        for name, raw in sorted(services.items()):
            props = dict(raw) if isinstance(raw, dict) else {}
            downstream = props.get("downstream", ()) if isinstance(props, dict) else ()
            for target in sorted(str(item) for item in downstream if str(item)):
                if self.store.get_node(f"service:{target}", version=version) is None:
                    self._node(node_id=f"service:{target}", entity_type="Service", name=target, source="service_topology", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={"implicit": True})
                self._edge(source_id=f"service:{name}", target_id=f"service:{target}", relation="DEPENDS_ON", source="service_topology", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        return self.store

    @staticmethod
    def _yaml_scalar(value: str) -> str:
        value = value.strip().strip('"\'')
        return value

    @classmethod
    def _yaml_metadata_name(cls, lines: list[str]) -> str:
        metadata_index = next((i for i, line in enumerate(lines) if re.match(r"^\s*metadata:\s*$", line)), None)
        if metadata_index is None:
            return ""
        base_indent = len(lines[metadata_index]) - len(lines[metadata_index].lstrip())
        for line in lines[metadata_index + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                break
            match = re.match(r"^\s*name:\s*(.+?)\s*$", line)
            if match:
                return cls._yaml_scalar(match.group(1))
        return ""

    @classmethod
    def _yaml_values(cls, lines: list[str], key: str) -> list[str]:
        return [cls._yaml_scalar(match.group(1)) for line in lines if (match := re.match(rf"^\s*-?\s*{re.escape(key)}:\s*(.+?)\s*$", line))]

    def from_kubernetes_yaml(self, content: str, *, source_id: str = "kubernetes.yaml", version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None) -> InMemoryStaticGraph:
        for document in re.split(r"(?m)^---\s*$", content):
            lines = document.splitlines()
            kind_match = next((re.match(r"^\s*kind:\s*(.+?)\s*$", line) for line in lines if re.match(r"^\s*kind:\s*", line)), None)
            if not kind_match:
                continue
            kind = self._yaml_scalar(kind_match.group(1))
            name = self._yaml_metadata_name(lines)
            if not name:
                continue
            namespace_match = next((re.match(r"^\s*namespace:\s*(.+?)\s*$", line) for line in lines if re.match(r"^\s*namespace:\s*", line)), None)
            namespace = self._yaml_scalar(namespace_match.group(1)) if namespace_match else ""
            base_props = {"namespace": namespace, "kind": kind}
            if kind == "Service":
                self._node(node_id=f"service:{name}", entity_type="Service", name=name, source="kubernetes_yaml", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={**base_props, "ports": self._yaml_values(lines, "port")})
                for port in self._yaml_values(lines, "port"):
                    api_id = f"api:{name}:{port}"
                    self._node(node_id=api_id, entity_type="API", name=f"{name}:{port}", source="kubernetes_yaml", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={"port": port, "namespace": namespace})
                    self._edge(source_id=f"service:{name}", target_id=api_id, relation="OWNS_API", source="kubernetes_yaml", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
            elif kind == "Deployment":
                images = self._yaml_values(lines, "image")
                deployment_id = f"deployment:{name}"
                self._node(node_id=deployment_id, entity_type="Deployment", name=name, source="kubernetes_yaml", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={**base_props, "images": images})
                if self.store.get_node(f"service:{name}", version=version) is not None:
                    self._edge(source_id=f"service:{name}", target_id=deployment_id, relation="DEPLOYED_AS", source="kubernetes_yaml", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        return self.store

    def from_repository_index(self, index: Any, *, repository: str = "repository", version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None, source_id: str | None = None) -> InMemoryStaticGraph:
        source_id = source_id or repository
        repository_id = f"repository:{repository}"
        self._node(node_id=repository_id, entity_type="Repository", name=repository, source="repository_ast", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        paths: set[str] = set()
        try:
            for sf in index.fs.iter_files():
                paths.add(str(sf.rel))
        except Exception:
            pass
        try:
            symbols = index.symbols("", limit=100)
        except Exception:
            symbols = []
        paths.update(str(row.get("path") or "") for row in symbols if row.get("path"))
        for path in sorted(paths):
            file_id = f"file:{path}"
            self._node(node_id=file_id, entity_type="File", name=path, source="repository_ast", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={"path": path})
            self._edge(source_id=repository_id, target_id=file_id, relation="IMPLEMENTED_BY", source="repository_ast", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
            module_name = path.split("/", 1)[0] if "/" in path else ""
            if module_name:
                module_id = f"module:{module_name}"
                self._node(node_id=module_id, entity_type="Module", name=module_name, source="repository_ast", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={"path_prefix": module_name})
                self._edge(source_id=module_id, target_id=file_id, relation="IMPLEMENTED_BY", source="repository_ast", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        for row in symbols:
            path = str(row.get("path") or "")
            qualified = str(row.get("qualified_name") or row.get("name") or "")
            if not path or not qualified:
                continue
            symbol_id = f"symbol:{path}:{qualified}"
            self._node(node_id=symbol_id, entity_type="Symbol", name=qualified, source="repository_ast", source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties={"path": path, "kind": row.get("kind"), "start_line": row.get("start_line"), "end_line": row.get("end_line")})
            self._edge(source_id=f"file:{path}", target_id=symbol_id, relation="CONTAINS_SYMBOL", source="repository_ast", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        # The existing AST index already records bounded call-site resolution.
        # Use only resolved definitions, so an unresolved name does not become
        # a fabricated graph node or a false CALLS fact.
        try:
            call_rows = index._fetchall(
                "SELECT path,caller_qualified_name,target_name,resolution_kind FROM calls"
            )
        except Exception:
            call_rows = []
        for path, caller, target, resolution_kind in call_rows:
            if resolution_kind not in {"exact", "import_resolved"}:
                continue
            caller_id = f"symbol:{path}:{caller}"
            if self.store.get_node(caller_id, version=version) is None:
                continue
            try:
                definitions = index.resolve_symbol(str(target))
            except Exception:
                definitions = []
            if len(definitions) != 1:
                continue
            definition = definitions[0]
            target_id = f"symbol:{definition['path']}:{definition['qualified_name']}"
            if self.store.get_node(target_id, version=version) is None:
                continue
            self._edge(source_id=caller_id, target_id=target_id, relation="CALLS", source="repository_ast", source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        return self.store

    def add_failure_mode(self, *, owner_id: str, failure_mode_id: str, name: str, source_id: str, source: str = "verified_knowledge", version: str | None = None, repo_commit: str | None = None, timestamp: str | None = None, properties: dict[str, Any] | None = None) -> InMemoryStaticGraph:
        if source not in {"verified_knowledge", "domain_document", "incident_memory_verified"}:
            raise ValueError("failure modes require an explicit non-evaluator knowledge source")
        owner = self.store.get_node(owner_id, version=version)
        if owner is None:
            raise ValueError(f"unknown failure mode owner: {owner_id}")
        self._node(node_id=f"failure:{failure_mode_id}", entity_type="FailureMode", name=name, source=source, source_id=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp, properties=properties)
        self._edge(source_id=owner_id, target_id=f"failure:{failure_mode_id}", relation="KNOWN_FAILURE_MODE", source=source, source_id_value=source_id, version=version, repo_commit=repo_commit, timestamp=timestamp)
        return self.store


class StaticKnowledgeGraphTools:
    """High-level graph port; no query-language surface is exposed."""

    def __init__(self, graph: InMemoryStaticGraph) -> None:
        self.graph = graph

    def get_neighbors(self, node_id: str, *, relation: RelationType | None = None, version: str | None = None) -> StaticGraphQueryResult:
        return self.graph.get_neighbors(node_id, relation=relation, version=version)

    def find_dependency_path(self, source_id: str, target_id: str, *, version: str | None = None, max_hops: int = 12) -> StaticGraphQueryResult:
        return self.graph.find_dependency_path(source_id, target_id, version=version, max_hops=max_hops)

    def get_failure_modes(self, owner_id: str, *, version: str | None = None) -> StaticGraphQueryResult:
        return self.graph.get_failure_modes(owner_id, version=version)

    def get_related_code(self, node_id: str, *, version: str | None = None) -> StaticGraphQueryResult:
        return self.graph.get_related_code(node_id, version=version)
