"""Gold-independent canonical CloudOps investigation Skill registry."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .loader import LocalSkillSource, LoadedSkill


class DomainAssetCatalog(BaseModel):
    """Validated projection of the pinned manifest used by every experiment arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: str
    manifest_sha256: str
    manifest_schema_version: str
    tool_ids: tuple[str, ...]
    skill_ids: tuple[str, ...]
    node_count: int = Field(ge=0)
    public_prior_node_count: int = Field(ge=0)
    excluded_node_count: int = Field(ge=0)
    excluded_control_surface_count: int = Field(ge=0)

    @classmethod
    def from_manifest(cls, path: str | Path) -> "DomainAssetCatalog":
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "domain-asset-manifest-v2":
            raise ValueError("DomainAssetManifest v2 is required")
        tools = data.get("official_tools", [])
        skills = data.get("official_skill_assets", [])
        if len(tools) != 12 or len(skills) != 12:
            raise ValueError("DomainAssetManifest must cover all 12 official Tools and Skills")
        if len({item.get("id") for item in tools}) != 12:
            raise ValueError("official Tool IDs must be unique")
        if len({item.get("id") for item in skills}) != 12:
            raise ValueError("official Skill IDs must be unique")
        nodes = [node for skill in skills for node in skill.get("nodes", ())]
        for item in [*tools, *skills, *nodes]:
            for field_name in ("source_sha256", "derived_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(item.get(field_name, ""))):
                    raise ValueError(
                        f"domain asset lacks {field_name}: {item.get('id')}"
                    )
        if any(
            item.get("disposition") not in {"DIRECT_REUSE", "THIN_ADAPTER"}
            for item in tools
        ):
            raise ValueError("official Tool has an unaudited disposition")
        if any(
            item.get("disposition") != "DIRECT_REUSE_PUBLIC_PRIOR"
            for item in skills
        ):
            raise ValueError("official Skill must be an audited public prior")
        valid_node_dispositions = {
            "DIRECT_REUSE_PUBLIC_PRIOR", "EXCLUDED_BENCHMARK_CONTROL",
        }
        if any(item.get("disposition") not in valid_node_dispositions for item in nodes):
            raise ValueError("official Skill node has an unaudited disposition")
        controls = data.get("excluded_control_surfaces", [])
        if not controls or any(
            item.get("disposition") != "EXCLUDED_BENCHMARK_CONTROL"
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("source_sha256", "")))
            for item in controls
        ):
            raise ValueError("benchmark control exclusions must be explicit and hashed")
        summary = data.get("audit_summary", {})
        public_prior = sum(
            item.get("disposition") == "DIRECT_REUSE_PUBLIC_PRIOR" for item in nodes
        )
        excluded = sum(
            item.get("disposition") == "EXCLUDED_BENCHMARK_CONTROL" for item in nodes
        )
        if (
            summary.get("node_count") != len(nodes)
            or summary.get("direct_public_prior_node_count") != public_prior
            or summary.get("excluded_skill_node_count") != excluded
        ):
            raise ValueError("DomainAssetManifest audit summary does not match node records")
        local = data.get("local_tool_catalog", {})
        local_source = path.parents[2] / str(local.get("source", ""))
        if not local_source.is_file():
            raise ValueError(f"local Tool catalog source is missing: {local_source}")
        if hashlib.sha256(local_source.read_bytes()).hexdigest() != local.get("source_sha256"):
            raise ValueError("local Tool catalog source hash drifted from the manifest")
        return cls(
            manifest_path=str(path),
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest_schema_version=data["schema_version"],
            tool_ids=tuple(item["id"] for item in tools),
            skill_ids=tuple(item["id"] for item in skills),
            node_count=len(nodes),
            public_prior_node_count=public_prior,
            excluded_node_count=excluded,
            excluded_control_surface_count=len(controls),
        )


class DomainSkill(BaseModel):
    """Audited public-prior Skill projected into the local control boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    objective: str
    signals: tuple[str, ...] = ()
    entry_nodes: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    expected_evidence: tuple[str, ...] = ()
    completion_rule: dict[str, Any] = Field(default_factory=dict)
    nodes: tuple[str, ...] = ()
    graph_nodes: tuple[dict[str, Any], ...] = ()
    source_sha256: str
    derived_sha256: str


class DomainSkillRegistry:
    """Single adapter for audited Cloud-OpsBench domain Skills.

    The registry owns domain guidance and tool affordances only. Runtime keeps
    permission, Evidence admission, and semantic state transitions.
    """

    def __init__(
        self,
        skills: tuple[DomainSkill, ...],
        *,
        manifest_path: str,
        manifest_sha256: str,
    ):
        self.skills = skills
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self._by_id = {item.skill_id: item for item in skills}

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
    ) -> "DomainSkillRegistry":
        path = Path(path).resolve()
        catalog = DomainAssetCatalog.from_manifest(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        source = LocalSkillSource()
        loaded_skills = source.discover()
        records: list[DomainSkill] = []
        for record in data["official_skill_assets"]:
            projection = record["registry_projection"]
            loaded: LoadedSkill | None = loaded_skills.get(str(record["id"]))
            if loaded is None:
                raise FileNotFoundError(
                    f"project Skill file is not available: {record['id']}"
                )
            metadata = dict(loaded.metadata)
            if metadata.get("source_sha256") != record["source_sha256"]:
                raise ValueError(
                    f"project Skill provenance drifted: {record['id']}"
                )
            if metadata.get("derived_sha256") != record["derived_sha256"]:
                raise ValueError(
                    f"project Skill projection hash drifted: {record['id']}"
                )
            if metadata.get("registry_projection") != projection:
                raise ValueError(
                    f"project Skill projection differs from manifest: {record['id']}"
                )
            nodes = metadata.get("nodes", {})
            if not isinstance(nodes, dict):
                raise ValueError(f"project Skill nodes must be a mapping: {record['id']}")
            node_records = {str(item["id"]): item for item in record.get("nodes", ())}
            if tuple(nodes) != tuple(node_records):
                raise ValueError(f"project Skill node order drifted: {record['id']}")
            for node_id, node_record in node_records.items():
                if _canonical_sha256(nodes[node_id]) != node_record["source_sha256"]:
                    raise ValueError(f"project Skill node provenance drifted: {record['id']}::{node_id}")
            node_ids = tuple(str(item) for item in nodes)
            signals = tuple(str(item) for item in (metadata.get("signals") or ()))
            objective = str(metadata.get("description") or projection["objective"])
            graph_nodes: list[dict[str, Any]] = []
            for node_id, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                goal = str(node.get("goal") or "").strip()
                actions = []
                for action in node.get("actions") or ():
                    if not isinstance(action, dict):
                        continue
                    actions.append({
                        "tool": str(action.get("tool") or ""),
                        "args": dict(action.get("args") or {}),
                    })
                evidence = tuple(str(item) for item in (node.get("evidence") or ()))
                graph_nodes.append({
                    "id": str(node_id),
                    "goal": goal,
                    "actions": actions,
                    "evidence": list(evidence),
                    "next": [str(item) for item in (node.get("next") or ())],
                })
            expected_allowed_tools = tuple(
                "read_file" if item == "read_file_when_source_workspace_is_bound" else str(item)
                for item in projection["allowed_tools"]
            )
            allowed_tools = tuple(str(item) for item in metadata.get(
                "runtime_allowed_tools", expected_allowed_tools
            ))
            if allowed_tools != expected_allowed_tools:
                raise ValueError(f"project Skill Tool adapter drifted: {record['id']}")
            records.append(DomainSkill(
                skill_id=str(record["id"]),
                objective=objective,
                signals=signals,
                entry_nodes=tuple(str(item) for item in (metadata.get("entry") or record.get("entry") or ())),
                allowed_tools=allowed_tools,
                expected_evidence=tuple(
                    str(item) for item in projection["expected_evidence"]
                ),
                completion_rule=dict(projection["completion_rule"]),
                nodes=node_ids or tuple(
                    str(item["id"]) for item in record.get("nodes", ())
                ),
                graph_nodes=tuple(graph_nodes),
                source_sha256=str(record["source_sha256"]),
                derived_sha256=str(record["derived_sha256"]),
            ))
        if tuple(item.skill_id for item in records) != catalog.skill_ids:
            raise ValueError("DomainSkillRegistry skill order drifted from manifest")
        return cls(
            tuple(records),
            manifest_path=catalog.manifest_path,
            manifest_sha256=catalog.manifest_sha256,
        )

    def load_skill(self, skill_id: str) -> DomainSkill:
        try:
            return self._by_id[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown audited domain Skill: {skill_id}") from exc

    def tool_names(self, skill_id: str) -> tuple[str, ...]:
        return self.load_skill(skill_id).allowed_tools

    def tool_names_for_context(
        self,
        skill_id: str,
        *,
        available_tools: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return the selected Skill's capability-available executable Tools."""
        skill = self.load_skill(skill_id)
        available = set(available_tools)
        return tuple(
            tool_name for tool_name in skill.allowed_tools
            if tool_name in available
        )

    def feasible_skill_ids(
        self,
        *,
        available_tools: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Filter canonical Skills by executable capability only.

        This method does not inspect incident text, hypotheses, Evidence, fault
        labels, or preferred Skills. Semantic selection belongs to the model.
        """
        available = set(available_tools)
        return tuple(
            skill.skill_id for skill in self.skills
            if available.intersection(skill.allowed_tools)
        )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
