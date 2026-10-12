"""Read-only model views of the canonical SKILL.md; no alternate Skill assets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .loader import LoadedSkill, LocalSkillSource


@dataclass(frozen=True, slots=True)
class SkillViews:
    skill_id: str
    skill_hash: str
    version: str
    card_json: str
    semantic_json: str
    action_json: str


def render_skill_views(skill: LoadedSkill, aliases: Mapping[str, str] | None = None) -> SkillViews:
    metadata = skill.metadata
    nodes = metadata["nodes"]
    identity = {
        "skill_id": skill.skill_id,
        "skill_hash": hashlib.sha256(skill.content.encode("utf-8")).hexdigest(),
        "skill_version": "canonical-skill-projection-v1",
    }
    common = {**identity, "description": skill.description,
              "signals": metadata.get("signals", []), "entry": metadata.get("entry", [])}
    action_names = tuple(dict.fromkeys(
        action["tool"] for node in nodes.values() for action in node.get("actions", [])
    ))
    card = {**common, "tool_affordances": action_names}
    semantic = {**common, "completion_rule": metadata["registry_projection"]["completion_rule"],
                "nodes": [{"id": node_id, "goal": node["goal"],
                           "expected_information": node.get("evidence", []),
                           "action_names": [action["tool"] for action in node.get("actions", [])],
                           "next": node.get("next", [])}
                          for node_id, node in nodes.items()]}
    alias_map = dict(aliases or {})
    action = {**identity, "entry": common["entry"],
              "nodes": [{"id": node_id, "goal": node["goal"],
                         "actions": [{"tool": item["tool"],
                                      "runtime_tool": alias_map.get(item["tool"], ""),
                                      "args_template": item.get("args", {})}
                                     for item in node.get("actions", [])],
                         "next": node.get("next", [])}
                        for node_id, node in nodes.items()]}
    encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return SkillViews(skill.skill_id, identity["skill_hash"], identity["skill_version"],
                      encode(card), encode(semantic), encode(action))


def load_skill_views(skill_ids: tuple[str, ...], aliases: Mapping[str, str] | None = None) -> tuple[SkillViews, ...]:
    source = LocalSkillSource().discover()
    return tuple(render_skill_views(source[skill_id], aliases) for skill_id in skill_ids)


def action_choices(views: tuple[SkillViews, ...], visible: set[str], *, companions=None) -> tuple[dict, ...]:
    """Exact turn-local choices derived only from frozen canonical projections.

    One ref represents one legal node/action pair, rather than two independently
    writable fields. Single-action nodes need no redundant action suffix.
    """
    rows = []
    for view in views:
        for node_index, node in enumerate(json.loads(view.action_json)["nodes"], 1):
            actions = [(i, item) for i, item in enumerate(node["actions"], 1)
                       if item["runtime_tool"] in visible]
            for action_index, item in actions:
                node_ref = f"{view.skill_hash[:8]}.N{node_index}"
                action_ref = f"A{action_index}" if len(node["actions"]) > 1 else None
                rows.append({"choice_ref": node_ref + (f".{action_ref}" if action_ref else ""),
                             "skill_id": view.skill_id, "node_id": node["id"],
                             "node_ref": node_ref, "action_ref": action_ref,
                             "operation": item["runtime_tool"], "description": item["tool"],
                             "args_template": item.get("args_template", {})})
                primary = rows[-1]
                for operation in (companions or {}).get(item['runtime_tool'], ()):
                    if operation in visible:
                        rows.append({**primary, 'choice_ref': node_ref + '.P.' + operation,
                                     'operation': operation, 'description': operation, 'args_template': {}})
    return tuple(rows)
