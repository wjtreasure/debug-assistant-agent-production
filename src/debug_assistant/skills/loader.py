from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

@dataclass(slots=True,frozen=True)
class LoadedSkill:
    name: str
    description: str
    content: str
    completion: str = ''
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def skill_id(self) -> str:
        return str(self.metadata.get('skill_id') or self.name)

class LocalSkillSource:
    def __init__(self,root:Path|None=None): self.root=root or Path(__file__).resolve().parent

    @staticmethod
    def _metadata(text: str) -> dict[str, Any]:
        """Read optional YAML front matter without changing Skill content."""
        if not text.startswith('---\n'):
            return {}
        match = re.match(r'^---\n(.*?)\n---(?:\n|$)', text, re.S)
        if match is None:
            raise ValueError('Skill front matter must be closed by a second --- line')
        value = yaml.load(match.group(1), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader)) or {}
        if not isinstance(value, dict):
            raise ValueError('Skill front matter must be a mapping')
        return value

    def discover(self)->dict[str,LoadedSkill]:
        out={}
        for p in sorted(self.root.glob('*/SKILL.md')):
            text=p.read_text(encoding='utf-8')
            name=p.parent.name
            metadata=self._metadata(text)
            desc=str(metadata.get('description') or '')
            if not desc:
                m=re.search(r'^description:\s*(.+)$',text,re.M)
                desc=m.group(1).strip() if m else ''
            completion_value=metadata.get('completion_rule', '')
            completion=str(completion_value) if completion_value else ''
            if not completion:
                cm=re.search(r'## Completion Criteria\s*(.*?)(?:\n## |\Z)',text,re.S)
                completion=cm.group(1).strip() if cm else ''
            out[name]=LoadedSkill(name,desc,text,completion,metadata)
        return out
    def load(self,name:str)->LoadedSkill|None:return self.discover().get(name)
