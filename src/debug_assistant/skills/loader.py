from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(slots=True,frozen=True)
class LoadedSkill:
    name:str; description:str; content:str; completion:str=''

class LocalSkillSource:
    def __init__(self,root:Path|None=None): self.root=root or Path(__file__).resolve().parent
    def discover(self)->dict[str,LoadedSkill]:
        out={}
        for p in sorted(self.root.glob('*/SKILL.md')):
            text=p.read_text(encoding='utf-8'); name=p.parent.name
            desc=''; completion=''
            m=re.search(r'^description:\s*(.+)$',text,re.M); desc=m.group(1).strip() if m else ''
            cm=re.search(r'## Completion Criteria\s*(.*?)(?:\n## |\Z)',text,re.S); completion=(cm.group(1).strip() if cm else '')
            out[name]=LoadedSkill(name,desc,text,completion)
        return out
    def load(self,name:str)->LoadedSkill|None:return self.discover().get(name)

class SkillLibrary:
    def __init__(self,source:LocalSkillSource|None=None):self.source=source or LocalSkillSource(); self.skills=self.source.discover()
    def render_catalog(self)->str:return '\n'.join(f'- {s.name}: {s.description}' for s in self.skills.values())
    def render_active(self,name:str|None)->str:
        if not name:return ''
        s=self.skills.get(name); return s.content if s else ''
