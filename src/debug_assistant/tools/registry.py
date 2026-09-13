from __future__ import annotations
import json
from typing import Any
from pydantic import ValidationError
from debug_assistant.contracts import compact_validation_error
from .repository import (
    RepoTreeTool, GrepTool, ReadFileTool, SymbolSearchTool, GitLogTool,
    GitShowTool, DiscoverTestsTool,
)
from .indexed import CodeSearchTool,IndexedSymbolSearchTool,InspectSymbolContextTool
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.paths import RepositoryPathResolver, RepositoryPathMatcher

# The bounded parallel policy is shared by the runtime guard and Planner prompt.
# Keep this next to the registered tool definitions so the two surfaces cannot drift.
PARALLEL_ALLOWED_TOOLS = frozenset({
    'read_file', 'grep', 'symbol_search', 'code_search', 'inspect_symbol_context',
})

class ToolRegistry:
    def __init__(self, repo_root, index=None, fs=None, *, code_search_default_mode='lexical'):
        self.fs=fs or SafeRepositoryFS(repo_root)
        self.path_resolver=RepositoryPathResolver(self.fs)
        self.path_matcher=RepositoryPathMatcher()
        kw={'fs':self.fs,'resolver':self.path_resolver,'matcher':self.path_matcher}
        symbol=IndexedSymbolSearchTool(index) if index is not None else SymbolSearchTool(repo_root,**kw)
        tools=[RepoTreeTool(repo_root,**kw),GrepTool(repo_root,**kw),ReadFileTool(repo_root,**kw),symbol,GitLogTool(repo_root,**kw),GitShowTool(repo_root,**kw),DiscoverTestsTool(repo_root,**kw)]
        if index is not None:
            tools.insert(2,CodeSearchTool(index, default_mode=code_search_default_mode))
            tools.insert(3,InspectSymbolContextTool(index))
        self._tools={t.spec.name:t for t in tools}

    def get(self,name): return self._tools.get(name)
    def specs(self): return [t.spec for t in self._tools.values()]

    def function_schemas(self):
        """Provider function-calling schemas; Pydantic remains the validation SSOT."""
        return [spec.function_schema() for spec in self.specs()]

    def is_parallel_safe(self, name: str) -> bool:
        spec = next((x for x in self.specs() if x.name == name), None)
        return bool(spec and (spec.parallel_safe or name in PARALLEL_ALLOWED_TOOLS))

    def validate_arguments(self,name:str,args:dict[str,Any]):
        tool=self.get(name)
        if not tool:
            return None,{"error_type":"unknown_tool","message":f"unknown tool: {name}","retryable":False}
        try:
            model=tool.spec.args_model.model_validate(args)
            return model.model_dump(exclude_none=True),None
        except ValidationError as exc:
            return None,{
                "error_type":"schema_validation",
                "tool":name,
                "message":"tool arguments failed schema validation",
                "details":compact_validation_error(exc),
                "allowed_fields":list(tool.spec.args_model.model_fields.keys()),
                "retryable":True,
            }


    def repair_arguments(self,name:str,args:dict[str,Any],error:dict[str,Any] | None=None):
        """Apply only deterministic, semantics-preserving mechanical repairs.

        V1.3.2.2 intentionally keeps this tiny. ``read_file`` already exposes a
        bounded ``line_count`` field, so there is no inclusive-range arithmetic to
        repair here. Ambiguous path/name/type errors are never guessed or repaired.
        """
        return None,None

    def render(self, compact: bool=False):
        blocks=[]
        for s in self.specs():
            schema=s.json_schema()
            props=schema.get('properties',{})
            required=set(schema.get('required',[]))
            fields=[]
            for name,meta in props.items():
                typ=meta.get('type') or (' | '.join(x.get('type','') for x in meta.get('anyOf',[])) if meta.get('anyOf') else 'value')
                default='' if name in required else f" optional default={meta.get('default')!r}"
                constraints=[]
                for k in ('minimum','maximum','minLength','maxLength'):
                    if k in meta: constraints.append(f"{k}={meta[k]}")
                fields.append(f"{name}:{typ}{default}"+(f" ({', '.join(constraints)})" if constraints else ''))
            if compact:
                blocks.append(f"- {s.name}({'; '.join(fields)}) — {s.description[:120]}")
            else:
                blocks.append(
                    f"- {s.name}: {s.description}\n"
                    f"  args: {{{'; '.join(fields)}}}\n"
                    f"  capability={s.capability}; cost={s.cost_class}; side_effect={s.side_effect}; output_limit={s.output_limit}"
                )
        return '\n'.join(blocks)
