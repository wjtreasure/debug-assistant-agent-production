from __future__ import annotations
from .repository import RepoTreeTool,GrepTool,ReadFileTool,SymbolSearchTool,GitLogTool,GitShowTool,DiscoverTestsTool
from .indexed import CodeSearchTool,IndexedSymbolSearchTool

class ToolRegistry:
    def __init__(self, repo_root, index=None):
        symbol=IndexedSymbolSearchTool(index) if index is not None else SymbolSearchTool(repo_root)
        tools=[RepoTreeTool(repo_root),GrepTool(repo_root),ReadFileTool(repo_root),symbol,GitLogTool(repo_root),GitShowTool(repo_root),DiscoverTestsTool(repo_root)]
        if index is not None: tools.insert(2,CodeSearchTool(index))
        self._tools={t.spec.name:t for t in tools}
    def get(self,name): return self._tools.get(name)
    def specs(self): return [t.spec for t in self._tools.values()]
    def render(self): return '\n'.join(f"- {s.name}: {s.description}; required={list(s.required_args)}; read_only={s.read_only}" for s in self.specs())
