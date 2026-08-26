from __future__ import annotations
import time
from pydantic import Field
from .base import Tool,ToolSpec,ToolArgs
from debug_assistant.models import ToolObservation

class CodeSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=40, ge=1, le=40)

class IndexedSymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=60, ge=1, le=60)

class CodeSearchTool(Tool):
    spec=ToolSpec('code_search','Hybrid full-text repository search over the task-scoped index.',CodeSearchArgs,'repository_search','light','none',14000)
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=40):
        t=time.time()
        try:
            rows=self.index.search(str(query),min(int(max_results),40)); text='\n'.join(f"{r['path']}: {r['snippet']}" for r in rows)
            return ToolObservation(self.spec.name,True,text,{'matches':len(rows),'truncated':len(rows)>=min(int(max_results),40),'retryable':False},None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{'retryable':False},type(e).__name__,(time.time()-t)*1000)

class IndexedSymbolSearchTool(Tool):
    spec=ToolSpec('symbol_search','Search the task-scoped AST symbol index.',IndexedSymbolSearchArgs,'repository_search','medium','none',12000)
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=60):
        t=time.time()
        try:
            rows=self.index.symbols(str(query),min(int(max_results),60)); text='\n'.join(f"{r['path']}:{r['start_line']}-{r['end_line']} {r['kind']} {r['name']}" for r in rows)
            return ToolObservation(self.spec.name,True,text,{'matches':len(rows),'truncated':len(rows)>=min(int(max_results),60),'retryable':False},None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{'retryable':False},type(e).__name__,(time.time()-t)*1000)
