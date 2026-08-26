from __future__ import annotations
import time
from .base import Tool,ToolSpec
from debug_assistant.models import ToolObservation

class CodeSearchTool(Tool):
    spec=ToolSpec('code_search','Hybrid full-text repository search over the task-scoped index.',('query',))
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=40,**kwargs):
        t=time.time()
        try:
            rows=self.index.search(str(query),int(max_results)); text='\n'.join(f"{r['path']}: {r['snippet']}" for r in rows)
            return ToolObservation(self.spec.name,True,text,{'matches':len(rows)},None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{},type(e).__name__,(time.time()-t)*1000)

class IndexedSymbolSearchTool(Tool):
    spec=ToolSpec('symbol_search','Search the task-scoped AST symbol index.',('query',))
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=60,**kwargs):
        t=time.time()
        try:
            rows=self.index.symbols(str(query),int(max_results)); text='\n'.join(f"{r['path']}:{r['start_line']}-{r['end_line']} {r['kind']} {r['name']}" for r in rows)
            return ToolObservation(self.spec.name,True,text,{'matches':len(rows)},None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{},type(e).__name__,(time.time()-t)*1000)
