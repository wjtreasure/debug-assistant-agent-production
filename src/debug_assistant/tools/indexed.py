from __future__ import annotations
import time
from dataclasses import asdict
from pydantic import Field
from .base import Tool,ToolSpec,ToolArgs
from debug_assistant.models import ToolObservation

class CodeSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    mode: str = Field(default='lexical', pattern='^(lexical|semantic|hybrid)$')
    max_results: int = Field(default=40, ge=1, le=40)

class IndexedSymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=60, ge=1, le=60)


class InspectSymbolContextArgs(ToolArgs):
    symbol: str = Field(min_length=1)
    file: str | None = None
    include_source: bool = False
    include_uncertain: bool = False
    max_callers: int = Field(default=5, ge=0, le=5)
    max_callees: int = Field(default=5, ge=0, le=5)
    max_source_chars: int = Field(default=12000, ge=1000, le=12000)

class CodeSearchTool(Tool):
    spec=ToolSpec(
        'code_search',
        "Search repository code. Use mode='lexical' for exact identifiers/error names, mode='semantic' for behavioral/conceptual descriptions without code identifiers, and mode='hybrid' when vocabulary is uncertain or lexical results were insufficient. Semantic candidates must be verified with read_file before becoming evidence.",
        CodeSearchArgs,'repository_search','medium','none',14000)
    def __init__(self,index): self.index=index
    def execute(self,query,mode='lexical',max_results=40):
        t=time.time()
        try:
            if hasattr(self.index,'search'):
                try:
                    result=self.index.search(str(query),mode=str(mode),limit=min(int(max_results),40))
                except TypeError:
                    result=self.index.search(str(query),min(int(max_results),40))
                if isinstance(result,tuple): rows,diag=result; metadata=asdict(diag) if hasattr(diag,'__dataclass_fields__') else {}
                else: rows=result; metadata={}
            else:
                rows=[]; metadata={}
            text='\n'.join(
                f"{r['path']}" + (f":{r.get('start_line')}-{r.get('end_line')}" if r.get('start_line') else '') +
                (f" [{r.get('symbol')}]" if r.get('symbol') else '') + f": {r.get('snippet','')}"
                for r in rows
            )
            metadata.update({'matches':len(rows),'truncated':len(rows)>=min(int(max_results),40),'retryable':False,'requested_mode':str(mode),'information_source':'candidate_retrieval'})
            return ToolObservation(self.spec.name,True,text,metadata,None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{'retryable':False,'requested_mode':str(mode)},type(e).__name__,(time.time()-t)*1000)

class IndexedSymbolSearchTool(Tool):
    spec=ToolSpec('symbol_search','Search the task-scoped AST symbol index.',IndexedSymbolSearchArgs,'repository_search','medium','none',12000)
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=60):
        t=time.time()
        try:
            rows=self.index.symbols(str(query),min(int(max_results),60)); text='\n'.join(f"{r['path']}:{r['start_line']}-{r['end_line']} {r['kind']} {r['name']}" for r in rows)
            return ToolObservation(self.spec.name,True,text,{'matches':len(rows),'truncated':len(rows)>=min(int(max_results),60),'retryable':False},None,(time.time()-t)*1000)
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{'retryable':False},type(e).__name__,(time.time()-t)*1000)


class InspectSymbolContextTool(Tool):
    spec=ToolSpec('inspect_symbol_context','Inspect one uniquely resolved Python symbol plus bounded direct callers/callees. Set include_source=true to return source snippets in the same tool result. Static call relations include an explicit resolution_kind.',InspectSymbolContextArgs,'repository_search','medium','none',16000)
    def __init__(self,index): self.index=(index.lexical if hasattr(index,'lexical') else index)
    def execute(self,symbol,file=None,include_source=False,include_uncertain=False,max_callers=5,max_callees=5,max_source_chars=12000):
        t=time.time()
        try:
            data=self.index.inspect_symbol_context(str(symbol),file,include_source=bool(include_source),include_uncertain=bool(include_uncertain),max_callers=int(max_callers),max_callees=int(max_callees),max_source_chars=int(max_source_chars))
            if not data.get('ok'):
                return ToolObservation(self.spec.name,False,str(data),{'retryable':False,**data},data.get('error_type'),(time.time()-t)*1000)
            import json
            return ToolObservation(self.spec.name,True,json.dumps(data,ensure_ascii=False,indent=2),{'retryable':False,'include_source':bool(include_source),'symbol_context':data,'information_source':'candidate_retrieval'},None,(time.time()-t)*1000)
        except Exception as e:
            return ToolObservation(self.spec.name,False,str(e),{'retryable':False},type(e).__name__,(time.time()-t)*1000)
