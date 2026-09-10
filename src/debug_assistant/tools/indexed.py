from __future__ import annotations
import json
import time
from dataclasses import asdict
from pydantic import Field
from .base import Tool,ToolSpec,ToolArgs
from .repository import (
    REPOSITORY_SOURCE_MAX_CHARS,
    REPOSITORY_SYMBOL_MAX_RESULTS,
    _symbol_match,
)
from debug_assistant.models import ToolObservation

class CodeSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    mode: str = Field(default='lexical', pattern='^(lexical|dense|semantic|hybrid|hybrid_ast)$')
    max_results: int = Field(default=40, ge=1, le=40)

class IndexedSymbolSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    max_results: int = Field(default=REPOSITORY_SYMBOL_MAX_RESULTS, ge=1, le=REPOSITORY_SYMBOL_MAX_RESULTS)


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
        "Search repository code. Use mode='lexical' for exact identifiers/error names, mode='semantic' for behavioral/conceptual descriptions without code identifiers, mode='hybrid' when vocabulary is uncertain, and mode='hybrid_ast' to refine Hybrid candidates with existing Python symbols/call relations/source ranges. Retrieval candidates must be verified with read_file before becoming evidence.",
        CodeSearchArgs,'repository_search','medium','none',14000)
    def __init__(self,index, *, default_mode='lexical'):
        self.index=index
        self.default_mode=str(default_mode).lower()

    def execute(self,query,mode=None,max_results=40):
        t=time.time()
        try:
            requested_mode = str(mode or self.default_mode).lower()
            engine_mode = 'semantic' if requested_mode == 'dense' else requested_mode
            if hasattr(self.index,'search'):
                try:
                    result=self.index.search(str(query),mode=engine_mode,limit=min(int(max_results),40))
                except TypeError:
                    result=self.index.search(str(query),min(int(max_results),40))
                if isinstance(result,tuple): rows,diag=result; metadata=asdict(diag) if hasattr(diag,'__dataclass_fields__') else {}
                else: rows=result; metadata={}
            else:
                rows=[]; metadata={}
            rendered=[]
            for r in rows:
                # Keep retrieval and structural metadata explicit.  The result
                # is still discovery-only; the model must use read_file before
                # any source-backed claim can become Evidence.
                rendered.append({
                    'path': r.get('path', ''),
                    'snippet': r.get('snippet', ''),
                    'retrieval_score': r.get('score'),
                    'retrieval_source': r.get('source', 'lexical'),
                    'matched_symbols': list(r.get('matched_symbols') or []),
                    'source_range': (
                        [r.get('start_line'), r.get('end_line')]
                        if r.get('start_line') is not None else None
                    ),
                    'structural_relevance': r.get('structural_relevance', 0.0),
                    'caller_count': r.get('caller_count', 0),
                    'callee_count': r.get('callee_count', 0),
                })
            metadata.update({'matches':len(rows),'truncated':len(rows)>=min(int(max_results),40),'retryable':False,'requested_mode':requested_mode,'effective_mode':metadata.get('effective_mode', engine_mode),'information_source':'candidate_retrieval'})
            payload = {
                'query': str(query),
                'results': rendered,
                'retrieval': {
                    'requested_mode': metadata.get('requested_mode'),
                    'effective_mode': metadata.get('effective_mode'),
                    'degraded': bool(metadata.get('degraded', False)),
                    'reason': metadata.get('reason', ''),
                    'lexical_candidates': metadata.get('lexical_candidates', 0),
                    'semantic_candidates': metadata.get('semantic_candidates', 0),
                    'rrf_k': metadata.get('rrf_k'),
                },
            }
            return ToolObservation(
                self.spec.name, True, json.dumps(payload, ensure_ascii=False),
                metadata, None, (time.time()-t)*1000,
            )
        except Exception as e:return ToolObservation(self.spec.name,False,str(e),{'retryable':False,'requested_mode':str(mode or self.default_mode)},type(e).__name__,(time.time()-t)*1000)

class IndexedSymbolSearchTool(Tool):
    spec=ToolSpec(
        'symbol_search',
        'Locate declared symbols from the task-scoped index and return bounded '
        'candidate context in the same result. Use read_file to verify the '
        'reported range or to read arbitrary ranges.',
        IndexedSymbolSearchArgs, 'repository_search', 'medium', 'none', 12000,
    )
    def __init__(self,index): self.index=index
    def execute(self,query,max_results=60):
        t=time.time()
        try:
            limit=min(int(max_results),REPOSITORY_SYMBOL_MAX_RESULTS)
            rows=self.index.symbols(str(query),limit)
            lexical=getattr(self.index,'lexical',self.index)
            output_limit=int(self.spec.output_limit or REPOSITORY_SOURCE_MAX_CHARS)
            budget=min(REPOSITORY_SOURCE_MAX_CHARS,max(1000,output_limit-output_limit//3))
            per_match=max(1000,budget // max(1,min(len(rows),6)))
            matches=[]
            for index,row in enumerate(rows):
                try:
                    lines=lexical.fs.read_text(row['path']).splitlines()
                except Exception:
                    lines=[]
                if index < 6 and lines:
                    match=_symbol_match(
                        path=row['path'], name=row['name'], kind=row['kind'],
                        start_line=row['start_line'], end_line=row['end_line'],
                        lines=lines, max_source_chars=per_match,
                    )
                    try:
                        context=lexical.inspect_symbol_context(
                            row['name'], row['path'], include_source=False,
                        )
                    except Exception:
                        context={}
                    if context.get('ok'):
                        match['callers']=context.get('callers') or []
                        match['callees']=context.get('callees') or []
                        match['relations_available']=bool(match['callers'] or match['callees'])
                else:
                    match={
                        'symbol':row['name'], 'name':row['name'], 'kind':row['kind'],
                        'file':row['path'], 'start_line':row['start_line'],
                        'end_line':row['end_line'], 'signature':'',
                        'source_context':'', 'source_context_start_line':None,
                        'source_context_end_line':None, 'source_context_truncated':True,
                        'truncated':True, 'callers':[], 'callees':[],
                        'relations_available':False, 'source_context_omitted':True,
                    }
                matches.append(match)
            payload={
                'query':str(query), 'matches':matches,
                'truncated':len(rows)>=limit or any(x.get('truncated') for x in matches),
                'source_context_budget_chars':budget,
            }
            metadata={
                'matches':len(matches), 'truncated':payload['truncated'],
                'retryable':False, 'information_source':'candidate_retrieval',
            }
            # A bounded context is useful for planning, but symbol_search is
            # still discovery.  The Planner must explicitly verify this range
            # with read_file before source becomes CODE Evidence.
            observation=ToolObservation(
                self.spec.name, True, json.dumps(payload, ensure_ascii=False), metadata,
                None, (time.time()-t)*1000,
            )
            return observation
        except Exception as e:
            return ToolObservation(self.spec.name,False,str(e),{'retryable':False},type(e).__name__,(time.time()-t)*1000)


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
