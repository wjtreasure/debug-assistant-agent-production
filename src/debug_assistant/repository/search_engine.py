from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class RetrievalDiagnostics:
    requested_mode:str
    effective_mode:str
    result_count:int
    semantic_available:bool
    degraded:bool=False
    reason:str=''
    lexical_candidates:int=0
    semantic_candidates:int=0
    rrf_k:int=60
    ast_available:bool=False
    ast_status:str='not_requested'
    ast_matches:int=0
    ast_relations:int=0
    ast_supported_candidates:int=0
    ast_unsupported_candidates:int=0


def reciprocal_rank_fusion(rankings:list[list[dict]], *, k:int=60, limit:int=20):
    """Fuse lexical file ranks and semantic chunk ranks at file granularity.

    Semantic search can return multiple chunks from one file; only the best-ranked
    chunk from each ranking contributes once. The representative row keeps useful
    chunk-level symbol/line metadata when available.
    """
    scores={}; docs={}
    for ranking in rankings:
        seen=set()
        for rank,row in enumerate(ranking,start=1):
            key=str(row.get('path') or row.get('chunk_id') or row.get('snippet','')[:80])
            if key in seen: continue
            seen.add(key)
            scores[key]=scores.get(key,0.0)+1.0/(k+rank)
            old=docs.get(key)
            # Prefer the chunk-aware row so the Agent can immediately read exact lines.
            if old is None or (row.get('start_line') and not old.get('start_line')):
                docs[key]=dict(row)
    ordered=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:limit]
    out=[]
    for key,score in ordered:
        row=dict(docs[key]); row['score']=score; row['source']='hybrid'; out.append(row)
    return out

class RepositorySearchEngine:
    def __init__(self, lexical_index, semantic_index=None, *, rrf_k:int=60, deadline=None):
        self.lexical=lexical_index; self.semantic=semantic_index; self.rrf_k=int(rrf_k); self.deadline=deadline
    def symbols(self,query,limit=60): return self.lexical.symbols(query,limit)
    def search(self,query:str,mode:str='lexical',limit:int=20):
        mode=str(mode).lower(); limit=max(1,min(int(limit),100)); semantic_available=bool(self.semantic and self.semantic.available)
        if mode=='lexical':
            rows=self.lexical.search(query,limit=limit)
            return rows,RetrievalDiagnostics(mode,'lexical',len(rows),semantic_available,lexical_candidates=len(rows),rrf_k=self.rrf_k)
        if mode=='semantic':
            if not semantic_available:
                rows=self.lexical.search(query,limit=limit)
                return rows,RetrievalDiagnostics(mode,'lexical',len(rows),False,True,'semantic_index_unavailable',lexical_candidates=len(rows),rrf_k=self.rrf_k)
            try:
                rows=self.semantic.search(query,limit=limit,deadline=self.deadline)
                return rows,RetrievalDiagnostics(mode,'semantic',len(rows),True,semantic_candidates=len(rows),rrf_k=self.rrf_k)
            except Exception as exc:
                rows=self.lexical.search(query,limit=limit)
                return rows,RetrievalDiagnostics(mode,'lexical',len(rows),True,True,f'semantic_query_failed:{type(exc).__name__}',lexical_candidates=len(rows),rrf_k=self.rrf_k)
        if mode in {'hybrid','hybrid_ast'}:
            lex=self.lexical.search(query,limit=max(limit,20))
            if not semantic_available:
                # ``hybrid_ast`` still has useful deterministic value when the
                # optional semantic index is unavailable: lexical/BM25 recall
                # remains the candidate boundary and the existing Python AST
                # pass may refine those candidates.  Never label this result
                # as Hybrid, because no RRF fusion actually happened.
                if mode == 'hybrid_ast':
                    refined, ast_diag = self.lexical.refine_hybrid_candidates(
                        lex[:limit], query, limit=limit,
                    )
                    ast_available = bool(ast_diag.get('ast_available'))
                    effective = 'lexical_ast' if ast_available else 'lexical'
                    reason = 'semantic_index_unavailable'
                    if not ast_available:
                        reason += ':ast_index_unavailable'
                    return refined, RetrievalDiagnostics(
                        mode, effective, len(refined), False, True, reason,
                        lexical_candidates=len(lex), rrf_k=self.rrf_k,
                        ast_available=ast_available,
                        ast_status=str(ast_diag.get('ast_status') or 'unavailable'),
                        ast_matches=int(ast_diag.get('matched_symbol_count') or 0),
                        ast_relations=int(ast_diag.get('relations_used') or 0),
                        ast_supported_candidates=int(ast_diag.get('supported_candidate_count') or 0),
                        ast_unsupported_candidates=int(ast_diag.get('unsupported_candidate_count') or 0),
                    )
                return lex[:limit],RetrievalDiagnostics(mode,'lexical',min(len(lex),limit),False,True,'semantic_index_unavailable',lexical_candidates=len(lex),rrf_k=self.rrf_k)
            try:
                sem=self.semantic.search(query,limit=max(limit,20),deadline=self.deadline)
            except Exception as exc:
                if mode == 'hybrid_ast':
                    refined, ast_diag = self.lexical.refine_hybrid_candidates(
                        lex[:limit], query, limit=limit,
                    )
                    ast_available = bool(ast_diag.get('ast_available'))
                    effective = 'lexical_ast' if ast_available else 'lexical'
                    reason = f'semantic_query_failed:{type(exc).__name__}'
                    if not ast_available:
                        reason += ':ast_index_unavailable'
                    return refined, RetrievalDiagnostics(
                        mode, effective, len(refined), True, True, reason,
                        lexical_candidates=len(lex), rrf_k=self.rrf_k,
                        ast_available=ast_available,
                        ast_status=str(ast_diag.get('ast_status') or 'unavailable'),
                        ast_matches=int(ast_diag.get('matched_symbol_count') or 0),
                        ast_relations=int(ast_diag.get('relations_used') or 0),
                        ast_supported_candidates=int(ast_diag.get('supported_candidate_count') or 0),
                        ast_unsupported_candidates=int(ast_diag.get('unsupported_candidate_count') or 0),
                    )
                return lex[:limit],RetrievalDiagnostics(mode,'lexical',min(len(lex),limit),True,True,f'semantic_query_failed:{type(exc).__name__}',lexical_candidates=len(lex),rrf_k=self.rrf_k)
            rows=reciprocal_rank_fusion([lex,sem],k=self.rrf_k,limit=limit)
            if mode=='hybrid':
                return rows,RetrievalDiagnostics(mode,'hybrid',len(rows),True,False,'',len(lex),len(sem),self.rrf_k)
            refined, ast_diag=self.lexical.refine_hybrid_candidates(
                rows, query, limit=limit,
            )
            ast_available=bool(ast_diag.get('ast_available'))
            ast_status=str(ast_diag.get('ast_status') or 'unavailable')
            degraded=not ast_available
            reason='' if ast_available else 'ast_index_unavailable'
            return refined,RetrievalDiagnostics(
                mode, 'hybrid_ast' if ast_available else 'hybrid', len(refined), True,
                degraded, reason, len(lex), len(sem), self.rrf_k,
                ast_available, ast_status,
                int(ast_diag.get('matched_symbol_count') or 0),
                int(ast_diag.get('relations_used') or 0),
                int(ast_diag.get('supported_candidate_count') or 0),
                int(ast_diag.get('unsupported_candidate_count') or 0),
            )
        raise ValueError("mode must be one of: lexical, semantic, hybrid, hybrid_ast")
