from __future__ import annotations
from pathlib import Path
import json,tempfile
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.repository.chunks import build_chunk_manifest
from debug_assistant.repository.embeddings import EmbeddingCache,SiliconFlowEmbeddingProvider
from debug_assistant.repository.semantic_index import SemanticIndex
from debug_assistant.repository.search_engine import RepositorySearchEngine

def _norm(p):return str(p).replace('\\','/').lstrip('./')

def evaluate_retrieval(tasks_root,config,modes=('lexical','semantic','hybrid'),top_k=10):
    root=Path(tasks_root); rows=[]
    semcfg=config.harness.semantic_search
    for d in sorted(root.iterdir()):
        if not (d/'task.json').exists() or not (d/'ground_truth.json').exists():continue
        meta=json.loads((d/'task.json').read_text()); workspace=meta.get('workspace')
        if not workspace:continue
        issue=(d/'issue.md').read_text(encoding='utf-8'); gold=json.loads((d/'ground_truth.json').read_text()); gold_files={_norm(x) for x in gold.get('files',[])}
        fs=SafeRepositoryFS(workspace)
        with tempfile.TemporaryDirectory(prefix='debug-retrieval-') as td:
            idx=RepositoryIndex(Path(workspace),Path(td)/'lexical.sqlite',fs=fs); idx.build()
            sem=None; cache=None
            if semcfg.enabled and semcfg.api_key:
                provider=SiliconFlowEmbeddingProvider(api_key=semcfg.api_key,model=semcfg.model,base_url=semcfg.base_url,dimension=semcfg.dimension,timeout=semcfg.timeout,batch_size=semcfg.batch_size,max_retries=semcfg.max_retries,max_isolation_depth=semcfg.max_isolation_depth)
                cache=EmbeddingCache(Path(semcfg.cache_path)); sem=SemanticIndex(build_chunk_manifest(fs,max_embedding_tokens=semcfg.max_embedding_tokens),provider,cache); sem.build()
            engine=RepositorySearchEngine(idx,sem,rrf_k=semcfg.rrf_k)
            for mode in modes:
                found,diag=engine.search(issue,mode=mode,limit=top_k)
                paths=[_norm(x.get('path','')) for x in found]
                rank=next((i+1 for i,p in enumerate(paths) if p in gold_files),None)
                rows.append({'task_id':d.name,'mode':mode,'hit':int(rank is not None),'mrr':0.0 if rank is None else 1/rank,'rank':rank,'gold_files':sorted(gold_files),'paths':paths,'diagnostics':__import__('dataclasses').asdict(diag) if hasattr(diag,'__dataclass_fields__') else {}})
            idx.close()
            if cache:cache.close()
    agg={}
    for mode in modes:
        mr=[r for r in rows if r['mode']==mode]; n=len(mr) or 1
        agg[mode]={'n':len(mr),f'file_recall@{top_k}':sum(r['hit'] for r in mr)/n,'mrr':sum(r['mrr'] for r in mr)/n}
    return {'aggregate':agg,'cases':rows}
