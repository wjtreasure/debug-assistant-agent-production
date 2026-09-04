from pathlib import Path
import numpy as np
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.chunks import build_chunk_manifest
from debug_assistant.repository.embeddings import EmbeddingCache
from debug_assistant.repository.semantic_index import SemanticIndex
from debug_assistant.repository.search_engine import RepositorySearchEngine, reciprocal_rank_fusion
from debug_assistant.repository.index import RepositoryIndex

class FakeEmbeddingProvider:
    provider_name='fake'; model='fake-v1'; dimension=3
    def embed_documents(self,texts): return [self.embed_query(x) for x in texts]
    def embed_query(self,text):
        t=text.lower()
        # semantic synonym: compatibility/schema map to same axis even if exact terms differ.
        if 'compatib' in t or 'schema' in t or 'contract' in t:return [1.0,0.0,0.0]
        if 'parser' in t:return [0.0,1.0,0.0]
        return [0.0,0.0,1.0]

def test_manifest_and_semantic_index_are_snapshot_scoped(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'a.py').write_text('def check_contract_compatibility(x):\n    return bool(x)\n')
    fs=SafeRepositoryFS(repo); manifest=build_chunk_manifest(fs)
    cache=EmbeddingCache(tmp_path/'cache.sqlite')
    sem=SemanticIndex(manifest,FakeEmbeddingProvider(),cache); stats=sem.build()
    assert stats.status=='ready' and len(sem.chunks)==len(manifest.chunks)
    rows=sem.search('schema is unexpectedly rejected',limit=5)
    assert rows and rows[0]['path']=='a.py'
    cache.close()

def test_hybrid_rrf_uses_rank_not_raw_score(tmp_path):
    a=[{'path':'a.py','snippet':'a'},{'path':'b.py','snippet':'b'}]
    b=[{'path':'b.py','snippet':'b'},{'path':'c.py','snippet':'c'}]
    rows=reciprocal_rank_fusion([a,b],k=60,limit=3)
    assert rows[0]['path']=='b.py'

def test_search_engine_degrades_semantic_to_lexical(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'x.py').write_text('def exact_name():\n    return 1\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    engine=RepositorySearchEngine(idx,None,rrf_k=60)
    rows,diag=engine.search('exact_name',mode='semantic',limit=5)
    assert diag.degraded is True and diag.effective_mode=='lexical'
    assert rows and rows[0]['path']=='x.py'
    idx.close()
