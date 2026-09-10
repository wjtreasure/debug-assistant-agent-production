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


def test_hybrid_ast_refinement_stays_in_candidates_and_uses_call_relations(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'target.py').write_text(
        'def helper(value):\n    return value + 1\n\n'
        'def target(value):\n    return helper(value)\n\n'
        'def caller(value):\n    return target(value)\n',
        encoding='utf-8',
    )
    (repo/'global_match.py').write_text('def target(value):\n    return value\n', encoding='utf-8')
    (repo/'unrelated.py').write_text('def stable(value):\n    return value\n', encoding='utf-8')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        rows, diagnostics=idx.refine_hybrid_candidates(
            [
                {'path':'target.py','score':0.03,'source':'hybrid','snippet':'target'},
                {'path':'unrelated.py','score':0.02,'source':'hybrid','snippet':'other'},
            ],
            'target behavior', limit=2,
        )
        assert {row['path'] for row in rows} == {'target.py','unrelated.py'}
        target=next(row for row in rows if row['path']=='target.py')
        match=target['matched_symbols'][0]
        assert match['symbol']=='target'
        assert match['kind']=='FunctionDef'
        assert match['source_range']==[4,5]
        assert any(item['symbol']=='caller' for item in match['callers'])
        assert any(item['symbol']=='helper' for item in match['callees'])
        assert diagnostics['relations_used'] >= 2
        assert 'global_match.py' not in {row['path'] for row in rows}
    finally:
        idx.close()


def test_hybrid_ast_does_not_promote_weak_symbol_over_hybrid_rank(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'strong.py').write_text('def unrelated(value):\n    return value\n', encoding='utf-8')
    (repo/'weak.py').write_text('def target(value):\n    return value\n', encoding='utf-8')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        rows, diagnostics=idx.refine_hybrid_candidates(
            [
                {'path':'strong.py','score':0.030,'source':'hybrid'},
                {'path':'weak.py','score':0.029,'source':'hybrid'},
            ],
            'target', limit=2,
        )
        assert diagnostics['ast_status']=='applied'
        assert [row['path'] for row in rows] == ['strong.py','weak.py']
    finally:
        idx.close()


def test_hybrid_ast_explicitly_degrades_without_python_symbols(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'main.go').write_text('package demo\nfunc Run() {}\n', encoding='utf-8')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        rows, diagnostics=idx.refine_hybrid_candidates(
            [{'path':'main.go','score':0.03,'source':'hybrid'}], 'Run', limit=1,
        )
        assert [row['path'] for row in rows] == ['main.go']
        assert diagnostics['ast_available'] is False
        assert diagnostics['ast_status']=='unsupported_language'
    finally:
        idx.close()


def test_search_engine_hybrid_ast_refines_the_fused_candidate_set(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'target.py').write_text(
        'def helper(value):\n    return value + 1\n\n'
        'def target(value):\n    return helper(value)\n',
        encoding='utf-8',
    )
    (repo/'other.py').write_text('def unrelated(value):\n    return value\n', encoding='utf-8')

    class AvailableSemanticIndex:
        available=True

        def search(self, query, *, limit, deadline=None):
            return [{'path':'target.py','snippet':'target behavior'}]

    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        engine=RepositorySearchEngine(idx,AvailableSemanticIndex())
        rows, diagnostics=engine.search('target behavior', mode='hybrid_ast', limit=2)
        assert diagnostics.requested_mode=='hybrid_ast'
        assert diagnostics.effective_mode=='hybrid_ast'
        assert diagnostics.ast_status=='applied'
        assert diagnostics.ast_matches >= 1
        assert [row['path'] for row in rows] == ['target.py']
        assert rows[0]['matched_symbols'][0]['source_range']==[4,5]
    finally:
        idx.close()
