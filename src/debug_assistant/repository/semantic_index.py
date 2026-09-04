from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .chunks import ChunkManifest, CodeChunk
from .embeddings import EmbeddingProvider, EmbeddingCache, EmbeddingError, EmbeddingInputError


@dataclass(slots=True)
class SemanticBuildStats:
    status:str='unavailable'
    chunks:int=0
    cache_hits:int=0
    api_embeddings:int=0
    error:str=''
    failed_path:str=''
    failed_symbol:str=''
    failed_start_line:int|None=None
    failed_end_line:int|None=None
    provider_requests:int=0
    provider_retries:int=0
    provider_failures:int=0
    provider_latency_ms:float=0.0
    provider_isolated_batches:int=0
    smoke_tested:bool=False
    backend:str=''


class SemanticIndex:
    """Task-scoped dense index. Cache is content-addressed; membership is manifest-scoped."""
    def __init__(self, manifest:ChunkManifest, provider:EmbeddingProvider, cache:EmbeddingCache|None=None, chunker_version:str=''):
        self.manifest=manifest; self.provider=provider; self.cache=cache; self.chunker_version=chunker_version or manifest.chunker_version
        self.matrix:np.ndarray|None=None; self.faiss_index=None; self.backend='numpy'; self.chunks:list[CodeChunk]=[]; self.stats=SemanticBuildStats()

    def _key(self,c:CodeChunk)->str:
        actual_hash=hashlib.sha256(c.embedding_text().encode('utf-8',errors='ignore')).hexdigest()
        raw='|'.join([self.provider.provider_name,self.provider.model,str(self.provider.dimension),self.chunker_version,actual_hash])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _provider_telemetry(self)->dict:
        s=getattr(self.provider,'stats',None)
        return {
            'provider_requests':int(getattr(s,'requests',0) or 0),
            'provider_retries':int(getattr(s,'retries',0) or 0),
            'provider_failures':int(getattr(s,'failures',0) or 0),
            'provider_latency_ms':float(getattr(s,'latency_ms',0.0) or 0.0),
            'provider_isolated_batches':int(getattr(s,'isolated_batches',0) or 0),
            'smoke_tested':bool(getattr(s,'smoke_tests',0) or 0),
        }

    def build(self, *, deadline=None)->SemanticBuildStats:
        vectors=[]; missing=[]; missing_pos=[]; hits=0
        for idx,c in enumerate(self.manifest.chunks):
            v=self.cache.get(self._key(c),self.provider.dimension) if self.cache else None
            if v is None:
                missing.append(c.embedding_text()); missing_pos.append(idx); vectors.append(None)
            else:
                vectors.append(v); hits+=1
        generated=[]
        try:
            # Separate provider/config failure from repository-input failure. Only pay
            # this extra call when the current manifest actually has cache misses.
            if missing and hasattr(self.provider,'smoke_test'):
                self.provider.smoke_test(deadline=deadline) if deadline is not None else self.provider.smoke_test()
            if missing:
                generated=(self.provider.embed_documents(missing,deadline=deadline) if deadline is not None else self.provider.embed_documents(missing))
            else:
                generated=[]
            if len(generated)!=len(missing): raise EmbeddingError('semantic index build incomplete')
            for pos,v in zip(missing_pos,generated):
                vectors[pos]=v
            if len(vectors)!=len(self.manifest.chunks) or any(v is None for v in vectors):
                raise EmbeddingError('semantic index missing embeddings')
            arr=np.asarray(vectors,dtype=np.float32)
            if arr.ndim!=2 or arr.shape[1]!=self.provider.dimension: raise EmbeddingError('semantic matrix dimension mismatch')
            norms=np.linalg.norm(arr,axis=1,keepdims=True); norms[norms==0]=1.0; arr=arr/norms
            self.matrix=arr; self.chunks=list(self.manifest.chunks)
            try:
                import faiss
                fi=faiss.IndexFlatIP(self.provider.dimension); fi.add(arr)
                self.faiss_index=fi; self.backend='faiss'
            except Exception:
                self.faiss_index=None; self.backend='numpy'
            # Cache writes happen only after the whole manifest has produced a valid matrix.
            for pos,v in zip(missing_pos,generated):
                if self.cache: self.cache.put(self._key(self.manifest.chunks[pos]),v)
            telem=self._provider_telemetry()
            self.stats=SemanticBuildStats('ready',len(self.chunks),hits,len(generated),'',backend=self.backend,**telem)
        except Exception as exc:
            failed={}
            if isinstance(exc,EmbeddingInputError) and 0 <= exc.input_index < len(missing_pos):
                c=self.manifest.chunks[missing_pos[exc.input_index]]
                failed={'failed_path':c.path,'failed_symbol':c.qualified_name or c.symbol or '',
                        'failed_start_line':c.start_line,'failed_end_line':c.end_line}
            self.matrix=None; self.chunks=[]; self.faiss_index=None
            telem=self._provider_telemetry()
            self.stats=SemanticBuildStats('unavailable',len(self.manifest.chunks),hits,0,str(exc),**failed,**telem)
        return self.stats

    @property
    def available(self)->bool:
        return self.matrix is not None and bool(self.chunks)

    def search(self,query:str,limit:int=20, *, deadline=None):
        if not self.available: return []
        if deadline is not None and deadline.expired():
            raise EmbeddingError('semantic query exceeded run deadline')
        q=np.asarray((self.provider.embed_query(query,deadline=deadline) if deadline is not None else self.provider.embed_query(query)),dtype=np.float32)
        if q.shape!=(self.provider.dimension,): raise EmbeddingError('query embedding dimension mismatch')
        n=float(np.linalg.norm(q)); q=q/(n or 1.0)
        nlimit=max(1,int(limit))
        if self.faiss_index is not None:
            vals,inds=self.faiss_index.search(q.reshape(1,-1).astype(np.float32),nlimit)
            pairs=[(int(i),float(v)) for i,v in zip(inds[0],vals[0]) if int(i)>=0]
        else:
            scores=self.matrix @ q; idx=np.argsort(-scores)[:nlimit]; pairs=[(int(i),float(scores[int(i)])) for i in idx]
        out=[]
        for i,score in pairs:
            c=self.chunks[i]
            out.append({'chunk_id':c.chunk_id,'path':c.path,'symbol':c.qualified_name or c.symbol,'start_line':c.start_line,'end_line':c.end_line,'snippet':c.content[:1200],'score':score,'source':'semantic','backend':self.backend})
        return out
