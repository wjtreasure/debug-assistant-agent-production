from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json, math, random, sqlite3, time
import httpx


class EmbeddingError(RuntimeError):
    pass


class EmbeddingDeadlineExceeded(EmbeddingError):
    pass


class EmbeddingHTTPError(EmbeddingError):
    def __init__(self, status_code:int, message:str, *, retryable:bool=False):
        super().__init__(message)
        self.status_code=int(status_code)
        self.retryable=bool(retryable)


class EmbeddingInputError(EmbeddingError):
    def __init__(self, input_index:int, message:str):
        super().__init__(message)
        self.input_index=int(input_index)


class EmbeddingProvider(Protocol):
    model: str
    dimension: int
    provider_name: str
    def embed_documents(self,texts:list[str])->list[list[float]]: ...
    def embed_query(self,text:str)->list[float]: ...


class EmbeddingCache:
    def __init__(self,path:Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS embeddings(cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL, dimension INT NOT NULL, created_at REAL NOT NULL)')
        self.conn.commit()
    def get(self,key:str,dimension:int):
        row=self.conn.execute('SELECT vector,dimension FROM embeddings WHERE cache_key=?',(key,)).fetchone()
        if not row or int(row[1])!=int(dimension): return None
        try: return [float(x) for x in json.loads(row[0])]
        except Exception: return None
    def put(self,key:str,vector:list[float]):
        self.conn.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)',(key,json.dumps(vector,separators=(',',':')),len(vector),time.time())); self.conn.commit()
    def close(self): self.conn.close()


@dataclass(slots=True)
class EmbeddingCallStats:
    requests:int=0
    retries:int=0
    failures:int=0
    latency_ms:float=0.0
    isolated_batches:int=0
    smoke_tests:int=0


class SiliconFlowEmbeddingProvider:
    provider_name='siliconflow'
    def __init__(self, *, api_key:str, model:str='BAAI/bge-m3', base_url:str='https://api.siliconflow.cn/v1', dimension:int=1024, timeout:float=60.0, batch_size:int=16, max_retries:int=3, max_isolation_depth:int=5):
        self.api_key=api_key; self.model=model; self.base_url=base_url.rstrip('/'); self.dimension=int(dimension); self.timeout=float(timeout)
        self.batch_size=max(1,int(batch_size)); self.max_retries=max(0,int(max_retries)); self.max_isolation_depth=max(0,int(max_isolation_depth)); self.stats=EmbeddingCallStats()
        if not api_key: raise EmbeddingError('SiliconFlow embedding API key is empty')

    @staticmethod
    def _validate_text(text:str)->None:
        if not isinstance(text,str) or not text.strip():
            raise EmbeddingError('embedding input must be a non-empty string')
        if '\x00' in text:
            raise EmbeddingError('embedding input contains NUL byte')

    def _call_batch(self,texts:list[str], *, deadline=None)->list[list[float]]:
        for text in texts: self._validate_text(text)
        payload={'model':self.model,'input':(texts[0] if len(texts)==1 else texts),'encoding_format':'float'}
        headers={'Authorization':f'Bearer {self.api_key}','Content-Type':'application/json'}
        last=None
        for attempt in range(self.max_retries+1):
            if deadline is not None and deadline.expired():
                raise EmbeddingDeadlineExceeded('embedding request exceeded run deadline')
            started=time.time(); self.stats.requests+=1
            try:
                timeout=self.timeout if deadline is None else deadline.effective_timeout(self.timeout)
                if timeout <= 0: raise EmbeddingDeadlineExceeded('embedding request exceeded run deadline')
                with httpx.Client(timeout=timeout) as client:
                    r=client.post(self.base_url+'/embeddings',headers=headers,json=payload)
                self.stats.latency_ms += (time.time()-started)*1000
                if r.status_code in (400,401,403):
                    self.stats.failures+=1
                    raise EmbeddingHTTPError(r.status_code,f'embedding API non-retryable HTTP {r.status_code}: {r.text[:500]}',retryable=False)
                if r.status_code==429 or r.status_code>=500:
                    last=EmbeddingHTTPError(r.status_code,f'embedding API retryable HTTP {r.status_code}: {r.text[:500]}',retryable=True)
                else:
                    r.raise_for_status(); data=r.json().get('data') or []
                    rows=sorted(data,key=lambda x:int(x.get('index',0)))
                    vecs=[x.get('embedding') for x in rows]
                    if len(vecs)!=len(texts): raise EmbeddingError(f'embedding count mismatch: expected {len(texts)}, got {len(vecs)}')
                    for v in vecs:
                        if not isinstance(v,list) or len(v)!=self.dimension or not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in v):
                            raise EmbeddingError(f'invalid embedding dimension/content; expected dimension={self.dimension}')
                    return [[float(x) for x in v] for v in vecs]
            except EmbeddingHTTPError:
                raise
            except (httpx.TimeoutException,httpx.TransportError) as exc:
                self.stats.latency_ms += (time.time()-started)*1000; last=EmbeddingError(f'embedding transport error: {exc}')
            except httpx.HTTPStatusError as exc:
                last=EmbeddingError(f'embedding HTTP error: {exc}')
            if attempt < self.max_retries:
                if deadline is not None and deadline.expired():
                    raise EmbeddingDeadlineExceeded('embedding retry exceeded run deadline')
                self.stats.retries+=1
                delay=min(8.0,(2**attempt)+random.random()*0.25)
                if deadline is not None:
                    delay=min(delay,deadline.remaining())
                if delay > 0: time.sleep(delay)
        self.stats.failures+=1
        raise last or EmbeddingError('embedding request failed')

    def smoke_test(self, *, deadline=None)->list[float]:
        self.stats.smoke_tests+=1
        return self._call_batch(['debug assistant semantic search smoke test'],deadline=deadline)[0]

    def _embed_batch_isolated(self,texts:list[str],*,base_index:int,depth:int,deadline=None)->list[list[float]]:
        try:
            return self._call_batch(texts,deadline=deadline)
        except EmbeddingHTTPError as exc:
            if exc.status_code != 400:
                raise
            if len(texts)==1:
                raise EmbeddingInputError(base_index,f'provider rejected embedding input at index {base_index}: {exc}') from exc
            if depth >= self.max_isolation_depth:
                raise EmbeddingInputError(base_index,f'embedding batch isolation depth exceeded at index {base_index}; batch_size={len(texts)}') from exc
            self.stats.isolated_batches+=1
            mid=len(texts)//2
            left=self._embed_batch_isolated(texts[:mid],base_index=base_index,depth=depth+1,deadline=deadline)
            right=self._embed_batch_isolated(texts[mid:],base_index=base_index+mid,depth=depth+1,deadline=deadline)
            return left+right

    def embed_documents(self,texts:list[str], *, deadline=None)->list[list[float]]:
        out=[]
        for i in range(0,len(texts),self.batch_size):
            out.extend(self._embed_batch_isolated(texts[i:i+self.batch_size],base_index=i,depth=0,deadline=deadline))
        return out

    def embed_query(self,text:str, *, deadline=None)->list[float]:
        return self._call_batch([text],deadline=deadline)[0]
