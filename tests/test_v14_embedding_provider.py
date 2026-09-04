import httpx
import pytest
from debug_assistant.repository.embeddings import SiliconFlowEmbeddingProvider, EmbeddingError

class Resp:
    def __init__(self,status,data=None,text='err'):
        self.status_code=status; self._data=data or {}; self.text=text
    def json(self): return self._data
    def raise_for_status(self):
        if self.status_code>=400:
            raise httpx.HTTPStatusError('bad',request=httpx.Request('POST','https://x'),response=httpx.Response(self.status_code))

class Client:
    queue=[]
    def __init__(self,*a,**k): pass
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def post(self,*a,**k): return self.queue.pop(0)

def test_embedding_401_fails_fast(monkeypatch):
    Client.queue=[Resp(401)]
    monkeypatch.setattr(httpx,'Client',Client)
    p=SiliconFlowEmbeddingProvider(api_key='x',dimension=3,max_retries=3)
    with pytest.raises(EmbeddingError): p.embed_query('x')
    assert p.stats.requests==1

def test_embedding_429_retries_then_succeeds(monkeypatch):
    Client.queue=[Resp(429),Resp(200,{'data':[{'index':0,'embedding':[1,0,0]}]})]
    monkeypatch.setattr(httpx,'Client',Client); monkeypatch.setattr('time.sleep',lambda *_:None)
    p=SiliconFlowEmbeddingProvider(api_key='x',dimension=3,max_retries=2)
    assert p.embed_query('x')==[1.0,0.0,0.0]
    assert p.stats.retries==1 and p.stats.requests==2
