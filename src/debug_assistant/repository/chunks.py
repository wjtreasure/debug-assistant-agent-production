from __future__ import annotations
from dataclasses import dataclass, asdict
import ast, hashlib, math
from .safe_fs import SafeRepositoryFS, TEXT_SUFFIXES

CHUNKER_VERSION='ast-symbol-v2-sized'


@dataclass(slots=True, frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    language: str
    symbol: str | None
    qualified_name: str | None
    kind: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    content: str
    content_hash: str
    embedding_text_hash: str
    parent_symbol: str | None = None
    part_index: int = 0
    part_count: int = 1

    def embedding_text(self) -> str:
        parts=[f'File: {self.path}',f'Kind: {self.kind}']
        if self.qualified_name: parts.append(f'Symbol: {self.qualified_name}')
        if self.signature: parts.append(f'Signature: {self.signature}')
        if self.docstring: parts.append(f'Docstring: {self.docstring}')
        if self.part_count > 1: parts.append(f'Chunk-Part: {self.part_index+1}/{self.part_count}')
        parts.append('Code:\n'+self.content)
        return '\n'.join(parts)


@dataclass(slots=True)
class ChunkManifest:
    chunker_version: str
    chunks: list[CodeChunk]
    digest: str

    def to_dict(self):
        return {'chunker_version':self.chunker_version,'digest':self.digest,'chunks':[asdict(c) for c in self.chunks]}


def _sha(text:str)->str:
    return hashlib.sha256(text.encode('utf-8',errors='ignore')).hexdigest()


def estimate_embedding_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for code/text embedding payloads.

    Code tends to tokenize more densely than prose. len/3 intentionally errs on the
    safe side so a provider's 8k-token hard limit is not approached by our 6k budget.
    """
    return max(1, math.ceil(len(text) / 3.0))


def _signature(node, lines:list[str]) -> str:
    try:
        return lines[node.lineno-1].strip()[:500]
    except Exception:
        return ''


def _embedding_text(path:str,kind:str,qname:str|None,signature:str,doc:str,content:str,part_index:int=0,part_count:int=1)->str:
    parts=[f'File: {path}',f'Kind: {kind}']
    if qname: parts.append(f'Symbol: {qname}')
    if signature: parts.append(f'Signature: {signature}')
    if doc: parts.append(f'Docstring: {doc}')
    if part_count > 1: parts.append(f'Chunk-Part: {part_index+1}/{part_count}')
    parts.append('Code:\n'+content)
    return '\n'.join(parts)


def _make_chunk(*,path:str,language:str,symbol:str|None,qname:str|None,kind:str,start:int,end:int,signature:str,doc:str,content:str,parent_symbol:str|None=None,part_index:int=0,part_count:int=1)->CodeChunk:
    ch=_sha(content)
    emb=_embedding_text(path,kind,qname,signature,doc,content,part_index,part_count)
    eh=_sha(emb)
    cid=_sha(f'{path}:{qname or ""}:{start}:{end}:{part_index}:{ch}')[:24]
    return CodeChunk(cid,path,language,symbol,qname,kind,start,end,signature,doc,content,ch,eh,parent_symbol,part_index,part_count)


def _split_symbol(*,path:str,lines:list[str],symbol:str,qname:str,kind:str,start:int,end:int,signature:str,doc:str,max_embedding_tokens:int)->list[CodeChunk]:
    content='\n'.join(lines[start-1:end])
    preview=_embedding_text(path,kind,qname,signature,doc,content)
    if estimate_embedding_tokens(preview) <= max_embedding_tokens:
        return [_make_chunk(path=path,language='python',symbol=symbol,qname=qname,kind=kind,start=start,end=end,signature=signature,doc=doc,content=content,parent_symbol=qname)]

    # Preserve the symbol identity while subdividing only its body. Build bounded
    # line groups greedily so no generated embedding text exceeds the configured budget.
    ranges=[]; s=start
    while s <= end:
        hi=s; chosen=s
        while hi <= end:
            candidate_text='\n'.join(lines[s-1:hi])
            probe=_embedding_text(path,kind,qname,signature if s==start else '',doc if s==start else '',candidate_text,len(ranges),1)
            if estimate_embedding_tokens(probe) > max_embedding_tokens:
                # If a single pathological line exceeds the estimate, isolate it so
                # provider-side validation can report that exact source range.
                chosen=hi if hi==s else hi-1
                break
            chosen=hi
            hi+=1
        ranges.append((s,min(chosen,end)))
        s=min(chosen,end)+1

    count=len(ranges); out=[]
    for idx,(lo,hi) in enumerate(ranges):
        part='\n'.join(lines[lo-1:hi])
        out.append(_make_chunk(
            path=path,language='python',symbol=symbol,qname=qname,kind=kind,
            start=lo,end=hi,signature=signature if idx==0 else '',doc=doc if idx==0 else '',
            content=part,parent_symbol=qname,part_index=idx,part_count=count,
        ))
    return out


def _python_chunks(path:str,text:str,max_embedding_tokens:int) -> list[CodeChunk]:
    lines=text.splitlines()
    try: tree=ast.parse(text)
    except Exception: return []
    parents={}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent): parents[child]=parent
    out=[]
    for node in ast.walk(tree):
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): continue
        start=int(node.lineno); end=int(getattr(node,'end_lineno',start) or start)
        q=[node.name]; p=parents.get(node)
        while p is not None:
            if isinstance(p,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): q.append(p.name)
            p=parents.get(p)
        qname='.'.join(reversed(q)); doc=(ast.get_docstring(node,clean=True) or '')[:2000]
        out.extend(_split_symbol(
            path=path,lines=lines,symbol=node.name,qname=qname,kind=type(node).__name__,
            start=start,end=end,signature=_signature(node,lines),doc=doc,max_embedding_tokens=max_embedding_tokens,
        ))
    return out


def _window_chunks(path:str,text:str,window:int=100,overlap:int=20,max_embedding_tokens:int=6000) -> list[CodeChunk]:
    lines=text.splitlines(); out=[]
    if not lines: return out
    step=max(1,window-overlap)
    for s0 in range(0,len(lines),step):
        e0=min(len(lines),s0+window); content='\n'.join(lines[s0:e0])
        if not content.strip(): continue
        # Shrink a window if a very long source file makes the conservative estimate exceed the budget.
        while e0 > s0+1 and estimate_embedding_tokens(_embedding_text(path,'window',None,'','',content)) > max_embedding_tokens:
            e0=max(s0+1,s0+(e0-s0)//2); content='\n'.join(lines[s0:e0])
        out.append(_make_chunk(path=path,language='text',symbol=None,qname=None,kind='window',start=s0+1,end=e0,signature='',doc='',content=content))
        if e0>=len(lines): break
    return out


def build_chunk_manifest(fs:SafeRepositoryFS,max_file_bytes:int=1_000_000,max_embedding_tokens:int=6000, *, deadline=None) -> ChunkManifest:
    max_embedding_tokens=max(256,int(max_embedding_tokens))
    chunks=[]
    for sf in fs.iter_files(suffixes=TEXT_SUFFIXES,max_file_bytes=max_file_bytes):
        if deadline is not None and deadline.expired():
            raise TimeoutError('chunk manifest build exceeded run deadline')
        try: text=sf.path.read_text(encoding='utf-8',errors='ignore')
        except (OSError,UnicodeError): continue
        if sf.path.suffix.lower()=='.py':
            cs=_python_chunks(sf.rel,text,max_embedding_tokens)
            if not cs: cs=_window_chunks(sf.rel,text,max_embedding_tokens=max_embedding_tokens)
        else:
            cs=_window_chunks(sf.rel,text,max_embedding_tokens=max_embedding_tokens)
        chunks.extend(cs)
    stable='\n'.join(f'{c.chunk_id}:{c.embedding_text_hash}' for c in sorted(chunks,key=lambda x:x.chunk_id))
    return ChunkManifest(CHUNKER_VERSION,chunks,_sha(stable))
