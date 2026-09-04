from __future__ import annotations
from pathlib import Path
import ast, sqlite3, time, threading
from .safe_fs import SafeRepositoryFS, TEXT_SUFFIXES


class IndexDeadlineExceeded(TimeoutError):
    pass

class RepositoryIndex:
    """Task-scoped lexical + AST symbol/call-site index over one safe snapshot."""
    def __init__(self,repo_root:Path,db_path:Path,fs:SafeRepositoryFS|None=None):
        self.repo_root=Path(repo_root).resolve(); self.fs=fs or SafeRepositoryFS(self.repo_root)
        self.db_path=Path(db_path); self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.db_path,check_same_thread=False)
        self._lock=threading.RLock()
        self.fts=True
        try:self.conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(path, content, tokenize="unicode61")')
        except sqlite3.OperationalError:
            self.fts=False; self.conn.execute('CREATE TABLE IF NOT EXISTS files_fts(path TEXT PRIMARY KEY, content TEXT)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS symbols(path TEXT, name TEXT, qualified_name TEXT, kind TEXT, start_line INT, end_line INT)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS calls(path TEXT, caller_name TEXT, caller_qualified_name TEXT, line INT, expression TEXT, target_name TEXT, resolution_kind TEXT)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_calls_target ON calls(target_name)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_qualified_name)')

    @staticmethod
    def _qname(node, parents):
        q=[node.name]; p=parents.get(node)
        while p is not None:
            if isinstance(p,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):q.append(p.name)
            p=parents.get(p)
        return '.'.join(reversed(q))

    def build(self,max_file_bytes=1_000_000, *, deadline=None):
        self.conn.execute('DELETE FROM files_fts'); self.conn.execute('DELETE FROM symbols'); self.conn.execute('DELETE FROM calls')
        files=symbols=calls=0; started=time.time()
        for sf in self.fs.iter_files(suffixes=TEXT_SUFFIXES,max_file_bytes=max_file_bytes):
            if deadline is not None and deadline.expired():
                self.conn.rollback()
                raise IndexDeadlineExceeded('repository index build exceeded run deadline')
            try:
                text=sf.path.read_text(encoding='utf-8',errors='ignore'); rel=sf.rel
                self.conn.execute('INSERT INTO files_fts(path,content) VALUES (?,?)',(rel,text)); files+=1
                if sf.path.suffix.lower()!='.py':continue
                try:tree=ast.parse(text)
                except Exception:continue
                parents={}
                for parent in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent):parents[child]=parent
                defined=set()
                for n in ast.walk(tree):
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                        qname=self._qname(n,parents); defined.add(n.name)
                        self.conn.execute('INSERT INTO symbols VALUES (?,?,?,?,?,?)',(rel,n.name,qname,type(n).__name__,n.lineno,getattr(n,'end_lineno',n.lineno))); symbols+=1
                imports={}
                for n in ast.walk(tree):
                    if isinstance(n,ast.Import):
                        for a in n.names:imports[a.asname or a.name.split('.')[0]]=a.name
                    elif isinstance(n,ast.ImportFrom):
                        mod=n.module or ''
                        for a in n.names:imports[a.asname or a.name]=f'{mod}.{a.name}'.strip('.')
                for n in ast.walk(tree):
                    if not isinstance(n,ast.Call):continue
                    p=parents.get(n); owner=None
                    while p is not None:
                        if isinstance(p,(ast.FunctionDef,ast.AsyncFunctionDef)):
                            owner=p;break
                        p=parents.get(p)
                    if owner is None:continue
                    try:expr=ast.unparse(n.func)
                    except Exception:expr=''
                    target=''; kind='dynamic'
                    if isinstance(n.func,ast.Name):
                        target=n.func.id
                        if target in defined:kind='exact'
                        elif target in imports:kind='import_resolved'
                        else:kind='local_name'
                    elif isinstance(n.func,ast.Attribute):
                        target=n.func.attr
                        root=n.func.value.id if isinstance(n.func.value,ast.Name) else ''
                        kind='import_resolved' if root in imports else 'attribute_unresolved'
                    self.conn.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?)',(rel,owner.name,self._qname(owner,parents),n.lineno,expr,target,kind)); calls+=1
            except (OSError,UnicodeError,ValueError):pass
        self.conn.commit(); return {'files':files,'symbols':symbols,'calls':calls,'build_ms':(time.time()-started)*1000,'fts5':self.fts}

    def _fetchall(self,sql,args=()):
        # RepositoryIndex is read from bounded Tool workers after build.  Serialize
        # access to the shared SQLite connection so completion order cannot corrupt
        # cursor/connection state. RLock is required because inspect_symbol_context
        # calls resolve_symbol -> symbols while already performing indexed reads.
        with self._lock:
            return self.conn.execute(sql,args).fetchall()

    def search(self,query,limit=40):
        limit=min(int(limit),100); rows=[]
        if self.fts:
            q=' OR '.join(x.replace('"','') for x in str(query).split() if x) or str(query)
            try:rows=self._fetchall('SELECT path, snippet(files_fts,1,"[","]"," … ",18), bm25(files_fts) FROM files_fts WHERE files_fts MATCH ? ORDER BY bm25(files_fts) LIMIT ?',(q,limit))
            except sqlite3.OperationalError:rows=[]
        if not rows:
            like=f'%{query}%'; rows=[(*r,0.0) for r in self._fetchall('SELECT path, substr(content,1,800) FROM files_fts WHERE content LIKE ? OR path LIKE ? LIMIT ?',(like,like,limit))]
        return [{'path':r[0],'snippet':r[1],'score':float(r[2]),'source':'lexical'} for r in rows]

    def symbols(self,query,limit=60):
        like=f'%{query}%'; rows=self._fetchall('SELECT path,name,qualified_name,kind,start_line,end_line FROM symbols WHERE name LIKE ? OR qualified_name LIKE ? ORDER BY length(name),path LIMIT ?',(like,like,min(int(limit),100)))
        return [{'path':r[0],'name':r[1],'qualified_name':r[2],'kind':r[3],'start_line':r[4],'end_line':r[5]} for r in rows]

    def resolve_symbol(self,symbol:str,file:str|None=None):
        rows=self.symbols(symbol,100)
        exact=[r for r in rows if symbol in {r['name'],r['qualified_name']} or r['qualified_name'].endswith('.'+symbol)]
        if file:exact=[r for r in exact if r['path']==file]
        uniq={(r['path'],r['qualified_name'],r['start_line'],r['end_line']):r for r in exact}
        return list(uniq.values())

    def _source(self,path,start,end,max_chars):
        try:lines=self.fs.read_text(path,max_bytes=1_000_000).splitlines(); a=max(1,int(start)); b=min(len(lines),int(end)); txt='\n'.join(f'{i:5d} | {lines[i-1]}' for i in range(a,b+1)); return txt[:max_chars]
        except Exception:return ''

    def inspect_symbol_context(self,symbol,file=None,*,include_source=False,include_uncertain=False,max_callers=5,max_callees=5,max_source_chars=12000):
        resolved=self.resolve_symbol(symbol,file)
        if len(resolved)!=1:
            return {'ok':False,'error_type':'ambiguous_symbol' if resolved else 'symbol_not_found','candidates':resolved}
        definition=resolved[0]; q=definition['qualified_name']; name=definition['name']
        allowed={'exact','import_resolved'} if not include_uncertain else {'exact','import_resolved','local_name','attribute_unresolved','dynamic'}
        cr=self._fetchall('SELECT path,caller_name,caller_qualified_name,line,expression,target_name,resolution_kind FROM calls WHERE target_name=? ORDER BY path,line LIMIT 100',(name,))
        callers=[{'path':r[0],'symbol':r[2] or r[1],'call_line':r[3],'expression':r[4],'resolution_kind':r[6]} for r in cr if r[6] in allowed][:max_callers]
        er=self._fetchall('SELECT path,caller_name,caller_qualified_name,line,expression,target_name,resolution_kind FROM calls WHERE path=? AND caller_qualified_name=? ORDER BY line LIMIT 100',(definition['path'],q))
        callees=[{'path':r[0],'caller_symbol':r[2] or r[1],'call_line':r[3],'expression':r[4],'symbol':r[5],'resolution_kind':r[6]} for r in er if r[6] in allowed][:max_callees]
        # Resolve callee definition uniquely where possible.
        for c in callees:
            rs=self.resolve_symbol(c['symbol'])
            if len(rs)==1:c.update({'definition_path':rs[0]['path'],'start_line':rs[0]['start_line'],'end_line':rs[0]['end_line']})
        remaining=max(0,int(max_source_chars))
        if include_source:
            src=self._source(definition['path'],definition['start_line'],definition['end_line'],remaining); definition=dict(definition,source_range=[definition['start_line'],definition['end_line']],source_code=src); remaining=max(0,remaining-len(src))
            for c in callers:
                rs=self.resolve_symbol(c['symbol'],c['path']); rr=rs[0] if len(rs)==1 else None
                if rr and remaining:
                    src=self._source(rr['path'],rr['start_line'],rr['end_line'],remaining); c.update(source_range=[rr['start_line'],rr['end_line']],source_code=src); remaining=max(0,remaining-len(src))
            for c in callees:
                if c.get('start_line') and remaining:
                    src=self._source(c['definition_path'],c['start_line'],c['end_line'],remaining); c.update(source_range=[c['start_line'],c['end_line']],source_code=src); remaining=max(0,remaining-len(src))
        return {'ok':True,'definition':definition,'callers':callers,'callees':callees,'include_source':bool(include_source),'source_chars_used':max_source_chars-remaining}

    def close(self):
        with self._lock:self.conn.close()
