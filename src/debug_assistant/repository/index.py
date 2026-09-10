from __future__ import annotations
from pathlib import Path
import ast, re, sqlite3, time, threading
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
            # Issue text is arbitrary user/provider input. Passing whitespace
            # chunks directly to FTS5 makes punctuation such as ``:``/``(``
            # become query syntax and can invalidate the whole MATCH clause.
            # Keep the existing OR-style lexical retrieval, but build a safe
            # token query instead of falling back to an impossible full-text
            # LIKE match for a multi-line issue.
            raw_terms = re.findall(r"[\w]+", str(query), flags=re.UNICODE)
            terms = []
            seen = set()
            for term in raw_terms:
                key = term.casefold()
                if len(term) < 2 or key in seen:
                    continue
                seen.add(key)
                terms.append(term)
            # Very long stack traces can contain thousands of unique tokens.
            # Prefer longer, identifier-like terms while retaining original
            # order; this keeps the query bounded and preserves useful names.
            if len(terms) > 64:
                selected = sorted(
                    enumerate(terms), key=lambda item: (-len(item[1]), item[0]),
                )[:64]
                terms = [term for _, term in sorted(selected)]
            q = " OR ".join(f'"{term}"' for term in terms)
            try:
                if q:
                    rows=self._fetchall('SELECT path, snippet(files_fts,1,"[","]"," … ",18), bm25(files_fts) FROM files_fts WHERE files_fts MATCH ? ORDER BY bm25(files_fts) LIMIT ?',(q,limit))
            except sqlite3.OperationalError:rows=[]
        if not rows:
            like=f'%{query}%'; rows=[(*r,0.0) for r in self._fetchall('SELECT path, substr(content,1,800) FROM files_fts WHERE content LIKE ? OR path LIKE ? LIMIT ?',(like,like,limit))]
        return [{'path':r[0],'snippet':r[1],'score':float(r[2]),'source':'lexical'} for r in rows]

    def symbols(self,query,limit=60):
        like=f'%{query}%'; rows=self._fetchall('SELECT path,name,qualified_name,kind,start_line,end_line FROM symbols WHERE name LIKE ? OR qualified_name LIKE ? ORDER BY length(name),path LIMIT ?',(like,like,min(int(limit),100)))
        return [{'path':r[0],'name':r[1],'qualified_name':r[2],'kind':r[3],'start_line':r[4],'end_line':r[5]} for r in rows]

    def _symbols_in_files(self, files, *, limit_per_file=100):
        """Return indexed Python declarations restricted to candidate files.

        This deliberately does not use the global ``symbols()`` result as a
        candidate generator.  A refinement pass must inspect the files already
        recalled by Hybrid; a repository-wide symbol hit is not a new retrieval
        candidate.
        """
        paths=tuple(dict.fromkeys(str(path) for path in files if str(path)))
        if not paths:
            return []
        placeholders=','.join('?' for _ in paths)
        rows=self._fetchall(
            f'SELECT path,name,qualified_name,kind,start_line,end_line '
            f'FROM symbols WHERE path IN ({placeholders}) '
            f'ORDER BY path,length(name),start_line',
            paths,
        )
        grouped={path:[] for path in paths}
        for row in rows:
            if len(grouped[row[0]]) < max(1,int(limit_per_file)):
                grouped[row[0]].append({
                    'path':row[0], 'name':row[1], 'qualified_name':row[2],
                    'kind':row[3], 'start_line':row[4], 'end_line':row[5],
                })
        return [item for path in paths for item in grouped[path]]

    def resolve_symbol(self,symbol:str,file:str|None=None):
        rows=self.symbols(symbol,100)
        exact=[r for r in rows if symbol in {r['name'],r['qualified_name']} or r['qualified_name'].endswith('.'+symbol)]
        if file:exact=[r for r in exact if r['path']==file]
        uniq={(r['path'],r['qualified_name'],r['start_line'],r['end_line']):r for r in exact}
        return list(uniq.values())

    def refine_hybrid_candidates(self, rows, query, *, limit=20,
                                 max_symbols_per_file=8,
                                 max_callers=3, max_callees=3):
        """Refine Hybrid files with bounded, existing Python AST evidence.

        The input rows are the complete candidate boundary.  This method never
        adds a file returned by a global symbol lookup.  It enriches each
        candidate with declaration ranges and, when a symbol resolves uniquely,
        direct caller/callee metadata from the existing call table.  Source is
        intentionally not read here; runtime verification must use ``read_file``.
        """
        started=time.monotonic()
        candidate_rows=[]; seen=set()
        for row in list(rows or [])[:max(1,int(limit))]:
            path=str(row.get('path') or '')
            if not path or path in seen:
                continue
            seen.add(path)
            candidate_rows.append(dict(row))
        if not candidate_rows:
            return [], {
                'ast_available':False, 'ast_status':'unavailable',
                'candidate_count':0, 'matched_symbol_count':0,
                'supported_candidate_count':0, 'unsupported_candidate_count':0,
                'relations_used':0, 'elapsed_ms':int((time.monotonic()-started)*1000),
            }

        supported_paths={
            row['path'] for row in candidate_rows
            if str(row['path']).casefold().endswith('.py')
        }
        unsupported_count=len(candidate_rows)-len(supported_paths)
        if not supported_paths:
            return candidate_rows[:max(1,int(limit))], {
                'ast_available':False, 'ast_status':'unsupported_language',
                'candidate_count':len(candidate_rows), 'matched_symbol_count':0,
                'supported_candidate_count':0,
                'unsupported_candidate_count':unsupported_count,
                'relations_used':0, 'elapsed_ms':int((time.monotonic()-started)*1000),
            }

        # The index is Python-AST based.  If it has no declarations, callers
        # receive the unchanged Hybrid result with an explicit degradation state.
        if not self._fetchall('SELECT 1 FROM symbols LIMIT 1'):
            return candidate_rows[:max(1,int(limit))], {
                'ast_available':False, 'ast_status':'unavailable',
                'candidate_count':len(candidate_rows), 'matched_symbol_count':0,
                'supported_candidate_count':len(supported_paths),
                'unsupported_candidate_count':unsupported_count,
                'relations_used':0, 'elapsed_ms':int((time.monotonic()-started)*1000),
            }

        tokens=[]; token_set=set()
        for token in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b',str(query)):
            key=token.casefold()
            if key not in token_set:
                token_set.add(key); tokens.append(key)
            if len(tokens)>=64:
                break
        symbols=self._symbols_in_files(
            [row['path'] for row in candidate_rows],
            limit_per_file=max_symbols_per_file*4,
        )
        by_path={}
        for symbol in symbols:
            name=str(symbol.get('name') or '').casefold()
            qualified=str(symbol.get('qualified_name') or '').casefold()
            components={part for part in qualified.split('.') if part}
            exact_name=name in token_set
            qualified_component=bool(components & token_set)
            qualified_query=bool('.' in qualified and qualified in str(query).casefold())
            if not (exact_name or qualified_component):
                continue
            strength=0.65 if exact_name else 0.45
            if qualified_query:
                strength+=0.15
            item=dict(symbol)
            item['_match_strength']=min(1.0,strength)
            by_path.setdefault(symbol['path'],[]).append(item)

        total_matches=0; relation_count=0
        for row in candidate_rows:
            matched=sorted(
                by_path.get(row['path'],[]),
                key=lambda item:(-float(item['_match_strength']),
                                 len(str(item.get('qualified_name') or '')),
                                 int(item.get('start_line') or 0)),
            )[:max(1,int(max_symbols_per_file))]
            public=[]
            for symbol in matched:
                query_symbol=str(symbol.get('qualified_name') or symbol.get('name') or '')
                relation={}
                try:
                    relation=self.inspect_symbol_context(
                        query_symbol, row['path'], include_source=False,
                        max_callers=max_callers, max_callees=max_callees,
                    )
                except Exception:
                    relation={}
                callers=list(relation.get('callers') or []) if relation.get('ok') else []
                callees=list(relation.get('callees') or []) if relation.get('ok') else []
                relation_total=len(callers)+len(callees)
                relation_count+=relation_total
                structural=min(
                    1.0,
                    float(symbol['_match_strength'])
                    + min(0.20, 0.05*relation_total)
                    + (0.10 if len(matched)>1 else 0.0),
                )
                public.append({
                    'symbol':query_symbol,
                    'name':symbol.get('name'),
                    'qualified_name':symbol.get('qualified_name'),
                    'kind':symbol.get('kind'),
                    'file':symbol.get('path'),
                    'source_range':[int(symbol['start_line']),int(symbol['end_line'])],
                    'start_line':int(symbol['start_line']),
                    'end_line':int(symbol['end_line']),
                    'callers':callers,
                    'callees':callees,
                    'structural_relevance':structural,
                })
            total_matches+=len(public)
            row['matched_symbols']=public
            row['source_ranges']=[item['source_range'] for item in public]
            row['structural_relevance']=max(
                (item['structural_relevance'] for item in public), default=0.0,
            )
            row['caller_count']=sum(len(item['callers']) for item in public)
            row['callee_count']=sum(len(item['callees']) for item in public)

        if total_matches:
            # Calibrate the structural bonus to the largest existing Hybrid
            # score.  RRF scores are intentionally small; an absolute bonus
            # would otherwise let one weak symbol outrank a strong Hybrid hit.
            # A lone exact declaration is useful location metadata, but is not
            # enough structural evidence to change the Hybrid order.  Call
            # relations or multiple matches must provide the extra signal.
            hybrid_scores=[float(row.get('score') or 0.0) for row in candidate_rows]
            reference=max(hybrid_scores,default=0.0)
            if reference<=0.0:
                reference=1.0
                hybrid_scores=[1.0/(index+1) for index in range(len(candidate_rows))]
            for index,row in enumerate(candidate_rows):
                base=float(row.get('score') or 0.0)
                if base<=0.0:
                    base=hybrid_scores[index]
                relevance=float(row.get('structural_relevance') or 0.0)
                calibrated=max(0.0,min(1.0,(relevance-0.65)/0.35))
                row['score']=base + reference*0.10*calibrated
                if row.get('structural_relevance'):
                    row['source']=f"{row.get('source','hybrid')}+ast_refinement"
            candidate_rows.sort(
                key=lambda row:(-float(row.get('score') or 0.0),str(row.get('path') or '')),
            )
            status='applied'
        else:
            status='available_no_match'
        for row in candidate_rows:
            row.pop('_match_strength',None)
        return candidate_rows[:max(1,int(limit))], {
            'ast_available':True, 'ast_status':status,
            'candidate_count':len(candidate_rows),
            'matched_symbol_count':total_matches,
            'supported_candidate_count':len(supported_paths),
            'unsupported_candidate_count':unsupported_count,
            'relations_used':relation_count,
            'elapsed_ms':int((time.monotonic()-started)*1000),
        }

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
