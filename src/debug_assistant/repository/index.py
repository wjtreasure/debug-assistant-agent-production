from __future__ import annotations
from pathlib import Path
import ast, hashlib, os, sqlite3, time

IGNORED={'.git','.venv','venv','node_modules','dist','build','__pycache__','.tox','.mypy_cache','.pytest_cache'}
TEXT_SUFFIXES={'.py','.md','.rst','.txt','.toml','.yaml','.yml','.json','.ini','.cfg','.c','.h','.cc','.cpp','.hpp','.java','.js','.ts','.tsx','.go','.rs'}

class RepositoryIndex:
    """Lightweight task-scoped hybrid index: SQLite FTS5 for text + AST symbol table for Python."""
    def __init__(self,repo_root:Path,db_path:Path):
        self.repo_root=Path(repo_root).resolve(); self.db_path=Path(db_path); self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.db_path)
        self.fts=True
        try:
            self.conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(path, content, tokenize="unicode61")')
        except sqlite3.OperationalError:
            self.fts=False; self.conn.execute('CREATE TABLE IF NOT EXISTS files_fts(path TEXT PRIMARY KEY, content TEXT)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS symbols(path TEXT, name TEXT, kind TEXT, start_line INT, end_line INT)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)')

    def build(self,max_file_bytes=1_000_000):
        self.conn.execute('DELETE FROM files_fts'); self.conn.execute('DELETE FROM symbols')
        files=0; symbols=0; started=time.time()
        for p in self.repo_root.rglob('*'):
            if not p.is_file() or any(x in IGNORED for x in p.parts) or p.suffix.lower() not in TEXT_SUFFIXES: continue
            try:
                if p.stat().st_size>max_file_bytes: continue
                text=p.read_text(encoding='utf-8',errors='ignore'); rel=str(p.relative_to(self.repo_root)).replace('\\','/')
                self.conn.execute('INSERT INTO files_fts(path,content) VALUES (?,?)',(rel,text)); files+=1
                if p.suffix=='.py':
                    try: tree=ast.parse(text)
                    except Exception: tree=None
                    if tree:
                        for n in ast.walk(tree):
                            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                                self.conn.execute('INSERT INTO symbols VALUES (?,?,?,?,?)',(rel,n.name,type(n).__name__,n.lineno,getattr(n,'end_lineno',n.lineno))); symbols+=1
            except (OSError,UnicodeError): pass
        self.conn.commit(); return {'files':files,'symbols':symbols,'build_ms':(time.time()-started)*1000,'fts5':self.fts}

    def search(self,query,limit=40):
        limit=min(int(limit),100); rows=[]
        if self.fts:
            q=' OR '.join(x.replace('"','') for x in query.split() if x) or query
            try: rows=self.conn.execute('SELECT path, snippet(files_fts,1,"[","]"," … ",18), bm25(files_fts) FROM files_fts WHERE files_fts MATCH ? ORDER BY bm25(files_fts) LIMIT ?',(q,limit)).fetchall()
            except sqlite3.OperationalError: rows=[]
        if not rows:
            like=f"%{query}%"; rows=[(*r,0.0) for r in self.conn.execute('SELECT path, substr(content,1,800) FROM files_fts WHERE content LIKE ? OR path LIKE ? LIMIT ?',(like,like,limit)).fetchall()]
        return [{'path':r[0],'snippet':r[1],'score':r[2]} for r in rows]

    def symbols(self,query,limit=60):
        like=f"%{query}%"; rows=self.conn.execute('SELECT path,name,kind,start_line,end_line FROM symbols WHERE name LIKE ? ORDER BY length(name),path LIMIT ?',(like,min(int(limit),100))).fetchall()
        return [{'path':r[0],'name':r[1],'kind':r[2],'start_line':r[3],'end_line':r[4]} for r in rows]

    def close(self): self.conn.close()
