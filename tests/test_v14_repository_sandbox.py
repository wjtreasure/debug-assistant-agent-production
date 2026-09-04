from pathlib import Path
import os
from debug_assistant.tools.repository import GrepTool, SymbolSearchTool
from debug_assistant.repository.index import RepositoryIndex

def test_grep_and_index_do_not_follow_outside_symlink(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); outside=tmp_path/'secret.py'; outside.write_text('ULTRA_SECRET_MARKER=1\n')
    link=repo/'leak.py'
    try: link.symlink_to(outside)
    except OSError: return
    g=GrepTool(repo).execute('ULTRA_SECRET_MARKER')
    assert g.ok and 'ULTRA_SECRET_MARKER' not in g.content
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build(); rows=idx.search('ULTRA_SECRET_MARKER')
    assert rows==[]; idx.close()
