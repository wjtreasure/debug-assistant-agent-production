from pathlib import Path
from debug_assistant.repository.index import RepositoryIndex

def test_task_index(tmp_path):
    repo=tmp_path/'repo'; (repo/'pkg').mkdir(parents=True); (repo/'pkg/a.py').write_text('def alpha_value(x):\n    return x + 1\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); stats=idx.build()
    assert stats['files']==1
    assert any(x['name']=='alpha_value' for x in idx.symbols('alpha'))
    assert any(x['path']=='pkg/a.py' for x in idx.search('alpha_value'))
    idx.close()
