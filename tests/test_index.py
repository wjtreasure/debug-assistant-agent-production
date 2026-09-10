from pathlib import Path
from debug_assistant.repository.index import RepositoryIndex

def test_task_index(tmp_path):
    repo=tmp_path/'repo'; (repo/'pkg').mkdir(parents=True); (repo/'pkg/a.py').write_text('def alpha_value(x):\n    return x + 1\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); stats=idx.build()
    assert stats['files']==1
    assert any(x['name']=='alpha_value' for x in idx.symbols('alpha'))
    assert any(x['path']=='pkg/a.py' for x in idx.search('alpha_value'))
    idx.close()


def test_lexical_search_accepts_punctuated_multiline_issue_text(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'schema.py').write_text(
        "def _invoke_field_validators(data):\n"
        "    return data['value']\n"
    )
    idx = RepositoryIndex(repo, tmp_path / 'idx.sqlite')
    idx.build()
    issue = "TypeError: 'NoneType' object is not subscriptable\n"
    issue += "Failure at _invoke_field_validators(data=result)."
    assert any(row['path'] == 'schema.py' for row in idx.search(issue))
    idx.close()
