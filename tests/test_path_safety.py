from debug_assistant.tools.registry import ToolRegistry

def test_path_escape_blocked(tmp_path):
    t=ToolRegistry(tmp_path).get('read_file')
    o=t.execute(path='../secret')
    assert not o.ok
