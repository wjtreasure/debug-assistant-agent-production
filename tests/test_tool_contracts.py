from debug_assistant.tools.registry import ToolRegistry


def test_all_tools_have_typed_schema(tmp_path):
    tools=ToolRegistry(tmp_path)
    specs=tools.specs()
    assert specs
    assert all(s.args_model is not None for s in specs)
    assert all(isinstance(s.json_schema(),dict) for s in specs)


def test_render_exposes_read_file_schema(tmp_path):
    text=ToolRegistry(tmp_path).render()
    assert 'read_file' in text
    assert 'start_line' in text
    assert 'end_line' in text
    assert 'cost=light' in text


def test_read_file_output_is_bounded(tmp_path):
    p=tmp_path/'a.py'
    p.write_text('\n'.join(str(i) for i in range(1,401)))
    tool=ToolRegistry(tmp_path).get('read_file')
    obs=tool.execute(path='a.py',start_line=1,end_line=200)
    assert obs.ok
    assert obs.metadata['end_line']==200
    assert len(obs.content.splitlines())==200
