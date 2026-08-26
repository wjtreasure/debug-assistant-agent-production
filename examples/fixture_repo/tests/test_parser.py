from src.parser import parse_value

def test_upper_boundary():
    assert parse_value("9") == 9
