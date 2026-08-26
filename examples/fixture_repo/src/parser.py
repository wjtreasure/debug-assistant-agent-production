def parse_value(text: str) -> int:
    value = int(text)
    if value < 0 or value >= 9:  # bug fixture: upper boundary should allow 9
        raise ValueError("out of range")
    return value
