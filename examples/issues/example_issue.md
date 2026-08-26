Parser rejects a valid value at the upper boundary. Values from 0 through 9 should be accepted, but parse_value("9") raises ValueError. The regression appears to be in the numeric range validation.
