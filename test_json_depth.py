def _validate_raw_json_depth(content: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > 128:
                raise ValueError(f"judge response JSON nesting exceeds maximum depth of 128")
        elif char in "]}":
            depth -= 1
    return depth

print(_validate_raw_json_depth("]} { "* 130))
