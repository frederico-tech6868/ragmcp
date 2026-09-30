import json
import pytest

from src.tool_parser import parse_tool_call

def test_parse_tool_call_valid_json():
    input_json = '{"tool": "example", "action": "test"}'
    result = parse_tool_call(input_json)
    assert result == {"tool": "example", "action": "test"}

def test_parse_tool_call_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call('invalid json')