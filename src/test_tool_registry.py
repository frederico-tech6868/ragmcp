import pytest
from src.tool_registry import verify_code_safety_isolated, secure_register_tool, DYNAMIC_TOOLS

def test_verify_code_safety_isolated_compiles():
    code = """
def add(a, b):
    return a + b
"""
    is_safe, feedback = verify_code_safety_isolated(code)
    assert is_safe
    assert "Verification Success" in feedback

def test_verify_code_safety_isolated_syntax_error():
    code = """
def add(a, b)
    return a + b
"""
    is_safe, feedback = verify_code_safety_isolated(code)
    assert not is_safe
    assert "Compilation Failed [SyntaxError]" in feedback

def test_secure_register_tool_registers_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # it writes ./dynamic_extensions
    code = """
def add(a, b):
    return a + b
"""
    result = secure_register_tool({"name": "add", "code": code})

    assert "hot-mapped successfully" in result
    assert DYNAMIC_TOOLS["add"](2, 3) == 5
    assert (tmp_path / "dynamic_extensions" / "ext_add.py").read_text() == code

def test_secure_register_tool_missing_name():
    args = {"code": "def add(a, b): return a + b"}
    result = secure_register_tool(args)
    assert result == "Registration Error: Parameters 'name' and 'code' are mandatory."


def test_secure_register_tool_missing_code():
    assert secure_register_tool({"name": "add"}).startswith("Registration Error")
