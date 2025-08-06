import pytest
from bishop._scrub import validate_code, code_checker


safe_code = """
def foobar(x):
    \"\"\"
    foobar
    \"\"\"
    # return sigmoid of the input
    return torch.nn.functional.sigmoid(x)
"""

unsafe_code_with_import_and_os = """
def foobar(x):
    \"\"\"
    foobar
    \"\"\"
    # return sigmoid of the input
    import os
    return os.listdir()
"""


unsafe_code_with_eval = """
def foobar(x):
    \"\"\"
    foobar
    \"\"\"
    # return sigmoid of the input
    eval("print('i was able to eval something!')")
    return "foo"
"""



def check_validate_safe_code():
    code = """
def foobar(x):
    return torch.nn.functional.sigmoid(x)
"""
    assert validate_code(code)

def check_validate_throws_error_on_unsafe_code():
    code = """
def foobar(x):
    import os
    return torch.nn.functional.sigmoid(x)
"""
    with pytest.raises(Exception):
        _ = validate_code(code)

def check_validate_throws_error_on_unsafe_code_2():
    code = """
def foobar(x):
    directory_listing = os.listdir()
    return torch.nn.functional.sigmoid(x)
"""
    with pytest.raises(Exception):
        _ = validate_code(code)



def test_code_checker_safe_code():
    assert code_checker(safe_code) == "pass"


def test_code_checker_unsafe_code_with_import():
    response = code_checker(unsafe_code_with_import_and_os)
    assert response != "pass"
    assert "os" in response



def test_code_checker_unsafe_code_with_eval():
    response = code_checker(unsafe_code_with_eval)
    assert response != "pass"
    assert "eval" in response
