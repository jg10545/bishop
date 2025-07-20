import ast

def get_user_validation(code):
    """
    Function to pause experiment until a human signs off on the code. Returns
    True if user hits "enter" and raises an exception otherwise.

    :code: string
    """
    print(code)
    response = input("Hit enter to approve code; otherwise leave a message explaining disapproval")
    if len(response.strip()) == 0:
        return True
    else:
        raise Exception(response)
    

def no_imports_check(code):
    if "import" in code.lower():
        raise Exception("no import statements allowed!")
    else:
        return True
    
def validate_code(code: str) -> bool:
    # 1. Syntax Validation
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        raise Exception(f"unable to compile: {e}")

    # 2. AST Analysis
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            raise Exception("disallowed nodes in parse tree")

    # 3. Content Filtering
    disallowed_modules = ['os', 'sys', 'subprocess']
    for module in disallowed_modules:
        if f"import {module}" in code or f"from {module} import" in code or "f{module}." in code:
            raise Exception("importing 'os', 'sys', and 'subprocess' not allowed")

    return True

def code_checker(code:str, human_in_loop:bool=False):
    """
    Input a string containing some code. Return "pass" or "fail" depending on
    whether the code passes our checks.
    """
    result = "pass"
    if not code.strip().startswith("def"):
        result = "fail: should start with a function definition"
    elif not code.split("\n")[-1].strip().startswith("return"):
        result = "fail: function needs to return something!"
    elif "#" not in code:
        result = "fail: please comment your code"
    elif '"""' not in code:
        result = "fail: please include a docstring"
    elif "import" in code:
        result = "fail: imports not permitted"
    elif ("os." in code)|("sys." in code)|("subprocess" in code):
        result = "fail: os, sys, and subprocess calls not permitted"
    elif "eval(" in code:
        result = "fail: eval() not permitted"
    elif "exec(" in code:
        result = "fail: exec() not permited"
    if human_in_loop & (result == "pass"):
        print(code)
        answer = input("Press enter if this code is OK; otherwise explain the problem:")
        if len(answer.strip()) == 0:
            result = f"fail: {answer}"
    print(result)
    return result
