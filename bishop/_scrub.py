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