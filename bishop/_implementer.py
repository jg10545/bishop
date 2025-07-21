import dspy

from ._scrub import code_checker

class ImplementerSig(dspy.Signature):
    """
    Input a hypothesis for the next experiment, and implement it as a Python function using Pytorch. The
    torch library is already loaded so you don't need to import anything. Just return the function as a string,
    with comments but no markdown or anything.

    Iterate on the code until it passes code_checker().
    """
    background:str = dspy.InputField()
    hypothesis:str = dspy.InputField()
    function_name:str = dspy.InputField()
    code:str = dspy.OutputField()


def get_implementer(max_iters=25):
    """
    NOTE: if you require a human in the loop, run final code scrub on Implementer output.

    :human_in_loop: bool; if True require human authentication to continue
    """
    return dspy.ReAct(ImplementerSig, tools=[code_checker], max_iters=max_iters)