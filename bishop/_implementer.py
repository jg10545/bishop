import dspy

from ._scrub import code_checker

class ImplementerSig(dspy.Signature):
    """
    Input a hypothesis for the next experiment, and implement a test for it as a Python 
    function. Iterate on the code until it passes code_checker().
    """
    background:str = dspy.InputField()
    hypothesis:str = dspy.InputField()
    constraints:str = dspy.InputField()
    function_name:str = dspy.InputField()
    code:str = dspy.OutputField()


def get_implementer(max_iters=25):
    """
    NOTE: if you require a human in the loop, run final code scrub on Implementer output.

    :human_in_loop: bool; if True require human authentication to continue
    """
    return dspy.ReAct(ImplementerSig, tools=[code_checker], max_iters=max_iters)