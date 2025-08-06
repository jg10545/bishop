import dspy
import warnings

from ._scrub import code_checker, _strip_markdown_from_code

class CoderSig(dspy.Signature):
    """
    Input background and a plan and implement the plan as a Python 
    function. Iterate on the code until it passes code_checker().

    As soon as your code receives a "pass" from code_checker(), return 
    it without any further modifications.
    """
    background:str = dspy.InputField()
    plan:str = dspy.InputField()
    constraints:str = dspy.InputField()
    function_name:str = dspy.InputField()
    code:str = dspy.OutputField()

class Coder(dspy.Module):
    """
    General-purpose coding agent
    """
    def __init__(self, max_iters:int=25, human_in_loop:bool=True,
                 verbose:bool=False):
        """
        :max_iters: max number of ReAct iterations to query dataset for analysis
        :human_in_loop: if True, pass to a human before marking complete
        :df: pandas DataFrame to use for analysis
        :verbose: if True, print out each stage of analysis
        """
        self.max_iters = max_iters
        self.human_in_loop = human_in_loop
        self.verbose = verbose
        self.react = dspy.ReAct(CoderSig, tools=[self.validate_code], 
                      max_iters=max_iters)
        
    def validate_code(self, code:str) -> str:
        """
        Use a single line of pandas code to probe the dataset
        """
        if self.verbose:
            print(f"code: {code}")
        result = code_checker(code, human_in_loop=self.human_in_loop)
        if result == "pass":
            self._code_passed_check = code
        if self.verbose:
            print(f"code-checker result: {result}")
        return result
    
    def forward(self, background:str, plan:str, function_name:str, 
                constraints:str="None", **kwargs) -> dspy.Prediction:
        """
        write code and make sure it's OK to run
        """
        self._code_passed_check = False
        code = self.react(background=background,
                          plan=plan,
                          function_name=function_name,
                          constraints=constraints)
        if self._code_passed_check:
            if _strip_markdown_from_code(code.code) != self._code_passed_check:
                warnings.warn(f"why did the code change???\npassed check: {self._code_passed_check}\nreturned: {code.code}")
            return dspy.Prediction(code=self._code_passed_check)
        else:
            raise Exception(f"Coder failed to pass checks!\n{code.code}")
