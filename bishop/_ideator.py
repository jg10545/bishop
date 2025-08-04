import dspy
import typing

class IdeatorSig(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, specializing in data analysis. It is your
    ethical and professional duty to pose difficult questions and challenge assumptions.

    Input the background/context for our research and the history of our previous experiments- for each,
    a hypothesis and a summary of our analysis after testing the hypothesis. Formulate a set of new
    hypotheses that could potentially explain our data. Try to be as creative as possible, to push our
    research in bold new directions.

    The next step will be to implement a strategy to test one or more of these hypotheses so make sure 
    they're clear, detailed,and testable!
    """
    background:str = dspy.InputField(desc="The context and goal of the research project")
    history:str = dspy.InputField(desc="Overview of what we've tried so far, possibly including feedback from your supervisor")
    hypotheses:typing.List[str] = dspy.OutputField(desc="Hypotheses to motivate the next experiment")


class CriticSig(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, specializing in data analysis. It is your
    ethical and professional duty to pose difficult questions and challenge assumptions.

    You have been asked to review a colleague's proposals for his research project. Input the background
    for the research, history of previous experiments, and set of hypotheses that could be used to 
    motivate the next steps. For each hypothesis, provide detailed constructive criticism. If you see
    an opportunity to improve the hypothesis, provide the revised version. Finally, score the viability of 
    each revised hypothesis on a scale of 1-5.

    The next step will be to implement a strategy to test one or more of these hypotheses so make sure 
    they're clear, detailed,and testable!
    """
    background:str = dspy.InputField(desc="The context and goal of the research project")
    history:str = dspy.InputField(desc="Overview of what we've tried so far, possibly including feedback from your supervisor")
    hypotheses:typing.List[str] = dspy.InputField(desc="Your colleagues' hypothesis")
    criticism:typing.List[str] = dspy.OutputField(desc="Your criticism of each hypothesis")
    revised_hypotheses:typing.List[str] = dspy.OutputField(desc="Updated list of hypotheses, factoring in your constructive criticism")
    score:typing.List[int] = dspy.OutputField()
