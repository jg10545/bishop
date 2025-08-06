import dspy
import typing



class PlannerSigDEPRECATED(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, specializing in data analysis. It is your
    ethical and professional duty to pose difficult questions and challenge assumptions.

    Input the background/context for our research and the history of our previous experiments- for each,
    a hypothesis and a summary of our analysis after testing the hypothesis. Formulate a new
    hypothesis, using what you've learned from this history! Try to be as creative as possible, to push our
    research in bold new directions.

    The next step will be to implement a strategy to test the hypothesis so make sure it's clear, detailed,
    and testable!
    """
    background:str = dspy.InputField(desc="The context and goal of the research project")
    history:str = dspy.InputField(desc="Overview of what we've tried so far, possibly including feedback from your supervisor")
    hypothesis:str = dspy.OutputField(desc="Hypothesis to motivate the next experiment")


class PlannerSig(dspy.Signature):
    """
    You are a senior researcher overseeing a laboratory, helping the junior scientists plan out
    the next phase of their work.

    Input the background for the research, history of previous experiments and their results, and a list
    of potential hypotheses to test out next. Choose a hypothesis to follow up on, and write a more detailed
    plan for your proteges to follow.
    """
    background:str = dspy.InputField(desc="The context and goal of the research project")
    history:str = dspy.InputField(desc="Overview of what we've tried so far, possibly including feedback from your supervisor")
    hypotheses:typing.List[str] = dspy.InputField(desc="Your colleagues' hypothesis")
    constraints:str = dspy.InputField(desc="Implementation constraints the plan must adhere to")
    final_hypothesis:str = dspy.OutputField(desc="Your chosen hypothesis to follow up on next")
    title:str = dspy.OutputField(desc="A clear, succinct title for this experiment")
    plan:str = dspy.OutputField(desc="A detailed plan for your lab to follow when implementing this test")