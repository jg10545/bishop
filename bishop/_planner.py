import dspy



class PlannerSig(dspy.Signature):
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


def get_planner():
    """
    Return a dspy ChainOfThought agent to plan the next phase of the experiment.
    """
    return dspy.ChainOfThought(PlannerSig)