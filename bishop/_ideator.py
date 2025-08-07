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
    You are a curious and rigorous AI scientist, and have been asked to provide constructive feedback
    on your colleague's research. Be critical but fair, so that your feedback will help your colleague
    improve their work.

    Input the background/context for the research program and the history of our previous experiments;
    for each, a hypothesis, a summary of analysis after the experiment, and possibly comments from your
    supervisor. Provide detailed feedback for your colleague's current hypothesis.
    """
    background:str = dspy.InputField()
    history:str = dspy.InputField()
    hypothesis:str = dspy.InputField()
    
    novelty:str = dspy.OutputField(desc="How creative or novel is the hypothesis? Is it sufficiently different from the previous ones?")
    detail:str = dspy.OutputField(desc="Is the hypothesis detailed enough that we could implement it?")
    alignment:str = dspy.OutputField(desc="How well does the hypothesis address the patterns seen in analysis of previous results?")
    practicality:str = dspy.OutputField(desc="Does the hypothesis conform to the limitations laid out in the background?")
    feedback:str = dspy.OutputField(desc="Any other feedback or criticism on the hypothesis. Did your colleague give a high-level overview, an equation if relevant, and explain why they're suggesting this approach?")

class ReActIdeatorSig(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, currently planning the next phase of your research.

    Input the background/context for the research program and the history of our previous experiments;
    for each, a hypothesis, a summary of analysis after the experiment, and possibly comments from your
    supervisor. Formulate a clear hypothesis informed by this previous work. Try to be as creative as 
    possible, to push our research in bold new directions. Avoid small changes like adjusting 
    hyperparameters or weights, and focus instead on fundamentally different approaches. The next step 
    will be to implement a strategy to test the hypotheses so make sure it's clear, detailed, and 
    testable!
    
    When you've formulated your hypothesis, submit it to your colleague for constructive criticism. Use that
    feedback to iterate on your hypothesis until it's good enough to test!
    """
    background:str = dspy.InputField()
    history:str = dspy.InputField()
    hypothesis:str = dspy.OutputField(desc="Final hypothesis, including high-level summary, equation if relevant, and explanation of why you're proposing this solution")





class ReActIdeator(dspy.Module):
    """
    
    """
    def __init__(self, max_iters=5, verbose=False):
        """
        
        """
        self.verbose=verbose
        self.critic = dspy.ChainOfThought(CriticSig)
        self.ideator = dspy.ReAct(ReActIdeatorSig, tools=[self._get_criticism], max_iters=max_iters)

    def _get_criticism(self, hypothesis):
        if self.verbose:
            print("hypothesis:", hypothesis)
        criticism = self.critic(background=self.background, history=self._history, hypothesis=hypothesis).feedback
        if self.verbose:
            print("criticism:", criticism)
        return criticism
    
    def forward(self, background, history):
        self.background = background
        self._history = history
        result = self.ideator(background=background, history=history)
        return result