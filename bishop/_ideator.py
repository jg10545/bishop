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


class _CriticSig(dspy.Signature):
    """
    You are a lead scientist at a top research institution, and have been asked to provide feedback on
    your colleague's research. Your criticism should be harsh but thorough and fair, to make sure your
    colleague catches any weak points in his analysis early on. 

    Use the background for the research program and history of experiments run so far to provide context
    for your criticism of your colleague's idea for his next step.
    """
    background:str = dspy.InputField()
    history:str = dspy.InputField()
    idea:str = dspy.InputField()
    
    novelty:str = dspy.OutputField(desc="How creative or novel is the idea? Is it sufficiently different from the previous ones?")
    detail:str = dspy.OutputField(desc="Is the idea detailed enough that we could implement it?")
    alignment:str = dspy.OutputField(desc="How well does the idea address the patterns seen in analysis of previous results?")
    practicality:str = dspy.OutputField(desc="Does the idea conform to the limitations laid out in the background?")
    feedback:str = dspy.OutputField(desc="Any other feedback or criticism on the idea. Did your colleague give a high-level overview, an equation if relevant, and explain why they're suggesting this approach?")


class CriticSig(dspy.Signature):
    """
    You are a lead scientist at a top research institution, and have been asked to provide feedback on
    your colleague's research. Your criticism should be harsh but thorough and fair, to make sure your
    colleague catches any weak points in his analysis early on. 

    Use the background for the research program and history of experiments run so far to provide context
    for your criticism. If the idea is too similar to a previous one, make your colleague differentiate them
    more. If there are issues in previous analysis not covered by the idea, call them out specifically! If
    the idea is too vague or makes unsubstantiated claims, make your colleague show their work!

    Remember that you are giving critical feedback on an idea BEFORE we test it experimentally! ONLY
    critique the idea 
    """
    background:str = dspy.InputField()
    history:str = dspy.InputField()
    idea:str = dspy.InputField()
    
    novelty:str = dspy.OutputField(desc="How creative or novel is the idea? Is it sufficiently different from the previous ones? Call out specific experiments that your colleague may need to differentiate from.")
    detail:str = dspy.OutputField(desc="Is the idea detailed enough that we could implement it?")
    alignment:str = dspy.OutputField(desc="How well does the idea address the patterns seen in analysis of previous results? Call out specific previous experiments and analysis.")
    feedback:str = dspy.OutputField()


class ReActIdeatorSig(dspy.Signature):
    """
    You are scientist at a top research instution and are currently planning the next experiment
    for your program.

    Using the background for your research and history of previous experiments as context, generate a 
    novel and idea for the next experiment. Try to be as creative as possible, to push our
    research in bold new directions. Avoid small changes like adjusting hyperparameters or weights, 
    and focus instead on fundamentally different approaches informed by what we've tried so far. The next 
    step will be to implement this idea so make sure it's clear, detailed, and testable!
    
    When you've formulated your idea, submit it to your colleague for constructive criticism. Use that
    feedback to iterate on the idea until it's ready to test!
    """
    background:str = dspy.InputField()
    history:str = dspy.InputField()
    idea_title:str = dspy.OutputField(desc="Concise but descriptive name for the idea, to easily differentiate it from others.")
    idea_summary:str = dspy.OutputField(desc="Short technical explanation of the idea; 2-3 sentences max plus an equation if necessary")
    idea_explanation:str = dspy.OutputField(desc="Detailed explanation of the idea, including SPECIFICS on how it intends to address analysis of previous ideas")
    #idea:str = dspy.OutputField(desc="Final idea, including high-level summary, equation if relevant, and explanation of why you're proposing this solution")





class ReActIdeator(dspy.Module):
    """
    dspy Module that attempts to generate better hypotheses by simulating a conversation
    between an "ideator" agent and a "critic" agent.
    """
    def __init__(self, max_iters:int=5, verbose:bool=False):
        """
        :max_iters: int; maximum number of times to iterate between ideator and critic
        :verbose: bool; whether to print out the interactions as they happen
        """
        self.counter = 0
        self.verbose=verbose
        self.critic = dspy.ChainOfThought(CriticSig)
        self.ideator = dspy.ReAct(ReActIdeatorSig, tools=[self._get_criticism], max_iters=max_iters)

    def _get_criticism(self, idea):
        if self.verbose:
            print(f"({self.counter}) idea:", idea)
        criticism = self.critic(background=self.background, history=self._history, idea=idea).feedback
        if self.verbose:
            print(f"(self.counter) criticism:", criticism)
        self.counter += 1
        return criticism
    
    def forward(self, background, history):
        self.counter = 0
        self.background = background
        self._history = history
        result = self.ideator(background=background, history=history)
        return result