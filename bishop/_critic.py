import dspy
import mlflow
import json

from ._main import Laboratory
from ._coder import Coder
from ._analyst import Analyst
from ._ideator import ReActIdeator
from ._mlflow import get_runs_as_json


class LaboratoryWithIdeaCritic(Laboratory):
    """
    This class runs fully automated in silico research.

    You can configure the details of the research program using the experiment function and prompts. To change
    the workflow that the laboratory follows, subclass Laboratory and overwrite two functions:

    * setup() initializes all the agents and any other data structures you need
    * run_one_experiment() runs a single end-to-end experiment.

    For now assume we're using the same LLM for everything. Could potentially make that configurable in the
    future, but would require changing a few things.

    Before creating the Laboratory object:
    * initialize a dspy.LM object for the LLM you're using
    * call dspy.configure(lm=lm, track_usage=True)
    * configure MLFlow with mlflow.set_tracking_uri() and mlflow.set_experiment()

    IN THIS VERSION: skip the Planner agent; not sure that was really helping. But allow the Ideator to interact
    with a critic agent to refine its hypothesis.
    """

    def setup(self):
        """
        Initialize all the agents and any other metadata we'll need. To customize the lab workflow, subclass 
        Laboratory and overwrite this function and run_one_experiment(). Add each agents to the self.agents 
        dictionary.

        This is also a good place to check the prompts the user passes to make sure the right stuff is included.
        """
        # create each agent we'll need
        self.agents["ideator"] = ReActIdeator(verbose=self.verbose)
        #self.agents["planner"] = dspy.ChainOfThought(PlannerSig)
        self.agents["coder"] = Coder(human_in_loop=self.human_in_loop, verbose=self.verbose)
        self.agents["analyst"] = Analyst(verbose=self.verbose)
        # identify the columns we'll need from mlflow to report on the history of the experiments
        self.mlflow_column_mapping = {
            #"params.ideator.hypothesis":"hypothesis",
            #"params.planner.title":"title",
            "params.ideator.idea_title":"title",
            "params.ideator.idea_summary":"summary",
            f"metrics.{self.metric_name}":f"{self.metric_name}",
            #"params.analyst.answer":"analysis",
            "params.analyst.summary":"analysis",
            "tags.status":"status",
            "tags.comment":"comment"
        }
        for k in ["background", "analysis_question", "function_name", "constraints"]:
            assert k in self.prompts, f"Missing prompt {k}"

        description = f"""
        # Background
        {self.prompts["background"]}

        # Analysis Question
        {self.prompts["analysis_question"]}

        # Implementation Constraints
        {self.prompts["constraints"]}
        """
        mlflow.set_experiment_tag("mlflow.note.content", description)


    def run_one_experiment(self, **kwargs):
        """
        Run one full experiment. To customize the lab workflow, subclass Laboratory and overwrite this
        function and setup().

        :idea: should be a dictionary with keys "title" and "summary"
        """
        p = self.prompts
        outdict = {}
        # review previous work and generate ideas as a list of hypotheses
        # TRY SOMETHING DIFFERENT for this one: only return completed runs from history
        history = self._get_history(status="complete")
        #history = get_runs_as_json(self.experiment_name, self.mlflow_column_mapping)
        #history = [h for h in history if h["status"] == "complete"]
        #history = json.dumps(history)
        # select a hypothesis from the ideas and generate a plan to test it
        if "idea" not in kwargs:

            idea = self._call_agent("ideator", background=p["background"],
                                    history=history
                                    )
            #outdict["idea"] = idea.idea
            outdict["idea_title"] = idea.idea_title
            outdict["idea_summary"] = idea.idea_summary
            outdict["idea_explanation"] = idea.idea_explanation
            if self.verbose:
                print("final idea:", idea)
        else:
            for k in ["title", "summary"]:
                if k not in kwargs["idea"]:
                    assert False, f"missing key {k} from idea dictionary"
            #idea = {"idea":kwargs["idea"]}
            mlflow.log_params({"ideator.idea_"+k:kwargs["idea"][k] for k in kwargs["idea"]})
        # implement plan as python code
        if "code" not in kwargs:
            code = self._call_agent("coder", background=p["background"],
                                    plan=idea["idea_explanation"], 
                                    function_name=p["function_name"],
                                    constraints=p["constraints"]).code
        else:
            code = kwargs["code"]
            mlflow.log_param("coder.code", kwargs["code"])
        outdict["code"] = code
        # run the experiment
        results = self.experiment_fn(code)
        mlflow.log_metric(self.metric_name, results[self.metric_name])
        # write up an analysis of the results
        analysis = self._call_agent("analyst", df=results["df"],
                                        question=p["analysis_question"],
                                        background=p["background"])
        #outdict["analysis"] = analysis.answer
        outdict["analysis_report"] = analysis.report
        outdict["analysis_summary"] = analysis.summary
        return outdict