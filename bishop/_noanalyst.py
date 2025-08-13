import dspy
import mlflow
import json

from ._main import Laboratory
from ._coder import Coder
from ._analyst import Analyst
from ._ideator import AIScientistIdeatorSig
from ._mlflow import get_runs_as_json


class LaboratoryWithNoAnalyst(Laboratory):
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

    IN THIS VERSION: skip the Analyst agent; use the ideation agent from "AI Scientist"
    """

    def setup(self):
        """
        Initialize all the agents and any other metadata we'll need. To customize the lab workflow, subclass 
        Laboratory and overwrite this function and run_one_experiment(). Add each agents to the self.agents 
        dictionary.

        This is also a good place to check the prompts the user passes to make sure the right stuff is included.
        """
        # create each agent we'll need
        self.agents["ideator"] = dspy.Predict(AIScientistIdeatorSig)
        #self.agents["planner"] = dspy.ChainOfThought(PlannerSig)
        self.agents["coder"] = Coder(human_in_loop=self.human_in_loop, verbose=self.verbose)
        # identify the columns we'll need from mlflow to report on the history of the experiments
        self.mlflow_column_mapping = {
            "params.ideator.experiment":"experiment",
            "params.ideator.idea_title":"title",
            f"metrics.{self.metric_name}":f"{self.metric_name}",
            "tags.status":"status",
            "tags.comment":"comment"
        }
        for k in ["background", "function_name", "constraints"]:
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
        if "idea" not in kwargs:

            idea = self._call_agent("ideator", task_description=p["background"],
                                    previous_ideas=history
                                    )
            for k in idea.keys():
                outdict[f"idea_{k}"] = idea[k]
            if self.verbose:
                print("final idea:", idea)
        else:
            for k in ["title", "name", "experiment"]:
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
        return outdict