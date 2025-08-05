import dspy
import mlflow
import logging
import json

from dspy.utils.usage_tracker import track_usage

from typing import Union, Callable

from ._ideator import IdeatorSig
from ._analyst import Analyst
from ._planner import PlannerSig
from ._coder import Coder
from ._mlflow import get_runs_as_json

MLFLOW_PARAM_TOKEN_LIMIT = 6000

PRICING = {
    "gpt4.1":[2., 8.],
    "kimi_k2":[0.15, 2.5],
    "Llama4_Maverick":[0.18, 0.6], # Lambda labs pricing
    "DeepSeek_R1":[0.5,2.18], # Lambda labs pricing
}


class Laboratory(dspy.Module):
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
    """
    def __init__(self, lm, experiment_fn, experiment_name, metric_name, prompts, human_in_loop:True, verbose=False):
        """
        :lm: dspy.LM object; the language model used by the agents in this experiment
        :experiment_fn: python function that handles all the details of running the actual experiment.
            * It should input a string containing the LLM-written python function for this run
            * It should output a dictionary containing the output metric and "df", a pandas dataframe of results to send
                to the analyst agent
        :experiment_name: string; name of the mlflow experiment
        :metric_name: string; name of the performance metric to be maximized/minimized
        :prompts: dictionary of contextual prompts for the different agents. By default this should include things like
            the background for the experiment and constraints for the python function
        :human_in_loop: bool; if True, require human review before running the machine-generated experiment code
        :verbose: bool; if True, print stuff out at every stage.
        """
        self.lm = lm
        self.model = lm.model
        self.experiment_name = experiment_name
        self.metric_name = metric_name
        self.prompts = prompts
        self.experiment_fn = experiment_fn
        self.human_in_loop = human_in_loop
        self.verbose = verbose

        # set up all our agents
        self.agents = {}
        self.usage = {}
        self.setup()

    def setup(self):
        """
        Initialize all the agents and any other metadata we'll need. To customize the lab workflow, subclass 
        Laboratory and overwrite this function and run_one_experiment(). Add each agents to the self.agents 
        dictionary.

        This is also a good place to check the prompts the user passes to make sure the right stuff is included.
        """
        # create each agent we'll need
        self.agents["ideator"] = dspy.ChainOfThought(IdeatorSig)
        self.agents["planner"] = dspy.ChainOfThought(PlannerSig)
        self.agents["coder"] = Coder(human_in_loop=self.human_in_loop, verbose=self.verbose)
        self.agents["analyst"] = Analyst(verbose=self.verbose)
        # identify the columns we'll need from mlflow to report on the history of the experiments
        self.mlflow_column_mapping = {
            "params.planner.final_hypothesis":"hypothesis",
            "params.planner.title":"title",
            f"metrics.{self.metric_name}":f"{self.metric_name}",
            "params.analyst.answer":"analysis",
            "tags.status":"status",
            "tags.comment":"comment"
        }
        for k in ["background", "analysis_question", "function_name", "constraints"]:
            assert k in self.prompts, f"Missing prompt {k}"

        description = f"""
        # Background
        {self.prompts["background"]}

        # Analysis Question
        {self.prompts["result_analysis_question"]}

        # Implementation Constraints
        {self.prompts["implementation_constraints"]}
        """
        mlflow.set_experiment_tag("mlflow.note.content", self._get_description())


    def run_one_experiment(self, **kwargs):
        """
        Run one full experiment. To customize the lab workflow, subclass Laboratory and overwrite this
        function and setup().
        """
        p = self.prompts
        outdict = {}
        # review previous work and generate ideas as a list of hypotheses
        history = json.dumps(get_runs_as_json(self.experiment_name, self.mlflow_column_mapping))
        # select a hypothesis from the ideas and generate a plan to test it
        if "plan" not in kwargs:
            ideas = self._call_agent("ideator", background=p["background"],
                                    history=history
                                    )
            outdict["hypotheses"] = ideas.hypotheses
            if self.verbose:
                print(ideas.hypotheses)
            plan = self._call_agent("planner", background=p["background"],
                                    history=history,
                                    hypotheses=ideas.hypotheses,
                                    constraints=p["constraints"])
            
            if self.verbose:
                print(plan.title)
                print(plan.final_hypothesis)
                print(plan.plan)
        else:
            plan = {"plan":kwargs["plan"]}
            mlflow.log_param("planner.plan", kwargs["plan"])
        for k in plan.keys():
            outdict[k] = plan[k]

        # implement plan as python code
        if "code" not in kwargs:
            code = self._call_agent("coder", background=p["background"],
                                    plan=plan["plan"], 
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
        outdict["analysis"] = analysis.answer
        return outdict


    def _call_agent(self, name, **kwargs):
        """
        Wrapper function for calling an agent; handles some additional logging and stuff
        """
        # run inputs through the agent
        with track_usage() as usage_tracker:
            outputs = self.agents[name](**kwargs)
        self.usage[name] = usage_tracker.get_total_tokens()
        # log every output to MLflow
        for k in outputs.keys():
            if k != "trajectory":
                self.log_param(f"{name}.{k}", outputs[k])
        return outputs

    def _log_usage(self):
        """
        Log total token usage as mlflow metrics and usage broken out by agent as an artifact.
        Also make some estimates of what it would cost to run this through several hosted APIs.
        """
        completion_tokens = 0
        prompt_tokens = 0
        for agent in self.usage:
            for k in self.usage[agent]:
                completion_tokens += self.usage[agent][k]['completion_tokens']
                prompt_tokens += self.usage[agent][k]['prompt_tokens']
        
        mlflow.log_metric("completion_tokens", completion_tokens)
        mlflow.log_metric("prompt_tokens", prompt_tokens)
        mlflow.log_dict(self.usage, "ml_usage.yaml")
        for p in PRICING:
            cost = PRICING[p][0]*prompt_tokens/1e6 + PRICING[p][1]*completion_tokens/1e6
            mlflow.log_metric(f"cost_estimate_{p}", cost)


    def forward(self, **kwargs):
        with mlflow.start_run():
            mlflow.set_tag("status", "incomplete")
            mlflow.set_tag("comment", "none")
            try:
                outputs = self.run_one_experiment(**kwargs)
                mlflow.set_tag("status", "complete")
            except Exception as e:
                mlflow.set_tag("status", "error")
                mlflow.log_param("error_msg", e)
                assert False, e

            
            self._log_usage()
        return dspy.Prediction(**outputs)

    def log_param(self, key, value):
        """
        Wrapper function to manage logging (possibly long) text responses to mlflow
        """
        if not isinstance(value, str):
            value = str(value)
        if len(value) > MLFLOW_PARAM_TOKEN_LIMIT:
            logging.warning(f"parameter {key} is above the max token limit for MLFlow. Recording only the first {MLFLOW_PARAM_TOKEN_LIMIT} characters.")
        mlflow.log_param(key, value[:MLFLOW_PARAM_TOKEN_LIMIT])

