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


class Laboratory(dspy.Module):
    """
    For now assume we're using the same LLM for everything. Could potentially make that configurable in the
    future, but would require changing a few things.
    """
    def __init__(self, lm, experiment_fn, experiment_name, metric_name, prompts, human_in_loop:True, verbose=False):
        """
        """
        #dspy.configure(track_usage=True, lm=lm)
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
        """
        # create each agent we'll need
        self.agents["ideator"] = dspy.ChainOfThought(IdeatorSig)
        self.agents["planner"] = dspy.ChainOfThought(PlannerSig)
        self.agents["coder"] = Coder(human_in_loop=self.human_in_loop, verbose=self.verbose)
        self.agents["analyst"] = Analyst(verbose=self.verbose)
        # identify the columns we'll need from mlflow to report on the 
        self.mlflow_column_mapping = {
            #"params.hypothesis":"hypothesis",
            "params.planner.final_hypothesis":"hypothesis",
            "params.planner.title":"title",
            f"metrics.{self.metric_name}":f"{self.metric_name}",
            #"params.analysis":"analysis",
            "params.analyst.answer":"analysis",
            "tags.status":"status",
            "tags.comment":"comment"
        }
        for k in ["background", "analysis_question", "function_name", "constraints"]:
            assert k in self.prompts, f"Missing prompt {k}"


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
        print("analysis keys:", analysis.keys())
        outdict["analysis"] = analysis.answer
        return outdict


    def _call_agent(self, name, **kwargs):
        # run inputs through the agent
        with track_usage() as usage_tracker:
            outputs = self.agents[name](**kwargs)
        #outputs.set_lm_usage(usage_tracker.get_total_tokens())
        self.usage[name] = usage_tracker.get_total_tokens()
        # log every output to MLflow
        for k in outputs.keys():
            if k != "trajectory":
                self.log_param(f"{name}.{k}", outputs[k])
        # track usage statistics
        #self.usage[name] = outputs.get_lm_usage()
        return outputs

    def _log_usage(self):
        """
        Log total token usage as mlflow metrics and usage broken out by agent as an artifact.
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


class Experimenter(dspy.Module):
    """
    
    """

    def __init__(self, experiment_fn:Callable[str,dict], background:str, metric_name:str, 
                 result_analysis_question:str, implementation_constraints:str, 
                 mlflow_tracking_uri:str, experiment_name:str,
                 analyst_max_iters:int=25, 
                 implementer_max_iters:int=25, human_in_loop:bool=True, strict:bool=True,
                 function_name:str="test_fn"):
        """
        :experiment_fn: python function that inputs a string containing code and outputs a 
            dictionary of results. This is where the actual experiment happens!
        :background: string; prompt giving the background/context/goal of the project
        :metric_name: string; name of the success metric. Should appear in the experiment_fn 
            output.
        :result_analysis_question: string; prompt to pass an Analyst agent for reviewing the
            results of an experiment
        :implementation_constraints: string; prompt to give a Coder agent to constrain how it
            implements a function for the next experiment.
        :analyst_max_iters: int; number of times Analyst agent can query the dataset
        :implementer_max_iters: int; number of times Coder agent can try writing code for the 
            next experiment. Current run will end if unable to pass automated checks this many 
            times.
        :human_in_loop: bool; if True, require human review before running the Coder agent's
            code
        :strict: bool; if True, restrict the Analyst agent to only use a manually-whitelisted
            subset of the pandas API
        :function_name: string; name of the function Coder agent should write.
        """
        self.experiment_fn = experiment_fn
        self.background = background
        self.metric_name = metric_name
        self.result_analysis_question = result_analysis_question
        self.implementation_constraints = implementation_constraints
        self.experiment_name = experiment_name
        self.analyst_max_iters = analyst_max_iters
        self.implementer_max_iters = implementer_max_iters
        self.human_in_loop = human_in_loop
        self.strict = strict
        self.function_name = function_name

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.set_experiment_tag("mlflow.note.content", self._get_description())

        self.planner = get_planner()
        self.implementer = get_implementer(max_iters=implementer_max_iters)

        self.mlflow_column_mapping = {
            "params.hypothesis":"hypothesis",
            f"metrics.{metric_name}":f"{metric_name}",
            "params.analysis":"analysis",
            "tags.status":"status",
            "tags.comment":"comment"
        }

    def _get_description(self):
        return f"""
        # Background
        {self.background}

        # Analysis Question
        {self.result_analysis_question}

        # Implementation Constraints
        {self.implementation_constraints}
        """

    def log_param(self, key, value):
        if len(value) > MLFLOW_PARAM_TOKEN_LIMIT:
            logging.warning(f"parameter {key} is above the max token limit for MLFlow. Recording only the first {MLFLOW_PARAM_TOKEN_LIMIT} characters.")
        mlflow.log_param(key, value[:MLFLOW_PARAM_TOKEN_LIMIT])


    def forward(self, 
                hypothesis:Union[str,None]=None,
                code:Union[str,None]=None) -> dspy.Prediction:
        """
        
        """
        with mlflow.start_run():
            mlflow.set_tag("status", "incomplete")
            mlflow.set_tag("comment", "none")
            mlflow.log_param("model", dspy.settings.get("lm").model)
            if hypothesis is None:
                # pull history
                history = get_runs_as_json(self.experiment_name, self.mlflow_column_mapping, round_to=3)
                hyp = self.planner(background=self.background,
                                   history=json.dumps(history))
                self.log_param("hypothesis_cot", hyp.reasoning)
                hypothesis = hyp.hypothesis
            self.log_param("hypothesis", hypothesis)
            if code is None:
                # Implementer inputs background and hypothesis and writes code   
                implementation = self.implementer(background=self.background,
                                                  hypothesis=hypothesis,
                                                  constraints=self.implementation_constraints,
                                                  function_name=self.function_name)
                code = _strip_markdown_from_code(implementation.code)
                # If human-in-loop enabled, get permission to run AFTER Implementer is finished
                codecheck = code_checker(code, self.human_in_loop)
                if codecheck != "pass":
                    mlflow.set_tag("status", "fail")
                    raise Exception(f"generated code did not pass final check: {codecheck}")
            self.log_param("code", code)
            # run experiment loop with code
            try:
                results = self.experiment_fn(code)
            except Exception as e:
                mlflow.set_tag("status", "fail")
                mlflow.log_param("error_message", e)
                return False
            mlflow.log_metric(self.metric_name, results[self.metric_name])
            # pull detailed results from this run
            if "df" in results:
                q = f"{self.result_analysis_question}\nHypothesis: {hypothesis}"
                analyst = get_analyst(results["df"], strict=self.strict, max_iters=self.analyst_max_iters)
                result_analysis = analyst(question=q, 
                                          background=self.background, 
                                          description=results["df"].describe().to_markdown())
                self.log_param("analysis", result_analysis.answer)
            
            mlflow.set_tag("status", "complete")
        return dspy.Prediction(hypothesis=hypothesis, code=code, analysis=result_analysis.answer)
