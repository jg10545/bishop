import dspy
import mlflow
import logging
import json

from typing import Union, Callable

from ._analyst import get_analyst
from ._planner import get_planner
from ._implementer import get_implementer  
from ._scrub import code_checker
from ._mlflow import get_runs_as_json


MLFLOW_PARAM_TOKEN_LIMIT = 6000

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
            "tags.status":"status"
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
            if hypothesis is None:
                # pull history
                history = get_runs_as_json(self.experiment_name, self.mlflow_column_mapping, round_to=3)
                # analyst inputs background, goes through history and reports and conclusions it can draw
                # MAYBE implement that later
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
                code = implementation.code
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
