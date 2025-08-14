import numpy as np
import pandas as pd
import mlflow


def get_runs_as_json(experiment, mapping, round_to=None, max_runs=25, **kwargs):
    """
    Query all the runs from an MLFlow experiment and return them as
    a JSON for in-context learning.

    :experiment: string; name of the experiment to pull from
    :mapping: dict where keys and values are strings; which columns to use from the MLFlow results
        and what to rename them
    :round_to: int or None; round numerical metrics to this many decimal places
    :max_runs: int; if more runs than this are returned, take a random sample of this size
    :kwargs: use to filter results

    """
    def _round(x):
        if round_to is not None:
            if isinstance(x, float):
                return round(x, round_to)
        return x
    df = mlflow.search_runs(experiment_names=[experiment])
    output = []
    for e,r in df.iterrows():
        if r.get('tags.mlflow.parentRunId', None) is None:
            output.append(
                {mapping[k]:_round(r.get(k,"None")) for k in mapping}
            )
    for k in kwargs:
            output = [o for o in output if o.get(k) == kwargs[k]]
            
    if len(output) > max_runs:
        output = np.random.choice(output, size=max_runs, replace=False).tolist()
    return output

def get_dataframe_from_mlflow_artifact(run_id=None, artifact_path=None):
    """
    Download a CSV file from an MLFlow artifact and return as a pandas DataFrame

    :run_id: string; ID of the run to pull from
    :
    """
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="eval_results.csv")
    eval_results = pd.read_csv(path)