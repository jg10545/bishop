import pandas as pd
import mlflow


def get_runs_as_json(experiment, mapping, round_to=None):
    """
    Query all the runs from an MLFlow experiment and return them as
    a JSON for in-context learning.

    :experiment: string; name of the experiment to pull from
    :mapping: dict where keys and values are strings; which columns to use from the MLFlow results
        and what to rename them
    :round_to: int or None; round numerical metrics to this many decimal places
    """
    def _round(x):
        if round_to is not None:
            if isinstance(x, float):
                return round(x, round_to)
        return x
    df = mlflow.search_runs(experiment_names=[experiment])
    output = []
    for e,r in df.iterrows():
        output.append(
            {mapping[k]:_round(r[k]) for k in mapping}
        )
    return output

def get_dataframe_from_mlflow_artifact(run_id=None, artifact_path=None):
    """
    Download a CSV file from an MLFlow artifact and return as a pandas DataFrame

    :run_id: string; ID of the run to pull from
    :
    """
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="eval_results.csv")
    eval_results = pd.read_csv(path)