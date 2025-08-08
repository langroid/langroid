import pandas as pd


class SharedData:
    df_results = pd.DataFrame
    df: pd.DataFrame | str
    task_type: str  # "regression" or "classification"
    metric_name: str
    score: float

    def __init__(self, df: pd.DataFrame | str):
        self.df = df
