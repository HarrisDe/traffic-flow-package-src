# traffic_flow/inference/data_cleaner.py
import numpy as np
import pandas as pd
from ..preprocessing.cleaning import (
    clean_and_cast,
    filter_and_interpolate_extremes,
    smooth_speeds,
)

class InferenceDataCleaner:
    def __init__(self, state: dict, sensor_col="sensor_id", value_col="value"):
        self.thr        = state["relative_threshold"]
        self.window     = state["smoothing_window"]
        self.use_median = state["use_median"]
        self.sensor_col = sensor_col
        self.value_col  = value_col

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = clean_and_cast(df_raw, value_col=self.value_col)
        df = filter_and_interpolate_extremes(
            df,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            threshold=self.thr,
        )
        df = smooth_speeds(
            df,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            window_size=self.window,
            use_median=self.use_median,
        )
        return df