# traffic_flow_package_src/features/gman_features.py
"""
Feature-engineering helper to merge horizon-specific GMAN predictions.
Compatible with Python 3.6.
"""
from typing import Optional, List
import numpy as np
import pandas as pd
from .base import FeatureTransformer

class GMANPredictionAdder(FeatureTransformer):
    """
    Merge one GMAN-prediction dataframe into the sensor dataframe.

    Parameters
    ----------
    df_gman
        GMAN output for a *single* horizon.
    sensor_col, datetime_col, prediction_col, prediction_date_col
        Column names.
    convert_to_delta
        If *True* (default) replace raw GMAN speed by the
        **delta = gman_speed - measured_speed**.
    keep_target_date
        Keep the ``gman_target_date`` column if it exists.

    Notes
    -----
    The class is deliberately *stateless*: it returns a **copy**
    of the input dataframe with the GMAN columns added.
    """

    def __init__(
        self,
        df_gman: pd.DataFrame,
        *,
        sensor_col: str = "sensor_id",
        datetime_col: str = "date",
        prediction_col: str = "gman_prediction",
        prediction_date_col: str = "gman_prediction_date",
        convert_to_delta: bool = True,
        keep_target_date: bool = True,
        drop_missing: bool = False
    ) -> None:
        self.df_gman = df_gman.copy()
        self.sensor_col = sensor_col
        self.datetime_col = datetime_col
        self.prediction_col = prediction_col
        self.prediction_date_col = prediction_date_col
        self.convert_to_delta = convert_to_delta
        self.keep_target_date = keep_target_date
        self.drop_missing = drop_missing

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
        """Return a *copy* of *df* with GMAN columns merged in."""
        base = df.copy()

        # Ensure comparable dtypes
        base[self.datetime_col] = pd.to_datetime(base[self.datetime_col])
        self.df_gman[self.prediction_date_col] = pd.to_datetime(
            self.df_gman[self.prediction_date_col]
        )

        # Trim GMAN to the base timeframe
        min_t, max_t = base[self.datetime_col].min(), base[self.datetime_col].max()
        gman_trim = self.df_gman[
            (self.df_gman[self.prediction_date_col] >= min_t)
            & (self.df_gman[self.prediction_date_col] <= max_t)
        ]

        cols: List[str] = [
            self.sensor_col,
            self.prediction_date_col,
            self.prediction_col,
        ]
        if self.keep_target_date and "gman_target_date" in gman_trim.columns:
            cols.append("gman_target_date")

        gman_trim = gman_trim[cols]

        # Merge on (timestamp, sensor)
        merged = pd.merge(
            base,
            gman_trim,
            how="left",
            left_on=[self.datetime_col, self.sensor_col],
            right_on=[self.prediction_date_col, self.sensor_col],
        )

        if not self.keep_target_date:
            merged.drop(columns=[self.prediction_date_col], inplace=True)

        # Drop rows without a GMAN prediction
        if self.drop_missing:
            merged = merged.dropna(subset=[self.prediction_col])

        # Convert to delta speed if requested
        if self.convert_to_delta:
            merged["{}_orig".format(self.prediction_col)] = merged[self.prediction_col]
            merged[self.prediction_col] = (
                merged[self.prediction_col] - merged[value_col]
            )

        return merged