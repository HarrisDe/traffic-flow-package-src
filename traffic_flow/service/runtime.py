from __future__ import annotations
import joblib
from typing import Dict, Any, Tuple, Optional
from ..inference.inference_pipeline import TrafficInferencePipeline
from ..inference.prediction_protocol import make_prediction_frame
from ..utils.helper_utils import LoggingMixin
import numpy as np
import pandas as pd

from ..inference.inference_pipeline import TrafficInferencePipeline
from ..inference.prediction_protocol import make_prediction_frame
from ..utils.helper_utils import LoggingMixin  # adjust import to your project

class InferenceRuntime(LoggingMixin):
    """
    Load a saved artifact (model + states), rebuild the inference pipeline,
    and provide a predict_df() that matches the training/offline canonical
    output, with an option to trim warm-up rows for lag features.

    Canonical prediction frame columns:
      - sensor_id (or whatever schema_state names it)
      - input_time
      - prediction_time
      - y_pred_delta
      - horizon
      - y_pred_total  (when add_total=True)
    """

    def __init__(self, artifact_path: str, *, keep_datetime: bool = False) -> None:
        super().__init__(disable_logs=False)

        bundle: Dict[str, Any] = joblib.load(artifact_path)
        self.states: Dict[str, Any]       = bundle["states"]
        self.model                        = bundle["model"]
        self.horizon: int                 = int(bundle.get("horizon", 15))
        self.feature_cols: list[str]      = bundle["feature_cols"]

        schema = self.states.get("schema_state", {})
        self.sensor_col: str   = schema.get("sensor_col", "sensor_id")
        self.datetime_col: str = schema.get("datetime_col",
                                           self.states["datetime_state"]["datetime_col"])
        self.value_col: str    = schema.get("value_col", "value")

        # Rebuild inference pipeline from states
        self.pipeline = TrafficInferencePipeline(self.states, keep_datetime=keep_datetime)

    # ------------------------------- public API ------------------------------- #

    def predict_df(
        self,
        df_raw: pd.DataFrame,
        *,
        df_for_ML: Optional[pd.DataFrame] = None,
        trim_warmup: bool = False,
        add_total: bool = True,
        add_y_act: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform raw rows -> features -> model predictions.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw rows with at least (datetime_col, sensor_col, value_col).
        trim_warmup : bool, default True
            If True, drop the first `lag_steps` rows per sensor *before*
            feature generation to avoid undefined lag values (no warm-up buffer).
            This mirrors the notebook parity check you ran.
        add_total : bool, default True
            If True, include y_pred_total = y_pred_delta + current value.

        Returns
        -------
        pred_df : pd.DataFrame
            Canonical prediction frame (see class docstring).
        feats : pd.DataFrame
            Engineered features aligned to `pred_df`.
        """
        if add_y_act and df_for_ML is None:
            raise ValueError("`df_for_ML` must be provided when `add_y_act=True` to access actual target values.")
        
        # 1) deterministic sort
        if (self.datetime_col in df_raw.columns) and (self.sensor_col in df_raw.columns):
            df_sorted = df_raw.sort_values(
                [self.datetime_col, self.sensor_col],
                kind="mergesort"
            ).reset_index(drop=True)
        else:
            raise ValueError(
                f"df_raw must contain columns {self.datetime_col!r} and {self.sensor_col!r}"
            )

        if (df_for_ML is not None) and (self.datetime_col in df_for_ML.columns) and (self.sensor_col in df_for_ML.columns):
            df_for_ML = df_for_ML.sort_values(
                [self.datetime_col, self.sensor_col],
                kind="mergesort"
            ).reset_index(drop=True)
        else:
            raise ValueError(
                f"df_for_ML must must not be none and contain columns {self.datetime_col!r} and {self.sensor_col!r}. Use TrafficDataPipelinOrchestrator to create it."
            )
        # 2) optional warm-up trim 
        if trim_warmup:
            df_sorted = self._trim_warmup_rows(df_sorted)

        # 3) features -> model
        feats = self.pipeline.transform(df_sorted).reindex(columns=self.feature_cols, copy=False)
        pred_delta = self.model.predict(feats)

        # 4) canonical prediction frame (includes timestamp math & optional totals)
        pred_df = make_prediction_frame(
            df_raw=df_sorted,
            df_for_ML= df_for_ML,
            feats=feats,
            pred_delta=pred_delta,
            states=self.states,
            horizon_min=self.horizon,
            add_total=add_total,
            add_y_act=add_y_act,

        )
        return pred_df, feats

    def predict_total_series(self, df_raw: pd.DataFrame, *, trim_warmup: bool = True) -> pd.Series:
        """
        Convenience: return only y_pred_total as a Series (aligned to features).
        """
        pred_df, _ = self.predict_df(df_raw, trim_warmup=trim_warmup, add_total=True)
        return pred_df["y_pred_total"]
    
    
    

    # ------------------------------ internals -------------------------------- #

    def _trim_warmup_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the first `lag_steps` rows per sensor so inference matches an
        offline pipeline that also discards those rows (no warm-up buffer).

        Notes
        -----
        - `lag_steps` comes from the saved state (states['lag_state']['lags']).
        - Requires the frame to be sorted by (datetime_col, sensor_col).
        - Returns a new frame; does not mutate the input.
        """
        lag_steps = int(self.states.get("lag_state", {}).get("lags", 0))
        if lag_steps <= 0:
            return df.copy()

        g = df.groupby(self.sensor_col, sort=False)
        rank = g.cumcount()
        kept = rank >= lag_steps
        return df.loc[kept].reset_index(drop=True)

class InferenceRuntime_orig(LoggingMixin):
    """
    Loads the artifact (model + states), rebuilds the inference pipeline,
    and exposes predict_df() that returns (delta_predictions, engineered_features).
    """
    def __init__(self, artifact_path: str,
                 disable_logs: bool = False,
                 keep_datetime: bool = False) -> None:
        super().__init__(disable_logs=disable_logs)
        bundle: Dict[str, Any] = joblib.load(artifact_path)
        self.states       = bundle["states"]
        self.model        = bundle["model"]
        self.horizon      = int(bundle.get("horizon", 15))
        self.feature_cols = bundle['states']["feature_cols"]
        schema = self.states.get("schema_state", {})
        self.sensor_col   = schema.get("sensor_col", "sensor_id")
        self.datetime_col = schema.get("datetime_col", self.states["datetime_state"]["datetime_col"])
        self.df_raw = None
        self.feats = None
        self.feats_num = None
        self.df_all = None
        self.keep_datetime = keep_datetime

        # Rebuild inference pipeline from states; model trained without datetime
        self.pipeline = TrafficInferencePipeline(self.states, keep_datetime=self.keep_datetime, disable_logs=disable_logs)

    def predict_df(self, df_raw: pd.DataFrame, trim_warmup: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Transform raw rows -> features -> model predictions.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw rows with at least (datetime_col, sensor_col, value_col).
        trim_warmup : bool, default True
            If True, drop the first `lag_steps` rows per sensor *before*
            feature generation to avoid undefined lag values (no warm-up buffer).
            This mirrors the notebook parity check you ran.
        add_total : bool, default True
            If True, include y_pred_total = y_pred_delta + current value.

        Returns
        -------
        pred_df : pd.DataFrame
            Canonical prediction frame (see class docstring).
        feats : pd.DataFrame
            Engineered features aligned to `pred_df`.
        """
        
        if (self.datetime_col in df_raw.columns) and (self.sensor_col in df_raw.columns):
            print('Sorting values of df_raw by datetime and sensor')
            df_raw = df_raw.sort_values([self.datetime_col, self.sensor_col], kind="mergesort").reset_index(drop=True)
        self.df_raw = df_raw.copy()
        feats = self.pipeline.transform(self.df_raw.copy())
        self.feats_after_transform = feats.copy()
        if trim_warmup:
                feats = self._trim_warmup_rows(feats)
        self.feats = feats
        feats_num = feats.reindex(columns=self.feature_cols, copy=False)
        self.feats_num = feats_num
        pred_delta = self.model.predict(feats_num)
        self.df_all = feats.copy()
        self.df_all['y_pred_delta'] = pred_delta
        self.df_all['y_pred_total'] = pred_delta + feats['value']
        
        return pd.Series(pred_delta), feats
    
    def predict_df_total_speed(self, df_raw,trim_warmup: bool = False) -> pd.Series:
        """
        Convenience: return only y_pred_total as a Series (aligned to features).
        """
        pred_delta_speed,feats = self.predict_df(df_raw, trim_warmup=trim_warmup)
        #self.df_all['y_pred_total_speed'] = pred_delta_speed + feats['value']
        return pred_delta_speed + feats['value']
    
    

    
    def _trim_warmup_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the first `lag_steps` rows per sensor so inference matches an
        offline pipeline that also discards those rows (no warm-up buffer).

        Notes
        -----
        - `lag_steps` comes from the saved state (states['lag_state']['lags']).
        - Requires the frame to be sorted by (datetime_col, sensor_col).
        - Returns a new frame; does not mutate the input.
        """
        lag_steps = int(self.states.get("lag_state", {}).get("lags", 0))
        if lag_steps <= 0:
            return df.copy()
        
        self._log(f"[_trim_warmup_rows] Dropping lag steps from the DataFrame.Initial df shape: {df.shape}")
        g = df.groupby(self.sensor_col, sort=False)
        rank = g.cumcount()
        kept = rank >= lag_steps
        df = df.loc[kept].reset_index(drop=True)
        self._log(f"[_trim_warmup_rows]After dropping lag steps, df shape: {df.shape}")
        return df
    
    
