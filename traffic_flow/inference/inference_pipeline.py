# traffic_flow/inference/pipeline.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List

# shared helpers
from ..preprocessing.cleaning import (
    clean_and_cast,
    filter_and_interpolate_extremes,
    smooth_speeds,
)

# feature transformers (all already have from_state)
from ..features.sensor_encoder       import OrdinalSensorEncoder, MeanSensorEncoder
from ..features.calendar_features    import DateTimeFeatureEngineer
from ..features.adjacent_features    import AdjacentSensorFeatureAdder
from ..features.temporal_features         import TemporalLagFeatureAdder
from ..features.congestion_threshold  import PerSensorCongestionFlagger
from ..features.congestion_outlier_features  import GlobalOutlierFlagger
from ..features.historical_reference_features import PreviousWeekdayWindowFeatureEngineer
from ..features.misc_features      import WeatherFeatureDropper
from ..preprocessing.dtypes import enforce_dtypes

# class TrafficInferencePipeline:
#     """
#     Stateless, *read-only* version of TrafficDataPipelineOrchestrator.
#     Construct it with the dict returned by `export_states()` and call
#     `.transform(raw_df)` to obtain the model-ready feature frame.
#     """

#     # --------------------------------------------------------------
#     def __init__(self, states: Dict[str, Any]) -> None:
#         self.states = states

#         # rebuild feature transformers -----------------------------
#         sen_state  = states["sensor_encoder_state"]
#         if sen_state["type"] == "ordinal":
#             self.sensor_enc = OrdinalSensorEncoder.from_state(sen_state)
#         else:
#             self.sensor_enc = MeanSensorEncoder.from_state(sen_state)

#         self.dt_fe       = DateTimeFeatureEngineer.from_state(states["datetime_state"])
#         self.adj_fe      = AdjacentSensorFeatureAdder.from_state(states["adjacency_state"])
#         self.lag_fe      = TemporalLagFeatureAdder.from_state(states["lag_state"])
#         self.cong_flag   = PerSensorCongestionFlagger.from_state(states["congestion_state"])
#         self.outlier_flag= GlobalOutlierFlagger.from_state(states["outlier_state"])
#         self.weather_drop= WeatherFeatureDropper.from_state(states["weather_state"])
#         self.prev_day_fe = PreviousWeekdayWindowFeatureEngineer.from_state(states["previous_day_state"])

#         # meta ------------------------------------------------------
#         self.clean_cfg    = states["clean_state"]
#         self.expected_cols: List[str] = states["feature_cols"]

#     # --------------------------------------------------------------
#     def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
#         """
#         Parameters
#         ----------
#         df_raw : pd.DataFrame
#             New data in the SAME schema as the original raw parquet
#             (sensor_id, date, value, …).

#         Returns
#         -------
#         df_feats : pd.DataFrame
#             Exactly the columns the XGB model expects (order preserved).
#         """
#         cfg = self.clean_cfg

#         # ---------- 1. replicate cleaning -------------------------
#         df = clean_and_cast(df_raw, value_col="value")
#         df = filter_and_interpolate_extremes(
#                 df,
#                 sensor_col="sensor_id",
#                 value_col="value",
#                 threshold=cfg["relative_threshold"],
#         )
#         df = smooth_speeds(
#                 df,
#                 sensor_col="sensor_id",
#                 value_col="value",
#                 window_size=cfg["smoothing_window"],
#                 use_median=cfg["use_median"],
#         )

#         # ---------- 2. feature stack (same order as training) -----
#         df = self.sensor_enc.transform(df)
#         df = self.dt_fe.transform(df)
#         df = self.adj_fe.transform(df)
#         df = self.lag_fe.transform(df)
#         df = self.cong_flag.transform(df)
#         df = self.outlier_flag.transform(df)
#         df = self.weather_drop.transform(df)
#         if self.prev_day_fe is not None:
#             df = self.prev_day_fe.transform(df)

#         # ---------- 3. final column alignment ---------------------
#         # Reindex guarantees *order* and drops any accidental extras
#         df_final = df.reindex(columns=self.expected_cols, copy=False)

#         # Safety check in case upstream data is missing sensors, etc.
#         missing = df_final.columns[df_final.isna().all()]
#         if len(missing):
#             df_final[missing] = 0.0          # or any sentinel you prefer

#         return self._canonical_sort(df_final)
    
    
#     # -----------------------------------------------------------
    # def _canonical_sort(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Match the sort order used during training."""
    #     return (
    #         df.sort_values([self.states["datetime_state"]["datetime_col"],
    #                         "sensor_uid"])          # after encoding
    #         .reset_index(drop=True)
    #      )
        
class TrafficInferencePipeline:
    """
    Stateless, read-only pipeline built from the dict returned by
    TrafficDataPipelineOrchestrator.export_states().

    Parameters
    ----------
    states : dict
        Output of export_states().
    keep_datetime : bool, default False
        If True, the datetime column will be re-attached to the output for
        debugging/equality checks (even if it was dropped during training).
    """

    def __init__(self, states: Dict[str, Any], keep_datetime: bool = False) -> None:
        self.states = states
        self.keep_datetime = keep_datetime

        # --- rebuild transformers ---------------------------------
        sen_state = states["sensor_encoder_state"]
        if sen_state["type"] == "ordinal":
            self.sensor_enc = OrdinalSensorEncoder.from_state(sen_state)
        else:
            self.sensor_enc = MeanSensorEncoder.from_state(sen_state)

        self.dt_fe    = DateTimeFeatureEngineer.from_state(states["datetime_state"])
        self.adj_fe   = AdjacentSensorFeatureAdder.from_state(states["adjacency_state"])
        self.lag_fe   = TemporalLagFeatureAdder.from_state(states["lag_state"])
        self.cong_flag    = PerSensorCongestionFlagger.from_state(states["congestion_state"])
        self.outlier_flag = GlobalOutlierFlagger.from_state(states["outlier_state"])
        self.weather_drop = WeatherFeatureDropper.from_state(states["weather_state"])

        # optional features may be absent depending on training config
        self.prev_day_fe: Optional[PreviousWeekdayWindowFeatureEngineer] = None
        if states.get("previous_day_state"):
            self.prev_day_fe = PreviousWeekdayWindowFeatureEngineer.from_state(
                states["previous_day_state"]
            )

        # --- meta --------------------------------------------------
        self.clean_cfg: Dict[str, Any] = states["clean_state"]
        self.expected_cols: List[str]   = states["feature_cols"]
        self.dtype_schema: Dict[str, str] = states.get("dtype_schema", {})
        self.dt_col: str = states["datetime_state"]["datetime_col"]

    # --------------------------------------------------------------
    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data (same schema as training raw) into the
        model-ready feature matrix with the exact column order and dtypes.
        """
        cfg = self.clean_cfg

        # ---------- 1) replicate cleaning --------------------------
        df = clean_and_cast(df_raw, value_col="value")
        df = filter_and_interpolate_extremes(
            df, sensor_col="sensor_id", value_col="value",
            threshold=cfg["relative_threshold"],
        )
        df = smooth_speeds(
            df, sensor_col="sensor_id", value_col="value",
            window_size=cfg["smoothing_window"], use_median=cfg["use_median"],
        )

        # ---------- 2) feature stack (same order as training) ------
        df = self.sensor_enc.transform(df)
        df = self.dt_fe.transform(df)
        df = self.adj_fe.transform(df)
        df = self.lag_fe.transform(df)
        df = self.cong_flag.transform(df)
        df = self.outlier_flag.transform(df)
        df = self.weather_drop.transform(df)
        if self.prev_day_fe is not None:
            df = self.prev_day_fe.transform(df)

        # ---------- 2.5) canonical row order -----------------------
        # Match the training sort order (by datetime then encoded sensor).
        # Sort BEFORE column reindex so we still have dt column available.
        sort_cols = [c for c in (self.dt_col, "sensor_uid") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        # ---------- 3) final column alignment & dtypes -------------
        df_final = df.reindex(columns=self.expected_cols, copy=False)

        # Apply the dtype schema saved at training-time
        if self.dtype_schema:
            df_final = enforce_dtypes(df_final, self.dtype_schema)

        # Optionally re-attach datetime column for debugging/equality checks
        if self.keep_datetime and self.dt_col not in df_final.columns and self.dt_col in df.columns:
            # put it first for convenience
            df_final.insert(0, self.dt_col, df[self.dt_col].values)

        # Safety: fill any all-null columns (shouldn't happen normally)
        missing = df_final.columns[df_final.isna().all()]
        if len(missing):
            df_final[missing] = 0

        return df_final

#

# class TrafficInferencePipeline:
#     """
#     Stateless, read-only replica of *TrafficDataPipelineOrchestrator*.

#     Parameters
#     ----------
#     states : dict
#         The dictionary returned by `TrafficDataPipelineOrchestrator.export_states()`.
#     keep_datetime : bool, default ``False``
#         If *True* the original datetime column is kept in the returned frame
#         (handy for debugging / A–B checks).  
#         If *False* it is dropped so the output contains **exactly** the columns
#         used by the model.
#     """

#     # ----------------------------------------------------------
#     def __init__(self, states: Dict[str, Any], *, keep_datetime: bool = False) -> None:
#         self.states        = states
#         self.keep_datetime = keep_datetime

#         # ── rebuild transformers ───────────────────────────────
#         sen_state = states["sensor_encoder_state"]
#         self.sensor_enc = (
#             OrdinalSensorEncoder.from_state(sen_state)
#             if sen_state["type"] == "ordinal"
#             else MeanSensorEncoder.from_state(sen_state)
#         )

#         self.dt_fe        = DateTimeFeatureEngineer.from_state(states["datetime_state"])
#         self.adj_fe       = AdjacentSensorFeatureAdder.from_state(states["adjacency_state"])
#         self.lag_fe       = TemporalLagFeatureAdder.from_state(states["lag_state"])
#         self.cong_flag    = PerSensorCongestionFlagger.from_state(states["congestion_state"])
#         self.outlier_flag = GlobalOutlierFlagger.from_state(states["outlier_state"])
#         self.weather_drop = WeatherFeatureDropper.from_state(states["weather_state"])
#         self.prev_day_fe  = PreviousWeekdayWindowFeatureEngineer.from_state(
#             states["previous_day_state"]
#         )

#         # ── meta ───────────────────────────────────────────────
#         self.clean_cfg      = states["clean_state"]
#         self.expected_cols: List[str] = states["feature_cols"]
#         self.dt_col         = self.dt_fe.datetime_col

#     # ----------------------------------------------------------
#     def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
#         cfg = self.clean_cfg

#         # 1. replicate cleaning --------------------------------
#         df = clean_and_cast(df_raw, value_col="value")
#         df = filter_and_interpolate_extremes(
#             df,
#             sensor_col="sensor_id",
#             value_col="value",
#             threshold=cfg["relative_threshold"],
#         )
#         df = smooth_speeds(
#             df,
#             sensor_col="sensor_id",
#             value_col="value",
#             window_size=cfg["smoothing_window"],
#             use_median=cfg["use_median"],
#         )

#         # 2. feature stack (same order as training) ------------
#         df = self.sensor_enc.transform(df)
#         df = self.dt_fe.transform(df)
#         df = self.adj_fe.transform(df)
#         df = self.lag_fe.transform(df)
#         df = self.cong_flag.transform(df)
#         df = self.outlier_flag.transform(df)
#         df = self.weather_drop.transform(df)
#         if self.prev_day_fe is not None:
#             df = self.prev_day_fe.transform(df)

#         # 3. canonical sort (match training order) -------------
#         df = self._canonical_sort(df)

#         # 4. drop / keep datetime ------------------------------
#         dt_series = None
#         if not self.keep_datetime and self.dt_col in df.columns:
#             df = df.drop(columns=[self.dt_col])
#         elif self.keep_datetime:
#             # stash a copy because reindex will discard it
#             dt_series = df[self.dt_col]

#         # 5. definitive column order ---------------------------
#         df_final = df.reindex(columns=self.expected_cols, copy=False)

#         # optionally put datetime back as the first column
#         if self.keep_datetime:
#             df_final.insert(0, self.dt_col, dt_series)

#         # fill any all-NaN columns (e.g. unseen sensors)
#         missing = df_final.columns[df_final.isna().all()]
#         if len(missing):
#             df_final[missing] = 0.0

#         return df_final

#     # ----------------------------------------------------------
#     def _canonical_sort(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Impose the `(datetime, sensor_uid)` order used during training."""
#         return (
#             df.sort_values([self.dt_col, "sensor_uid"])
#               .reset_index(drop=True)
#         )