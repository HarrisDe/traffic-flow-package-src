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


class TrafficInferencePipeline:
    """
    Stateless, *read-only* version of TrafficDataPipelineOrchestrator.
    Construct it with the dict returned by `export_states()` and call
    `.transform(raw_df)` to obtain the model-ready feature frame.
    """

    # --------------------------------------------------------------
    def __init__(self, states: Dict[str, Any]) -> None:
        self.states = states

        # rebuild feature transformers -----------------------------
        sen_state  = states["sensor_encoder_state"]
        if sen_state["type"] == "ordinal":
            self.sensor_enc = OrdinalSensorEncoder.from_state(sen_state)
        else:
            self.sensor_enc = MeanSensorEncoder.from_state(sen_state)

        self.dt_fe       = DateTimeFeatureEngineer.from_state(states["datetime_state"])
        self.adj_fe      = AdjacentSensorFeatureAdder.from_state(states["adjacency_state"])
        self.lag_fe      = TemporalLagFeatureAdder.from_state(states["lag_state"])
        self.cong_flag   = PerSensorCongestionFlagger.from_state(states["congestion_state"])
        self.outlier_flag= GlobalOutlierFlagger.from_state(states["outlier_state"])
        self.weather_drop= WeatherFeatureDropper.from_state(states["weather_state"])
        self.prev_day_fe = PreviousWeekdayWindowFeatureEngineer.from_state(states["previous_day_state"])

        # meta ------------------------------------------------------
        self.clean_cfg    = states["clean_state"]
        self.expected_cols: List[str] = states["feature_cols"]

    # --------------------------------------------------------------
    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_raw : pd.DataFrame
            New data in the SAME schema as the original raw parquet
            (sensor_id, date, value, …).

        Returns
        -------
        df_feats : pd.DataFrame
            Exactly the columns the XGB model expects (order preserved).
        """
        cfg = self.clean_cfg

        # ---------- 1. replicate cleaning -------------------------
        df = clean_and_cast(df_raw, value_col="value")
        df = filter_and_interpolate_extremes(
                df,
                sensor_col="sensor_id",
                value_col="value",
                threshold=cfg["relative_threshold"],
        )
        df = smooth_speeds(
                df,
                sensor_col="sensor_id",
                value_col="value",
                window_size=cfg["smoothing_window"],
                use_median=cfg["use_median"],
        )

        # ---------- 2. feature stack (same order as training) -----
        df = self.sensor_enc.transform(df)
        df = self.dt_fe.transform(df)
        df = self.adj_fe.transform(df)
        df = self.lag_fe.transform(df)
        df = self.cong_flag.transform(df)
        df = self.outlier_flag.transform(df)
        df = self.weather_drop.transform(df)
        if self.prev_day_fe is not None:
            df = self.prev_day_fe.transform(df)

        # ---------- 3. final column alignment ---------------------
        # Reindex guarantees *order* and drops any accidental extras
        df_final = df.reindex(columns=self.expected_cols, copy=False)

        # Safety check in case upstream data is missing sensors, etc.
        missing = df_final.columns[df_final.isna().all()]
        if len(missing):
            df_final[missing] = 0.0          # or any sentinel you prefer

        return df_final