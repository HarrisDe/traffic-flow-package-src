# traffic_flow/inference/pipeline.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional

# shared helpers
from ...preprocessing.cleaning import (
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
from ..features.calendar_cyclical_features import PredictionTimeCyclicalFeatureEngineer
from ..features.momentum_features import MomentumFeatureEngineer
from ..features.adjacent_features_congestion import AdjacentSensorFeatureAdderCongestion
from ..features.upstream_shifted_features import UpstreamTravelTimeShiftedFeatures
from ...preprocessing.dtypes import enforce_dtypes
from .prediction_protocol import make_prediction_frame
from ...utils.helper_utils import LoggingMixin
        
class TrafficInferencePipeline(LoggingMixin):
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

    def __init__(self, states: Dict[str, Any], 
                 keep_datetime: bool = False,
                 disable_logs: bool = False,) -> None:
        super().__init__(disable_logs=disable_logs)
        self.states = states
        self.keep_datetime = keep_datetime
        schema = states.get("schema_state", {})
        self.sensor_col   = schema.get("sensor_col", "sensor_id")
        self.datetime_col = schema.get("datetime_col", states["datetime_state"]["datetime_col"])
        self.value_col    = schema.get("value_col", "value")
        self.row_order    = schema.get("row_order", [self.datetime_col, self.sensor_col])
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
        # --- optional: adjacent congestion ---
        self.adj_cong: Optional[AdjacentSensorFeatureAdderCongestion] = None

        # prefer a consistent key name; support either for backward compat
        adj_cong_state = states.get("adjacent_congestion_state")
        if adj_cong_state:
            self.adj_cong = AdjacentSensorFeatureAdderCongestion.from_state(adj_cong_state)
        
        self.momentum_fe = None
        mom_state = states.get("momentum_state")
        if mom_state is not None:
            self.momentum_fe = MomentumFeatureEngineer.from_state(mom_state)
        self.pred_time_cyc_fe: Optional[PredictionTimeCyclicalFeatureEngineer] = None
        pt_state = states.get("prediction_time_cyc_state", None)
        if pt_state is not None:
            self.pred_time_cyc_fe = PredictionTimeCyclicalFeatureEngineer.from_state(pt_state)
        
        self.upstream_shifted_fe = None
        upstream_shifted_state = states.get("upstream_shifted_state") 
        if upstream_shifted_state is not None:
            self.upstream_shifted_fe = UpstreamTravelTimeShiftedFeatures.from_state(
                states["upstream_shifted_state"])

        # --- meta --------------------------------------------------
        self.clean_cfg: Dict[str, Any] = states["clean_state"]
        self.expected_cols: List[str]   = states["feature_cols"]
        self.dtype_schema: Dict[str, str] = states.get("dtype_schema", {})
  

    # --------------------------------------------------------------
    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data (same schema as training raw) into the
        model-ready feature matrix with the exact column order and dtypes.
        """
        
        # order = self.states.get("row_order", [self.dt_col, "sensor_id"])
        # present = [c for c in order if c in df.columns]
        # df = df.sort_values(present, kind="mergesort").reset_index(drop=True)
        
        cfg = self.clean_cfg
        

        # ---------- 1) replicate cleaning --------------------------
        df = clean_and_cast(df_raw, value_col=self.value_col)
        df = filter_and_interpolate_extremes(
            df, sensor_col=self.sensor_col, value_col=self.value_col,
            threshold=cfg["relative_threshold"],
        )

        if cfg.get("smooth_speeds", True):
            df = smooth_speeds(df, sensor_col=self.sensor_col, value_col=self.value_col,
                            window_size=cfg["smoothing_window"],
                            use_median=cfg["use_median"])
        
    
        
   
        present = [c for c in self.row_order if c in df.columns]
        if present:
            df = df.sort_values(present, kind="mergesort").reset_index(drop=True)

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
        if self.adj_cong is not None:
            df = self.adj_cong.transform(df)
        if self.momentum_fe is not None:
            df = df.sort_values(by=[self.sensor_col, self.datetime_col], kind="mergesort")
            df = self.momentum_fe.transform(df)
            df = df.sort_values(by=present, kind="mergesort").reset_index(drop=True)
        if self.upstream_shifted_fe is not None:
            df = df.sort_values(by=[self.sensor_col, self.datetime_col], kind="mergesort")
            df = self.upstream_shifted_fe.transform(df)
            df = df.sort_values(by=present, kind="mergesort").reset_index(drop=True)
            
        if self.pred_time_cyc_fe is not None:
            df = self.pred_time_cyc_fe.transform(df)

 

        # ---------- 3) final column alignment & dtypes -------------
        df_final = df.reindex(columns=self.expected_cols, copy=False)

        # Apply the dtype schema saved at training-time
        if self.dtype_schema:
            df_final = enforce_dtypes(df_final, self.dtype_schema)

        # Optionally re-attach datetime column for debugging/equality checks
        if self.keep_datetime and self.datetime_col not in df_final.columns and self.datetime_col in df.columns:
            # put it first for convenience
            df_final.insert(0, self.datetime_col, df[self.datetime_col].values)
            df_final.insert(1, self.sensor_col, df[self.sensor_col].values)
        
        if not self.keep_datetime and self.datetime_col in df_final.columns:
            # drop it if not needed
            to_drop = [c for c in (self.datetime_col, self.sensor_col) if c in df_final.columns]
            if to_drop:
                df_final = df_final.drop(columns=[self.datetime_col,self.sensor_col])

        # Safety: fill any all-null columns (shouldn't happen normally)
        missing = df_final.columns[df_final.isna().all()]
        if len(missing):
            print(f"Warning: {len(missing)} columns missing in final output, filling with 0s: {missing.tolist()}")
            df_final[missing] = 0

        return df_final
    
   

