# traffic_flow/service/runtime.py
from __future__ import annotations
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from ..inference.inference_pipeline import TrafficInferencePipeline
from ..inference.prediction_protocol import make_prediction_frame
class InferenceRuntime:
    """
    Loads the artifact (model + states), rebuilds the inference pipeline,
    and exposes predict_df() that returns (delta_predictions, engineered_features).
    """
    def __init__(self, artifact_path: str):
        bundle: Dict[str, Any] = joblib.load(artifact_path)
        self.states       = bundle["states"]
        self.model        = bundle["model"]
        self.horizon      = int(bundle.get("horizon", 15))
        self.feature_cols = bundle["feature_cols"]

        # Rebuild inference pipeline from states; model trained without datetime
        self.pipeline = TrafficInferencePipeline(self.states, keep_datetime=False)

    # def predict_df(self, df_raw: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    #     feats = self.pipeline.transform(df_raw)
    #     feats = feats.reindex(columns=self.feature_cols, copy=False)
    #     pred_delta = self.model.predict(feats)
    #     return pd.Series(pred_delta), feats
    
    # def predict_df_total_speed(self, df_raw):
    #     pred_delta_speed,feats = self.predict_df(df_raw)
    #     return pred_delta_speed + feats['value']
    
    
    def predict_df(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (canonical_prediction_frame, engineered_features)
        """
        feats = self.pipeline.transform(df_raw)
        feats = feats.reindex(columns=self.feature_cols, copy=False)
        pred_delta = self.model.predict(feats)
        pred_df = make_prediction_frame(
            df=df_raw,
            feats=feats,
            pred_delta=pred_delta,
            states=self.states,
            horizon_min=self.horizon,
            add_total=True,
            sensor_col="sensor_id",
            add_y_act=True
            
        )
        return pred_df, feats