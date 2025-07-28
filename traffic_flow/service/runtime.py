# traffic_flow/service/runtime.py
from __future__ import annotations
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from ..inference.inference_pipeline import TrafficInferencePipeline

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

    def predict_df(self, df_raw: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        feats = self.pipeline.transform(df_raw)
        feats = feats.reindex(columns=self.feature_cols, copy=False)
        pred_delta = self.model.predict(feats)
        return pd.Series(pred_delta), feats