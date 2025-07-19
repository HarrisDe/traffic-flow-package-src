import pandas as pd
from typing import List, Tuple, Dict, Any
from .base import BaseFeatureTransformer



# traffic_flow/features/congestion_features.py
from typing import Dict, Any
import pandas as pd
from .congestion_threshold import PerSensorCongestionFlagger
from .congestion_outlier_features  import GlobalOutlierFlagger
from .base import BaseFeatureTransformer

class CongestionFeatureEngineer(BaseFeatureTransformer):
    """
    Thin wrapper that sequentially applies two independent transformers
    for backward compatibility with existing pipeline code.
    """

    def __init__(self, **kwargs):
        super().__init__(disable_logs=kwargs.get("disable_logs", False))
        self.cong_ = PerSensorCongestionFlagger(**kwargs)
        self.outl_ = GlobalOutlierFlagger(
            lower_bound = kwargs.get("lower_bound", 0.01),
            upper_bound = kwargs.get("upper_bound", 0.99),
            value_col   = kwargs.get("value_col", "value"),
            sensor_col  = kwargs.get("sensor_col", "sensor_id"),
            disable_logs= kwargs.get("disable_logs", False),
        )
        self.feature_names_out_ = ["is_congested", "is_outlier"]

    # sklearn API
    def fit(self, X: pd.DataFrame, y=None):
        self.cong_.fit(X)
        self.outl_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.cong_.transform(X)
        X = self.outl_.transform(X)
        return X

    # persistence
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "congestion_bundle",
            "congestion_state": self.cong_.export_state(),
            "outlier_state":    self.outl_.export_state(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "CongestionFeatureEngineer":
        inst = cls()  # dummy init
        inst.cong_ = PerSensorCongestionFlagger.from_state(state["congestion_state"])
        inst.outl_ = GlobalOutlierFlagger.from_state(state["outlier_state"])
        return inst