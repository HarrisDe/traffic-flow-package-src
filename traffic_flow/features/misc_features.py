from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from .base import  FeatureTransformer, SensorEncodingStrategy, BaseFeatureTransformer


    
class WeatherFeatureDropper(BaseFeatureTransformer):
    """
    Removes the weather columns specified at construction time.

    •  Learns *nothing* ⇒ `fit()` only records which of the requested
       columns are actually present in the training frame.
    •  Provides `export_state()` / `from_state()` so the exact same
       columns are dropped during inference.
    """

    def __init__(
        self,
        *,
        weather_cols: List[str] | None = None,
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)
        self.requested_cols = list(weather_cols or [])
        self.cols_to_drop_: List[str] = []
        self.fitted_ = False
        self.feature_names_out_: List[str] = []   # empty → we *remove* cols

    # -----------------------------------------------------------------
    def fit(self, X, y=None):
        self.cols_to_drop_ = [c for c in self.requested_cols if c in X.columns]
        self.fitted_ = True
        self._log(f"Will drop {self.cols_to_drop_ or 'no columns'}")
        return self

    def transform(self, X: pd.DataFrame):
        if not self.fitted_:
            raise RuntimeError("Call fit() or from_state() first.")

        return X.drop(columns=self.cols_to_drop_, errors="ignore")

    # ------------------------ persistence ----------------------------
    def export_state(self) -> Dict[str, Any]:
        return {"type": "weather_dropper", "cols_to_drop": self.cols_to_drop_}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "WeatherFeatureDropper":
        inst = cls(weather_cols=state["cols_to_drop"])
        inst.cols_to_drop_ = state["cols_to_drop"]
        inst.fitted_ = True
        return inst

