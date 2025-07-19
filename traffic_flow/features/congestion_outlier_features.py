import pandas as pd
from typing import List, Tuple, Dict, Any
from .base import BaseFeatureTransformer


# traffic_flow/features/outlier_flagger.py
from typing import Dict, Any, List
import pandas as pd
from .base import BaseFeatureTransformer

class GlobalOutlierFlagger(BaseFeatureTransformer):
    """
    Generates 'is_outlier': Flags values as outliers based on global quantile boundaries from training data.

    Args:
        lower_bound (float): Lower quantile threshold for outlier detection.
        upper_bound (float): Upper quantile threshold for outlier detection.
        disable_logs (bool): If True, suppress logging.
    """

    def __init__(
        self,
        *,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        value_col: str = "value",
        sensor_col: str = "sensor_id",
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value_col   = value_col
        self.sensor_col  = sensor_col

        self._lower_val: float = float("nan")
        self._upper_val: float = float("nan")
        self.feature_names_out_ = ["is_outlier"]
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        if "test_set" not in X.columns:
            raise ValueError("Need 'test_set' column to learn outlier thresholds.")
        train = X[~X["test_set"]]
        self._lower_val = float(train[self.value_col].quantile(self.lower_bound))
        self._upper_val = float(train[self.value_col].quantile(self.upper_bound))
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() or from_state() first.")
        df = X.copy()
        df["is_outlier"] = (
            (df[self.value_col] < self._lower_val) |
            (df[self.value_col] > self._upper_val)
        ).astype(int)
        return df

    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "global_outlier",
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "value_col": self.value_col,
            "sensor_col": self.sensor_col,
            "lower_val": self._lower_val,
            "upper_val": self._upper_val,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "GlobalOutlierFlagger":
        inst = cls(
            lower_bound = state["lower_bound"],
            upper_bound = state["upper_bound"],
            value_col   = state["value_col"],
            sensor_col  = state["sensor_col"],
        )
        inst._lower_val = state["lower_val"]
        inst._upper_val = state["upper_val"]
        inst.fitted_ = True
        return inst
    

