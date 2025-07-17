import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from .base import SensorEncodingStrategy


# class OrdinalSensorEncoder(SensorEncodingStrategy):
#     """
#     Encodes sensor IDs using ordinal integers (e.g., sensor_1 → 0, sensor_2 → 1, ...).
#     Uses only the training set to compute the encoding.
#     """

#     def __init__(self, sensor_col: str = "sensor_id",new_sensor_col: str = "sensor_uid", disable_logs: bool = False):
#         super().__init__(sensor_col=sensor_col, new_sensor_col = new_sensor_col, disable_logs=disable_logs)
#         self.mapping: Dict[str, int] = {}

#     def encode(self, df: pd.DataFrame) -> pd.DataFrame:
#         self._log("Applying ordinal encoding to sensor IDs.")

#         if 'test_set' not in df.columns:
#             raise ValueError("Column 'test_set' is required to determine training rows.")

#         if not self.mapping:
#             train_df = df[~df['test_set']]
#             self.train_df = train_df.copy()
#             unique_ids = sorted(train_df[self.sensor_col].unique())
#             self.mapping = {sid: idx for idx, sid in enumerate(unique_ids)}
#             self._log(f"Generated ordinal mapping for {len(self.mapping)} sensors.")

#         df[self.new_sensor_col] = df[self.sensor_col].map(self.mapping).fillna(-1).astype(int)
#         return df
    
class OrdinalSensorEncoder(SensorEncodingStrategy):
    """
    Map each sensor_id to an integer code.
    Learned from *training* subset (rows where test_set == False).
    Unseen sensors at inference receive -1.
    """

    def __init__(
        self,
        *,
        sensor_col: str = "sensor_id",
        new_sensor_col: str = "sensor_uid",
        disable_logs: bool = False,
    ):
        super().__init__(sensor_col=sensor_col, new_sensor_col=new_sensor_col, disable_logs=disable_logs)
        self.mapping_: Dict[Any, int] = {}  # learned in fit
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y=None):
        if "test_set" not in X.columns:
            raise ValueError("OrdinalSensorEncoder.fit() requires 'test_set' column to identify training rows.")
        train_df = X[~X["test_set"]]
        unique_ids = sorted(pd.Series(train_df[self.sensor_col]).unique())
        self.mapping_ = {sid: idx for idx, sid in enumerate(unique_ids)}
        self.fitted_ = True
        self._log(f"Ordinal mapping learned for {len(self.mapping_)} sensors.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("OrdinalSensorEncoder not fitted. Call fit() or use from_state().")
        X = X.copy()
        X[self.new_sensor_col] = (
            X[self.sensor_col].map(self.mapping_).fillna(-1).astype(int)  # type: ignore[arg-type]
        )
        return X

    # ---- persistence (for saving/loading in inference) ----
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "ordinal",
            "sensor_col": self.sensor_col,
            "new_sensor_col": self.new_sensor_col,
            "mapping": self.mapping_,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "OrdinalSensorEncoder":
        inst = cls(
            sensor_col=state["sensor_col"],
            new_sensor_col=state["new_sensor_col"],
        )
        inst.mapping_ = state["mapping"]
        inst.fitted_ = True
        return inst


# class MeanSensorEncoder(SensorEncodingStrategy):
#     """
#     Encodes sensor IDs using their average speed (or other value) computed on the training data.
#     """

#     def __init__(self, sensor_col: str = "sensor_id", new_sensor_col: str = "sensor_uid", disable_logs: bool = False):
#         super().__init__(sensor_col=sensor_col, new_sensor_col = new_sensor_col, disable_logs=disable_logs)
#         self.mapping: Dict[str, float] = {}
#         self.train_df = None

#     def encode(self, df: pd.DataFrame) -> pd.DataFrame:
#         self._log("Applying mean encoding to sensor IDs using training data.")

#         if 'test_set' not in df.columns:
#             raise ValueError("Column 'test_set' is required to determine training rows.")

#         if not self.mapping:
#             train_df = df[~df['test_set']]
#             self.train_df = train_df.copy()
#             self.mapping = train_df.groupby(self.sensor_col)["value"].mean().to_dict()
#             self._log(f"Generated mean mapping for {len(self.mapping)} sensors.")

#         df[self.new_sensor_col] = df[self.sensor_col].map(self.mapping).fillna(-1.0)
#         return df



class MeanSensorEncoder(SensorEncodingStrategy):
    """
    Encode sensors by mean of the `value` column computed over training rows.
    Unseen sensors → global mean (or -1.0 fallback; choose behaviour).
    """

    def __init__(
        self,
        *,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        new_sensor_col: str = "sensor_uid",
        disable_logs: bool = False,
        unseen_strategy: str = "global_mean",  # or "sentinel"
        sentinel_value: float = -1.0,
    ):
        super().__init__(sensor_col=sensor_col, new_sensor_col=new_sensor_col, disable_logs=disable_logs)
        self.value_col = value_col
        self.unseen_strategy = unseen_strategy
        self.sentinel_value = sentinel_value

        self.mapping_: Dict[Any, float] = {}
        self.global_mean_: float = float("nan")
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        if "test_set" not in X.columns:
            raise ValueError("MeanSensorEncoder.fit() requires 'test_set' column to identify training rows.")
        train_df = X[~X["test_set"]]
        self.mapping_ = train_df.groupby(self.sensor_col)[self.value_col].mean().to_dict()
        self.global_mean_ = float(train_df[self.value_col].mean())
        self.fitted_ = True
        self._log(f"Mean encoding learned for {len(self.mapping_)} sensors. Global mean={self.global_mean_:.2f}.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("MeanSensorEncoder not fitted. Call fit() or use from_state().")
        X = X.copy()
        encoded = X[self.sensor_col].map(self.mapping_)  # type: ignore[arg-type]

        if self.unseen_strategy == "global_mean":
            encoded = encoded.fillna(self.global_mean_)
        else:  # sentinel
            encoded = encoded.fillna(self.sentinel_value)

        X[self.new_sensor_col] = encoded.astype(float)
        return X

    # ---- persistence (for saving/loading in inference) ----
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "mean",
            "sensor_col": self.sensor_col,
            "value_col": self.value_col,
            "new_sensor_col": self.new_sensor_col,
            "unseen_strategy": self.unseen_strategy,
            "sentinel_value": self.sentinel_value,
            "mapping": self.mapping_,
            "global_mean": self.global_mean_,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MeanSensorEncoder":
        inst = cls(
            sensor_col=state["sensor_col"],
            value_col=state["value_col"],
            new_sensor_col=state["new_sensor_col"],
            unseen_strategy=state["unseen_strategy"],
            sentinel_value=state["sentinel_value"],
        )
        inst.mapping_ = state["mapping"]
        inst.global_mean_ = state["global_mean"]
        inst.fitted_ = True
        return inst





