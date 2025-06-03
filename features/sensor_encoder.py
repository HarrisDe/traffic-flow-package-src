import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from .base import SensorEncodingStrategy


class OrdinalSensorEncoder(SensorEncodingStrategy):
    """
    Encodes sensor IDs using ordinal integers (e.g., sensor_1 → 0, sensor_2 → 1, ...).
    Uses only the training set to compute the encoding.
    """

    def __init__(self, sensor_col: str = "sensor_id",new_sensor_col: str = "sensor_uid", disable_logs: bool = False):
        super().__init__(sensor_col=sensor_col, new_sensor_col = new_sensor_col, disable_logs=disable_logs)
        self.mapping: Dict[str, int] = {}

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Applying ordinal encoding to sensor IDs.")

        if 'test_set' not in df.columns:
            raise ValueError("Column 'test_set' is required to determine training rows.")

        if not self.mapping:
            train_df = df[~df['test_set']]
            unique_ids = sorted(train_df[self.sensor_col].unique())
            self.mapping = {sid: idx for idx, sid in enumerate(unique_ids)}
            self._log(f"Generated ordinal mapping for {len(self.mapping)} sensors.")

        df[self.new_sensor_col] = df[self.sensor_col].map(self.mapping).fillna(-1).astype(int)
        return df


class MeanSensorEncoder(SensorEncodingStrategy):
    """
    Encodes sensor IDs using their average speed (or other value) computed on the training data.
    """

    def __init__(self, sensor_col: str = "sensor_uid", new_sensor_col: str = "sensor_id", disable_logs: bool = False):
        super().__init__(sensor_col=sensor_col, new_sensor_col = new_sensor_col, disable_logs=disable_logs)
        self.mapping: Dict[str, float] = {}

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Applying mean encoding to sensor IDs using training data.")

        if 'test_set' not in df.columns:
            raise ValueError("Column 'test_set' is required to determine training rows.")

        if not self.mapping:
            train_df = df[~df['test_set']]
            self.mapping = train_df.groupby(self.sensor_col)["value"].mean().to_dict()
            self._log(f"Generated mean mapping for {len(self.mapping)} sensors.")

        df[self.new_sensor_col] = df[self.sensor_col].map(self.mapping).fillna(-1.0)
        return df



class OneHotSensorEncoder(SensorEncodingStrategy):
    """
    Encodes sensor IDs using one-hot encoding, based on training data only.
    Ensures consistent columns across the full dataset.
    """

    def __init__(self,sensor_col: str = "sensor_id", new_sensor_col: Optional[str] = None, disable_logs: bool = False):
        super().__init__(sensor_col=sensor_col, new_sensor_col = new_sensor_col, disable_logs=disable_logs)
        self.train_columns: Optional[List[str]] = None

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Applying one-hot encoding to sensor IDs (fit on train set).")

        if "test_set" not in df.columns:
            raise ValueError("Column 'test_set' is required to distinguish training data.")

        # Fit on training data
        train_df = df[~df["test_set"]]
        train_encoded = pd.get_dummies(train_df[self.sensor_col], drop_first=True,prefix=self.sensor_col)
        self.train_columns = train_encoded.columns.tolist()
        self._log(f"Generated {len(self.train_columns)} one-hot columns from train set.")

        # Transform full dataset
        full_encoded = pd.get_dummies(df[self.sensor_col],drop_first=True, prefix=self.sensor_col)

        # Align columns to match training set encoding
        for col in self.train_columns:
            if col not in full_encoded.columns:
                full_encoded[col] = 0
        full_encoded = full_encoded[self.train_columns]

        df_out = df.copy()
        for col in full_encoded.columns:
            df_out[col] = full_encoded[col]
        # Drop original sensor_col and concatenate one-hot columns
        #df = pd.concat([df.reset_index(drop=True), full_encoded.reset_index(drop=True)], axis=1)

        return df_out


