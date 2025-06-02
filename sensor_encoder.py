# sensor_encoder.py
from abc import ABC, abstractmethod
import pandas as pd
from .helper_utils import LoggingMixin

class SensorEncodingStrategy(ABC):
    """
    Abstract base class for all sensor encoding strategies.
    """
    @abstractmethod
    def encode(self, df: pd.DataFrame, col: str, is_train: pd.Series) -> pd.DataFrame:
        """
        Encodes the given column in the dataframe.

        Args:
            df (pd.DataFrame): Full dataset.
            col (str): Column name to encode.
            is_train (pd.Series): Boolean mask for training rows.

        Returns:
            pd.DataFrame: Modified dataframe with encoded column.
        """
        pass


class OrdinalSensorEncoder(SensorEncodingStrategy, LoggingMixin):
    """
    Encodes sensor IDs with ordinal integers.
    """
    def __init__(self, disable_logs=False):
        super().__init__(disable_logs)
        self.mapping = {}

    def encode(self, df, col, is_train):
        self._log("Applying ordinal encoding to sensor IDs.")
        if not self.mapping:
            unique_ids = sorted(df.loc[is_train, col].unique())
            self.mapping = {sid: idx for idx, sid in enumerate(unique_ids)}
            self._log(f"Generated ordinal mapping for {len(self.mapping)} sensors.")
        df[col] = df[col].map(self.mapping)
        return df


class MeanSensorEncoder(SensorEncodingStrategy, LoggingMixin):
    """
    Encodes sensor IDs using the average speed value from the training set.
    """
    def __init__(self, disable_logs=False):
        super().__init__(disable_logs)
        self.mapping = {}

    def encode(self, df, col, is_train):
        self._log("Applying mean encoding to sensor IDs using training data.")
        if not self.mapping:
            self.mapping = df.loc[is_train].groupby(col)['value'].mean().to_dict()
            self._log(f"Generated mean mapping for {len(self.mapping)} sensors.")
        df[col] = df[col].map(self.mapping)
        return df


class OneHotSensorEncoder(SensorEncodingStrategy, LoggingMixin):
    """
    Applies one-hot encoding to the sensor column.
    """
    def __init__(self, disable_logs=False):
        super().__init__(disable_logs)

    def encode(self, df, col, is_train):
        self._log("Applying one-hot encoding to sensor IDs.")
        return pd.get_dummies(df, columns=[col])
    



    
    


