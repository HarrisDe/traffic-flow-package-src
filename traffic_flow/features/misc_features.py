from typing import List, Tuple, Optional
import pandas as pd
from .base import  FeatureTransformer, SensorEncodingStrategy



class WeatherFeatureDropper(FeatureTransformer):
    """
    Drops weather-related columns from the DataFrame, if present.
    """

    def __init__(self, weather_cols: List[str] = None, disable_logs: bool = False):
        super().__init__(disable_logs)
        self.weather_cols = weather_cols or []

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        self._log("Dropping weather-related features.")
        before_cols = set(df.columns)
        df = df.drop(columns=[col for col in self.weather_cols if col in df.columns], errors='ignore')
        dropped = list(before_cols - set(df.columns))
        self._log(f"Dropped columns: {dropped}")
        return df, dropped

class MiscellaneousFeatureEngineerDeprecated(FeatureTransformer):
    """
    Applies sensor ID encoding and removes unwanted meta features like weather columns.
    
    Args:
        encoder (SensorEncodingStrategy): Strategy to encode sensor identifiers.
        weather_cols (Optional[List[str]]): Weather-related columns to drop.
        disable_logs (bool): If True, disables logging.
    """

    def __init__(self,
                 encoder: SensorEncodingStrategy,
                 weather_cols: Optional[List[str]] = None,
                 disable_logs: bool = False):
        super().__init__(disable_logs)
        self.encoder = encoder
        self.weather_cols = weather_cols or []

    def encode_sensor_ids(self, df: pd.DataFrame, is_train: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Encodes sensor identifiers using the provided strategy."""
        self._log("Encoding sensor IDs using provided strategy.")
        df = self.encoder.encode(df, is_train)
        return df, [self.encoder.sensor_col]

    def drop_weather_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Drops weather-related features if they exist."""
        self._log("Dropping weather-related columns.")
        before_cols = set(df.columns)
        df = df.drop(columns=[col for col in self.weather_cols if col in df.columns], errors='ignore')
        dropped = list(before_cols - set(df.columns))
        self._log(f"Dropped columns: {dropped}")
        return df, dropped

    def transform(self, df: pd.DataFrame, is_train: pd.Series, drop_weather: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies all transformations.

        Args:
            df (pd.DataFrame): Input data.
            is_train (pd.Series): Boolean Series indicating training samples.
            drop_weather (bool): Whether to remove weather-related columns.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Transformed DataFrame and list of added/dropped feature names.
        """
        df, enc_cols = self.encode_sensor_ids(df, is_train)
        if drop_weather:
            df, dropped_cols = self.drop_weather_features(df)
            return df, enc_cols + dropped_cols
        return df, enc_cols