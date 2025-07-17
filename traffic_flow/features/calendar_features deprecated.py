import pandas as pd
from typing import List, Tuple
from .base import FeatureTransformer


class DateTimeFeatureEngineer(FeatureTransformer):
    """
    Adds calendar-related features extracted from a datetime column.

    Features include:
    - Hour of the day
    - Day of the week
    - Month of the year
    - Weekend indicators (is_saturday, is_sunday)

    Args:
        datetime_col (str): Name of the datetime column in the DataFrame.
        disable_logs (bool): If True, disables logging output.
    """

    def __init__(self, datetime_col: str = 'datetime', disable_logs: bool = False):
        super().__init__(disable_logs)
        self.datetime_col = datetime_col

    def add_hour_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Adds an 'hour' column from the datetime."""
        self._log("Adding 'hour' column.")
        df['hour'] = df[self.datetime_col].dt.hour
        return df, ['hour']

    def add_day_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Adds a 'day' column (0=Monday, 6=Sunday) from the datetime."""
        self._log("Adding 'day' column.")
        df['day'] = df[self.datetime_col].dt.dayofweek
        return df, ['day']

    def add_month_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Adds a 'month' column (1=January, ..., 12=December) from the datetime."""
        self._log("Adding 'month' column.")
        df['month'] = df[self.datetime_col].dt.month
        return df, ['month']

    def add_weekend_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Adds binary weekend indicator columns:
        - is_saturday
        - is_sunday
        """
        self._log("Adding 'is_saturday' and 'is_sunday' columns.")
        df['is_saturday'] = (df[self.datetime_col].dt.dayofweek == 5).astype(int)
        df['is_sunday'] = (df[self.datetime_col].dt.dayofweek == 6).astype(int)
        return df, ['is_saturday', 'is_sunday']

    def transform(self, df: pd.DataFrame, add_month: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies all datetime-related feature transformations to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with a datetime column.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Transformed DataFrame and list of new feature columns.
        """
        self._log("Starting calendar feature engineering.")

        new_features: List[str] = []

        df, hour_cols = self.add_hour_column(df)
        new_features += hour_cols

        df, day_cols = self.add_day_column(df)
        new_features += day_cols

        if add_month:
            df, month_cols = self.add_month_column(df)
            new_features += month_cols

        df, weekend_cols = self.add_weekend_columns(df)
        new_features += weekend_cols

        self._log(f"Completed calendar features: {new_features}")
        return df, new_features