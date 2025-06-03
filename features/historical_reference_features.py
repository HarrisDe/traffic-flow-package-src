import pandas as pd
import numpy as np
from typing import List, Tuple
from .base import FeatureTransformer

class PreviousWeekdayValueFeatureEngineer(FeatureTransformer):
    """
    Adds a feature representing the value of each sensor from the previous non-weekend day,
    shifted forward by a specified horizon (in minutes). The feature uses a pivoted lookup table
    and supports optional filtering based on weekday alignment.

    Example use case:
    - At 10:00 AM on Tuesday, this feature can reference the value at 10:15 AM on Monday (horizon = 15).

    Parameters:
        datetime_col (str): Name of datetime column.
        sensor_col (str): Name of sensor identifier column.
        value_col (str): Name of the target value column.
        horizon_minutes (int): Time offset to apply after identifying the previous weekday.
        strict_weekday_match (bool): If True, keeps only Mon–Fri → Mon–Fri pairs.
        disable_logs (bool): Suppress logging.
    """

    def __init__(
        self,
        datetime_col: str = 'date',
        sensor_col: str = 'sensor_id',
        value_col: str = 'value',
        horizon_minutes: int = 15,
        strict_weekday_match: bool = True,
        disable_logs: bool = False
    ):
        super().__init__(disable_logs)
        self.datetime_col = datetime_col
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.horizon_minutes = horizon_minutes
        self.strict_weekday_match = strict_weekday_match
        self.new_column_name = f'prev_weekday_value_h{self.horizon_minutes}'

    def _get_previous_weekdays(self, dates: pd.Series) -> pd.Series:
        """
        Vectorized computation of previous weekday shifted by the given horizon.

        Args:
            dates (pd.Series): Input datetime series.

        Returns:
            pd.Series: Adjusted datetime for value lookup.
        """
        self._log("Computing adjusted timestamps for previous weekdays.")
        prev = dates - pd.Timedelta(days=1)
        prev = prev.mask(prev.dt.weekday == 6, prev - pd.Timedelta(days=2))  # Sunday → Friday
        prev = prev.mask(prev.dt.weekday == 5, prev - pd.Timedelta(days=1))  # Saturday → Friday
        return prev + pd.Timedelta(minutes=self.horizon_minutes)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transforms the dataframe by adding a feature from the previous weekday value.

        Args:
            df (pd.DataFrame): Input dataframe with timestamps and sensor values.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Updated dataframe and list of added feature names.
        """
        self._log(f"Adding historical reference feature: {self.new_column_name}")
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        # Create pivot table for fast multi-index lookup
        self._log("Creating pivot table for fast sensor-time value retrieval.")
        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        stacked = pivot.stack()
        stacked.index.names = [self.datetime_col, self.sensor_col]

        # Create adjusted timestamps for lookup
        df['current_dayofweek'] = df[self.datetime_col].dt.weekday
        df['lookup_time'] = self._get_previous_weekdays(df[self.datetime_col])
        df['lookup_dayofweek'] = df['lookup_time'].dt.weekday

        # Filter to ensure both source and lookup times are weekdays (if requested)
        if self.strict_weekday_match:
            valid_mask = (df['current_dayofweek'] < 5) & (df['lookup_dayofweek'] < 5)
            self._log("Filtering to keep only weekday-to-weekday lookup pairs.")
        else:
            valid_mask = pd.Series(True, index=df.index)

        # Perform lookup from pivoted data using MultiIndex
        self._log("Performing value lookups using reindex.")
        lookup_index = list(zip(df['lookup_time'], df[self.sensor_col]))
        lookup_values = stacked.reindex(lookup_index).values
        df[self.new_column_name] = np.where(valid_mask, lookup_values, np.nan)

        # Clean temporary columns
        df.drop(columns=['current_dayofweek', 'lookup_time', 'lookup_dayofweek'], inplace=True)

        self._log(f"Feature '{self.new_column_name}' successfully added.")
        return df, [self.new_column_name]