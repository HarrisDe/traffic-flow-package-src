import pandas as pd
from typing import List, Optional, Tuple
from .base import FeatureTransformer 


    
class TemporalLagFeatureAdder(FeatureTransformer):
    """
    Adds lag-based features to capture past values or relative differences for each sensor.

    Args:
        lags (int): Number of lags to compute.
        relative (bool): Whether to compute relative differences instead of absolute lags.
        fill_nans_value (float): Value to fill for initial NaNs.
        disable_logs (bool): If True, suppress logging.
        sensor_col (str): Column indicating sensor identifier.
        value_col (str): Column containing target value.
        datetime_col (str): Column containing datetime.
        epsilon (float): Small value to prevent division by zero in relative diffs.
    """

    def __init__(
        self,
        lags: int = 3,
        relative: bool = False,
        fill_nans_value: float = -1,
        disable_logs: bool = False,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        datetime_col: str = "datetime",
        epsilon: float = 1e-5
    ):
        super().__init__(disable_logs)
        self.lags = lags
        self.relative = relative
        self.fill_nans_value = fill_nans_value
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.epsilon = epsilon
        self.new_columns: List[str] = []

    def transform(
        self,
        df: pd.DataFrame,
        current_smoothing: Optional[str] = None,
        prev_smoothing: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Adds lag or relative lag features to the dataset.

        Args:
            df (pd.DataFrame): Input dataframe with traffic data.
            current_smoothing (str): Current smoothing strategy applied (optional).
            prev_smoothing (str): Previous smoothing strategy applied (optional).

        Returns:
            Tuple[pd.DataFrame, List[str]]: Transformed dataframe and list of new column names.
        """
        self._log(f"Adding {'relative' if self.relative else 'absolute'} lags (lags={self.lags})")

        prefix = "relative_diff_lag" if self.relative else "lag"
        expected_cols = [f"{prefix}{i}" for i in range(1, self.lags + 1)]
        existing_cols = [col for col in df.columns if col.startswith(prefix)]
        to_drop = list(set(existing_cols) - set(expected_cols))

        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            self._log(f"Dropped excess lag columns: {to_drop}")

        for i in range(1, self.lags + 1):
            col_name = f"{prefix}{i}"
            if col_name in df.columns and current_smoothing == prev_smoothing:
                self._log(f"Skipping already existing column: {col_name}")
                self.new_columns.append(col_name)
                continue

            shifted = df.groupby(self.sensor_col)[self.value_col].shift(i)

            if self.relative:
                df[col_name] = (df[self.value_col] - shifted) / (shifted + self.epsilon)
            else:
                df[col_name] = shifted - df[self.value_col]

            df[col_name].fillna(self.fill_nans_value, inplace=True)
            self.new_columns.append(col_name)

        return df, self.new_columns
