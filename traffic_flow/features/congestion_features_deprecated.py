import pandas as pd
from typing import List, Tuple, Dict, Any
from .base import BaseFeatureTransformer


class CongestionFeatureEngineer(BaseFeatureTransformer):
    """
    Adds congestion-related features based on per-sensor speed thresholds and outlier detection.
    
    Features added:
    - 'is_congested': Marks whether the value falls below a dynamic threshold during peak hours.
    - 'is_outlier': Flags values as outliers based on global quantile boundaries from training data.

    Args:
        hour_start (int): Start of peak congestion window (inclusive).
        hour_end (int): End of peak congestion window (inclusive).
        quantile_threshold (float): Quantile level used per sensor for congestion threshold.
        quantile_percentage (float): Percentage multiplier applied to the quantile threshold.
        lower_bound (float): Lower quantile threshold for outlier detection.
        upper_bound (float): Upper quantile threshold for outlier detection.
        disable_logs (bool): If True, suppress logging.
    """

    def __init__(
        self,
        hour_start: int = 6,
        hour_end: int = 19,
        quantile_threshold: float = 0.9,
        quantile_percentage: float = 0.65,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        disable_logs: bool = False
    ):
        super().__init__(disable_logs)
        self.hour_start = hour_start
        self.hour_end = hour_end
        self.quantile_threshold = quantile_threshold
        self.quantile_percentage = quantile_percentage
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform_congestion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Adds a binary 'is_congested' column if a value is below a per-sensor dynamic threshold
        during specified peak hours, using training data only.

        Raises:
            ValueError: If required columns 'hour' or 'test_set' are missing.

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with 'is_congested' feature added,
                                            and the name of the added column.
        """
        self._log("Adding congestion feature based on thresholds.")

        if 'hour' not in df.columns:
            raise ValueError("Column 'hour' is missing. Run DateTimeFeatureEngineer first.")
        if 'test_set' not in df.columns:
            raise ValueError("Column 'test_set' is missing. Ensure data has been split.")

        mask = (
            (df['hour'] >= self.hour_start) &
            (df['hour'] <= self.hour_end) &
            (~df['test_set'])
        )

        congestion_thr = df[mask].groupby('sensor_id')['value'].quantile(self.quantile_threshold)
        thresholds = df['sensor_id'].map(congestion_thr) * self.quantile_percentage
        df['is_congested'] = (df['value'] < thresholds).astype(int)

        return df, ['is_congested']

    def add_outlier_flags(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Adds a binary 'is_outlier' column based on global lower and upper value percentiles
        from the training set only.

        Raises:
            ValueError: If 'test_set' column is missing.

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with 'is_outlier' flag and column name.
        """
        self._log("Flagging outliers in training set.")

        if 'test_set' not in df.columns:
            raise ValueError("Missing 'test_set' column. Split your data first.")

        train_df = df[~df['test_set']]
        lower_val = train_df['value'].quantile(self.lower_bound)
        upper_val = train_df['value'].quantile(self.upper_bound)

        self._log(f"Using outlier thresholds: lower={lower_val}, upper={upper_val}")

        df['is_outlier'] = ((df['value'] < lower_val) | (df['value'] > upper_val)).astype(int)
        return df, ['is_outlier']

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies both congestion threshold logic and outlier detection to the dataframe.

        Args:
            df (pd.DataFrame): Input DataFrame with required preprocessed columns.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Updated DataFrame and all new feature column names.
        """
        df, congestion_cols = self.transform_congestion(df)
        df, outlier_cols = self.add_outlier_flags(df)
        return df, congestion_cols + outlier_cols
    
    

