import pandas as pd
from typing import List, Tuple
from .base import BaseFeatureTransformer



class DateTimeFeatureEngineer(BaseFeatureTransformer):
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
    def __init__(
        self,
        datetime_col: str = "datetime",
        *,
        add_hour: bool = True,
        add_day: bool = True,
        add_month: bool = False,
        add_weekend: bool = True,
        drop_original: bool = False,
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)
        self.datetime_col = datetime_col
        self.add_hour = add_hour
        self.add_day = add_day
        self.add_month = add_month
        self.add_weekend = add_weekend
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        if self.datetime_col not in X.columns:
            raise ValueError(
                f"DateTimeFeatureEngineer: '{self.datetime_col}' not found in columns."
            )
        out: List[str] = []
        if self.add_hour: out.append("hour")
        if self.add_day: out.append("day")
        if self.add_month: out.append("month")
        if self.add_weekend: out.extend(["is_saturday", "is_sunday"])
        self.feature_names_out_ = out
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")

        if self.add_hour:
            X["hour"] = dt.dt.hour
        if self.add_day:
            X["day"] = dt.dt.dayofweek
        if self.add_month:
            X["month"] = dt.dt.month
        if self.add_weekend:
            dow = dt.dt.dayofweek
            X["is_saturday"] = (dow == 5).astype(int)
            X["is_sunday"]   = (dow == 6).astype(int)

        if self.drop_original:
            X.drop(columns=[self.datetime_col], inplace=True)

        return X