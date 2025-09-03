import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .base import BaseFeatureTransformer


class PredictionTimeCyclicalFeatureEngineer(BaseFeatureTransformer):
    """
    Add sin/cos cyclical encodings for:
      - CURRENT time (row's datetime)
      - PREDICTION time (row's datetime + horizon_min)

    Feature names are stored in self.feature_names_out_ (no tuple return from transform).

    Parameters
    ----------
    datetime_col : str
        Name of datetime column (tz-naive or tz-aware Pandas datetimes).
    horizon_min : int
        Forecast horizon in minutes (used to compute prediction/target time).
    add_day : bool, default True
        Whether to encode day-of-week as sin/cos.
    add_hour : bool, default True
        Whether to encode hour-of-day as sin/cos.
    add_minute : bool, default True
        Whether to encode minute-of-hour as sin/cos.
    include_current_time : bool, default True
        If True, add *_curr features (for the input timestamp).
    include_forecast_time : bool, default True
        If True, add *_tgt features (for the prediction timestamp).
    """

    def __init__(
        self,
        *,
        datetime_col: str,
        horizon_min: int,
        add_day: bool = True,
        add_hour: bool = True,
        add_minute: bool = True,
        include_current_time: bool = True,
        include_forecast_time: bool = True,
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.datetime_col = datetime_col
        self.horizon_min = int(horizon_min)
        self.add_day = bool(add_day)
        self.add_hour = bool(add_hour)
        self.add_minute = bool(add_minute)
        self.include_current_time = bool(include_current_time)
        self.include_forecast_time = bool(include_forecast_time)
        self.feature_names_out_: List[str] = []
        self.fitted_ = True  # stateless

    # --------------------------- helpers ---------------------------- #
    @staticmethod
    def _cyc_enc(values: pd.Series, period: int, prefix: str) -> Tuple[pd.Series, pd.Series, List[str]]:
        ang = 2.0 * np.pi * (values.astype("float32") / period)
        s = np.sin(ang).astype("float32")
        c = np.cos(ang).astype("float32")
        return s, c, [f"{prefix}_sin", f"{prefix}_cos"]

    def _encode_dt(self, dt: pd.Series, suffix: str) -> Tuple[pd.DataFrame, List[str]]:
        cols: List[str] = []
        out = pd.DataFrame(index=dt.index)

        if self.add_day:
            s, c, names = self._cyc_enc(dt.dt.dayofweek, 7, f"dow_{suffix}")
            out[names[0]] = s; out[names[1]] = c; cols.extend(names)
        if self.add_hour:
            s, c, names = self._cyc_enc(dt.dt.hour, 24, f"hour_{suffix}")
            out[names[0]] = s; out[names[1]] = c; cols.extend(names)
        if self.add_minute:
            s, c, names = self._cyc_enc(dt.dt.minute, 60, f"minute_{suffix}")
            out[names[0]] = s; out[names[1]] = c; cols.extend(names)

        return out, cols

    # --------------------------- API ------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        if self.datetime_col not in df.columns:
            raise KeyError(f"Column '{self.datetime_col}' not found.")

        dt = pd.to_datetime(df[self.datetime_col], errors="coerce")
        if dt.isna().any():
            raise ValueError(f"NaT detected in '{self.datetime_col}' after to_datetime().")

        added_cols: List[str] = []
        frames: List[pd.DataFrame] = [df]

        if self.include_current_time:
            enc_curr, cols_curr = self._encode_dt(dt, "curr")
            frames.append(enc_curr); added_cols.extend(cols_curr)

        if self.include_forecast_time:
            dt_tgt = dt + pd.to_timedelta(self.horizon_min, unit="m")
            enc_tgt, cols_tgt = self._encode_dt(dt_tgt, "tgt")
            frames.append(enc_tgt); added_cols.extend(cols_tgt)

        df_out = pd.concat(frames, axis=1)
        self.feature_names_out_ = added_cols
        return df_out

    # --------------------------- persistence ------------------------ #
    def export_state(self) -> Dict[str, Any]:
        return {
            "datetime_col": self.datetime_col,
            "horizon_min": self.horizon_min,
            "add_day": self.add_day,
            "add_hour": self.add_hour,
            "add_minute": self.add_minute,
            "include_current_time": self.include_current_time,
            "include_forecast_time": self.include_forecast_time,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PredictionTimeCyclicalFeatureEngineer":
        inst = cls(
            datetime_col=state["datetime_col"],
            horizon_min=state["horizon_min"],
            add_day=state.get("add_day", True),
            add_hour=state.get("add_hour", True),
            add_minute=state.get("add_minute", True),
            include_current_time=state.get("include_current_time", True),
            include_forecast_time=state.get("include_forecast_time", True),
        )
        inst.fitted_ = True
        return inst