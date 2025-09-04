# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, List, Tuple
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class MomentumFeatureEngineer(BaseFeatureTransformer):
    """
    Add momentum-style features that help a tabular model react earlier to sharp changes.
    (No simple lags/deltas here, since you already have TemporalLagFeatureAdder.)

    Features
    --------
    1) Rolling OLS slope over windows W (in rows):         slope_w{W}
       - Fast and leak-safe; computed per-sensor via rolling().apply(raw=True).

    2) EWMA mean & EWMA slope for half-lives H:            ewm_mean_h{H}, ewm_slope_h{H}
       - EWMA slope is the first difference of the EWMA mean (captures momentum).

    3) Volatility over windows W:                          std_w{W}, mad_w{W}
       - MAD is a robust dispersion estimate.

    4) Recent extrema and distances over windows W:        roll_min_w{W}, roll_max_w{W},
                                                           dist_to_min_w{W}, dist_to_max_w{W}

    5) Time since threshold (minutes) for thresholds T:    mins_since_below_{T}
       - Number of rows since value < T (scaled by minutes_per_row).

    6) Optional binary fast-drop flag:                     is_dropping_fast_w{W}_thr{THR}
       - 1 if slope_w{W} < THR (THR negative), else 0.

    Parameters
    ----------
    sensor_col : str
    value_col : str
    datetime_col : str
    slope_windows : Sequence[int]
        Rolling window sizes (rows) for OLS slope.
    ewm_halflives : Sequence[float]
        Half-lives for EWMA mean/slope.
    vol_windows : Sequence[int]
        Windows (rows) for std/MAD.
    minmax_windows : Sequence[int]
        Windows (rows) for rolling min/max & distances.
    thresholds_kph : Sequence[float]
        Speed thresholds (kph) for "time since below threshold".
    minutes_per_row : float
        If your sampling is 1 row = 1 minute, keep 1.0; otherwise set accordingly.
    drop_fast_flag : bool
        If True, add the fast-drop binary flag using fast_flag_window & fast_flag_thresh.
    fast_flag_window : int
        Window (rows) used for slope in the fast-drop flag.
    fast_flag_thresh : float
        Threshold on slope (negative). Example: -0.8.
    fill_nans_value : float
        Fill value for initial NaNs produced by rolling/EWMA.
    epsilon : float
        Small constant for numeric stability.
    """

    def __init__(
        self,
        *,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        datetime_col: str = "date",
        slope_windows: Sequence[int] = (5, 10, 15, 30),
        ewm_halflives: Sequence[float] = (5.0, 10.0),
        vol_windows: Sequence[int] = (10, 30),
        minmax_windows: Sequence[int] = (15, 30),
        thresholds_kph: Sequence[float] = (70.0, 80.0, 90.0),
        minutes_per_row: float = 1.0,
        drop_fast_flag: bool = True,
        fast_flag_window: int = 5,
        fast_flag_thresh: float = -1.0,
        fill_nans_value: float = -1.0,
        epsilon: float = 1e-6,
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.datetime_col = datetime_col

        self.slope_windows = tuple(int(w) for w in slope_windows)
        self.ewm_halflives = tuple(float(h) for h in ewm_halflives)
        self.vol_windows = tuple(int(w) for w in vol_windows)
        self.minmax_windows = tuple(int(w) for w in minmax_windows)
        self.thresholds_kph = tuple(float(t) for t in thresholds_kph)

        self.minutes_per_row = float(minutes_per_row)
        self.drop_fast_flag = bool(drop_fast_flag)
        self.fast_flag_window = int(fast_flag_window)
        self.fast_flag_thresh = float(fast_flag_thresh)

        self.fill_nans_value = float(fill_nans_value)
        self.epsilon = float(epsilon)

        self.feature_names_out_: List[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # sklearn-style API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in (self.sensor_col, self.value_col) if c not in X.columns]
        if missing:
            raise ValueError(f"MomentumFeatureEngineer: missing columns {missing}")
        self.fitted_ = True
        
        print(f"MomentumFeatureEngineer: configured with slope_windows:{ self.slope_windows },\
              ewm_halflives:{ self.ewm_halflives },vol_windows:{ self.vol_windows },\
              minmax_windows:{ self.minmax_windows },thresholds_kph:{ self.thresholds_kph },\
              ,fast_flag_window:{ self.fast_flag_window },drop_fast_flag:{ self.drop_fast_flag }")
        return self

    # ------------------------------- helpers --------------------------- #
    @staticmethod
    def _rolling_ols_slope_func(window: int):
        """
        Return a fast function f(arr) that computes OLS slope for x = [0..window-1].
        slope = cov(x,y) / var(x)
        """
        n = float(window)
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denom = (n * sum_x2 - sum_x * sum_x)
        if denom == 0:
            denom = 1.0  # numeric safety

        def f(y: np.ndarray) -> float:
            # y is length=window, dtype float
            y = y.astype(np.float64, copy=False)
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            num = (n * sum_xy - sum_x * sum_y)
            return float(num / denom)

        return f

    @staticmethod
    def _rolling_mad(arr: np.ndarray) -> float:
        med = np.median(arr)
        return float(np.median(np.abs(arr - med)))

    def _per_sensor_ewm(self, s: pd.Series, halflife: float) -> pd.Series:
        # Independent EWMA per sensor
        return s.groupby(level=0).apply(
            lambda g: g.droplevel(0).ewm(halflife=halflife, adjust=False).mean()
        )

    def _time_since_below_threshold(
        self, df: pd.DataFrame, threshold: float
    ) -> pd.Series:
        """
        Minutes since last time value < threshold, per sensor. NaN if never below.
        """
        s = df[self.value_col]
        is_below = s < threshold

        # Compute last index where is_below was True, per sensor
        # Build a per-sensor integer index (0..n-1) to diff against
        idx_local = df.groupby(self.sensor_col).cumcount()
        last_true_idx = pd.Series(
            np.where(is_below.values, idx_local.values, np.nan),
            index=df.index,
        )
        last_true_idx = last_true_idx.groupby(df[self.sensor_col]).ffill()

        since = idx_local - last_true_idx
        since = since.astype(float)
        since[~np.isfinite(since)] = np.nan
        # Convert rows to minutes
        since_minutes = since * self.minutes_per_row
        return since_minutes

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() (or from_state()) before transform().")
        if not {self.sensor_col, self.value_col}.issubset(X.columns):
            raise ValueError("MomentumFeatureEngineer: required columns missing.")

        df = X.copy()
        feat_cols: List[str] = []

        # Ensure sorting by time within sensor if datetime provided
        if self.datetime_col in df.columns:
            df = df.sort_values([self.sensor_col, self.datetime_col])

        g = df.groupby(self.sensor_col, sort=False)
        s = df[self.value_col]

        # 1) Rolling OLS slope
        for w in self.slope_windows:
            col = f"slope_w{w}"
            func = self._rolling_ols_slope_func(w)
            # groupby.rolling returns a MultiIndex; use .values to align back
            slope_vals = (
                g[self.value_col]
                .rolling(window=w, min_periods=w)
                .apply(func, raw=True)
                .reset_index(level=0, drop=True)
            )
            df[col] = slope_vals.fillna(self.fill_nans_value).astype(np.float32)
            feat_cols.append(col)

        # 2) EWMA mean and EWMA slope (diff of EWMA mean)
        # Use a multi-index (sensor, row) for group-wise ewm
        s_by_sensor = s.copy()
        s_by_sensor.index = pd.MultiIndex.from_arrays(
            [df[self.sensor_col].values, df.index.values],
            names=[self.sensor_col, "_row"]
        )
        for h in self.ewm_halflives:
            m_col = f"ewm_mean_h{int(h) if float(h).is_integer() else h}"
            d_col = f"ewm_slope_h{int(h) if float(h).is_integer() else h}"

            ewm_mean = self._per_sensor_ewm(s_by_sensor, halflife=float(h))
            ewm_mean.index = ewm_mean.index.get_level_values("_row")
            ewm_mean = ewm_mean.reindex(df.index)

            df[m_col] = ewm_mean.astype(np.float32)
            df[d_col] = ewm_mean.diff().fillna(self.fill_nans_value).astype(np.float32)

            # Fill initial NaNs in mean too (keep actual means if you prefer NaNs)
            df[m_col] = df[m_col].fillna(self.fill_nans_value).astype(np.float32)

            feat_cols.extend([m_col, d_col])

        # 3) Volatility (std, MAD)
        for w in self.vol_windows:
            std_col = f"std_w{w}"
            mad_col = f"mad_w{w}"
            df[std_col] = (
                g[self.value_col].rolling(window=w, min_periods=w).std()
                .reset_index(level=0, drop=True)
                .fillna(self.fill_nans_value)
                .astype(np.float32)
            )
            df[mad_col] = (
                g[self.value_col].rolling(window=w, min_periods=w)
                .apply(lambda a: self._rolling_mad(a), raw=True)
                .reset_index(level=0, drop=True)
                .fillna(self.fill_nans_value)
                .astype(np.float32)
            )
            feat_cols.extend([std_col, mad_col])

        # 4) Recent extrema + distances
        for w in self.minmax_windows:
            min_col = f"roll_min_w{w}"
            max_col = f"roll_max_w{w}"
            dmin_col = f"dist_to_min_w{w}"
            dmax_col = f"dist_to_max_w{w}"

            roll_min = (
                g[self.value_col].rolling(window=w, min_periods=w).min()
                .reset_index(level=0, drop=True)
            )
            roll_max = (
                g[self.value_col].rolling(window=w, min_periods=w).max()
                .reset_index(level=0, drop=True)
            )

            df[min_col] = roll_min.fillna(self.fill_nans_value).astype(np.float32)
            df[max_col] = roll_max.fillna(self.fill_nans_value).astype(np.float32)
            df[dmin_col] = (df[self.value_col] - df[min_col]).astype(np.float32)
            df[dmax_col] = (df[max_col] - df[self.value_col]).astype(np.float32)

            # Fill any NaNs from arithmetic if present
            for c in (dmin_col, dmax_col):
                df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(self.fill_nans_value)

            feat_cols.extend([min_col, max_col, dmin_col, dmax_col])

        # 5) Minutes since last below-threshold (per threshold)
        for thr in self.thresholds_kph:
            col = f"mins_since_below_{int(thr) if float(thr).is_integer() else thr}"
            df[col] = (
                g.apply(lambda sub: self._time_since_below_threshold(sub, threshold=float(thr)))
                 .reset_index(level=0, drop=True)
                 .fillna(self.fill_nans_value)
                 .astype(np.float32)
            )
            feat_cols.append(col)

        # 6) Optional fast-drop flag from a short slope window
        if self.drop_fast_flag:
            # If the corresponding slope column doesn't exist yet (e.g., window not in slope_windows),
            # compute it on the fly; otherwise just threshold the existing column.
            slope_col = f"slope_w{self.fast_flag_window}"
            if slope_col not in df.columns:
                func = self._rolling_ols_slope_func(self.fast_flag_window)
                tmp = (
                    g[self.value_col]
                    .rolling(window=self.fast_flag_window, min_periods=self.fast_flag_window)
                    .apply(func, raw=True)
                    .reset_index(level=0, drop=True)
                )
                df[slope_col] = tmp.fillna(self.fill_nans_value).astype(np.float32)
                feat_cols.append(slope_col)

            flag_col = f"is_dropping_fast_w{self.fast_flag_window}_thr{self.fast_flag_thresh}"
            df[flag_col] = (df[slope_col] < self.fast_flag_thresh).astype(np.int8)
            feat_cols.append(flag_col)

        # Record names in the canonical place
        self.feature_names_out_ = feat_cols
        return df

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "momentum_pack",
            "sensor_col": self.sensor_col,
            "value_col": self.value_col,
            "datetime_col": self.datetime_col,
            "slope_windows": list(self.slope_windows),
            "ewm_halflives": list(self.ewm_halflives),
            "vol_windows": list(self.vol_windows),
            "minmax_windows": list(self.minmax_windows),
            "thresholds_kph": list(self.thresholds_kph),
            "minutes_per_row": self.minutes_per_row,
            "drop_fast_flag": self.drop_fast_flag,
            "fast_flag_window": self.fast_flag_window,
            "fast_flag_thresh": self.fast_flag_thresh,
            "fill_nans_value": self.fill_nans_value,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MomentumFeatureEngineer":
        inst = cls(
            sensor_col=state["sensor_col"],
            value_col=state["value_col"],
            datetime_col=state.get("datetime_col", "date"),
            slope_windows=state.get("slope_windows", (5, 10, 15, 30)),
            ewm_halflives=state.get("ewm_halflives", (5.0, 10.0)),
            vol_windows=state.get("vol_windows", (10, 30)),
            minmax_windows=state.get("minmax_windows", (15, 30)),
            thresholds_kph=state.get("thresholds_kph", (70.0, 80.0, 90.0)),
            minutes_per_row=state.get("minutes_per_row", 1.0),
            drop_fast_flag=state.get("drop_fast_flag", True),
            fast_flag_window=state.get("fast_flag_window", 5),
            fast_flag_thresh=state.get("fast_flag_thresh", -1.0),
            fill_nans_value=state.get("fill_nans_value", -1.0),
            epsilon=state.get("epsilon", 1e-6),
        )
        inst.fitted_ = True
        return inst