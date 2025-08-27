from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import pandas as pd




def make_demand_context_feature_fn(
    *,
    datetime_col: str = "Datetime",
    value_col: str = "PJME_MW",
    windows: Iterable[int] = (24, 168),
    add_deviation: bool = True,
    add_zscore: bool = True,
    add_ratio: bool = False,
    min_periods: Optional[int] = None,  # None -> use each window size
    include_business_hour: bool = True,
    include_weekend_flag: bool = True,
    holiday_dates: Optional[Iterable[str]] = None,  # e.g. ["2017-12-25", "2018-01-01"]
) -> callable:
    """
    Build a callable(df)->DataFrame that adds rolling demand context features.

    For each window w in `windows`, the function computes rolling mean μ_w and std σ_w
    using only past/current values at time t. It then adds:
      - deviation:  value - μ_w
      - zscore:     (value - μ_w) / σ_w
      - ratio:      value / μ_w

    It also adds light calendar flags (business hours, weekend) and optional holiday flag.

    All outputs are numeric float32 and aligned row-for-row with the input df.

    Parameters
    ----------
    datetime_col : str
        Name of the datetime column in df.
    value_col : str
        Name of the load column in df.
    windows : Iterable[int]
        Rolling window sizes in hours (e.g., 24 for daily, 168 for weekly).
    add_deviation, add_zscore, add_ratio : bool
        Which transforms to include.
    min_periods : Optional[int]
        Minimum periods for rolling stats. If None, uses each window size w.
        Use 1 to avoid early NaNs, or w to be stricter.
    include_business_hour, include_weekend_flag : bool
        Add simple calendar indicators.
    holiday_dates : Optional[Iterable[str]]
        A list of holiday dates (YYYY-MM-DD) to flag, if available.

    Returns
    -------
    callable
        A function you can pass as `custom_feature_fn` to TFWindowedDatasetBuilder.
    """
    windows = tuple(int(w) for w in windows)

    def _feature_fn(df: pd.DataFrame) -> pd.DataFrame:
        d = df[[datetime_col, value_col]].copy()
        dt = pd.to_datetime(d[datetime_col], errors="coerce")
       
        # IMPORTANT: keep the same index so assignments align row-for-row
        s = pd.Series(
        np.asarray(d[value_col], dtype=np.float32),
        index=df.index,
        dtype="float32",
        copy=False,
        )
        
        out = pd.DataFrame(index=df.index)  # keep same row order/length

        # Rolling stats & derived transforms
        for w in windows:
            mp = w if min_periods is None else int(min_periods)
            # pandas rolling preserves alignment; convert to float32 at the end
            mu = s.rolling(window=w, min_periods=mp).mean()
            sd = s.rolling(window=w, min_periods=mp).std()

            if add_deviation:
                out[f"dev_{w}h"] = (s - mu).astype("float32").fillna(0.0)
            if add_zscore:
                # avoid division by zero; fill remaining NaNs with 0
                z = (s - mu) / sd.replace(0.0, np.nan)
                out[f"z_{w}h"] = z.astype("float32").fillna(0.0)
            if add_ratio:
                r = s / mu.replace(0.0, np.nan)
                out[f"ratio_{w}h"] = r.astype("float32").fillna(0.0)

        # Simple calendar flags (numeric)
        if include_business_hour:
            out["is_business_hour"] = ((dt.dt.hour >= 8) & (dt.dt.hour <= 18)).astype("float32")
        if include_weekend_flag:
            out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")

        if holiday_dates is not None:
            hol = pd.to_datetime(pd.Index(holiday_dates)).normalize()
            out["is_holiday"] = dt.dt.normalize().isin(hol).astype("float32")

        return out
    
    _feature_fn.__name__ = "demand_context_features"
    _feature_fn._dc_config = {
        "datetime_col": datetime_col,
        "value_col": value_col,
        "windows": list(windows),
        "add_deviation": add_deviation,
        "add_zscore": add_zscore,
        "add_ratio": add_ratio,
        "min_periods": (None if min_periods is None else int(min_periods)),
        "include_business_hour": include_business_hour,
        "include_weekend_flag": include_weekend_flag,
        "holiday_dates_count": (len(tuple(holiday_dates)) if holiday_dates is not None else 0),}

    return _feature_fn

def adapt_single_series_context_fn(single_series_fn, *, datetime_col: str = "date", ref_col: str = "value_ref"):
    """
    Wrap a callable(df_single)->DataFrame into df_wide->DataFrame by
    creating a reference series (row-wise mean across sensors).
    """
    def _wide_fn(df_wide: pd.DataFrame) -> pd.DataFrame:
        sensors = [c for c in df_wide.columns if c not in (datetime_col, "test_set")]
        tmp = pd.DataFrame({
            datetime_col: pd.to_datetime(df_wide[datetime_col]),
            ref_col: df_wide[sensors].mean(axis=1).astype(np.float32)
        }, index=df_wide.index)
        # Call the single-series function (it must be configured to expect ref_col)
        return single_series_fn(tmp)
    _wide_fn.__name__ = f"wide_{getattr(single_series_fn, '__name__', 'custom')}"
    return _wide_fn

def make_per_sensor_context_fn(
    *,
    datetime_col: str = "date",
    windows: Iterable[int] = (60, 1440),  # minutes: 1h, 1d 1-min data
    add_deviation: bool = True,
    add_zscore: bool = True,
    add_ratio: bool = False,
    min_periods: Optional[int] = None,
    include_business_hour: bool = True,
    include_weekend_flag: bool = True,
) -> callable:
    """
    Returns df_wide -> DataFrame (numeric) with per-sensor rolling features.
    Windows are in *rows* (minutes for your data).
    """
    windows = tuple(int(w) for w in windows)

    def _fn(df_wide: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df_wide[datetime_col], errors="coerce")
        sensors = [c for c in df_wide.columns if c not in (datetime_col, "test_set")]
        vals = df_wide[sensors].astype(np.float32)

        out = []

        for w in windows:
            mp = w if (min_periods is None) else int(min_periods)
            mu = vals.rolling(window=w, min_periods=mp).mean()
            sd = vals.rolling(window=w, min_periods=mp).std()

            if add_deviation:
                dev = (vals - mu).astype(np.float32)
                dev.columns = [f"{c}__dev_{w}m" for c in sensors]
                out.append(dev.fillna(0.0))

            if add_zscore:
                z = (vals - mu) / sd.replace(0.0, np.nan)
                z = z.astype(np.float32)
                z.columns = [f"{c}__z_{w}m" for c in sensors]
                out.append(z.fillna(0.0))

            if add_ratio:
                r = vals / mu.replace(0.0, np.nan)
                r = r.astype(np.float32)
                r.columns = [f"{c}__ratio_{w}m" for c in sensors]
                out.append(r.fillna(0.0))

        # light calendar flags (global, numeric)
        flags = pd.DataFrame(index=df_wide.index)
        if include_business_hour:
            flags["is_business_hour"] = ((dt.dt.hour >= 8) & (dt.dt.hour <= 18)).astype("float32")
        if include_weekend_flag:
            flags["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
        out.append(flags)

        return pd.concat(out, axis=1) if out else pd.DataFrame(index=df_wide.index)

    _fn.__name__ = "per_sensor_context_features"
    return _fn