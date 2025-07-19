"""
Shared, stateless utilities used by BOTH the training data-loader
and the inference pre-processor.  They do NO I/O, NO train/test split.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

Float = np.float32

# ------------------------------------------------------------ #
# 1) numeric coercion
# ------------------------------------------------------------ #
def clean_value(v: Any) -> Float | None:
    """Coerce strings like '1 000,5' → 1000.5 and cast to float32."""
    try:
        if isinstance(v, float):
            return np.float32(round(v, 2))
        return np.float32(round(float(str(v).replace(" ", ".")), 2))
    except (ValueError, AttributeError):
        return None            # will be dropped/interpolated later

def clean_and_cast(df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[value_col] = df[value_col].apply(clean_value).astype(Float)
    return df

# ------------------------------------------------------------ #
# 2) extreme-change filter
# ------------------------------------------------------------ #
def filter_and_interpolate_extremes(
    df: pd.DataFrame,
    *,
    sensor_col: str,
    value_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Replace Δ% > threshold with NaN then interpolate."""
    df = df.copy()
    pct = df.groupby(sensor_col)[value_col].pct_change().abs()
    mask = pct > threshold
    df.loc[mask, value_col] = np.nan
    df[value_col] = (
        df.groupby(sensor_col)[value_col]
          .transform(lambda s: s.interpolate().ffill().bfill())
    )
    return df

# ------------------------------------------------------------ #
# 3) smoothing
# ------------------------------------------------------------ #
def smooth_speeds(
    df: pd.DataFrame,
    *,
    sensor_col: str,
    value_col: str,
    window_size: int,
    use_median: bool,
) -> pd.DataFrame:
    df = df.copy()
    def roll(s):
        r = s.rolling(window_size, min_periods=1)
        return r.median() if use_median else r.mean()

    df[value_col] = (
        df.groupby(sensor_col)[value_col].transform(roll).ffill().bfill()
    )
    method = "med" if use_median else "mean"
    df.attrs["smoothing_id"] = f"win{window_size}_{method}"
    return df