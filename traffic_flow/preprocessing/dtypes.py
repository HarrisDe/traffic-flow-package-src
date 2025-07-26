# traffic_flow/preprocessing/dtypes.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Dict, Iterable

# ---- Rule-based policy for lightweight types -----------------------
# These patterns can be expanded later new feature families are added.
_FLOAT32_PREFIXES = (
    r"^value$",
    r"^longitude$",
    r"^latitude$",
    r"^lag\d+$",
    r"^relative_diff_lag\d+$",
    r"^downstream_sensor_\d+$",
    r"^upstream_sensor_\d+$",
    r"^prev_wd_.*",                # previous weekday features
    r"^gman_prediction$",
    r"^weather_.*",                # if you keep weather columns
)

_INT8_COLUMNS = ("hour", "day", "is_saturday", "is_sunday", "is_congested", "is_outlier")
_INT32_COLUMNS = ("sensor_uid",)   # safe for many sensors; change to int16 if you’re sure


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) is not None for p in patterns)

def build_dtype_schema(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a canonical, lightweight dtype schema for `df` based on rules above.
    Returns a pandas-compatible astype() mapping, e.g. {"value": "float32", ...}.
    """
    schema: Dict[str, str] = {}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # leave datetimes as they are
            continue

        if col in _INT8_COLUMNS:
            # ensure there are no NaNs; cast after filling elsewhere if needed
            schema[col] = "int8"
            continue

        if col in _INT32_COLUMNS:
            schema[col] = "int32"
            continue

        if _matches_any(col, _FLOAT32_PREFIXES):
            schema[col] = "float32"
            continue

        # Fallbacks:
        if pd.api.types.is_float_dtype(df[col]):
            schema[col] = "float32"
        elif pd.api.types.is_integer_dtype(df[col]):
            # Downcast integers safely using pandas logic
            # (this keeps tiny columns as int8/int16 where possible)
            try:
                down = pd.to_numeric(df[col], downcast="integer")
                schema[col] = str(down.dtype)
            except Exception:
                schema[col] = "int32"
        else:
            # leave object/categorical/etc. alone (shouldn't exist in model inputs)
            pass

    return schema


def enforce_dtypes(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a dtype schema safely, only for columns present.
    """
    to_apply = {c: dt for c, dt in schema.items() if c in df.columns}
    if not to_apply:
        return df
    return df.astype(to_apply, copy=False)