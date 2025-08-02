# traffic_flow/inference/prediction_protocol.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Optional

def make_prediction_frame(
    *,
    df: pd.DataFrame,
    feats: pd.DataFrame,
    pred_delta: np.ndarray | pd.Series,
    states: dict,
    horizon_min: int,
    add_total: bool = True,
    add_y_act: bool = False,
    datetime_col: str = 'datetime_col',
    sensor_col: str = "sensor_id",
    value_col: str = 'value',
    target_col: str = 'target_total_speed'
) -> pd.DataFrame:
    """
    Build a canonical prediction DataFrame so every producer (training,
    local inference, API) returns the same structure.

    Returns columns:
      sensor_id, input_time, prediction_time, y_pred_delta, y_pred_total, horizon
    """
    # Ensure alignment to prediction length
    n = int(len(pred_delta))
    df = df.iloc[:n].copy()
    feats = feats.iloc[:n].copy()
    dt_col = states["datetime_state"][datetime_col]  # usually "date"
    dt = pd.to_datetime(df[dt_col], errors="coerce")

    out = pd.DataFrame(
        {
            "sensor_id": df[sensor_col],
            "input_time": dt.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "prediction_time": (dt + pd.to_timedelta(horizon_min, unit="m"))
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .to_numpy(),
            "y_pred_delta": np.asarray(pred_delta, dtype=float),
            "horizon": int(horizon_min)
        }
    )

    if add_total:
        if value_col not in feats.columns:
            raise ValueError("Cannot compute y_pred_total (missing 'value' in features).")
        out["y_pred_total"] = out["y_pred_delta"] + feats["value"].to_numpy(dtype=float)

    # Canonical sort (matches your training sorting convention)
    out = out.sort_values(["input_time", "sensor_id"], kind="mergesort").reset_index(drop=True)
    
    if add_y_act:
        if target_col not in df.columns:
            raise ValueError("Cannot compute y_pred_total (missing 'target_total_speed' in features).")
        out["y_act_total"] = df[target_col].to_numpy(dtype=float)
    return out