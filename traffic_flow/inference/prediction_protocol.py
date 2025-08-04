# traffic_flow/inference/prediction_protocol.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Optional
import warnings

def make_prediction_frame(
    *,
    df_raw: pd.DataFrame,
    df_for_ML: Optional[pd.DataFrame] = None,
    feats: Optional[pd.DataFrame] = None,
    pred_delta: np.ndarray | pd.Series,
    states: dict,
    horizon_min: int,
    add_total: bool = True,
    add_y_act: bool = True,
    target_col: str = 'target_total_speed'
) -> pd.DataFrame:
    """
    Construct a standardized prediction DataFrame.

    This function ensures consistent prediction output formatting across 
    different sources (e.g., training, local inference, API).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input DataFrame used for prediction (must include datetime and sensor ID).
    df_for_ML : Optional[pd.DataFrame], default=None
        Processed input DataFrame used in the ML pipeline.
        Required if `feats` is not provided or if `add_y_act=True` (since it contains ground truth values).
    feats : Optional[pd.DataFrame], default=None
        Feature DataFrame used to compute predictions. Required if `df_for_ML` is not provided.
    pred_delta : np.ndarray or pd.Series
        The model's predicted deltas to apply to the current value to get the total prediction.
    states : dict
        Dictionary of pipeline metadata, including schema and datetime info.
    horizon_min : int
        Forecasting horizon in minutes (used to compute `prediction_time`).
    add_total : bool, default=True
        Whether to compute and include the column `y_pred_total = y_pred_delta + current_value`.
    add_y_act : bool, default=True
        Whether to include the actual target values (`y_act_total`) from `df_for_ML[target_col]`.
        If set to True, `df_for_ML` must be provided and contain `target_col`.
    target_col : str, default='target_total_speed'
        Name of the column in `df_for_ML` containing actual target values.

    Returns
    -------
    pd.DataFrame
        A canonical prediction DataFrame with columns:
        - sensor_id
        - input_time
        - prediction_time
        - y_pred_delta
        - y_pred_total (optional)
        - y_act_total (optional)

    Raises
    ------
    ValueError
        If both `df_for_ML` and `feats` are missing.
        If `add_total=True` but `value_col` is missing in the selected features.
        If `add_y_act=True` but `df_for_ML` is missing or lacks `target_col`.
    """

    if df_for_ML is None and feats is None:
        raise ValueError("At least one of df_for_ML or feats must be provided.")

    if df_for_ML is not None and 'test_set' in df_for_ML.columns:
        df_for_ML = df_for_ML[df_for_ML['test_set']]
    
    # Ensure alignment to the length of predictions
    n = int(len(pred_delta))
    df_raw = df_raw.iloc[:n].copy()
    df_for_ML = df_for_ML.iloc[:n].copy() if df_for_ML is not None else None
    feats = feats.iloc[:n].copy() if feats is not None else None

    # Retrieve column names from state dictionary
    dt_col = states.get("datetime_state", {}).get("datetime_col", "date")
    sensor_col = states.get("schema_state", {}).get("sensor_col", "sensor_id")
    value_col = states.get("schema_state", {}).get("value_col", "value")

    # Convert input datetime column to pandas datetime format
    dt = pd.to_datetime(df_raw[dt_col], errors="coerce")

    # Construct the output DataFrame
    out = pd.DataFrame({
        "input_time": dt,
        "sensor_id": df_raw[sensor_col],
        "prediction_time": dt + pd.to_timedelta(horizon_min, unit="m"),
        "y_pred_delta": np.asarray(pred_delta, dtype=float),
    })

    # Optionally compute and include the total predicted speed
    if add_total:
        if feats is not None:
            if value_col not in feats.columns:
                raise ValueError("Cannot compute y_pred_total (missing 'value' in features).")
            out["y_pred_total"] = out["y_pred_delta"] + feats[value_col].to_numpy(dtype=float)
        elif df_for_ML is not None:
            out["y_pred_total"] = out["y_pred_delta"] + df_for_ML[value_col].to_numpy(dtype=float)
        else:
            raise ValueError("Cannot compute y_pred_total (no features provided).")

    # Sort by input time and sensor ID (to match training convention)
    out = out.sort_values(["input_time", "sensor_id"], kind="mergesort").reset_index(drop=True)

    # Optionally include the actual observed values
    if add_y_act:
        if df_for_ML is None or target_col not in df_for_ML.columns:
            raise ValueError(f"Cannot compute y_act_total (missing {target_col} in df_for_ML).")
        #out["y_act_total"] = df_for_ML[target_col].to_numpy(dtype=float)
        # Merge with out using prediction_time + sensor_id
        df_for_ML.rename(columns={target_col: "y_act_total"}, inplace=True)
        out = out.merge(df_for_ML[["prediction_time",sensor_col,"y_act_total"]], on=["prediction_time", sensor_col], how="left", validate="one_to_one")
        if out["y_act_total"].isna().any():
            warnings.warn("Some rows could not be matched to actual targets (y_act_total is NaN).")

    return out




# def make_prediction_frame(
#     *,
#     df_raw: pd.DataFrame,
#     df_for_ML: pd.DataFrame = None,
#     feats: pd.DataFrame = None,
#     pred_delta: np.ndarray | pd.Series,
#     states: dict,
#     horizon_min: int,
#     add_total: bool = True,
#     add_y_act: bool = True,
#     target_col: str = 'target_total_speed'
# ) -> pd.DataFrame:
#     """
#     Build a canonical prediction DataFrame so every producer (training,
#     local inference, API) returns the same structure.

#     Returns columns:
#       sensor_id, input_time, prediction_time, y_pred_delta, y_pred_total, horizon
#     """
    
#     if df_for_ML is None and feats is None:
#         raise ValueError("At least one of df_for_ML or feats must be provided.")
    
    
#     # Ensure alignment to prediction length
#     n = int(len(pred_delta))
#     df_raw = df_raw.iloc[:n].copy()
#     df_for_ML  = df_for_ML.iloc[:n].copy() if df_for_ML is not None else None
#     feats = feats.iloc[:n].copy() if feats is not None else None
#     dt_col = states["datetime_state"].get('datetime_col','date') # usually date
#     sensor_col = states["schema_state"].get('sensor_col','sensor_id')  # usually sensor_id
#     value_col = states["schema_state"].get('value_col', 'value')
#     print(f"dt_col: {dt_col}")
#     dt = pd.to_datetime(df_raw[dt_col], errors="coerce")

#     out = pd.DataFrame(
#         {   
#             "input_time": dt,
#             "sensor_id": df_raw[sensor_col],
#             #"input_time": dt.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
#             "prediction_time": (dt + pd.to_timedelta(horizon_min, unit="m")),
#             # .dt.strftime("%Y-%m-%d %H:%M:%S")
#             # .to_numpy(),
#             "y_pred_delta": np.asarray(pred_delta, dtype=float),
#         }
#     )

#     if add_total:
#         if feats is not None:
#             if value_col not in feats.columns:
#                 raise ValueError("Cannot compute y_pred_total (missing 'value' in features).")
#             out["y_pred_total"] = out["y_pred_delta"] + feats["value"].to_numpy(dtype=float)
#         elif df_for_ML is not None:
#             out["y_pred_total"] = out["y_pred_delta"] + df_for_ML[value_col].to_numpy(dtype=float)
#         else:
#             raise ValueError("Cannot compute y_pred_total (no features provided).")

#     # Canonical sort (matches training sorting convention)
#     out = out.sort_values(["input_time", "sensor_id"], kind="mergesort").reset_index(drop=True)
    
#     if add_y_act:
#         if target_col not in df_for_ML.columns:
#             raise ValueError(f"Cannot compute y_act_total (missing {target_col} in features).")
#         out["y_act_total"] = df_for_ML[target_col].to_numpy(dtype=float)
#     return out