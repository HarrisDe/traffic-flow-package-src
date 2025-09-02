from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..utils.helper_utils import LoggingMixin
from .plotting import ScatterPlotterSeaborn
from .plotting import TimeSeriesPlotter




def assemble_results_dataframe(
    *,
    meta: dict,
    y_pred: np.ndarray,
    value_col: str,                      # sensor name, e.g. "RWS01_MONIBAS_0040vwe0633ra"
) -> pd.DataFrame:
    """
    Build a per-sensor dataframe aligned on:
      - date             -> issue time t   (when the model made the prediction)
      - prediction_time  -> target time t+H (what the forecast refers to)
      - <sensor>_at_issued_time -> sensor's actual at t
      - <sensor>               -> sensor's actual at t+H   (y_true)
      - <sensor>_pred          -> forecast for t+H issued at t

    Expected meta (as you already pass in TrafficDeepExperiment.run()):
      meta = {
        "y":             y_true_target_space (N, 1),
        "base_times":    base_times (N,),              # issue time t
        "base_values":   base_values (N,),             # sensor actual at t
        "pred_times":    {h_label: target_times (N,)}, # t+H
        "h_labels":      [h_label],
        "target_mode":   "delta" or "abs",
      }
    """
    h_label = meta["h_labels"][0]
    base_times  = pd.to_datetime(meta["base_times"])                  # t
    target_times = pd.to_datetime(meta["pred_times"][h_label])        # t+H
    base_vals   = np.asarray(meta["base_values"]).reshape(-1)         # (N,)
    y_true      = np.asarray(meta["y"]).reshape(-1)                   # (N,)
    y_pred      = np.asarray(y_pred).reshape(-1)                      # (N,)

    # Convert to absolute space if needed
    if str(meta.get("target_mode", "abs")).lower() == "delta":
        y_true_abs = base_vals + y_true
        y_pred_abs = base_vals + y_pred
    else:
        y_true_abs = y_true
        y_pred_abs = y_pred

    df = pd.DataFrame({
        "date":             base_times,          # issue time t
        "prediction_time":  target_times,        # target time t+H
        f"{value_col}_at_issued_time": base_vals,
        value_col:          y_true_abs,          # actual at t+H
        f"{value_col}_pred": y_pred_abs,         # forecast for t+H issued at t
    })

    # Keep stable ordering
    df = df.sort_values(["date", "prediction_time"], kind="stable").reset_index(drop=True)
    return df




def _valid_pairs(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    rounding: int = 2,
    eps: float = 1e-12,
    is_naive= False
) -> Dict[str, float]:
    """
    Compute MAE, MedianAE, RMSE, MAPE, SMAPE (and optionally DTW) over a 1D series.

    MAPE ignores positions where y_true is zero. SMAPE treats 0/0 as zero contribution.
    When show_dtw is True, the function also returns Dynamic Time Warping distance
    between the two sequences (computed with tslearn).
    """
    y_true, y_pred = _valid_pairs(y_true, y_pred)

    ae = np.abs(y_true - y_pred)
    mae = np.mean(ae)
    medae = np.median(ae)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denom_mape = np.where(y_true == 0, np.nan, np.abs(y_true))
    mape = np.nanmean(ae / denom_mape) * 100.0

    denom_smape = np.abs(y_true) + np.abs(y_pred)
    ratio = np.where(denom_smape == 0, 0.0, (2.0 * ae) / (denom_smape + eps))
    smape = np.mean(ratio) * 100.0

    if is_naive:
        out = {"MAE_naive": mae, "MedianAE_naive": medae, "RMSE_naive": rmse, "MAPE_naive": mape, "SMAPE_naive": smape}
    else:
        out = {"MAE": mae, "MedianAE": medae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}

    return {k: float(np.round(v, rounding)) for k, v in out.items()}


class ModelEvaluator(LoggingMixin): 
    """
    Evaluate a single-horizon results frame at the **target time (prediction_time = t+H)**.

    This class aligns true vs predicted on the *same* target timestamp and computes:
    MAE, MedianAE, RMSE, MAPE, SMAPE. It also computes a naive persistence baseline
    when available.

    Supported result schemas
    ------------------------
    1) Sensor schema (from the new assemble_results_dataframe):
       - date                        -> issue time t
       - prediction_time             -> target time t+H
       - <sensor>                    -> actual at t+H
       - <sensor>_pred               -> forecast for t+H, issued at t
       - <sensor>_at_issued_time     -> actual at t   (used as naive baseline)

    2) Generic schema (old style):
       - y_true_abs_<H>, y_pred_abs_<H> (or reconstructed from deltas)
       - value_col column may hold base/current value at t (used as naive baseline)
       - prediction_time_<H> or prediction_time may be present; otherwise datetime_col is used

    Notes
    -----
    * Errors are computed at the **target time** (prediction_time), never at the issue time.
      Example: at 00:00 you forecast for 00:15. We compare that forecast to the actual at 00:15.
    * NaNs and zero-denominators in MAPE/SMAPE are handled in `compute_metrics`.
    """

    def __init__(
        self,
        *,
        df_res: pd.DataFrame,
        horizon: str,
        value_col: str,
        datetime_col: str = "date",                 # issue time column name (fallback axis)
        prediction_time_col: Optional[str] = None,  # if None, it will be auto-detected
        test_flag_col: str = "test_set",
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        self.df_res = df_res.copy()
        self.horizon = str(horizon)
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.test_flag_col = test_flag_col

        # Filter to test rows if present
        if self.test_flag_col in self.df_res.columns:
            self.df_res = self.df_res[self.df_res[self.test_flag_col].astype(bool)].reset_index(drop=True)

        # Pick the time column for *evaluation at target time*
        self.prediction_time_col = prediction_time_col or self._choose_prediction_time_col(self.df_res, self.horizon)

        # Detect schema & extract arrays
        (
            self.y_true_abs,
            self.y_pred_abs,
            self.y_pred_naive_abs,  # may be None
        ) = self._extract_series(self.df_res, self.horizon, self.value_col)

        # Store time axis (target time preferred)
        self.time = pd.to_datetime(self.df_res[self.prediction_time_col])

        # Outputs
        self.metrics_: Dict[str, float] = {}
        self.naive_metrics_: Dict[str, float] = {}

    # --------------------------- public API --------------------------- #
    def evaluate(self, *, rounding: int = 2) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute metrics (model + naive when available) on absolute values,
        aligned by the target time column.
        """
        self.metrics_ = compute_metrics(self.y_true_abs, self.y_pred_abs, rounding=rounding)

        naive: Dict[str, float] = {}
        if self.y_pred_naive_abs is not None:
            naive = compute_metrics(self.y_true_abs, self.y_pred_naive_abs, rounding=rounding, is_naive=True)

        self.naive_metrics_ = naive
        self._log(f"[Evaluator] Metrics      : {self.metrics_}")
        if naive:
            self._log(f"[Evaluator] Naive metrics: {self.naive_metrics_}")
        else:
            self._log("[Evaluator] Naive metrics: (not available)")
        return self.metrics_, self.naive_metrics_

    # -------------------------- internals --------------------------- #
    @staticmethod
    def _choose_prediction_time_col(df: pd.DataFrame, horizon: str) -> Optional[str]:
        """Prefer explicit prediction-time columns; fallback to None."""
        for cand in (f"prediction_time_{horizon}", "prediction_time", "prediction_datetime"):
            if cand in df.columns:
                return cand
        return None

    @staticmethod
    def _resolve_absolute_targets_generic(df: pd.DataFrame, horizon: str, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic schema: return (y_true_abs, y_pred_abs), reconstructing from deltas if needed.
        """
        true_abs_col = f"y_true_abs_{horizon}"
        pred_abs_col = f"y_pred_abs_{horizon}"
        if true_abs_col in df.columns and pred_abs_col in df.columns:
            yt = df[true_abs_col].astype(float).to_numpy()
            yp = df[pred_abs_col].astype(float).to_numpy()
            return yt, yp

        # Fallback from deltas
        true_d_col = f"y_true_delta_{horizon}"
        pred_d_col = f"y_pred_delta_{horizon}"
        if value_col not in df.columns or true_d_col not in df.columns or pred_d_col not in df.columns:
            raise ValueError(
                "Cannot resolve absolute targets. Expected either "
                f"('{true_abs_col}','{pred_abs_col}') or "
                f"('{value_col}','{true_d_col}','{pred_d_col}')."
            )
        base = df[value_col].astype(float).to_numpy()
        yt = base + df[true_d_col].astype(float).to_numpy()
        yp = base + df[pred_d_col].astype(float).to_numpy()
        return yt, yp

    def _extract_series(
        self,
        df: pd.DataFrame,
        horizon: str,
        value_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Detect the schema and extract:
            y_true_abs, y_pred_abs, y_pred_naive_abs (or None)
        All arrays are aligned row-wise (same length, same target times).
        """
        # ---- Sensor schema (preferred when sensor columns exist) ----
        sensor_pred_col = f"{value_col}_pred"
        sensor_issue_col = f"{value_col}_at_issued_time"
        if value_col in df.columns and sensor_pred_col in df.columns:
            # Evaluate at target time; df rows are already aligned on (date, prediction_time)
            y_true_abs = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
            y_pred_abs = pd.to_numeric(df[sensor_pred_col], errors="coerce").to_numpy()

            # Naive persistence if available (actual at issued time t)
            y_naive = None
            if sensor_issue_col in df.columns:
                y_naive = pd.to_numeric(df[sensor_issue_col], errors="coerce").to_numpy()

            return y_true_abs, y_pred_abs, y_naive

        # ---- Generic schema (old style) ----
        y_true_abs, y_pred_abs = self._resolve_absolute_targets_generic(df, horizon, value_col)

        # Naive from base/current value if present
        y_naive = None
        if value_col in df.columns:
            y_naive = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

        return y_true_abs, y_pred_abs, y_naive
    
    


