from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd




def assemble_results_dataframe(
    *,
    meta: Dict,
    y_pred: np.ndarray,
    datetime_col: str = "Datetime",
    target_mode: Optional[str] = None,
    value_col: str = "PJME_MW",
) -> pd.DataFrame:
    """
    Assemble a *wide* evaluation-ready DataFrame from test metadata and predictions.

    Behavior depends on the target mode:
      - If target_mode == "absolute":
          y_true_<h>, y_pred_<h> are absolute values (MW).
          Also adds y_true_delta_<h>, y_pred_delta_<h> if base_values are present.

      - If target_mode == "delta":
          y_true_<h>, y_pred_<h> are deltas (MW-change).
          Also adds y_true_abs_<h>, y_pred_abs_<h> if base_values are present.

    Parameters
    ----------
    meta : Dict
        Metadata returned by TFWindowedDatasetBuilder.build_split(...). Expected keys:
        - "y"           : (N, n_outputs) ground-truth targets (absolute or delta)
        - "base_times"  : (N,) base timestamps (window endpoints)
        - "base_values" : (N,) values at base timestamps (optional but recommended)
        - "pred_times"  : dict[label] -> (N,) prediction timestamps per horizon
        - "h_labels"    : list[str] horizon labels
        - "target_mode" : "absolute" | "delta" (optional if provided explicitly)
    y_pred : np.ndarray
        (N, n_outputs) model predictions in the SAME space as meta["y"].
    datetime_col : str
        Name of the base timestamp column in the output dataframe.
    target_mode : Optional[str]
        If provided, overrides meta["target_mode"]. Must be "absolute" or "delta".

    Returns
    -------
    pd.DataFrame
        Wide dataframe with base time, per-horizon prediction times, and per-horizon
        y_true / y_pred in the requested target space, plus convenience columns to
        convert to the other space whenever base_values are available.
    """
    # Core fields
    y_true = meta["y"]
    base_times = pd.to_datetime(meta["base_times"])
    base_vals  = meta.get("base_values", None)
    h_labels: List[str] = list(meta["h_labels"])

    # Mode selection (explicit first, meta fallback second, default "absolute")
    mode = target_mode or meta.get("target_mode", "absolute")
    if mode not in ("absolute", "delta"):
        raise ValueError("target_mode must be 'absolute' or 'delta'")

    # Base column
    df = pd.DataFrame({datetime_col: base_times})

    # Add per-horizon columns
    for j, lbl in enumerate(h_labels):
        df[f"prediction_time_{lbl}"] = pd.to_datetime(meta["pred_times"][lbl])

        yt = y_true[:, j].astype(float)
        yp = y_pred[:, j].astype(float)

        if mode == "delta":
            # Keep deltas and, if possible, also reconstruct absolute
            df[f"y_true_delta_{lbl}"] = yt
            df[f"y_pred_delta_{lbl}"] = yp
            if base_vals is not None:
                base = np.asarray(base_vals, dtype=float)
                df[f"y_true_abs_{lbl}"] = base + yt
                df[f"y_pred_abs_{lbl}"] = base + yp
        else:
            # Keep absolutes and, if possible, also compute deltas
            df[f"y_true_abs_{lbl}"] = yt
            df[f"y_pred_abs_{lbl}"] = yp
            if base_vals is not None:
                base = np.asarray(base_vals, dtype=float)
                df[f"y_true_delta_{lbl}"] = yt - base
                df[f"y_pred_delta_{lbl}"] = yp - base
       
    if base_vals is not None:     
        df[value_col] = meta['base_values']
    return df



# energy_forecasting/evaluation.py
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


from ..utils.helper_utils import LoggingMixin
from ..plotting import ScatterPlotterSeaborn
from ..plotting import TimeSeriesPlotter


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
    Evaluate a single-horizon result frame and always plot **absolute** values.

    The frame is expected to come from `assemble_results_dataframe`. For a given
    `horizon` (e.g. "1h"), if absolute columns are present they are used directly;
    otherwise, absolute values are reconstructed as the sum of the current observed
    value and the delta values.

    The naive baseline is:
      - in absolute space: the current observed value column (zero-change baseline),
        which matches the behavior when the model was trained to predict deltas.
    """

    def __init__(
        self,
        *,
        df_res: pd.DataFrame,
        horizon: str,
        datetime_col: str = "Date",
        value_col: str = "value",
        test_flag_col: str = "test_set",
        prediction_time_col: Optional[str] = None,
        scatter_plotter: Optional[ScatterPlotterSeaborn] = None,
        ts_plotter: Optional[TimeSeriesPlotter] = None,
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        self.df_res = df_res.copy()
        self.horizon = str(horizon)
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.test_flag_col = test_flag_col

     
        # Choose a prediction-time column if present
        if prediction_time_col is None:
            for cand in (f"prediction_time_{self.horizon}", "prediction_time", "prediction_datetime"):
                if cand in self.df_res.columns:
                    prediction_time_col = cand
                    break
        self.prediction_time_col = prediction_time_col

        self.scatter_plotter = scatter_plotter or ScatterPlotterSeaborn()
        self.ts_plotter = ts_plotter or TimeSeriesPlotter()

        # Filter to test rows if flagged
        if self.test_flag_col in self.df_res.columns:
            self.df_res = self.df_res[self.df_res[self.test_flag_col].astype(bool)].reset_index(drop=True)

        # Resolve absolute columns (reconstruct if needed), then store arrays
        abs_true, abs_pred = self._resolve_absolute_targets(self.df_res, self.horizon, self.value_col)
        self.y_true_abs = abs_true.astype(float)
        self.y_pred_abs = abs_pred.astype(float)

        # Time axis
        if self.prediction_time_col and self.prediction_time_col in self.df_res.columns:
            self.time = pd.to_datetime(self.df_res[self.prediction_time_col])
        else:
            self.time = pd.to_datetime(self.df_res[self.datetime_col])

        # Naive baseline in absolute space: current observed value (zero-change)
        if self.value_col not in self.df_res.columns:
            raise ValueError(f"Column '{self.value_col}' not found in results frame.")
        self.y_pred_naive_abs = self.df_res[self.value_col].astype(float).to_numpy()

        # Outputs
        self.metrics_: Dict[str, float] = {}
        self.naive_metrics_: Dict[str, float] = {}

    # --------------------------- public API --------------------------- #
    def evaluate(self, *, rounding: int = 2) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute metrics on absolute values for model predictions and the naive baseline.
        Returns (metrics, naive_metrics).
        """
        self.metrics_ = compute_metrics(self.y_true_abs, self.y_pred_abs, rounding=rounding)
        self.naive_metrics_ = compute_metrics(self.y_true_abs, self.y_pred_naive_abs, rounding=rounding, 
                                              is_naive=True)
        self._log(f"[Evaluator] Metrics      : {self.metrics_}")
        self._log(f"[Evaluator] Naive metrics: {self.naive_metrics_}")
        return self.metrics_, self.naive_metrics_

    def plot_scatter(self, *, title: Optional[str] = None, alpha: float = 0.6) -> None:
        """Scatter plot of absolute y_true vs y_pred with x=y dashed line."""
        default_title = f"Predicted vs True (absolute, {self.horizon})"
        self.scatter_plotter.plot(self.y_true_abs, self.y_pred_abs, alpha=alpha, title=title or default_title)

    def plot_timeseries(self, *, title: Optional[str] = None, split_time=None) -> None:
        """Overlay absolute true vs predicted over time using Plotly."""
        default_title = f"Time series: true vs predicted ({self.horizon})"
        df_plot = pd.DataFrame({
            "t": self.time,
            "y_true": self.y_true_abs,
            "y_pred": self.y_pred_abs,
        })
        self.ts_plotter.plot(
            df_plot,
            time_col="t",
            y_true_col="y_true",
            y_pred_col="y_pred",
            split_time=split_time,
            title=title or default_title,
        )

    # -------------------------- internals --------------------------- #
    @staticmethod
    def _resolve_absolute_targets(df: pd.DataFrame, horizon: str, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (y_true_abs, y_pred_abs) for a horizon. When absolute columns are
        missing, reconstruct them as value_col + delta columns.
        """
        true_abs_col = f"y_true_abs_{horizon}"
        pred_abs_col = f"y_pred_abs_{horizon}"
        if true_abs_col in df.columns and pred_abs_col in df.columns:
            yt = df[true_abs_col].to_numpy()
            yp = df[pred_abs_col].to_numpy()
            return yt, yp

        # Fallback: reconstruct from deltas
        true_d_col = f"y_true_delta_{horizon}"
        pred_d_col = f"y_pred_delta_{horizon}"
        if value_col not in df.columns or true_d_col not in df.columns or pred_d_col not in df.columns:
            raise ValueError(
                f"Absolute columns not found and cannot reconstruct from deltas. "
                f"Expected either ('{true_abs_col}','{pred_abs_col}') or "
                f"('{value_col}','{true_d_col}','{pred_d_col}')."
            )
        base = df[value_col].astype(float).to_numpy()
        yt = base + df[true_d_col].astype(float).to_numpy()
        yp = base + df[pred_d_col].astype(float).to_numpy()
        return yt, yp
    
    



def assemble_results_dataframe(
    *,
    meta: Dict,
    y_pred: np.ndarray,
    datetime_col: str = "Datetime",
    target_mode: Optional[str] = None,
    value_col: str = "PJME_MW",
) -> pd.DataFrame:
    """
    Assemble a *wide* evaluation-ready DataFrame from test metadata and predictions.

    Behavior depends on the target mode:
      - If target_mode == "absolute":
          y_true_<h>, y_pred_<h> are absolute values.
          Also adds y_true_delta_<h>, y_pred_delta_<h> if base_values are present.

      - If target_mode == "delta":
          y_true_<h>, y_pred_<h> are deltas (future - current).
          Also adds y_true_abs_<h>, y_pred_abs_<h> if base_values are present.

    Parameters
    ----------
    meta : Dict
        Expected keys (vectorized over N samples):
          - "y"           : (N, n_outputs) ground-truth targets (absolute or delta)
          - "base_times"  : (N,) timestamps at window endpoints
          - "base_values" : (N,) or (N, F) values at window endpoints (optional but recommended)
          - "pred_times"  : dict[label] -> (N,) prediction timestamps per horizon
          - "h_labels"    : list[str] of horizon labels (e.g. ["15m"])
          - "target_mode" : "absolute" | "delta" (optional if provided explicitly)
    y_pred : np.ndarray
        (N, n_outputs) model predictions in the SAME space as meta["y"].
        A 1D array of shape (N,) is also accepted (treated as (N,1)).
    datetime_col : str
        Name of the base timestamp column in the output dataframe.
    target_mode : Optional[str]
        If provided, overrides meta["target_mode"]. Must be "absolute" or "delta".
    value_col : str
        Name to use for the *current* observed value column (i.e., the base value at window end).

    Returns
    -------
    pd.DataFrame
        Columns include:
          - datetime_col                                 (base/window end time)
          - prediction_time_<h>                          (one per horizon)
          - y_true_<h>, y_pred_<h>                       (delta or absolute, per mode)
          - y_true_abs_<h>, y_pred_abs_<h>               (if base_values provided and mode='delta')
          - y_true_delta_<h>, y_pred_delta_<h>           (if base_values provided and mode='absolute')
          - value_col                                    (base/current observed value, if base_values provided)
    """
    # --- pull meta ---
    if "y" not in meta:
        raise KeyError("meta['y'] (ground-truth targets) is required.")
    y_true = np.asarray(meta["y"])

    # normalize shapes to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    N, n_outputs = y_true.shape
    if y_pred.shape[0] != N:
        raise ValueError(f"y_true and y_pred must have same N; got {N} vs {y_pred.shape[0]}")
    if y_pred.shape[1] != n_outputs:
        raise ValueError(f"y_pred second dim must equal meta['y'].shape[1]={n_outputs}; got {y_pred.shape[1]}")

    # horizons & timestamps
    h_labels: List[str] = list(meta.get("h_labels", [])) or [str(i) for i in range(n_outputs)]
    if len(h_labels) != n_outputs:
        raise ValueError(f"len(h_labels)={len(h_labels)} must match n_outputs={n_outputs}")

    base_times = pd.to_datetime(meta["base_times"])
    if len(base_times) != N:
        raise ValueError(f"len(base_times)={len(base_times)} must equal N={N}")

    pred_times_dict = meta.get("pred_times", {})
    # basic sanity: ensure each label exists
    for lbl in h_labels:
        if lbl not in pred_times_dict:
            raise KeyError(f"meta['pred_times'] missing key '{lbl}'")
        if len(pred_times_dict[lbl]) != N:
            raise ValueError(f"pred_times['{lbl}'] length must be N={N}")

    base_vals = meta.get("base_values", None)
    if base_vals is not None:
        base_vals = np.asarray(base_vals)
        if base_vals.ndim == 2 and base_vals.shape[0] == N and base_vals.shape[1] == 1:
            base_vals = base_vals.ravel()  # (N,)
        # Allow (N,) for per-sensor usage; (N,F) is okay but caller should slice per output.
        if base_vals.ndim > 1 and base_vals.shape[0] != N:
            raise ValueError("base_values first dimension must be N")

    # choose mode
    mode = (target_mode or meta.get("target_mode", "absolute")).lower()
    if mode not in ("absolute", "delta"):
        raise ValueError("target_mode must be 'absolute' or 'delta'")

    # --- build output frame ---
    df = pd.DataFrame({datetime_col: base_times})

    for j, lbl in enumerate(h_labels):
        df[f"prediction_time_{lbl}"] = pd.to_datetime(pred_times_dict[lbl])

        yt = y_true[:, j].astype(float)
        yp = y_pred[:, j].astype(float)

        if mode == "delta":
            df[f"y_true_delta_{lbl}"] = yt
            df[f"y_pred_delta_{lbl}"] = yp
            if base_vals is not None:
                # base_vals may be (N,) in per-sensor use; if (N,F) caller should slice before calling
                base = base_vals if base_vals.ndim == 1 else base_vals[:, j]
                base = np.asarray(base, dtype=float)
                df[f"y_true_abs_{lbl}"] = base + yt
                df[f"y_pred_abs_{lbl}"] = base + yp
        else:  # mode == "absolute"
            df[f"y_true_abs_{lbl}"]  = yt
            df[f"y_pred_abs_{lbl}"]  = yp
            if base_vals is not None:
                base = base_vals if base_vals.ndim == 1 else base_vals[:, j]
                base = np.asarray(base, dtype=float)
                df[f"y_true_delta_{lbl}"] = yt - base
                df[f"y_pred_delta_{lbl}"] = yp - base

    if base_vals is not None:
        # store the *current* observed value at the base time (for convenience/naive baseline)
        # if base_vals is (N,F) and caller didn't slice, fall back to the first column
        if base_vals.ndim == 1:
            df[value_col] = base_vals.astype(float)
        else:
            df[value_col] = base_vals[:, 0].astype(float)

    return df