from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,  median_absolute_error
import numpy as np
import matplotlib.patheffects as PathEffects
#from .data_processing import TrafficFlowDataProcessing
from ...utils.helper_utils import normalize_data
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from time import time
import pandas as pd
from typing import Optional, Dict, Union
from ..inference.prediction_protocol import make_prediction_frame
from ...preprocessing.cleaning import clean_and_cast




class ModelEvaluator:
    """
    Evaluate regression models (XGBoost/.pkl or Keras/.h5) on a prepared test split.

    Key features
    ------------
    - Supports MAE, Median AE, RMSE, MAPE, SMAPE (symmetric) with NaN-safe handling.
    - Robust divide-by-zero protection via epsilon or zero-discard mask.
    - Optional de-normalization of model outputs when the target was normalized.
    - Optional reconstruction step for total-speed (e.g., y = delta + base) or GMAN error-correction.
    - Single source of truth for prediction and metric computation (no code duplication).
    - Canonical predictions helper to produce a standardized output DataFrame.

    Parameters
    ----------
    X_test : pandas.DataFrame or numpy.ndarray
        Feature matrix for the test split (aligned 1:1 with df_for_ML[test_set] rows).
        Must contain `value_col` if reconstruction uses it.
    df_for_ML : pandas.DataFrame
        The ML frame; rows filtered internally to df_for_ML['test_set'] == True.
        Should include columns: [date_col, sensor_col, 'prediction_time', value_col, 'target_total_speed']
        where applicable.
    y_train : array-like, optional
        Training targets, required if `y_is_normalized=True` (used to recover mean/std).
    y_test : array-like, optional
        Test targets BEFORE reconstruction. They will be copied and reconstructed internally.
    rounding : int, default=2
        Decimal rounding for reported metrics.
    discard_zero_mape : bool, default=False
        If True, MAPE calculations discard rows with y_true==0; else use epsilon smoothing.
    target_is_gman_error_prediction : bool, default=False
        If True, final target reconstruction uses: y = y_pred + df_for_ML['gman_prediction_orig'].
        Otherwise, y = y_pred + X_test[value_col].
    sensor_col : str, default='sensor_id'
        Name of the sensor column.
    date_col : str, default='date'
        Name of the datetime column in df_for_ML/df_raw.
    value_col : str, default='value'
        Name of the base value (e.g., last observed speed) used for reconstruction and naive baselines.
    y_is_normalized : bool, default=False
        If True, de-normalize model outputs using mean/std estimated from y_train.
    epsilon : float, default=1e-2
        Small positive constant for stable divisions (MAPE/SMAPE denominators).

    Notes
    -----
    - For Keras models, normalization of X can be auto-triggered when model_path contains "neural",
      or you can force it by passing `force_normalize_X=True` to `evaluate_model(...)`.
    """

    def __init__(
        self,
        X_test,
        df_for_ML,
        y_train=None,
        y_test=None,
        rounding=2,
        discard_zero_mape=False,
        target_is_gman_error_prediction=False,
        sensor_col='sensor_id',
        date_col='date',
        value_col='value',
        y_is_normalized=False,
        epsilon=1e-2,
    ):
        # Store core inputs
        self.X_test = X_test
        self.df_for_ML = df_for_ML[df_for_ML['test_set']]  # use only test rows
        self.y_train = y_train
        self.y_test = y_test

        # Config flags / columns
        self.rounding = int(rounding)
        self.discard_zero_mape = bool(discard_zero_mape)
        self.target_is_gman_error_prediction = bool(target_is_gman_error_prediction)
        self.y_is_normalized = bool(y_is_normalized)
        self.date_col = date_col
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.epsilon = float(epsilon)

        # Derived attributes
        self.df_predictions = None
        self.y_pred = None
        self.y_pred_before_reconstruction = None

        # If outputs were normalized, we need training stats to invert the transform
        if self.y_is_normalized:
            if y_train is None:
                raise ValueError("y_train must be provided for de-normalization when y_is_normalized=True.")
            self.y_mean = float(np.mean(y_train))
            self.y_std = float(np.std(y_train))
        else:
            self.y_mean = 0.0
            self.y_std = 1.0

        # Keep a copy of test targets before reconstruction (used for naive baselines)
        if self.y_test is None:
            raise ValueError("y_test must be provided.")
        self.y_test_before_reconstruction = np.asarray(self.y_test).copy()
        # Reconstruct final target (e.g., back to total speed)
        self.y_test = self.reconstruct_y(self.y_test_before_reconstruction)

        # Decide if we need special MAPE handling based on presence of zeros in y_true
        zero_percentage = self.calculate_discarded_percentage()
        self.calculate_mape_with_handling_zero_values = zero_percentage > 0.0

    # ---------------------------------------------------------------------
    #                         Utility / helpers
    # ---------------------------------------------------------------------

    def calculate_discarded_percentage(self) -> float:
        """
        Percentage of y_true values equal to zero after reconstruction.
        Used to decide whether to apply special MAPE handling.
        """
        y_true = np.asarray(self.y_test)
        zero_mask = (y_true == 0)
        return float(np.sum(zero_mask) / max(1, len(y_true)) * 100.0)

    def load_model_from_path(self, model_path: str):
        """
        Load a model from disk depending on extension.

        Parameters
        ----------
        model_path : str
            Path to a .h5 (Keras) or .pkl (pickle) model file.

        Returns
        -------
        Any
            Loaded model instance.
        """
        if model_path.endswith('.h5'):
            # Local import keeps TF optional until needed
            from tensorflow.keras.models import load_model  # type: ignore
            return load_model(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        raise ValueError(f"Unsupported model file: {model_path}")

    def reconstruct_y(self, y):
        """
        Reconstruct the final target space.

        If `target_is_gman_error_prediction` is True:
            y_final = y + df_for_ML['gman_prediction_orig']
        Else:
            y_final = y + X_test[value_col]

        Parameters
        ----------
        y : array-like
            Predicted (or true) deltas or normalized values.

        Returns
        -------
        numpy.ndarray
            Reconstructed values in the final target space.
        """
        y = np.asarray(y, dtype=float)
        if self.target_is_gman_error_prediction:
            base = np.asarray(self.df_for_ML['gman_prediction_orig'], dtype=float)
            return y + base
        else:
            base = np.asarray(self.X_test[self.value_col], dtype=float)
            return y + base

    def smape(self, y_true, y_pred):
        """
        Symmetric MAPE in percent, NaN-safe.

        Returns
        -------
        (float, float)
            (mean SMAPE %, std SMAPE %)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan
        yt = y_true[mask]
        yp = y_pred[mask]
        denom = (np.abs(yt) + np.abs(yp)) / 2.0
        smape_vals = np.abs(yp - yt) / np.where(denom == 0.0, 1.0, denom)
        return float(np.mean(smape_vals) * 100.0), float(np.std(smape_vals) * 100.0)

    # ---------------------------------------------------------------------
    #                       Centralized private logic
    # ---------------------------------------------------------------------

    def _predict_and_prepare(
        self,
        model,
        *,
        model_origin: Optional[str] = None,
        force_normalize_X: bool = False,
    ):
        """
        Single place to:
        1) Normalize X (if needed),
        2) Run model.predict,
        3) De-normalize y_pred (if configured),
        4) Reconstruct to final target space,
        5) Time inference and compute per-sample latency,
        6) Build df_predictions & persist self.y_pred.

        Returns
        -------
        (y_pred_rec, inference_time, inference_time_per_sample)
            y_pred_rec : numpy.ndarray
                Reconstructed predictions aligned to X_test rows.
            inference_time : float
            inference_time_per_sample : float
        """
        time_start = time()

        # Heuristic: if path contains "neural" or caller forces it, normalize X_test
        needs_norm = force_normalize_X or (isinstance(model_origin, str) and ("neural" in model_origin.lower()))
        if needs_norm:
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            raw_pred = model.predict(X_test_normalized)
        else:
            raw_pred = model.predict(self.X_test)

        # Ensure shape is (N,)
        raw_pred = np.asarray(raw_pred, dtype=float).squeeze()

        # If targets were normalized during training, invert
        if self.y_is_normalized:
            raw_pred = raw_pred * self.y_std + self.y_mean

        # Keep pre-reconstruction copy for baselines
        self.y_pred_before_reconstruction = raw_pred.copy()

        # Reconstruct to total (or GMAN-adjusted) target
        y_pred_rec = self.reconstruct_y(raw_pred)

        time_end = time()
        inference_time = float(time_end - time_start)
        inference_time_per_sample = float(inference_time / max(1, len(self.X_test)))

        # Build predictions DataFrame (only available columns are kept)
        cols = [c for c in [self.date_col, self.sensor_col, 'prediction_time', self.value_col, 'target_total_speed']
                if c in self.df_for_ML.columns]
        self.df_predictions = self.df_for_ML[cols].copy()
        self.df_predictions['y_pred'] = y_pred_rec

        # Persist
        self.y_pred = y_pred_rec
        self.df_for_ML['y_pred'] = self.y_pred

        return y_pred_rec, inference_time, inference_time_per_sample

    def _compute_metrics(self, y_pred_rec, inference_time, inference_time_per_sample):
        """
        Compute all metrics (and their std) in a single place, NaN-safe.

        Returns
        -------
        metrics, metrics_std, naive_metrics, naive_metrics_std : dict
            Rounded according to self.rounding.
        """
        # Extract aligned arrays
        y_true_full = np.asarray(self.y_test, dtype=float)
        y_pred_full = np.asarray(y_pred_rec, dtype=float)
        x_val_full = np.asarray(self.X_test[self.value_col], dtype=float)
        y_base_full = np.asarray(self.y_test_before_reconstruction, dtype=float)

        # Drop invalid rows
        mask = np.isfinite(y_true_full) & np.isfinite(y_pred_full)
        if not np.any(mask):
            raise ValueError("All y_true/y_pred are NaN/inf after filtering.")

        y_true = y_true_full[mask]
        y_pred = y_pred_full[mask]
        x_val = x_val_full[mask]
        y_base = y_base_full[mask]

        # Pointwise errors
        abs_errors = np.abs(y_true - y_pred)

        # Classic metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        median_ae = float(median_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))

        # SMAPE (and naive)
        smape, smape_std = self.smape(y_true, y_pred)
        naive_smape, naive_smape_std = self.smape(y_true, x_val)

        # MAPE and naive MAPE
        if self.calculate_mape_with_handling_zero_values:
            if self.discard_zero_mape:
                m2 = y_true != 0
                y_true_m = y_true[m2]
                y_pred_m = y_pred[m2]
                y_base_m = y_base[m2]
                x_val_m = x_val[m2]
            else:
                y_true_m = np.where(y_true == 0, self.epsilon, y_true)
                y_pred_m = y_pred
                y_base_m = np.where(y_base == 0, self.epsilon, y_base)
                x_val_m = x_val

            ape = np.abs(y_true_m - y_pred_m) / np.clip(np.abs(y_true_m), self.epsilon, np.inf)
            denom_naive = y_base_m + x_val_m
            naive_ape = np.abs(y_base_m) / np.clip(np.abs(denom_naive), self.epsilon, np.inf)

            # Guard against residual NaNs
            m_valid = np.isfinite(ape) & np.isfinite(naive_ape)
            ape = ape[m_valid]
            naive_ape = naive_ape[m_valid]

            mape = float(np.mean(ape))
            mape_std = float(np.std(ape))
            naive_mape = float(np.mean(naive_ape))
            naive_mape_std = float(np.std(naive_ape))
        else:
            denom = np.clip(np.abs(y_true), self.epsilon, np.inf)
            ape = np.abs(y_true - y_pred) / denom
            mape = float(np.mean(ape))
            mape_std = float(np.std(ape))

            denom_naive = np.clip(np.abs(y_true), self.epsilon, np.inf)
            naive_ape = np.abs(y_base) / denom_naive
            naive_mape = float(np.mean(naive_ape))
            naive_mape_std = float(np.std(naive_ape))

        # Assemble metric dicts
        metrics = {
            'MAE': mae,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'MAPE': mape * 100.0,
            'SMAPE': smape,
            'inference_time': float(inference_time),
            'inference_time_per_sample': float(inference_time_per_sample),
        }
        metrics_std = {
            'MAE_std': float(np.std(abs_errors)),
            'Median_AE_std': float(np.std(abs_errors)),
            'RMSE_std': float(np.std((y_true - y_pred) ** 2)),
            'MAPE_std': mape_std * 100.0,
            'SMAPE_std': smape_std,
        }

        # Naive (absolute deltas as errors)
        if self.target_is_gman_error_prediction:
            # Use df_for_ML column if provided for GMAN-delta targets
            naive_error_vals = np.abs(
                self.df_for_ML.loc[self.df_for_ML.index[mask], 'target_speed_delta'].to_numpy(dtype=float)
            )
        else:
            naive_error_vals = np.abs(y_base)

        naive_metrics = {
            'Naive_MAE': float(np.mean(naive_error_vals)),
            'Naive_Median_AE': float(np.median(naive_error_vals)),
            'Naive_RMSE': float(np.sqrt(np.mean(naive_error_vals ** 2))),
            'Naive_MAPE': naive_mape * 100.0,
            'Naive_SMAPE': naive_smape,
        }
        naive_metrics_std = {
            'Naive_MAE_std': float(np.std(naive_error_vals)),
            'Naive_Median_AE_std': float(np.std(naive_error_vals)),
            'Naive_RMSE_std': float(np.std(naive_error_vals ** 2)),
            'Naive_MAPE_std': float(naive_mape_std * 100.0),
            'Naive_SMAPE_std': float(naive_smape_std),
        }

        # Rounding
        if self.rounding is not None:
            r = int(self.rounding)
            metrics = {k: round(v, r) for k, v in metrics.items()}
            metrics_std = {k: round(v, r) for k, v in metrics_std.items()}
            naive_metrics = {k: round(v, r) for k, v in naive_metrics.items()}
            naive_metrics_std = {k: round(v, r) for k, v in naive_metrics_std.items()}

        return metrics, metrics_std, naive_metrics, naive_metrics_std

    # ---------------------------------------------------------------------
    #                          Public API methods
    # ---------------------------------------------------------------------

    def predict(self, model_path: Optional[str] = None):
        """
        Backward-compatible convenience method: load model (if path) and return reconstructed predictions.

        Parameters
        ----------
        model_path : str, optional
            If given, the model is loaded from disk. Otherwise, this method assumes you will
            call `evaluate_model(...)` directly with an in-memory model.

        Returns
        -------
        numpy.ndarray
            Reconstructed predictions (same length/order as X_test).
        """
        if model_path is None:
            raise ValueError("Provide a model_path for `predict`, or use `evaluate_model(model=...)`.")
        model = self.load_model_from_path(model_path)

        # Mirror the legacy behavior: normalize X when 'neural' appears in the path
        needs_norm = "neural" in model_path.lower()
        if needs_norm:
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            y_pred = model.predict(X_test_normalized).flatten()
        else:
            y_pred = model.predict(self.X_test)

        if self.y_is_normalized:
            y_pred = y_pred * self.y_std + self.y_mean

        self.y_pred_before_reconstruction = np.asarray(y_pred, dtype=float).copy()
        y_pred_rec = self.reconstruct_y(y_pred)
        self.y_pred = y_pred_rec
        return y_pred_rec

    def evaluate_model(
        self,
        model,
        *,
        print_results: bool = True,
        model_origin: Optional[str] = None,
        force_normalize_X: bool = False,
    ):
        """
        Evaluate an already-loaded model instance (no disk I/O).

        Parameters
        ----------
        model : Any
            In-memory model that exposes `.predict(X)`.
        print_results : bool, default=True
            If True, print metric dicts to stdout.
        model_origin : str, optional
            Optional string hint (e.g., file path) used to infer preprocessing (e.g., "neural").
        force_normalize_X : bool, default=False
            Force normalization of X_test prior to prediction (overrides model_origin heuristic).

        Returns
        -------
        dict
            {
              "metrics": {...},
              "metrics_std": {...},
              "naive_metrics": {...},
              "naive_metrics_std": {...}
            }
        """
        y_pred, inf_time, ips = self._predict_and_prepare(
            model,
            model_origin=model_origin,
            force_normalize_X=force_normalize_X,
        )
        metrics, metrics_std, naive_metrics, naive_metrics_std = self._compute_metrics(y_pred, inf_time, ips)

        if print_results:
            self.print_evaluation_results(metrics, metrics_std, naive_metrics, naive_metrics_std)

        return {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std,
        }

    def evaluate_model_from_path(
        self,
        model_path: Optional[str] = None,
        saved_model=None,
        print_results: bool = True,
        force_normalize_X: bool = False,
    ):
        """
        Load (if needed) and evaluate a model. This is a thin wrapper around `evaluate_model(...)`.

        Parameters
        ----------
        model_path : str, optional
            Path to a .h5 or .pkl model. Ignored if `saved_model` is provided.
        saved_model : Any, optional
            In-memory model. If provided, takes precedence over `model_path`.
        print_results : bool, default=True
            If True, print metric dicts to stdout.
        force_normalize_X : bool, default=False
            Force normalization of X_test prior to prediction.

        Returns
        -------
        dict
            Same structure as `evaluate_model(...)`.
        """
        model = None
        origin = model_path

        if (model_path is not None) and (saved_model is not None):
            warnings.warn("Both `model_path` and `saved_model` are provided. Using `saved_model` and ignoring `model_path`.")
            model = saved_model
        elif model_path is not None:
            model = self.load_model_from_path(model_path)
        else:
            model = saved_model

        if model is None:
            raise ValueError("Provide either `model_path` or `saved_model`.")

        return self.evaluate_model(
            model,
            print_results=print_results,
            model_origin=origin,
            force_normalize_X=force_normalize_X,
        )

    def print_evaluation_results(self, metrics, metrics_std, naive_metrics, naive_metrics_std):
        """
        Pretty-print metrics and their standard deviations (incl. naive baselines).
        """
        print("\n--- Evaluation Results ---")
        print("\nNaive Metrics:")
        print(naive_metrics)
        print("\nNaive Metrics Standard Deviations:")
        print(naive_metrics_std)
        print("\nMetrics:")
        print(metrics)
        print("\nMetrics Standard Deviations:")
        print(metrics_std)
        print("--------------------------\n")

    def calculate_mape_in_case_of_zero_values(self, y_pred):
        """
        (Legacy helper, kept for compatibility.)

        Compute (naive) MAPE while handling y_true==0 by either discarding or epsilon-smoothing.

        Parameters
        ----------
        y_pred : array-like
            Reconstructed predictions.

        Returns
        -------
        (mape, mape_std, naive_mape, naive_mape_std) : tuple of floats (fractions, not %)
        """
        if self.discard_zero_mape:
            mask = self.y_test != 0
            y_test = self.y_test[mask]
            y_pred = np.asarray(y_pred, dtype=float)[mask]
            y_test_base = self.y_test_before_reconstruction[mask]
            x_val = np.asarray(self.X_test[self.value_col], dtype=float)[mask]
        else:
            y_test = np.where(self.y_test == 0, self.epsilon, self.y_test)
            y_pred = np.asarray(y_pred, dtype=float)
            y_test_base = np.where(self.y_test_before_reconstruction == 0, self.epsilon, self.y_test_before_reconstruction)
            x_val = np.asarray(self.X_test[self.value_col], dtype=float)

        ape = np.abs(y_test - y_pred) / np.clip(np.abs(y_test), self.epsilon, np.inf)
        naive_ape = np.abs(y_test_base) / np.clip(np.abs(y_test_base + x_val), self.epsilon, np.inf)

        # Clean any residual NaN/inf
        m_valid = np.isfinite(ape) & np.isfinite(naive_ape)
        ape = ape[m_valid]
        naive_ape = naive_ape[m_valid]

        return float(np.mean(ape)), float(np.std(ape)), float(np.mean(naive_ape)), float(np.std(naive_ape))

    def to_canonical_predictions(
        self,
        *,
        df_raw: pd.DataFrame,
        model=None,
        model_path: Optional[str] = None,
        states: Optional[dict] = None,
        horizon_min: Optional[int] = None,
        add_total: bool = True,
    ) -> pd.DataFrame:
        """
        Produce the canonical prediction DataFrame for the *test* split
        using the evaluator's X_test and df_for_ML (already filtered to test_set).

        Columns in the returned DataFrame (from `make_prediction_frame`) typically include:
            sensor_id, input_time, prediction_time, y_pred_delta, y_pred_total, horizon, (and optionally y_act)

        Parameters
        ----------
        df_raw : pandas.DataFrame
            Original raw data (pre-ML features). Used by `make_prediction_frame` to map to canonical form.
        model : Any, optional
            In-memory model. If None, `model_path` must be provided.
        model_path : str, optional
            Path to a model; used if `model` is None.
        states : dict, optional
            Exported states from your data processing pipeline; used to interpret columns.
        horizon_min : int, optional
            Forecast horizon in minutes. If None, inferred from prediction_time - date.
        add_total : bool, default=True
            If True, include total prediction in the canonical frame.

        Returns
        -------
        pandas.DataFrame
            Canonical predictions aligned to the test split.
        """
        # Get model
        if model is None and model_path is not None:
            model = self.load_model_from_path(model_path)
        if model is None:
            raise ValueError("Provide a model or model_path.")

        # Resolve columns from states (fall back to evaluator defaults)
        value_col = self.value_col
        dt_col = self.date_col
        if states:
            if "sensor_encoder_state" in states:
                value_col = states["sensor_encoder_state"].get("value_col", value_col)
            if "datetime_state" in states:
                dt_col = states["datetime_state"].get("datetime_col", dt_col)

        # Clean and order df_raw
        df_raw = clean_and_cast(df_raw, value_col=value_col)
        df_raw[dt_col] = pd.to_datetime(df_raw[dt_col], errors="coerce")
        df_raw.sort_values(by=[dt_col, self.sensor_col], inplace=True)

        # Predict deltas on X_test (raw model outputs, not reconstructed)
        y_pred_delta = model.predict(self.X_test)

        # Infer horizon if not provided
        df_test = self.df_for_ML.copy()
        if horizon_min is None:
            if ("prediction_time" in df_test.columns) and (self.date_col in df_test.columns):
                delta = pd.to_datetime(df_test["prediction_time"]) - pd.to_datetime(df_test[self.date_col])
                horizon_min = int(round(delta.dt.total_seconds().median() / 60.0))
            else:
                horizon_min = 15
                warnings.warn(
                    "Neither horizon_min nor (date/prediction_time) available; falling back to 15 minutes."
                )

        # Build canonical frame
        pred_df = make_prediction_frame(
            df_raw=df_raw,
            df_for_ML=self.df_for_ML.rename(columns={self.date_col: "date"}),
            pred_delta=y_pred_delta,
            states=states,
            horizon_min=horizon_min,
            add_total=add_total,
            sensor_col=self.sensor_col,
            add_y_act=True,
        )
        return pred_df