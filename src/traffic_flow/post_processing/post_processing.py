
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm
from ..utils.helper_utils import *
from typing import Callable, Optional, Dict, Any, List, Union
from joblib import Parallel, delayed
from tqdm import tqdm
from pykalman import KalmanFilter
from typing import Optional
import re
# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # You can set this to DEBUG, WARNING, etc. as needed
)


def xgb_to_lstm_like_df(
    df: pd.DataFrame,
    *,
    sensor_col: str = "sensor_id",
    issue_time_col: str = "date",
    target_time_col: str = "prediction_time",
    actual_col: str = "target_total_speed",   # y_true at t+H
    pred_col: str = "y_pred",                  # model forecast for t+H issued at t
    issued_val_col: str = "value",             # actual at issue time t (persistence baseline)
    dtype: str = "float32",
    dedup_agg: str = "last",                   # or "mean"
) -> pd.DataFrame:
    """
    Convert long df_pred_xgb into wide df_pred layout:
      columns: date, prediction_time, <sensor>, <sensor>_pred, <sensor>_at_issued_time

    Mapping:
      - <sensor>                   <- target_total_speed   (actual at prediction_time = t+H)
      - <sensor>_pred              <- y_pred               (forecast for t+H, issued at date = t)
      - <sensor>_at_issued_time    <- value                (actual at issue time t)
    """
    # Keep only needed columns and ensure datetimes
    cols = [sensor_col, issue_time_col, target_time_col, actual_col, pred_col, issued_val_col]
    df = df[cols].copy()
    df[issue_time_col]  = pd.to_datetime(df[issue_time_col])
    df[target_time_col] = pd.to_datetime(df[target_time_col])

    # Deduplicate per (t, t+H, sensor) if any; choose how to aggregate
    gb = (df.sort_values([issue_time_col, target_time_col])
            .groupby([issue_time_col, target_time_col, sensor_col], as_index=False))
    if dedup_agg == "mean":
        dfu = gb.agg({actual_col: "mean", pred_col: "mean", issued_val_col: "mean"})
    else:
        dfu = gb.agg({actual_col: "last", pred_col: "last", issued_val_col: "last"})

    # Pivot each measure to wide
    wide_act  = dfu.pivot(index=[issue_time_col, target_time_col], columns=sensor_col, values=actual_col)
    wide_pred = dfu.pivot(index=[issue_time_col, target_time_col], columns=sensor_col, values=pred_col)
    wide_iss  = dfu.pivot(index=[issue_time_col, target_time_col], columns=sensor_col, values=issued_val_col)

    # Rename prediction & issued-time columns
    wide_pred.columns = [f"{c}_pred" for c in wide_pred.columns]
    wide_iss.columns  = [f"{c}_at_issued_time" for c in wide_iss.columns]

    # Join and tidy
    wide = pd.concat([wide_act, wide_pred, wide_iss], axis=1)

    # Order columns: date, prediction_time, then for each sensor: <s>, <s>_pred, <s>_at_issued_time
    sensors = list(wide_act.columns)  # preserves pivot order
    ordered = []
    for s in sensors:
        for c in (s, f"{s}_pred", f"{s}_at_issued_time"):
            if c in wide.columns:
                ordered.append(c)

    wide = wide[ordered].reset_index().sort_values([issue_time_col, target_time_col])

    # Optional downcast to save memory
    if dtype:
        num_cols = [c for c in wide.columns if c not in (issue_time_col, target_time_col)]
        wide[num_cols] = wide[num_cols].astype(dtype)

    # Standardize column names to match df_pred
    wide = wide.rename(columns={issue_time_col: "date", target_time_col: "prediction_time"})
    return wide


class PredictionCorrectionPerSensor:
    """
    Class to apply post-processing corrections to time-series predictions
    for a specific sensor. Methods assume that `y_pred` and `y_test` are already
    reconstructed speed values.
    """

    def __init__(
        self,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        sensor_uid: Optional[str],
        df_for_ML: Optional[pd.DataFrame] = None,
        rounding: int = 2,
    ) -> None:
        """
        Initialize the correction handler for a single sensor's time-series.

        Args:
            X_test: Test set DataFrame (must include 'sensor_uid' and 'value').
            y_test: Ground truth reconstructed speed values.
            sensor_uid: Sensor ID to isolate for correction.
            df_for_ML: Full ML dataset with training/test split (optional).
            rounding: Decimal rounding to apply to outputs.
        """
        if sensor_uid is not None:
            self.X_test_sensor = X_test.loc[X_test["sensor_uid"] == sensor_uid]
        else:
            self.X_test_sensor = X_test

        sensor_idx = self.X_test_sensor.index

        if not sensor_idx.isin(y_test.index).all():
            raise ValueError("Indices of y_test do not align with X_test for sensor.")

        self.y_test = y_test.loc[sensor_idx]

        if df_for_ML is not None:
            self.df_for_ML = df_for_ML.loc[df_for_ML["sensor_uid"] == sensor_uid]
            self.train_values = self.df_for_ML.loc[~self.df_for_ML["test_set"], "value"]
        else:
            self.df_for_ML = None
            self.train_values = None
            warnings.warn(
                "df_for_ML not provided. Some methods like constrain_predictions or kalman_smoothing will be disabled.",
                UserWarning,
            )

        self.rounding = rounding

    def _align_predictions_to_sensor(self, y_pred: pd.Series) -> pd.Series:
        """
        Align predictions to the current sensor's rows.

        Args:
            y_pred: Raw predictions for all sensors.

        Returns:
            Aligned predictions (only for current sensor).
        """
        sensor_idx = self.X_test_sensor.index
        return y_pred.loc[sensor_idx]

    def naive_based_correction(self, y_pred: pd.Series, naive_threshold: float = 0.5) -> pd.Series:
        """
        Replace predictions that deviate significantly from current values.

        Args:
            y_pred: Model predictions.
            naive_threshold: Relative deviation threshold.

        Returns:
            Corrected predictions as a Series.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        current_speed = self.X_test_sensor["value"].values
        deviation = np.abs(y_pred - current_speed) / current_speed
        mask = deviation > naive_threshold

        corrected = y_pred.copy()
        corrected[mask] = current_speed[mask]

        if self.rounding is not None:
            corrected = np.round(corrected, self.rounding)

        return pd.Series(corrected, index=self.X_test_sensor.index)

    def rolling_median_correction(self, y_pred: pd.Series, window_size: int = 3) -> pd.Series:
        """
        Smooth out spikes using rolling median.

        Args:
            y_pred: Model predictions.
            window_size: Rolling window size.

        Returns:
            Smoothed predictions as a Series.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        smoothed = y_pred.rolling(window=window_size, min_periods=1).median()

        if self.rounding is not None:
            smoothed = smoothed.round(self.rounding)

        return smoothed

    def ewma_smoothing(self, y_pred: pd.Series, span: int = 3) -> pd.Series:
        """
        Apply EWMA smoothing.

        Args:
            y_pred: Model predictions.
            span: EWMA span parameter.

        Returns:
            Smoothed predictions as a Series.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        smoothed = y_pred.ewm(span=span, adjust=False).mean()

        if self.rounding is not None:
            smoothed = smoothed.round(self.rounding)

        return smoothed

    def constrain_predictions(
        self,
        y_pred: pd.Series,
        min_speed: Optional[float] = None,
        max_speed: Optional[float] = None,
    ) -> pd.Series:
        """
        Constrain predictions to min/max observed values.

        Args:
            y_pred: Model predictions.
            min_speed: Optional fixed minimum value.
            max_speed: Optional fixed maximum value.

        Returns:
            Constrained predictions.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)

        if self.train_values is None:
            raise ValueError("df_for_ML was not provided or does not contain training data.")

        min_speed = min_speed if min_speed is not None else self.train_values.min()
        max_speed = max_speed if max_speed is not None else self.train_values.max()

        constrained = np.clip(y_pred, min_speed, max_speed)

        if self.rounding is not None:
            constrained = np.round(constrained, self.rounding)

        return pd.Series(constrained, index=self.X_test_sensor.index)

    def kalman_smoothing(
        self,
        y_pred: pd.Series,
        Q: float = 1.0,
        R: Optional[float] = None,
        handle_nans: bool = True,
    ) -> pd.Series:
        """
        Apply Kalman filter smoothing.

        Args:
            y_pred: Model predictions.
            Q: Process noise covariance.
            R: Measurement noise covariance. If None, it uses the variance of training values.
            handle_nans: Whether to auto-fill missing values.

        Returns:
            Kalman-filtered predictions.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)

        if handle_nans:
            y_pred = y_pred.fillna(method="ffill").fillna(method="bfill")
        elif y_pred.isna().any():
            raise ValueError("y_pred contains NaNs. Set handle_nans=True to fill.")

        if self.train_values is None:
            raise ValueError("df_for_ML was not provided or contains no training data.")

        obs_cov = R if R is not None else np.var(self.train_values)
        init_mean = np.mean(self.train_values)
        init_cov = np.var(self.train_values)

        kf = KalmanFilter(
            initial_state_mean=init_mean,
            initial_state_covariance=init_cov,
            transition_matrices=[1],
            observation_matrices=[1],
            observation_covariance=obs_cov,
            transition_covariance=Q,
        )

        smoothed, _ = kf.smooth(y_pred.values)

        if self.rounding is not None:
            smoothed = np.round(smoothed.flatten(), self.rounding)

        return pd.Series(smoothed.flatten(), index=self.X_test_sensor.index)

    def apply_all_corrections(
        self,
        y_pred: pd.Series,
        naive_threshold: float = 0.5,
        rolling_window: int = 3,
        ewma_span: int = 3,
    ) -> pd.Series:
        """
        Apply all corrections sequentially:
        1. Naive-based correction
        2. Rolling median
        3. EWMA
        4. Constraint to train min/max

        Args:
            y_pred: Raw model predictions.
            naive_threshold: Threshold for naive deviation filtering.
            rolling_window: Window size for median smoothing.
            ewma_span: Span for EWMA smoothing.

        Returns:
            Final corrected predictions.
        """
        y_pred_corrected = self.naive_based_correction(y_pred, naive_threshold)
        y_pred_corrected = self.rolling_median_correction(y_pred_corrected, rolling_window)
        y_pred_corrected = self.ewma_smoothing(y_pred_corrected, ewma_span)
        return self.constrain_predictions(y_pred_corrected)




class PredictionCorrectionPipeline:
    """
    Class to apply prediction post-processing to multiple sensors using PredictionCorrectionPerSensor.
    """

    def __init__(
        self,
        correction_class: Callable,
        method: str = "apply_all_corrections",
        rounding: int = 2,
        parallel: bool = False,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize the correction pipeline.

        Parameters:
        - correction_class: Callable class (e.g., PredictionCorrectionPerSensor)
        - method: Name of the method to apply from the correction class.
        - rounding: Decimal rounding applied to corrected predictions.
        - parallel: Whether to use parallel execution across sensors.
        - n_jobs: Number of parallel jobs (-1 uses all available cores).
        - verbose: Whether to show progress bar (only in sequential mode).
        """
        self.correction_class = correction_class
        self.method = method
        self.rounding = rounding
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _correct_single_sensor(
        self,
        sensor_id: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: pd.Series,
        df_for_ML: Optional[pd.DataFrame],
        method_args: Dict[str, Any]
    ) -> pd.Series:
        """
        Apply correction for a single sensor.

        Returns:
        - pd.Series: Corrected predictions for that sensor
        """
        corrector = self.correction_class(
            X_test=X_test,
            y_test=y_test,
            sensor_uid=sensor_id,
            df_for_ML=df_for_ML,
            rounding=self.rounding
        )
        method_to_call = getattr(corrector, self.method)
        return method_to_call(y_pred, **method_args)

    def apply(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: pd.Series,
        df_for_ML: Optional[pd.DataFrame] = None,
        **method_args
    ) -> pd.Series:
        """
        Apply post-processing to all sensors.

        Parameters:
        - X_test: pd.DataFrame with 'sensor_uid' column and test features
        - y_test: pd.Series of ground truth reconstructed speeds
        - y_pred: pd.Series of predicted reconstructed speeds
        - df_for_ML: Optional pd.DataFrame with original train/test indicators
        - method_args: Additional arguments passed to the correction method

        Returns:
        - pd.Series: Combined corrected predictions (same index as inputs)
        """
        # === Ensure input consistency ===
        assert X_test.index.equals(y_test.index), "X_test and y_test indices must match"
        assert X_test.index.equals(y_pred.index), "X_test and y_pred indices must match"

        sensor_ids = X_test["sensor_uid"].unique()

        # === Parallel or Sequential execution ===
        if self.parallel:
            corrected_list: List[pd.Series] = Parallel(n_jobs=self.n_jobs)(
                delayed(self._correct_single_sensor)(
                    sensor_id, X_test, y_test, y_pred, df_for_ML, method_args
                ) for sensor_id in sensor_ids
            )
        else:
            corrected_list = [
                self._correct_single_sensor(sensor_id, X_test, y_test, y_pred, df_for_ML, method_args)
                for sensor_id in tqdm(sensor_ids, desc="Post-processing sensors", disable=not self.verbose)
            ]

        # === Combine results ===
        return pd.concat(corrected_list).sort_index()
    
    
    
    
    
    
    
    
class PredictionCorrectionPerSensor_deprecated:
    """
    Class to apply post-processing corrections to time-series predictions.
    This class assumes y_pred and y_test are already reconstructed speed values.
    It's mainly to be used in order to ensure resulting sensor time-series after
    prediction correction are the same as the ones of the class PredictionCorrection
    (which applies at once the corrections to the complete dataset).
    """

    def __init__(self, X_test, y_test,sensor_uid, df_for_ML=None,rounding=2):
        """
        Parameters:
        - X_test: pd.DataFrame - Test features used to create predictions.
        - y_test: np.ndarray or pd.Series - Actual reconstructed speed values.
        - df_for_ML: pd.DataFrame (optional) - Additional data if needed.
        - rounding: int - Decimal rounding for corrected predictions.
        """
        
        if sensor_uid is not None:
            self.X_test_sensor  = X_test.loc[X_test['sensor_uid']==sensor_uid]
        else:
            self.X_test_sensor = X_test
        sensor_idx = self.X_test_sensor.index
        
        if not sensor_idx.isin(y_test.index).all():
            raise ValueError("Indices of y_test do not align with X_test.")
        self.y_test = y_test.loc[sensor_idx]
        if df_for_ML is not None:
            self.df_for_ML = df_for_ML.loc[df_for_ML['sensor_uid'] == sensor_uid]
            self.train_values = self.df_for_ML.loc[~self.df_for_ML['test_set'], 'value']
        else:
            self.df_for_ML = None
            self.train_values = None
            warnings.warn(
                "df_for_ML is None. Methods that rely on train_values (e.g., constrain_predictions, kalman_smoothing) will not work.",
                UserWarning
            )
        
        self.rounding = rounding
    
    
    def _align_predictions_to_sensor(self, y_pred):
        """
        Align y_pred to the current sensor's index if possible.

        Parameters:
        - y_pred: pd.Series, pd.DataFrame, or np.ndarray

        Returns:
        - np.ndarray: predictions aligned to self.X_test.index
        """
        sensor_idx = self.X_test_sensor.index
        y_pred = y_pred.loc[sensor_idx]
        self.y_pred_per_sensor = y_pred
        return y_pred

    def naive_based_correction(self, y_pred, naive_threshold=0.5):
        """
        Replace predictions deviating significantly from the naive prediction (current speed) by the naive prediction.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - threshold: float - Relative deviation threshold (e.g., 0.5 for 50%).

        Returns:
        - corrected_y_pred: np.ndarray - Corrected predictions.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        current_speed = self.X_test_sensor['value'].values
        deviation = np.abs(y_pred - current_speed) / current_speed
        
        mask = deviation > naive_threshold
        corrected_y_pred = y_pred.copy()
        corrected_y_pred[mask] = current_speed[mask]

        if self.rounding is not None:
            corrected_y_pred = np.round(corrected_y_pred, self.rounding)

        return pd.Series(corrected_y_pred, index=self.X_test_sensor.index)


    def rolling_median_correction(self, y_pred, window_size=3):
        """
        Smooth predictions using rolling median to avoid abrupt spikes.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - window_size: int - Window size for rolling median.

        Returns:
        - smoothed_y_pred: np.ndarray - Smoothed predictions.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        smoothed_y_pred = pd.Series(y_pred).rolling(window=window_size, center=False, min_periods=1).median().values

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred, self.rounding)

        return pd.Series(smoothed_y_pred, index=self.X_test_sensor.index)


    def ewma_smoothing(self, y_pred, span=3):
        """
        Smooth predictions using Exponential Weighted Moving Average (EWMA).

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - span: int - Span parameter for EWMA.

        Returns:
        - smoothed_y_pred: np.ndarray - Smoothed predictions.
        """
        
        y_pred = self._align_predictions_to_sensor(y_pred)
        smoothed_y_pred = pd.Series(y_pred).ewm(span=span, adjust=False).mean().values

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred, self.rounding)

        return pd.Series(smoothed_y_pred, index=self.X_test_sensor.index)

    def constrain_predictions(self, y_pred, min_speed=None, max_speed=None):
        """
        Constrain predictions within observed historical min/max speeds.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - min_speed: float (optional) - Minimum allowed speed.
        - max_speed: float (optional) - Maximum allowed speed.

        Returns:
        - constrained_y_pred: np.ndarray - Predictions constrained to historical bounds.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)
        if self.train_values is None:
            raise ValueError("df_for_ML was not provided or contains no training data.")

        min_speed = min_speed if min_speed is not None else self.train_values.min()
        max_speed = max_speed if max_speed is not None else self.train_values.max()

        constrained_y_pred = np.clip(y_pred, min_speed, max_speed)

        if self.rounding is not None:
            constrained_y_pred = np.round(constrained_y_pred, self.rounding)

        return pd.Series(constrained_y_pred, index=self.X_test_sensor.index)

    
    
    def kalman_smoothing(self, y_pred, Q: float = 1.0, R: Optional[float] = None, handle_nans: bool = True):
        """
        Apply Kalman filter smoothing to predictions.

        Parameters
        ----------
        y_pred : pd.Series or np.ndarray
            Predicted reconstructed speed values.
        Q : float, optional (default=1.0)
            Process (transition) noise covariance. Higher values allow faster changes.
        R : float, optional
            Observation (measurement) noise covariance. If None, uses variance of train values.
        handle_nans : bool, optional (default=True)
            If True, fills NaNs in y_pred using forward/backward fill.

        Returns
        -------
        pd.Series
            Smoothed predictions using Kalman filter.
        """
        y_pred = self._align_predictions_to_sensor(y_pred)

        # Handle missing values
        if handle_nans:
            y_pred = pd.Series(y_pred).fillna(method='ffill').fillna(method='bfill')
        else:
            if np.any(pd.isna(y_pred)) or np.any(np.isinf(y_pred)):
                raise ValueError("y_pred contains NaNs or infs. Set handle_nans=True to auto-fill.")

        if self.train_values is None:
            raise ValueError("df_for_ML was not provided or contains no training data.")

        # Estimate noise parameters
        observation_covariance = R if R is not None else np.var(self.train_values)
        initial_state_mean = np.mean(self.train_values)
        initial_state_covariance = np.var(self.train_values)

        # Build Kalman filter
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=[1],
            observation_matrices=[1],
            observation_covariance=observation_covariance,
            transition_covariance=Q
        )

        smoothed_y_pred, _ = kf.smooth(y_pred)

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred.flatten(), self.rounding)

        return pd.Series(smoothed_y_pred.flatten(), index=self.X_test_sensor.index)


    def apply_all_corrections(self, y_pred, naive_threshold=0.5, rolling_window=3, ewma_span=3):
        """
        Apply all corrections sequentially: naive-based, rolling median, EWMA, and constraints.

        Parameters:
        - y_pred: pd.Series or np.ndarray - Initial predictions to correct.
        - naive_threshold: float - Threshold for naive-based correction.
        - rolling_window: int - Window size for rolling median.
        - ewma_span: int - Span parameter for EWMA smoothing.

        Returns:
        - y_pred_final: pd.Series - Predictions after all corrections.
        """
        y_pred_corrected = self.naive_based_correction(y_pred, naive_threshold)
        y_pred_corrected = self.rolling_median_correction(y_pred_corrected, window_size=rolling_window)
        y_pred_corrected = self.ewma_smoothing(y_pred_corrected, span=ewma_span)
        y_pred_final = self.constrain_predictions(y_pred_corrected)

        return y_pred_final

   
    
    
class PredictionCorrection_deprecated:
    """Post-process sequential predictions sensor-by-sensor in a *vectorised* way.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Feature matrix used at inference time **for the test set only**.  Must
        contain a *value* column that holds the current speed.
    y_test : array-like
        Ground-truth reconstructed speed values (same length/order as
        ``X_test``).
    df_for_ML : pandas.DataFrame, optional
        Full modelling dataframe.  It must contain the boolean column
        ``test_set`` **and** a ``sensor_id`` column.  Training rows (``False``)
        are used to derive per-sensor statistics (min / max / variance).
    rounding : int, default=2
        Decimal precision applied to all returned arrays.  Use ``None`` to
        disable rounding.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(
        self,
        X_test: pd.DataFrame,
        y_test: Union[np.ndarray, pd.Series],
        df_for_ML: pd.DataFrame,
        rounding: Optional[int] = 2,
    ) -> None:
        # Store main arrays
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = np.asarray(y_test)
        self.rounding = rounding

        # ----------------------------------------------------------------
        # Extract sensor meta-data *once* (vector of length = test rows)
        # ----------------------------------------------------------------
        test_mask = df_for_ML["test_set"].values
        if not test_mask.any():
            raise ValueError("`df_for_ML` must contain at least one test row.")

        # Sensor ids aligned with y_pred/y_test order
        self.sensor_ids = df_for_ML.loc[test_mask, "sensor_id"].reset_index(drop=True)

        # Pre-compute per-sensor train statistics for fast vectorised use
        train_stats = (
            df_for_ML.loc[~test_mask, ["sensor_id", "value"]]
            .groupby("sensor_id")["value"]
            .agg(["min", "max", "var", "mean"])
        )
        self._min_by_sensor = train_stats["min"]
        self._max_by_sensor = train_stats["max"]
        self._var_by_sensor = train_stats["var"].fillna(train_stats["var"].mean())
        self._mean_by_sensor = train_stats["mean"]

        # Broadcast min / max to row level once for later *O(1)* look-ups
        self._row_min = self.sensor_ids.map(self._min_by_sensor).values
        self._row_max = self.sensor_ids.map(self._max_by_sensor).values

    # ------------------------------------------------------------------
    # Vectorised helpers
    # ------------------------------------------------------------------
    def _to_series(self, y_pred: np.ndarray, name: str = "pred") -> pd.Series:
        """Return *y_pred* as a pandas Series with sensor_id as second level."""
        s = pd.Series(y_pred, name=name)
        # Attach sensor_id as second level to enable fast group-wise ops
        s.index = pd.MultiIndex.from_arrays([s.index, self.sensor_ids], names=["row", "sensor"])
        return s

    # ------------------------------------------------------------------
    # Post-processing methods
    # ------------------------------------------------------------------
    def naive_based_correction(self, y_pred: np.ndarray, *, naive_threshold: float = 0.5) -> np.ndarray:
        """Clamp predictions that deviate *too much* from the current speed.

        The *current speed* per row is simply the value in ``X_test['value']``.
        Any prediction whose **relative absolute deviation** exceeds the given
        ``naive_threshold`` is replaced by that naive value.
        """
        current_speed = self.X_test["value"].values
        deviation = np.abs(y_pred - current_speed) / np.maximum(current_speed, 1e-9)
        mask = deviation > naive_threshold
        corrected = np.where(mask, current_speed, y_pred)
        return np.round(corrected, self.rounding) if self.rounding is not None else corrected

    def rolling_median_correction(self, y_pred: np.ndarray, *, window_size: int = 3) -> np.ndarray:
        """Apply a *per-sensor* centred rolling-median filter (spike removal)."""
        s = self._to_series(y_pred)
        # groupby+rolling keeps everything in compiled C loops – fast!
        smoothed = (
            s.groupby(level="sensor", sort=False)
            .rolling(window_size, center=False, min_periods=1)
            .median()
            .droplevel("sensor")  # back to single-level index
            .sort_index()          # restore original row order
            .values
        )
        return np.round(smoothed, self.rounding) if self.rounding is not None else smoothed

    def ewma_smoothing(self, y_pred: np.ndarray, *, span: int = 3) -> np.ndarray:
        """Per-sensor Exponentially Weighted Moving Average smoothing."""
        s = self._to_series(y_pred)
        smoothed = (
            s.groupby(level="sensor", sort=False)
            .apply(lambda x: x.droplevel("sensor").ewm(span=span, adjust=False).mean())
            .droplevel("sensor")
            .sort_index()
            .values
        )
        return np.round(smoothed, self.rounding) if self.rounding is not None else smoothed

    def constrain_predictions(
        self,
        y_pred: np.ndarray,
        *,
        min_speed: Optional[np.ndarray] = None,
        max_speed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Clip predictions to realistic *per-sensor* speed bounds."""
        mins = min_speed if min_speed is not None else self._row_min
        maxs = max_speed if max_speed is not None else self._row_max
        constrained = np.clip(y_pred, mins, maxs)
        return np.round(constrained, self.rounding) if self.rounding is not None else constrained

    # ------------------------------------------------------------------
    # Kalman smoothing (cannot be fully vectorised – lightweight loop)
    # ------------------------------------------------------------------
    def kalman_smoothing(self, y_pred: np.ndarray) -> np.ndarray:
        """Run a simple scalar Kalman filter **per sensor**.

        Notes
        -----
        • True full vectorisation is infeasible because each sensor keeps its
          own latent state.  We still avoid Python–level overhead by *pre-
          building* one filter configuration and streaming each sensor’s
          vector through it.
        • This step is optional and typically slower than the other
          corrections; consider disabling it if runtime is critical.
        """
        result = np.empty_like(y_pred, dtype=float)
        s = self._to_series(y_pred)

        # Shared filter parameters derived from *all* training data
        kf = KalmanFilter(
            transition_matrices=[1.0],
            observation_matrices=[1.0],
            transition_covariance=0.01,
        )

        for sensor_id, values in s.groupby(level="sensor", sort=False):
            vec = values.droplevel("sensor").values.reshape(-1, 1)
            # Set sensor-specific initial state using training stats
            kf.initial_state_mean = self._mean_by_sensor.loc[sensor_id]
            kf.initial_state_covariance = self._var_by_sensor.loc[sensor_id] or 1.0
            kf.observation_covariance = self._var_by_sensor.loc[sensor_id] or 1.0
            smoothed, _ = kf.smooth(vec)
            result[values.index.get_level_values("row")] = smoothed.ravel()

        return np.round(result, self.rounding) if self.rounding is not None else result

    # ------------------------------------------------------------------
    # Convenience orchestrator
    # ------------------------------------------------------------------
    def apply_all_corrections(
        self,
        y_pred: np.ndarray,
        *,
        naive_threshold: float = 0.5,
        rolling_window: int = 3,
        ewma_span: int = 3,
        clip: bool = True,
    ) -> np.ndarray:
        """Pipeline that chains the main corrections in a sensible order."""
        y = self.naive_based_correction(y_pred, naive_threshold=naive_threshold)
        y = self.rolling_median_correction(y, window_size=rolling_window)
        y = self.ewma_smoothing(y, span=ewma_span)
        if clip:
            y = self.constrain_predictions(y)
        return y