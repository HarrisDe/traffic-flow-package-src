
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from .constants import colnames
import random
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm
from .helper_utils import *
import pickle
import time
import json
from pykalman import KalmanFilter
import re
# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # You can set this to DEBUG, WARNING, etc. as needed
)

import numpy as np
import pandas as pd


class PredictionCorrection:
    """
    Class to apply post-processing corrections to time-series predictions.
    This class assumes y_pred and y_test are already reconstructed speed values.
    """

    def __init__(self, X_test, y_test, df_for_ML=None, rounding=2):
        """
        Parameters:
        - X_test: pd.DataFrame - Test features used to create predictions.
        - y_test: np.ndarray or pd.Series - Actual reconstructed speed values.
        - df_for_ML: pd.DataFrame (optional) - Additional data if needed.
        - rounding: int - Decimal rounding for corrected predictions.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.df_for_ML = df_for_ML
        self.rounding = rounding

    def naive_based_correction(self, y_pred, naive_threshold=0.5):
        """
        Replace predictions deviating significantly from the naive prediction (current speed) by the naive prediction.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - threshold: float - Relative deviation threshold (e.g., 0.5 for 50%).

        Returns:
        - corrected_y_pred: np.ndarray - Corrected predictions.
        """
        current_speed = self.X_test['value'].values
        deviation = np.abs(y_pred - current_speed) / current_speed
        
        mask = deviation > naive_threshold
        corrected_y_pred = y_pred.copy()
        corrected_y_pred[mask] = current_speed[mask]

        if self.rounding is not None:
            corrected_y_pred = np.round(corrected_y_pred, self.rounding)

        return corrected_y_pred

    def rolling_median_correction(self, y_pred, window_size=3):
        """
        Smooth predictions using rolling median to avoid abrupt spikes.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - window_size: int - Window size for rolling median.

        Returns:
        - smoothed_y_pred: np.ndarray - Smoothed predictions.
        """
        smoothed_y_pred = pd.Series(y_pred).rolling(window=window_size, center=True, min_periods=1).median().values

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred, self.rounding)

        return smoothed_y_pred

    def ewma_smoothing(self, y_pred, span=3):
        """
        Smooth predictions using Exponential Weighted Moving Average (EWMA).

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.
        - span: int - Span parameter for EWMA.

        Returns:
        - smoothed_y_pred: np.ndarray - Smoothed predictions.
        """
        smoothed_y_pred = pd.Series(y_pred).ewm(span=span, adjust=False).mean().values

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred, self.rounding)

        return smoothed_y_pred

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
        min_speed = min_speed if min_speed is not None else self.X_test['value'].min()
        max_speed = max_speed if max_speed is not None else self.X_test['value'].max()

        constrained_y_pred = np.clip(y_pred, min_speed, max_speed)

        if self.rounding is not None:
            constrained_y_pred = np.round(constrained_y_pred, self.rounding)

        return constrained_y_pred
    
    
    def kalman_smoothing(self, y_pred):
        """
        Apply Kalman filter smoothing to predictions.

        Parameters:
        - y_pred: np.ndarray - Predicted reconstructed speed values.

        Returns:
        - smoothed_y_pred: np.ndarray - Kalman-filtered predictions.
        """
        # Estimate initial parameters from training data
        train_values = self.df_for_ML.loc[~self.df_for_ML['test_set'], 'value'].values
        initial_state_mean = np.mean(train_values)
        initial_state_covariance = np.var(train_values)

        # Define Kalman filter
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=[1],  # assumes next state is similar to current
            observation_matrices=[1],
            observation_covariance=np.var(train_values),  # measurement noise
            transition_covariance=0.01  # small transition noise
        )

        smoothed_y_pred, _ = kf.smooth(y_pred)

        if self.rounding is not None:
            smoothed_y_pred = np.round(smoothed_y_pred.flatten(), self.rounding)

        return smoothed_y_pred

    def apply_all_corrections(self, y_pred, naive_threshold=0.5, rolling_window=3, ewma_span=3):
        """
        Apply all corrections sequentially: naive-based, rolling median, EWMA, and constraints.

        Parameters:
        - y_pred: np.ndarray - Initial predictions to correct.
        - naive_threshold: float - Threshold for naive-based correction.
        - rolling_window: int - Window size for rolling median.
        - ewma_span: int - Span parameter for EWMA smoothing.

        Returns:
        - y_pred_final: np.ndarray - Predictions after all corrections.
        """
        y_pred_corrected = self.naive_based_correction(y_pred, threshold=naive_threshold)
        y_pred_corrected = self.rolling_median_correction(y_pred_corrected, window_size=rolling_window)
        y_pred_corrected = self.ewma_smoothing(y_pred_corrected, span=ewma_span)
        y_pred_final = self.constrain_predictions(y_pred_corrected)

        return y_pred_final
