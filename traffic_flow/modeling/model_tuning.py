from calendar import c
import os
import sys
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from ..utils.helper_utils import normalize_data
import pickle
import warnings
import numpy as np
from time import time
from typing import Dict, Optional, Tuple, Union
import pandas as pd


class ModelTunerXGB:
    """
    A class for performing hyperparameter tuning and model evaluation for XGBoost regression models.
    Supports time series-aware cross-validation or standard k-fold.
    Automatically adjusts prediction batch size based on available GPU memory.

    Args:
        X_train: DataFrame
        X_test: DataFrame
        y_train: Series or 1D ndarray
        y_test: Series or 1D ndarray
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        random_state: int = 69,
        use_ts_split: bool = True,
        n_splits: int = 3,
        best_model_name_string_start: str = 'best_model_',
        model_path: Optional[str] = None,
        XGBoost_model_name: Optional[str] = None,
        predict_in_batches: bool = False,
        gpu_memory_gb: Optional[float] = 40.0  # Default to 40GB if unspecified
    ) -> None:
        """
        Initialize the ModelTuner instance.

        Args:
            X_train: DataFrame
            X_test: DataFrame
            y_train: Series or 1D ndarray
            y_test: Series or 1D ndarray
            random_state: Random seed for reproducibility.
            use_ts_split: Whether to use TimeSeriesSplit or KFold.
            n_splits: Number of cross-validation splits.
            best_model_name_string_start: Prefix for saved model filenames.
            model_path: Directory to save best models.
            XGBoost_model_name: Custom name for the XGBoost model.
            predict_in_batches: Whether to use batched prediction.
            gpu_memory_gb: Amount of GPU memory in GB (used to estimate batch size).
        Raises:
            TypeError: If any input is not the required type.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)}")
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be a pandas DataFrame, got {type(X_test)}")
        if not (isinstance(y_train, pd.Series) or (isinstance(y_train, np.ndarray) and y_train.ndim == 1)):
            raise TypeError(f"y_train must be a pandas Series or 1D numpy array, got {type(y_train)} with shape {getattr(y_train, 'shape', None)}")
        if not (isinstance(y_test, pd.Series) or (isinstance(y_test, np.ndarray) and y_test.ndim == 1)):
            raise TypeError(f"y_test must be a pandas Series or 1D numpy array, got {type(y_test)} with shape {getattr(y_test, 'shape', None)}")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_models = {}
        self.random_state = random_state
        self.use_ts_split = use_ts_split
        self.n_splits = n_splits
        self.best_model_name_string_start = best_model_name_string_start
        self.model_path = model_path if model_path else './models'
        os.makedirs(self.model_path, exist_ok=True)
        self.XGBoost_model_name = XGBoost_model_name or 'XGBoost'
        self.predict_in_batches = predict_in_batches
        self.batch_size = self._estimate_batch_size(gpu_memory_gb if gpu_memory_gb is not None else 40.0)
        self.best_model = None

    def _estimate_batch_size(self ,gpu_memory_gb: float, use_available_memory_ratio: float = 0.5) -> int:
        """
        Estimate batch size for prediction based on GPU memory.

        Args:
            gpu_memory_gb: Total available GPU memory in GB.

        Returns:
            Estimated batch size as an integer.
        """
        bytes_per_float32 = 4
        bytes_per_row = self.X_test.shape[1] * bytes_per_float32
        available_bytes = gpu_memory_gb * 1e9 * use_available_memory_ratio  # Use 50% of available GPU memory
        batch_size = int(available_bytes // bytes_per_row)
        return max(1000, min(batch_size, len(self.X_test)))

    def get_cv_splitter(self):
        """Return appropriate cross-validation splitter."""
        return TimeSeriesSplit(n_splits=self.n_splits) if self.use_ts_split else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def _calculate_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculate regression error metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Tuple containing MAE, MAE std, Median AE, Median AE std, RMSE, MAPE, MAPE std.
        """
        absolute_errors = np.abs(y_true - y_pred)
        mae = np.mean(absolute_errors)
        mae_std = np.std(absolute_errors)
        median_ae = np.median(absolute_errors)
        median_ae_std = np.std(absolute_errors)
        rmse = np.sqrt(np.mean(absolute_errors**2))
        safe_mape = np.where(y_true == 0, np.nan, absolute_errors / y_true) * 100
        mape = np.nanmean(safe_mape)
        mape_std = np.nanstd(safe_mape)
        return float(mae), float(mae_std), float(median_ae), float(median_ae_std), float(rmse), float(mape), float(mape_std)

    def _predict_in_batches(self, model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
        """
        Predict in smaller batches to avoid memory issues on large datasets.

        Args:
            model: Trained XGBoost model.
            X: Input features to predict.

        Returns:
            Numpy array of predictions.
        """
        n_rows = X.shape[0]
        preds = []
        for start in range(0, n_rows, self.batch_size):
            end = min(start + self.batch_size, n_rows)
            preds.append(model.predict(X[start:end]))
        return np.concatenate(preds)

    def tune_xgboost(
        self,
        model_name: Optional[str] = None,
        params: Optional[Dict] = None,
        use_gpu: bool = True,
        objective: Optional[str] = None,
        suppress_output: bool = False,
        n_jobs: int = -1
    ) -> Tuple[str, Dict, float, float]:
        """
        Perform hyperparameter tuning using GridSearchCV for an XGBoost model.

        Returns:
            Path to best saved model, best hyperparameters, retraining time, total tuning time.
        """
        model_name = model_name or self.XGBoost_model_name
        objective = objective or 'reg:squarederror'
        grid_params = params or {
            'max_depth': [10, 8, 6, 4],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [1000, 750, 500, 250]
        }

        model = self._create_xgboost_model(use_gpu, objective, suppress_output, n_jobs)
        cv_splitter = self.get_cv_splitter()

        grid = GridSearchCV(
            model, grid_params, scoring='neg_mean_absolute_error',
            cv=cv_splitter, verbose=0 if suppress_output else 3
        )

        start_total = time()
        grid.fit(self.X_train, self.y_train)
        end_total = time()
        total_time = end_total - start_total

        best_model = grid.best_estimator_

        # Retrain best model and time it
        start_train = time()
        best_model.fit(self.X_train, self.y_train)
        end_train = time()
        training_time = end_train - start_train

        best_model_path, best_params_ = self._finalize_model(best_model, grid.best_params_, model_name)

        if not suppress_output:
            print(f"Retraining time with best params: {training_time:.2f} seconds")
            print(f"Total hyperparameter tuning time: {total_time:.2f} seconds")
        
        self.best_model = best_model

        return best_model_path, best_params_, training_time, total_time

    def _create_xgboost_model(self, use_gpu: bool, objective: str, suppress_output: bool, n_jobs: int) -> xgb.XGBRegressor:
        """Create and return a configured XGBoost model."""
        if use_gpu:
            if n_jobs != 1:
                print("[WARNING] Overriding n_jobs to 1 for GPU usage.")
                n_jobs = 1
            return xgb.XGBRegressor(
                objective=objective, tree_method='gpu_hist', predictor='gpu_predictor',
                n_jobs=n_jobs, random_state=self.random_state, verbosity=0 if suppress_output else 1)
        return xgb.XGBRegressor(
            objective=objective, n_jobs=n_jobs, random_state=self.random_state, verbosity=0 if suppress_output else 1)

    def _finalize_model(self, best_model: xgb.XGBRegressor, best_params: Dict, model_name: str) -> Tuple[str, Dict]:
        """Save the best model and return its path and hyperparameters."""
        if self.predict_in_batches:
            if isinstance(self.X_test, (pd.DataFrame, pd.Series)):
                X_test_arr = self.X_test.values
            else:
                X_test_arr = np.asarray(self.X_test)
            y_pred = self._predict_in_batches(best_model, X_test_arr)
        else:
            if isinstance(self.X_test, (pd.DataFrame, pd.Series)):
                X_test_arr = self.X_test.values
            else:
                X_test_arr = np.asarray(self.X_test)
            y_pred = best_model.predict(X_test_arr)

        if isinstance(self.y_test, (pd.DataFrame, pd.Series)):
            y_test_arr = self.y_test.values
        else:
            y_test_arr = np.asarray(self.y_test)
        y_test_arr = np.asarray(y_test_arr)
        self._calculate_errors(y_test_arr, y_pred)

        # Compute naive baseline
        if isinstance(self.X_test, pd.DataFrame):
            X_test_value = self.X_test['value'].values
        elif isinstance(self.X_test, pd.Series):
            X_test_value = self.X_test.values
        else:
            # fallback for numpy structured array or dict-like
            try:
                X_test_value = self.X_test['value']
            except Exception:
                X_test_value = np.asarray(self.X_test)
        X_test_value = np.asarray(X_test_value)
        naive_pred = np.abs(X_test_value - (y_test_arr + X_test_value))
        self._calculate_errors(y_test_arr, naive_pred)

        best_model_path = self._save_model(model_name, best_model)
        return best_model_path, best_params

    def _save_model(self, model_name: str, model: xgb.XGBRegressor) -> str:
        """Save the trained model to disk and return its file path."""
        file_ext = 'pkl'
        file_name = f"{self.best_model_name_string_start}{model_name}.{file_ext}"
        file_path = os.path.join(self.model_path, file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        return file_path




