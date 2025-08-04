from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,  median_absolute_error
import numpy as np
from keras.models import load_model
import matplotlib.patheffects as PathEffects
#from .data_processing import TrafficFlowDataProcessing
from ..utils.helper_utils import normalize_data
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from time import time
import pandas as pd
from typing import Optional, Dict, Union
from ..inference.prediction_protocol import make_prediction_frame




class ModelEvaluator:
    """
    A class to evaluate saved regression models (XGBoost/.pkl or Keras/.h5) using common metrics,
    including SMAPE, MAE, RMSE, and MAPE. Handles normalization and speed reconstruction.
    """

    def __init__(self, X_test, df_for_ML, y_train=None, y_test=None, rounding=2,
                 discard_zero_mape=False, target_is_gman_error_prediction=False,
                 sensor_col = 'sensor_id',date_col = 'date',value_col = 'value',
                 y_is_normalized=False, epsilon=1e-2):
        self.X_test = X_test
        self.df_for_ML = df_for_ML[df_for_ML['test_set']]
        self.y_train = y_train
        self.rounding = int(rounding)
        self.discard_zero_mape = discard_zero_mape
        self.target_is_gman_error_prediction = target_is_gman_error_prediction
        self.y_is_normalized = y_is_normalized
        self.df_predictions = None
        self.date_col = date_col
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.epsilon = epsilon

        if self.y_is_normalized:
            if y_train is None:
                raise ValueError(
                    "y_train must be provided for de-normalization.")
            self.y_mean = np.mean(y_train)
            self.y_std = np.std(y_train)
        self.y_test = y_test
        self.y_test_before_reconstruction = y_test.copy()
        self.y_test = self.reconstruct_y(self.y_test)

        zero_percentage = self.calculate_discarded_percentage()
        self.calculate_mape_with_handling_zero_values = zero_percentage > 0

    def calculate_discarded_percentage(self):
        zero_mask = self.y_test == 0
        return (np.sum(zero_mask) / len(self.y_test)) * 100

    def load_model_from_path(self, model_path):
        if model_path.endswith('.h5'):
            return load_model(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        raise ValueError(f"Unsupported model file: {model_path}")

    def reconstruct_y(self, y):
        if self.target_is_gman_error_prediction:
            return y + self.df_for_ML['gman_prediction_orig']
        return y + self.X_test[self.value_col]


    def smape(self, y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape_vals = np.abs(y_pred - y_true) / np.where(denominator == 0, 1, denominator)
        return np.mean(smape_vals) * 100, np.std(smape_vals) * 100

    def evaluate_model_from_path(self, model_path = None,saved_model = None, print_results=True):
        
        if model_path is not None and saved_model is not None:
            import warnings
            warnings.warn("Both `model_path` and `saved_model` are provided. Using `saved_model` and ignoring `model_path`.")
            model = saved_model
        elif model_path is not None:
            model = self.load_model_from_path(model_path)
        else:
            model = saved_model
            
        if model_path is not None:
            model = self.load_model_from_path(model_path)
        else:
            model = saved_model
        time_start = time() # start measuring inference time after model is loaded to exclude model loading overhead time
        if model_path and 'neural' in model_path.lower():
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            y_pred = model.predict(X_test_normalized).flatten()
        else:
            y_pred = model.predict(self.X_test)

        if self.y_is_normalized:
            y_pred = y_pred * self.y_std + self.y_mean

        self.y_pred_before_reconstruction = y_pred.copy()
        y_pred = self.reconstruct_y(y_pred)
        time_end = time()
        self.df_predictions = self.df_for_ML[[self.date_col,self.sensor_col,'date_of_prediction',self.value_col,
                                              'target_total_speed']]
        self.df_predictions['y_pred'] = y_pred
        self.y_pred = y_pred
        inference_time = time_end - time_start
        self.df_for_ML['y_pred'] = self.y_pred
        inference_time_per_sample = inference_time / len(self.X_test)
        abs_errors = np.abs(self.y_test - y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        median_ae = median_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        smape, smape_std = self.smape(self.y_test, y_pred)
        naive_smape, naive_smape_std = self.smape(
            self.y_test, self.X_test[self.value_col])
        

        if self.calculate_mape_with_handling_zero_values:
            mape, mape_std, naive_mape, naive_mape_std = self.calculate_mape_in_case_of_zero_values(
                y_pred)
        else:
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
            mape_std = np.std(abs_errors / np.abs(self.y_test))
            naive_error = np.abs(self.y_test_before_reconstruction)
            naive_mape = np.mean(naive_error / np.abs(self.y_test))
            naive_mape_std = np.std(naive_error / np.abs(self.y_test))

        metrics = {
            'MAE': mae,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'MAPE': mape * 100,
            'SMAPE': smape,
            'inference_time':inference_time,
            'inference_time_per_sample': inference_time_per_sample
        }

        metrics_std = {
            'MAE_std': np.std(abs_errors),
            'Median_AE_std': np.std(abs_errors),
            'RMSE_std': np.std((self.y_test - y_pred) ** 2),
            'MAPE_std': mape_std * 100,
            'SMAPE_std': smape_std
        }

        if self.target_is_gman_error_prediction:
            naive_error = np.abs(self.df_for_ML['target_speed_delta'])
        else:
            naive_error = np.abs(self.y_test_before_reconstruction)

        naive_metrics = {
            'Naive_MAE': np.mean(naive_error),
            'Naive_Median_AE': np.median(naive_error),
            'Naive_RMSE': np.sqrt(np.mean(naive_error ** 2)),
            'Naive_MAPE': naive_mape * 100,
            'Naive_SMAPE': naive_smape
        }

        naive_metrics_std = {
            'Naive_MAE_std': np.std(naive_error),
            'Naive_Median_AE_std': np.std(naive_error),
            'Naive_RMSE_std': np.std(naive_error ** 2),
            'Naive_MAPE_std': naive_mape_std * 100,
            'Naive_SMAPE_std': naive_smape_std
        }

        if self.rounding is not None:
            metrics = {k: round(v, self.rounding) for k, v in metrics.items()}
            metrics_std = {k: round(v, self.rounding)
                           for k, v in metrics_std.items()}
            naive_metrics = {k: round(v, self.rounding)
                             for k, v in naive_metrics.items()}
            naive_metrics_std = {k: round(v, self.rounding)
                                 for k, v in naive_metrics_std.items()}

        if print_results:
            self.print_evaluation_results(
                metrics, metrics_std, naive_metrics, naive_metrics_std)

        return {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std
        }

    def print_evaluation_results(self, metrics, metrics_std, naive_metrics, naive_metrics_std):
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
        if self.discard_zero_mape:
            mask = self.y_test != 0
            y_test = self.y_test[mask]
            y_pred = y_pred[mask]
            y_test_base = self.y_test_before_reconstruction[mask]
            x_val = self.X_test[mask][self.value_col]
        else:
            y_test = np.where(self.y_test == 0, self.epsilon, self.y_test)
            y_pred = y_pred
            y_test_base = np.where(self.y_test_before_reconstruction ==
                                   0, self.epsilon, self.y_test_before_reconstruction)
            x_val = self.X_test[self.value_col]

        ape = np.abs(y_test - y_pred) / np.abs(y_test)
        naive_ape = np.abs(y_test_base / (y_test_base + x_val))

        return np.mean(ape), np.std(ape), np.mean(naive_ape), np.std(naive_ape)
    
    
    def to_canonical_predictions(
        self,
        *,
        model=None,
        model_path: Optional[str] = None,
        states: Optional[dict] = None,
        horizon_min: Optional[int] = None,
        add_total: bool = True,
    ) -> pd.DataFrame:
        """
        Produce the canonical prediction DataFrame for the *test* split using the
        evaluator's X_test and df_for_ML (which is already filtered to test_set).

        Returns columns:
        sensor_id, input_time, prediction_time, y_pred_delta, y_pred_total, horizon
        """
        # 1) Get model
        if model is None and model_path is not None:
            model = self.load_model_from_path(model_path)
        if model is None:
            raise ValueError("Provide a model or model_path.")

        # 2) Predict deltas on X_test
        y_pred_delta = model.predict(self.X_test)

        # 3) Build a minimal RAW-like frame aligned to X_test rows
        df_test = self.df_for_ML.copy()   # already test_set only
        # input_time for the row; if 'date' is missing, reconstruct from date_of_prediction - horizon
        if self.date_col in df_test.columns:
            raw_date = df_test[self.date_col].copy()
        else:
            if horizon_min is None:
                raise ValueError("horizon_min is required if 'date' is not present in df_for_ML.")
            raw_date = pd.to_datetime(df_test["date_of_prediction"]) - pd.to_timedelta(horizon_min, unit="m")

        df_raw_min = pd.DataFrame({
            self.sensor_col: df_test[self.sensor_col],
            self.date_col: pd.to_datetime(raw_date, errors="coerce"),
            self.value_col: self.X_test[self.value_col].to_numpy(dtype=float),
            'target_total_speed': df_test['target_total_speed']
        }, index=self.X_test.index)

        # 4) Infer horizon if not provided
        if horizon_min is None:
            if "date_of_prediction" in df_test.columns and self.date_col in df_test.columns:
                delta = pd.to_datetime(df_test["date_of_prediction"]) - pd.to_datetime(df_test[self.date_col])
                horizon_min = int(round(delta.dt.total_seconds().median() / 60.0))
            else:
                # reasonable fallback
                horizon_min = 15
                warnings.warn("Neither horizon_min nor the required column to calculate the horizon manually(date_of_prediction) is provided. Assuming horizon is 15 minutes.")

        if states is None:
            raise ValueError("states (from tdp.export_states()) must be provided.")
        print(f"df_raw_min_cols: {df_raw_min.columns}")
        # 5) Canonical frame
        pred_df = make_prediction_frame(
            df=self.df_for_ML.rename(columns={self.date_col: "date"}),  # column name expected by states
            feats=self.X_test,
            pred_delta=y_pred_delta,
            states=states,
            horizon_min=horizon_min,
            add_total=add_total,
            sensor_col=self.sensor_col,
            add_y_act=True
        )
        return pred_df
    
    
