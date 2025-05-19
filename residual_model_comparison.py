import pickle
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import matplotlib.patheffects as PathEffects
#from .data_processing import TrafficFlowDataProcessing
from .helper_utils import LoggingMixin
from .post_processing import PredictionCorrection
import seaborn as sns
sns.set_style('darkgrid')
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)




class ResidualModelEvaluator(LoggingMixin):
    def __init__(self, X_test, y_test, df_for_ML, disable_logs=False):
        super().__init__(disable_logs=disable_logs)
        self._log("Initializing ResidualModelEvaluator...")
        
        self.X_test = X_test.copy()
        self.y_test = y_test
        self.df_for_ML_raw = df_for_ML.copy()  # Keep original in case needed
        self.main_model_pred_total_speed = None  # Will be set in _align_inputs

        self._align_inputs()

    def _align_inputs(self):
        """
        Align y_test and df_for_ML to the index of X_test.
        Raises warning if 'test_set' column is not found.
        """
        # Align y_test
        if isinstance(self.y_test, (pd.Series, pd.DataFrame)):
            self.y_test = pd.Series(self.y_test, index=self.y_test.index).loc[self.X_test.index]
        else:
            raise ValueError("y_test must be a pandas Series or DataFrame with a proper index.")

        df = self.df_for_ML_raw.copy()

        # Filter to test set if available
        if 'test_set' in df.columns:
            df = df[df['test_set']]
            self._log(f"'test_set' column found. Using only test rows: {len(df)} remaining.")
        else:
            warnings.warn("No 'test_set' column found in df_for_ML. Assuming it already contains only test samples.")
            self._log("Warning: No 'test_set' column found. Using df_for_ML as-is.")

        # Align df to X_test
        df = df.loc[df.index.intersection(self.X_test.index)].copy()
        df = df.loc[self.X_test.index]

        if not df.index.equals(self.X_test.index):
            raise ValueError("Final index mismatch: df_for_ML and X_test must have identical indices.")

        self.df_for_ML = df
        self.main_model_pred_total_speed = df['main_model_prediction']
        self._log(f"Successfully aligned df_for_ML to {len(df)} rows.")

    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def compute_metrics(self, y_true, y_pred):
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        mae = np.mean(abs_errors)
        mae_std = np.std(abs_errors)
        median_ae = np.median(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_std = np.std(errors ** 2)

        # Safe MAPE computation
        safe_y_true = np.where(y_true == 0, 1e-3, y_true)
        mape = np.mean(abs_errors / np.abs(safe_y_true))
        mape_std = np.std(abs_errors / np.abs(safe_y_true))

        return {
            'MAE': mae,
            'MAE_std': mae_std,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'RMSE_std': rmse_std,
            'MAPE': mape * 100,
            'MAPE_std': mape_std * 100
        }
        
    
    def get_predictions(self,model_path):
        
        model = self.load_model(model_path)
        y_pred_residual = model.predict(self.X_test)
        y_pred_total_speed = self.main_model_pred_total_speed + y_pred_residual
        
        return y_pred_total_speed, y_pred_residual

    def evaluate(self, model_path):

        
        self._log("Evaluating residual model performance...")
        y_pred_total_speed_w_residual, y_pred_residual = self.get_predictions(model_path)
        residual_metrics = self.compute_metrics(self.y_test, y_pred_residual)

        self._log("Evaluating total (corrected) prediction performance...")
        true_total_speed = self.df_for_ML['target_total_speed']
        total_metrics = self.compute_metrics(true_total_speed,  y_pred_total_speed_w_residual)
        self.print_evaluation_results(residual_metrics,total_metrics)

        return residual_metrics, total_metrics
    
    
    def evaluate_with_threshold(self, model_path, threshold=3.0):
        """
        Evaluate the residual correction only where |residual prediction| > threshold.
        """
        self._log(f"Evaluating with residual correction threshold: {threshold:.2f} kph")
        model = self.load_model(model_path)
        y_pred_residual = model.predict(self.X_test)

        mask = np.abs(y_pred_residual) > threshold
        self._log(f"Correction will be applied to {np.sum(mask)} out of {len(mask)} samples.")

        corrected = self.main_model_pred_total_speed.copy()
        corrected[mask] += y_pred_residual[mask]

        true_total_speed = self.df_for_ML['target_total_speed']
        total_metrics = self.compute_metrics(true_total_speed, corrected)

        # Only compute residual metrics on the corrected subset
        residual_metrics = self.compute_metrics(self.y_test[mask], y_pred_residual[mask]) if np.any(mask) else {}

        self.print_evaluation_results(residual_metrics, total_metrics)
        return residual_metrics, total_metrics
    
    
    def print_evaluation_results(self, residual_metrics, total_metrics):
        """
        Print evaluation results in a structured format similar to the original ModelEvaluator.
        """
        print("\n--- Residual Evaluation Results ---")
        
        print("\nResidual Metrics (Residual Model Target ≈ y_true - y_pred):")
        print({k: round(v, 2) for k, v in residual_metrics.items()})
        
        print("\nTotal Model Metrics (Corrected Prediction ≈ Final Total Speed):")
        print({k: round(v, 2) for k, v in total_metrics.items()})
        
        print("--------------------------\n")

    def validate_target_computation(self):
        """
        Asserts: target_total_speed ≈ main_model_prediction + residual_target
        """
        true_total_speed = self.df_for_ML['target_total_speed']
        computed_total_speed = self.main_model_pred_total_speed + self.y_test
        assert np.allclose(true_total_speed, computed_total_speed, atol=1e-3), (
            "Residual validation failed: predicted + residual does not match true total speed."
        )
        self._log("CORRECT! Residual validation passed: prediction + residual ≈ true total speed.")