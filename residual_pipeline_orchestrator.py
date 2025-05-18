import pandas as pd
import numpy as np
import warnings
from traffic_flow_package_src.helper_utils import LoggingMixin

class ResidualTrafficPipelineOrchestrator(LoggingMixin):
    """
    Constructs a residual modeling dataset using predicted total speed,
    optional speed delta, and true speed, independent of model internals.
    """

    def __init__(
        self,
        X_test,
        y_pred_total_speed,
        y_true_total_speed,
        df,
        y_pred_speed_delta=None,
        disable_logs=False,
        date_col = 'date'
    ):
        super().__init__(disable_logs=disable_logs)
        self._log("Initializing ResidualTrafficPipelineOrchestrator...")

        self.X_original = X_test.copy()
        self.y_pred_total_speed = y_pred_total_speed
        self.y_pred_speed_delta = y_pred_speed_delta
        self.y_true_total_speed = y_true_total_speed
        self.date_col = date_col
        df = df.sort_values(by=self.date_col)
        self.df_orig = df.copy()
        self.df = df.loc[X_test.index]
        self.df.sort_values(by=self.date_col,inplace=True)
        if 'test_set' in self.df.columns:
            df.drop(columns=['test_set'],inplace=True)

        if not (len(self.X_original) == len(self.y_pred_total_speed) == len(self.y_true_total_speed)):
            raise ValueError("Input lengths of X_test, y_pred_total_speed, and y_true_total_speed must match.")

        if self.y_pred_speed_delta is not None and len(self.y_pred_speed_delta) != len(self.X_original):
            raise ValueError("Length of y_pred_speed_delta must match X_test if provided.")

    def run_pipeline(self,test_size = 0.33):
        """
        Computes residuals and performs time-aware train/test split.

        Returns:
        - X_train, X_test, y_train, y_test
        """
        self._log("Computing residuals (true - predicted total speed)...")
        residual = self.y_true_total_speed - self.y_pred_total_speed

        self._log("Creating residual dataset and injecting features...")
        self.df["main_model_prediction"] = self.y_pred_total_speed
        if self.y_pred_speed_delta is not None:
            self.df["main_model_speed_delta"] = self.y_pred_speed_delta
        else:
            warnings.warn("y_pred_speed_delta is None — 'main_model_speed_delta' will not be included as a feature.")
            self._log("Warning: y_pred_speed_delta is None. Feature will be excluded.")

        self.df["residual_target"] = residual

        self._log("Performing time-aware train/test split...")
        n = len(self.df)
        n_test = int(n * test_size)
        n_train = n - n_test

        self.df['test_set'] = False
        self.df['test_set'].iloc[n_train:] = True
        train_df = self.df[~self.df['test_set']].copy()
        test_df = self.df[self.df['test_set']].copy()

        X_train = train_df.drop(columns=["residual_target"])
        y_train = train_df["residual_target"]
        X_test = test_df.drop(columns=["residual_target"])
        y_test = test_df["residual_target"]
        
        cols_to_drop = ['sensor_id', 'target_total_speed','target',
                'target_speed_delta', 'date', 'sensor_id', 
                'test_set', 'gman_prediction_date', 'gman_target_date','date_of_prediction']
        
        for df in [X_train, X_test]:

            df = df.drop(
                columns=[col for col in cols_to_drop if col in df.columns], inplace=True)


        self._log(f"Split complete: {n_train} train rows, {n_test} test rows.")
        
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test
    
    
    
    def validate_target_computation(self):
        computed = self.df["main_model_prediction"] + self.df["residual_target"]
        actual = self.y_true_total_speed
        assert np.allclose(computed, actual, atol=1e-3), "Residual reconstruction failed."
        self._log("CORRECT! Residual target validated: prediction + residual ≈ true total speed.")
