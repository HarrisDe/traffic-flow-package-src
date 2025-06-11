import pandas as pd
import numpy as np
import warnings
from traffic_flow_package_src.utils.helper_utils import LoggingMixin



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
        date_col='date'
    ):
        super().__init__(disable_logs=disable_logs)
        self._log("Initializing ResidualTrafficPipelineOrchestrator...")

        # Store input
        self.X_original = X_test.copy()
        self.y_pred_total_speed = y_pred_total_speed
        self.y_true_total_speed = y_true_total_speed
        self.y_pred_speed_delta = y_pred_speed_delta
        self.date_col = date_col
        self._features_injected = False

        # Sort df and align to X_test index
        self.df_orig = df.sort_values(by=self.date_col).copy()
        self.df = self.df_orig.loc[X_test.index].copy()
        self.df.sort_values(by=self.date_col, inplace=True)

        # Validate input lengths
        if not (len(X_test) == len(y_pred_total_speed) == len(y_true_total_speed)):
            raise ValueError("Lengths of X_test, y_pred_total_speed, and y_true_total_speed must match.")

        if y_pred_speed_delta is not None and len(y_pred_speed_delta) != len(X_test):
            raise ValueError("Length of y_pred_speed_delta must match X_test if provided.")

        self._align_data()

    def _align_data(self):
        """Align predictions with sorted df index."""
        self._log("Aligning predictions with DataFrame index...")
        self.y_pred_total_speed = pd.Series(self.y_pred_total_speed, index=self.X_original.index).loc[self.df.index]
        self.y_true_total_speed = pd.Series(self.y_true_total_speed, index=self.X_original.index).loc[self.df.index]

        if self.y_pred_speed_delta is not None:
            self.y_pred_speed_delta = pd.Series(self.y_pred_speed_delta, index=self.X_original.index).loc[self.df.index]

    def _inject_features(self):
        """Inject prediction features and compute residual target."""
        if self._features_injected:
            self._log("Features already injected. Skipping.")
            return

        self._log("Injecting prediction features and computing residuals...")
        self.df["main_model_prediction"] = self.y_pred_total_speed
        self.df["residual_target"] = self.y_true_total_speed - self.y_pred_total_speed

        if self.y_pred_speed_delta is not None:
            self.df["main_model_speed_delta"] = self.y_pred_speed_delta
        else:
            warnings.warn("y_pred_speed_delta is None — 'main_model_speed_delta' will not be included.")
            self._log("y_pred_speed_delta is None; skipping feature.")

        self._features_injected = True

    def _drop_unused_columns(self, df):
        """Drop non-feature columns from X."""
        cols_to_drop = [
            'sensor_id', 'target_total_speed', 'target', 'target_speed_delta',
            'date', 'test_set', 'gman_prediction_date', 'gman_target_date', 'date_of_prediction'
        ]
        return df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    def run_pipeline(self, test_size=0.33):
        """
        Computes residuals and performs time-aware train/test split.
        Returns: X_train, X_test, y_train, y_test
        """
        self._inject_features()

        self._log("Performing time-aware train/test split...")
        n = len(self.df)
        n_test = int(n * test_size)
        n_train = n - n_test

        self.df['test_set'] = False
        self.df.iloc[n_train:, self.df.columns.get_loc('test_set')] = True

        train_df = self.df[~self.df['test_set']].copy()
        test_df = self.df[self.df['test_set']].copy()

        X_train = self._drop_unused_columns(train_df.drop(columns=["residual_target"]))
        y_train = train_df["residual_target"]
        X_test = self._drop_unused_columns(test_df.drop(columns=["residual_target"]))
        y_test = test_df["residual_target"]

        self._log(f"Split complete: {n_train} train rows, {n_test} test rows.")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test




# class ResidualTrafficPipelineOrchestrator(LoggingMixin):
#     """
#     Constructs a residual modeling dataset using predicted total speed,
#     optional speed delta, and true speed, independent of model internals.
#     """

#     def __init__(
#         self,
#         X_test,
#         y_pred_total_speed,
#         y_true_total_speed,
#         df,
#         y_pred_speed_delta=None,
#         disable_logs=False,
#         date_col = 'date'
#     ):
#         super().__init__(disable_logs=disable_logs)
#         self._log("Initializing ResidualTrafficPipelineOrchestrator...")
        
        
#         self.X_original = X_test.copy()
#         self.y_pred_total_speed = y_pred_total_speed
#         self.y_pred_speed_delta = y_pred_speed_delta
#         self.y_true_total_speed = y_true_total_speed
#         self.date_col = date_col
#         df = df.sort_values(by=self.date_col)
#         self.df_orig = df.copy()
#         self.df = df.loc[X_test.index]
#         self.df.sort_values(by=self.date_col,inplace=True)
#         if 'test_set' in self.df.columns:
#             df.drop(columns=['test_set'],inplace=True)

#         if not (len(self.X_original) == len(self.y_pred_total_speed) == len(self.y_true_total_speed)):
#             raise ValueError("Input lengths of X_test, y_pred_total_speed, and y_true_total_speed must match.")

#         if self.y_pred_speed_delta is not None and len(self.y_pred_speed_delta) != len(self.X_original):
#             raise ValueError("Length of y_pred_speed_delta must match X_test if provided.")

#     def run_pipeline(self,test_size = 0.33):
#         """
#         Computes residuals and performs time-aware train/test split.

#         Returns:
#         - X_train, X_test, y_train, y_test
#         """
#         self._log("Computing residuals (true - predicted total speed)...")
#         residual = self.y_true_total_speed - self.y_pred_total_speed

#         self._log("Creating residual dataset and injecting features...")
#         self.df["main_model_prediction"] = self.y_pred_total_speed
#         if self.y_pred_speed_delta is not None:
#             self.df["main_model_speed_delta"] = self.y_pred_speed_delta
#         else:
#             warnings.warn("y_pred_speed_delta is None — 'main_model_speed_delta' will not be included as a feature.")
#             self._log("Warning: y_pred_speed_delta is None. Feature will be excluded.")

#         self.df["residual_target"] = residual

#         self._log("Performing time-aware train/test split...")
#         n = len(self.df)
#         n_test = int(n * test_size)
#         n_train = n - n_test

#         self.df['test_set'] = False
#         self.df['test_set'].iloc[n_train:] = True
#         train_df = self.df[~self.df['test_set']].copy()
#         test_df = self.df[self.df['test_set']].copy()

#         X_train = train_df.drop(columns=["residual_target"])
#         y_train = train_df["residual_target"]
#         X_test = test_df.drop(columns=["residual_target"])
#         y_test = test_df["residual_target"]
        
#         cols_to_drop = ['sensor_id', 'target_total_speed','target',
#                 'target_speed_delta', 'date', 'sensor_id', 
#                 'test_set', 'gman_prediction_date', 'gman_target_date','date_of_prediction']
        
#         for df in [X_train, X_test]:

#             df = df.drop(
#                 columns=[col for col in cols_to_drop if col in df.columns], inplace=True)


#         self._log(f"Split complete: {n_train} train rows, {n_test} test rows.")
        
        
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test

#         return X_train, X_test, y_train, y_test
    
    
    
#     def validate_target_computation(self):
#         computed = self.df["main_model_prediction"] + self.df["residual_target"]
#         actual = self.y_true_total_speed
#         assert np.allclose(computed, actual, atol=1e-3), "Residual reconstruction failed."
#         self._log("CORRECT! Residual target validated: prediction + residual ≈ true total speed.")
