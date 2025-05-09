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
from .helper_utils import normalize_data
from .post_processing import PredictionCorrection
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




class ModelEvaluator:
    """
    A class to evaluate multiple models using saved model files (.h5 for Keras or .pkl for scikit-learn compatible models).
    It computes error metrics and their standard deviations for each model, with optional rounding.
    """

    def __init__(self, X_test, df_for_ML, y_train=None, y_test=None, rounding=2, prediction_correction=None,
                 discard_zero_mape=False, target_is_gman_error_prediction=True, y_is_normalized=False, epsilon=1e-2):
        """
        Initialize the ModelEvaluator with test data.

        Parameters
        ----------
        X_test : array-like
            Test feature set.
        y_test : array-like
            True values for the test set.
        rounding : int, optional
            Number of decimal places to round the results. If None, no rounding is applied.
        discard_zero_mape : bool, optional
            If True, discard zero values in y_test for MAPE calculation. Default is False.
        epsilon : float, optional
            Small value to replace zero values in y_test for MAPE calculation when discard_zero_mape is False. Default is 1e-10.
        """
        self.y_train = y_train
        self.X_test = X_test
        self.y_is_normalized = y_is_normalized
        self.target_is_gman_error_prediction = target_is_gman_error_prediction
        if self.y_is_normalized:
            if self.y_train is None:
                raise ValueError(
                    " y_train (to get the averages) must be provided if y_is_normalized is True")

            self.y_mean = np.mean(self.y_train)
            self.y_std = np.std(self.y_train)
            print(
                f"from y_train, y_mean is {self.y_mean}, y_std is {self.y_std}")
            y_mean_df_for_ML = df_for_ML.loc[
                df_for_ML['test_set'] == False, 'target'].mean()
            y_std_df_for_ML = df_for_ML.loc[
                df_for_ML['test_set'] == False, 'target'].std()
            print(
                f"from df_for_ML, y_mean is {y_mean_df_for_ML}, y_std is {y_std_df_for_ML}")
            

        #     # Apply Z-de-Normalization
        #     print('self.y_test is being de-normalized')
        #     self.y_test = self.y_test_norm * y_std + y_mean
        # else:
        #     if y_test is None:
        #         raise ValueError(
        #             "y_test must be provided if y_is_normalized is False")
        #     self.y_test = y_test
        self.y_test = y_test
        self.df_for_ML = df_for_ML[df_for_ML['test_set']]
        self.y_test_before_reconstruction = y_test.copy()
        self.y_test = self.reconstruct_y(self.y_test)

        self.rounding = int(rounding)
        self.discard_zero_mape = discard_zero_mape
        self.epsilon = epsilon
        zero_percentage_values = self.calculate_discarded_percentage()
        #print(f'Percentage of zero values in y_test: {zero_percentage_values:.2f}%')
        if zero_percentage_values > 0:
            self.calculate_mape_with_handling_zero_values = True

        else:
            self.calculate_mape_with_handling_zero_values = False
        self.prediction_correction = ( prediction_correction or PredictionCorrection(X_test, self.y_test, self.df_for_ML, rounding))

    def calculate_discarded_percentage(self):
        """
        Calculate the percentage of data points with zero values in y_test.

        Returns
        -------
        discarded_percentage : float
            The percentage of data points in y_test that are zero.
        """
        zero_mask = self.y_test == 0  # Identify where y_test is zero
        num_zeros = np.sum(zero_mask)  # Count zeros
        total_points = len(self.y_test)  # Total number of points
        discarded_percentage = (num_zeros / total_points) * \
            100  # Calculate percentage
        return discarded_percentage

    def load_model_from_path(self, model_path):
        """
        Load a model from the given path. 
        If .h5, load a Keras model.
        If .pkl, load a pickle-serialized model (e.g., scikit-learn, XGBoost).
        """
        if model_path.endswith('.h5'):
            # Keras model
            return load_model(model_path)
        elif model_path.endswith('.pkl'):
            # Pickled scikit-learn model
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(
                f"Unknown model file extension for {model_path}. Supported: .h5 (Keras), .pkl (pickle)")

    def reconstruct_y(self, y):
        """
        Reconstruct the y_test and y_pred values by adding the value column from X_test, in order to have the speed 
        instead of the delta-speed.
        """
        if self.target_is_gman_error_prediction:
            # In this case, y is the speed error prediction, and this is added to the speed prediction in order to get the speed (either actual or predicted via the second model's speed error prediction)
            return y + self.df_for_ML['gman_prediction_orig']
        else:
            return y + self.X_test['value']

    def get_predictions(self, model_path):

        model = self.load_model_from_path(model_path)
        if 'neural' in model_path.lower():
            print('Normalizing data because error is being calculated for ANN.')
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            y_pred = model.predict(X_test_normalized).flatten()
        else:
            #print('Calculating predictions...')
            y_pred = model.predict(self.X_test)

        if self.y_is_normalized:
            print(f'Mean of y_pred NORMALIZED is: {round(np.mean(y_pred),2)}')
            y_pred = y_pred * self.y_std + self.y_mean
            print(
                f'Mean of y_pred de-NORMALIZED is: {round(np.mean(y_pred),2)} kph of delta speed')
        self.y_pred_before_reconstruction = y_pred.copy()
        y_pred = self.reconstruct_y(y_pred)
        #print(f'Mean of y_pred AFTER RECONSTRUCTION is: {round(np.mean(y_pred),2)} kph of total speed')

        return y_pred, self.y_pred_before_reconstruction
    
    
    def evaluate_model_from_path_with_correction(
        self, 
        model_path, 
        apply_correction=True, 
        correction_method='naive_based_correction', 
        correction_kwargs={'naive_threshold': 0.7}, 
        print_results=True,
        return_preds = False
    ):
        """
        Evaluate a single model with options for applying prediction correction.

        Parameters:
        - model_path: str - Path to the model file (.pkl or .h5).
        - apply_correction: bool - Whether to apply prediction correction.
        - correction_method: str - Name of the method in PredictionCorrection class.
        - correction_kwargs: dict - Additional arguments for the correction method.
        - print_results: bool - Whether to print the evaluation results.

        Returns:
        - dict: Contains original and corrected prediction metrics and errors.
        """
        correction_kwargs = correction_kwargs or {}

        # Obtain original predictions (no correction applied yet)
        y_pred_original, _ = self.get_predictions(model_path)

        # Calculate metrics on original predictions
        original_results = self.calculate_metrics(y_pred_original, prefix='Original')

        corrected_results = None
        if apply_correction:
            # Dynamically get correction method from PredictionCorrection class
            correction_fn = getattr(self.prediction_correction, correction_method)
            
            # Apply prediction correction
            y_pred_corrected = correction_fn(y_pred_original.copy(), **correction_kwargs)
            
            # Calculate metrics on corrected predictions
            corrected_results = self.calculate_metrics(y_pred_corrected, prefix='Corrected')

        # Compile both original and corrected results into a structured dictionary
        results = {
            'original': original_results,
            'corrected': corrected_results
        }

        # Optionally print results clearly distinguishing original from corrected
        if print_results:
            print("\n=== Original (No Correction Applied) ===")
            self.print_evaluation_results(**original_results)
            
            if corrected_results:
                print("\n=== Corrected Predictions ===")
                self.print_evaluation_results(**corrected_results)
        
        if return_preds:
            return results, y_pred_original, y_pred_corrected

        return results
    
    def calculate_metrics(self, y_pred, prefix=''):
        """
        Calculate evaluation metrics and their standard deviations.

        Parameters:
        - y_pred: np.ndarray - Predicted values (already reconstructed).
        - prefix: str - Optional prefix to distinguish original and corrected metrics.

        Returns:
        - dict: Contains metrics, metrics_std, naive_metrics, naive_metrics_std.
        """
        # Calculate prediction errors
        errors = self.y_test - y_pred
        abs_errors = np.abs(errors)

        # Standard error metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        median_ae = median_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)

        # MAPE calculation with handling zeros if required
        if not self.calculate_mape_with_handling_zero_values:
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
        else:
            mape, _, _, _ = self.calculate_mape_in_case_of_zero_values(y_pred)

        # Compute standard deviations of errors
        mae_std = np.std(abs_errors)
        rmse_std = np.std(errors ** 2)
        mape_std = np.std(abs_errors / np.abs(self.y_test))

        # Naive metrics (assuming no speed change as naive prediction)
        naive_error = np.abs(self.y_test_before_reconstruction)
        naive_mae = np.mean(naive_error)
        naive_median_ae = np.median(naive_error)
        naive_rmse = np.sqrt(np.mean(naive_error ** 2))

        if not self.calculate_mape_with_handling_zero_values:
            naive_mape = np.mean(naive_error / np.abs(self.y_test))
        else:
            _, _, naive_mape, _ = self.calculate_mape_in_case_of_zero_values(y_pred)

        # Aggregating metrics with rounding
        metrics = {
            f'{prefix}_MAE': round(mae, self.rounding),
            f'{prefix}_Median_AE': round(median_ae, self.rounding),
            f'{prefix}_RMSE': round(rmse, self.rounding),
            f'{prefix}_MAPE': round(mape * 100, self.rounding),
        }

        naive_metrics = {
            f'{prefix}_Naive_MAE': round(naive_mae, self.rounding),
            f'{prefix}_Naive_Median_AE': round(naive_median_ae, self.rounding),
            f'{prefix}_Naive_RMSE': round(naive_rmse, self.rounding),
            f'{prefix}_Naive_MAPE': round(naive_mape * 100, self.rounding),
        }

        metrics_std = {
            f'{prefix}_MAE_std': round(mae_std, self.rounding),
            f'{prefix}_Median_AE_std': round(mae_std, self.rounding),
            f'{prefix}_RMSE_std': round(rmse_std, self.rounding),
            f'{prefix}_MAPE_std': round(mape_std * 100, self.rounding),
        }

        naive_metrics_std = {
            f'{prefix}_Naive_MAE_std': round(np.std(naive_error), self.rounding),
            f'{prefix}_Naive_Median_AE_std': round(np.std(naive_error), self.rounding),
            f'{prefix}_Naive_RMSE_std': round(np.std(naive_error ** 2), self.rounding),
            f'{prefix}_Naive_MAPE_std': round(np.std(naive_error / np.abs(self.y_test)) * 100, self.rounding),
        }

        return {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std
        }

    def evaluate_model_from_path(self, model_path, print_results=True):
        """
        Evaluate a single model (loaded from model_path) on test data and return four dictionaries:
        1. metrics: A dictionary of error metrics: MAE, Median Absolute Error, RMSE, MAPE.
        2. metrics_std: A dictionary of the standard deviations of the per-sample errors related to these metrics.
        3. naive_metrics: A dictionary of naive error metrics: Naive MAE, Naive Median AE, Naive RMSE, Naive MAPE.
        4. naive_metrics_std: A dictionary of the standard deviations of the naive errors.
        """

        # Get predictions (y_pred-->prediction of the total speed, y_pred_before_reconstruction --> prediction of the gman error)
        y_pred, y_pred_before_reconstruction = self.get_predictions(model_path)
        

        # Compute per-sample errors˝ (y_test is the actual total speed, y_pred is the predicted total speed)
        errors = self.y_test - y_pred
        abs_errors = np.abs(errors)

        # Model metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        #print(f"mae from numpy for {model_path} is: {np.mean(abs_errors)}")
        #print(f'mae std from numpy is {np.std(abs_errors)}')
        median_ae = median_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        if not self.calculate_mape_with_handling_zero_values:  # self.calculate_mape_with_handling_zero_values
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
            #print(f'mape from sklearn: {mape}')
            mape_manual = np.mean(abs_errors / self.y_test)
            #print(f'mape_manual: {mape_manual}')

        # Standard deviation calculations
        mae_std = np.std(abs_errors)
        # not sure if this is correct (rmse std should be same as mae std)
        rmse_std = np.std(errors**2)
        mape_std = np.std(abs_errors / np.abs(self.y_test))

        # Naive metrics
        # use y_test_before_reconstruction for naive (because y_test is originally the speed difference)
        if self.target_is_gman_error_prediction:
            naive_error = np.abs(self.df_for_ML['target_speed_delta'])
        else:
            naive_error = np.abs(self.y_test_before_reconstruction)
        naive_mae = np.mean(naive_error)
        naive_mae_std = np.std(naive_error)
        naive_median_ae = np.median(naive_error)
        naive_rmse = np.sqrt(np.mean(naive_error**2))
        # not sure if this is correct (rmse std should be same as mae std)
        naive_rmse_std = np.std(naive_error**2)
        if not self.calculate_mape_with_handling_zero_values:
            #print('Calculating mape without handling zero values')
            naive_mape = np.mean(naive_error / self.y_test)
            naive_mape_std = np.std(naive_error / self.y_test)
            naive_mape = np.mean(
                naive_error/np.abs((naive_error + self.X_test['value'])))
            naive_mape_std = np.std(
                naive_error/np.abs((naive_error + self.X_test['value'])))

        if self.calculate_mape_with_handling_zero_values:
            #print('Calculating mape with handling zero values')
            mape, mape_std, naive_mape, naive_mape_std = self.calculate_mape_in_case_of_zero_values(
                y_pred)

        # Collect all metrics
        metrics = {
            'MAE': mae,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'MAPE': mape*100
        }

        metrics_std = {
            'MAE_std': mae_std,
            'Median_AE_std': mae_std,  # Same as MAE (same distribution)
            # Doesn't make sense (std is of squared error)
            'RMSE_std': rmse_std,
            'MAPE_std': mape_std*100
        }

        naive_metrics = {
            'Naive_MAE': naive_mae,
            'Naive_Median_AE': naive_median_ae,
            'Naive_RMSE': naive_rmse,
            'Naive_MAPE': naive_mape*100
        }

        naive_metrics_std = {
            'Naive_MAE_std': naive_mae_std,
            'Naive_Median_AE_std': naive_mae_std,
            'Naive_RMSE_std': naive_rmse_std,
            'Naive_MAPE_std': naive_mape_std*100
        }

        # Apply rounding if specified
        if self.rounding is not None:
            metrics = {key: round(value, self.rounding)
                       for key, value in metrics.items()}
            metrics_std = {key: round(value, self.rounding)
                           for key, value in metrics_std.items()}
            naive_metrics = {key: round(value, self.rounding)
                             for key, value in naive_metrics.items()}
            naive_metrics_std = {key: round(
                value, self.rounding) for key, value in naive_metrics_std.items()}

        # If print_results is True, print the evaluation results starting with the naive metrics
        if print_results:
            self.print_evaluation_results(
                metrics, metrics_std, naive_metrics, naive_metrics_std)

        results = {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std
        }

        # return metrics, metrics_std, naive_metrics, naive_metrics_std
        return results

    def print_evaluation_results(self, metrics, metrics_std, naive_metrics, naive_metrics_std):
        """
        Print the evaluation results in a structured manner,
        starting with the naive metrics.
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
        Calculates mape in case of zero values in y_test (reconstructed so that it represents speed)
        """

        print('Calculating MAPE with handling of zero values in y_test.')
        # Handle MAPE calculation
        if self.discard_zero_mape:
            zero_mask = self.y_test == 0
            print(
                f"Discarding {np.sum(zero_mask)} zero values in y_test for MAPE calculation.")
            y_test_non_zero = self.y_test[~zero_mask]
            y_test_non_zero_before_reconstruction = self.y_test_before_reconstruction[
                ~zero_mask]
            X_test_non_zero = self.X_test[~zero_mask]
            y_pred_non_zero = y_pred[~zero_mask]
            ape = np.abs((y_test_non_zero - y_pred_non_zero) /
                         (y_test_non_zero))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_error_non_zero = np.abs(
                y_test_non_zero_before_reconstruction)
            naive_ape = np.abs(
                naive_error_non_zero / (y_test_non_zero_before_reconstruction + X_test_non_zero['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)
        else:
            y_test_safe = np.where(self.y_test == 0, self.epsilon, self.y_test)
            y_test_safe_before_reconstruction = np.where(
                self.y_test_before_reconstruction == 0, self.epsilon, self.y_test_before_reconstruction)
            ape = np.abs((y_test_safe - y_pred) / (y_test_safe))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_ape = np.abs((y_test_safe_before_reconstruction) /
                               (y_test_safe_before_reconstruction + self.X_test['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)

            return mape, mape_std, naive_mape, naive_mape_std

    def evaluate_all_models_from_paths(self, models_dict):
        """
        Evaluate multiple models given by their file paths on the test data and return four dictionaries:
        1. A dictionary of error metrics for each model.
        2. A dictionary of standard deviations for these metrics for each model.
        3. A dictionary of naive error metrics for each model.
        4. A dictionary of standard deviations for the naive metrics for each model.

        Parameters
        ----------
        models_dict : dict
            A dictionary where keys are model names and values are model file paths.

        Returns
        -------
        errors_all : dict
            errors_all[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_std : dict
            errors_all_std[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_naive : dict
            errors_all_naive[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        errors_all_naive_std : dict
            errors_all_naive_std[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        """
        errors_all = {}
        errors_all_std = {}
        errors_all_naive = {}
        errors_all_naive_std = {}

        for model_name, model_path in models_dict.items():
            # Evaluate the model
            metrics, metrics_std, naive_metrics, naive_metrics_std = self.evaluate_model_from_path(
                model_path)

            # Store results
            errors_all[model_name] = metrics
            errors_all_std[model_name] = metrics_std
            errors_all_naive[model_name] = naive_metrics
            errors_all_naive_std[model_name] = naive_metrics_std

        return errors_all, errors_all_std, errors_all_naive, errors_all_naive_std

class ModelEvaluator_deprecated:
    """
    A class to evaluate multiple models using saved model files (.h5 for Keras or .pkl for scikit-learn compatible models).
    It computes error metrics and their standard deviations for each model, with optional rounding.
    """

    def __init__(self, X_test, y_test, rounding=2, discard_zero_mape=False, epsilon=1e-2, y_test_is_normalized=False):
        """
        Initialize the ModelEvaluator with test data.

        Parameters
        ----------
        X_test : array-like
            Test feature set.
        y_test : array-like
            True values for the test set.
        rounding : int, optional
            Number of decimal places to round the results. If None, no rounding is applied.
        discard_zero_mape : bool, optional
            If True, discard zero values in y_test for MAPE calculation. Default is False.
        epsilon : float, optional
            Small value to replace zero values in y_test for MAPE calculation when discard_zero_mape is False. Default is 1e-10.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_is_normalized = y_test_is_normalized
        self.y_test_before_reconstruction = y_test.copy()
        self.y_test = self.reconstruct_y(self.y_test)
        print(
            f'Reconstrcuted errors of self.y_test, mean value is {np.mean(self.y_test)}')
        self.rounding = int(rounding)
        self.discard_zero_mape = discard_zero_mape
        self.epsilon = epsilon
        zero_percentage_values = self.calculate_discarded_percentage()
        print(
            f'Percentage of zero values in y_test: {zero_percentage_values:.2f}%')
        if zero_percentage_values > 0:
            self.calculate_mape_with_handling_zero_values = True

        else:
            self.calculate_mape_with_handling_zero_values = False

    def calculate_discarded_percentage(self):
        """
        Calculate the percentage of data points with zero values in y_test.

        Returns
        -------
        discarded_percentage : float
            The percentage of data points in y_test that are zero.
        """
        zero_mask = self.y_test == 0  # Identify where y_test is zero
        num_zeros = np.sum(zero_mask)  # Count zeros
        total_points = len(self.y_test)  # Total number of points
        discarded_percentage = (num_zeros / total_points) * \
            100  # Calculate percentage
        return discarded_percentage

    def load_model_from_path(self, model_path):
        """
        Load a model from the given path. 
        If .h5, load a Keras model.
        If .pkl, load a pickle-serialized model (e.g., scikit-learn, XGBoost).
        """
        if model_path.endswith('.h5'):
            # Keras model
            return load_model(model_path)
        elif model_path.endswith('.pkl'):
            # Pickled scikit-learn model
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(
                f"Unknown model file extension for {model_path}. Supported: .h5 (Keras), .pkl (pickle)")

    def reconstruct_y(self, y):
        """
        Reconstruct the y_test and y_pred values by adding the value column from X_test, in order to have the speed 
        instead of the delta-speed.
        """
        return y + self.X_test['value']

    def get_predictions(self, model_path):

        model = self.load_model_from_path(model_path)
        if 'neural' in model_path.lower():
            print('Normalizing data because error is being calculated for ANN.')
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            y_pred = model.predict(X_test_normalized).flatten()
        else:
            y_pred = model.predict(self.X_test)

        y_pred_before_reconstruction = y_pred.copy()
        y_pred = self.reconstruct_y(y_pred)
        print(f'Mean of y_pred is: {np.mean(round(y_pred,2))}')

        return y_pred, y_pred_before_reconstruction

    def evaluate_model_from_path(self, model_path,return_preds = False,print_results=True):
        """
        Evaluate a single model (loaded from model_path) on test data and return four dictionaries:
        1. metrics: A dictionary of error metrics: MAE, Median Absolute Error, RMSE, MAPE.
        2. metrics_std: A dictionary of the standard deviations of the per-sample errors related to these metrics.
        3. naive_metrics: A dictionary of naive error metrics: Naive MAE, Naive Median AE, Naive RMSE, Naive MAPE.
        4. naive_metrics_std: A dictionary of the standard deviations of the naive errors.
        """

        # Get predictions
        y_pred, y_pred_before_reconstruction = self.get_predictions(model_path)

        # Compute per-sample errors˝
        errors = self.y_test - y_pred
        abs_errors = np.abs(errors)

        # Model metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        print(f"mae from numpy for {model_path} is: {np.mean(abs_errors)}")
        print(f'mae std from numpy is {np.std(abs_errors)}')
        median_ae = median_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        if not self.calculate_mape_with_handling_zero_values:  # self.calculate_mape_with_handling_zero_values
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
            print(f'mape from sklearn: {mape}')
            mape_manual = np.mean(abs_errors / self.y_test)
            print(f'mape_manual: {mape_manual}')

        # Standard deviation calculations
        mae_std = np.std(abs_errors)
        # not sure if this is correct (rmse std should be same as mae std)
        rmse_std = np.std(errors**2)
        mape_std = np.std(abs_errors / np.abs(self.y_test))

        # Naive metrics
        # use y_test_before_reconstruction for naive (because y_test is originally the speed difference)
        naive_error = np.abs(self.y_test_before_reconstruction)
        naive_mae = np.mean(naive_error)
        naive_mae_std = np.std(naive_error)
        naive_median_ae = np.median(naive_error)
        naive_rmse = np.sqrt(np.mean(naive_error**2))
        # not sure if this is correct (rmse std should be same as mae std)
        naive_rmse_std = np.std(naive_error**2)
        if not self.calculate_mape_with_handling_zero_values:
            print('Calculating mape without handling zero values')
            naive_mape = np.mean(naive_error / self.y_test)
            naive_mape_std = np.std(naive_error / self.y_test)
            naive_mape = np.mean(
                naive_error/np.abs((naive_error + self.X_test['value'])))
            naive_mape_std = np.std(
                naive_error/np.abs((naive_error + self.X_test['value'])))

        if self.calculate_mape_with_handling_zero_values:
            print('Calculating mape with handling zero values')
            mape, mape_std, naive_mape, naive_mape_std = self.calculate_mape_in_case_of_zero_values(
                y_pred)

        # Collect all metrics
        metrics = {
            'MAE': mae,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'MAPE': mape*100
        }

        metrics_std = {
            'MAE_std': mae_std,
            'Median_AE_std': mae_std,  # Same as MAE (same distribution)
            # Doesn't make sense (std is of squared error)
            'RMSE_std': rmse_std,
            'MAPE_std': mape_std*100
        }

        naive_metrics = {
            'Naive_MAE': naive_mae,
            'Naive_Median_AE': naive_median_ae,
            'Naive_RMSE': naive_rmse,
            'Naive_MAPE': naive_mape*100
        }

        naive_metrics_std = {
            'Naive_MAE_std': naive_mae_std,
            'Naive_Median_AE_std': naive_mae_std,
            'Naive_RMSE_std': naive_rmse_std,
            'Naive_MAPE_std': naive_mape_std*100
        }

        # Apply rounding if specified
        if self.rounding is not None:
            metrics = {key: round(value, self.rounding)
                       for key, value in metrics.items()}
            metrics_std = {key: round(value, self.rounding)
                           for key, value in metrics_std.items()}
            naive_metrics = {key: round(value, self.rounding)
                             for key, value in naive_metrics.items()}
            naive_metrics_std = {key: round(
                value, self.rounding) for key, value in naive_metrics_std.items()}

        # If print_results is True, print the evaluation results starting with the naive metrics
        if print_results:
            self.print_evaluation_results(
                metrics, metrics_std, naive_metrics, naive_metrics_std)

        results = {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std
        }

        # return metrics, metrics_std, naive_metrics, naive_metrics_std
        
        if return_preds:
            return y_pred,results
        return results

    def print_evaluation_results(self, metrics, metrics_std, naive_metrics, naive_metrics_std):
        """
        Print the evaluation results in a structured manner,
        starting with the naive metrics.
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
        Calculates mape in case of zero values in y_test (reconstructed so that it represents speed)
        """

        print('Calculating MAPE with handling of zero values in y_test.')
        # Handle MAPE calculation
        if self.discard_zero_mape:
            zero_mask = self.y_test == 0
            print(
                f"Discarding {np.sum(zero_mask)} zero values in y_test for MAPE calculation.")
            y_test_non_zero = self.y_test[~zero_mask]
            y_test_non_zero_before_reconstruction = self.y_test_before_reconstruction[
                ~zero_mask]
            X_test_non_zero = self.X_test[~zero_mask]
            y_pred_non_zero = y_pred[~zero_mask]
            ape = np.abs((y_test_non_zero - y_pred_non_zero) /
                         (y_test_non_zero))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_error_non_zero = np.abs(
                y_test_non_zero_before_reconstruction)
            naive_ape = np.abs(
                naive_error_non_zero / (y_test_non_zero_before_reconstruction + X_test_non_zero['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)
        else:
            y_test_safe = np.where(self.y_test == 0, self.epsilon, self.y_test)
            y_test_safe_before_reconstruction = np.where(
                self.y_test_before_reconstruction == 0, self.epsilon, self.y_test_before_reconstruction)
            ape = np.abs((y_test_safe - y_pred) / (y_test_safe))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_ape = np.abs((y_test_safe_before_reconstruction) /
                               (y_test_safe_before_reconstruction + self.X_test['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)

            return mape, mape_std, naive_mape, naive_mape_std

    def evaluate_all_models_from_paths(self, models_dict):
        """
        Evaluate multiple models given by their file paths on the test data and return four dictionaries:
        1. A dictionary of error metrics for each model.
        2. A dictionary of standard deviations for these metrics for each model.
        3. A dictionary of naive error metrics for each model.
        4. A dictionary of standard deviations for the naive metrics for each model.

        Parameters
        ----------
        models_dict : dict
            A dictionary where keys are model names and values are model file paths.

        Returns
        -------
        errors_all : dict
            errors_all[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_std : dict
            errors_all_std[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_naive : dict
            errors_all_naive[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        errors_all_naive_std : dict
            errors_all_naive_std[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        """
        errors_all = {}
        errors_all_std = {}
        errors_all_naive = {}
        errors_all_naive_std = {}

        for model_name, model_path in models_dict.items():
            # Evaluate the model
            metrics, metrics_std, naive_metrics, naive_metrics_std = self.evaluate_model_from_path(
                model_path)

            # Store results
            errors_all[model_name] = metrics
            errors_all_std[model_name] = metrics_std
            errors_all_naive[model_name] = naive_metrics
            errors_all_naive_std[model_name] = naive_metrics_std

        return errors_all, errors_all_std, errors_all_naive, errors_all_naive_std


class ModelEvaluatorGMAN_deprecated:
    """
    A class to evaluate multiple models using saved model files (.h5 for Keras or .pkl for scikit-learn compatible models).
    It computes error metrics and their standard deviations for each model, with optional rounding.
    """

    def __init__(self, X_test, y_test, df_for_ML, rounding=2, discard_zero_mape=False, epsilon=1e-2):
        """
        Initialize the ModelEvaluator with test data.

        Parameters
        ----------
        X_test : array-like
            Test feature set.
        y_test : array-like
            True values for the test set.
        rounding : int, optional
            Number of decimal places to round the results. If None, no rounding is applied.
        discard_zero_mape : bool, optional
            If True, discard zero values in y_test for MAPE calculation. Default is False.
        epsilon : float, optional
            Small value to replace zero values in y_test for MAPE calculation when discard_zero_mape is False. Default is 1e-10.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.df_for_ML = df_for_ML[df_for_ML['test_set']]
        self.y_test_before_reconstruction = y_test.copy()
        self.y_test = self.reconstruct_y(self.y_test)
        print(
            f'Reconstrcuted errors of self.y_test, mean value is {np.mean(self.y_test)}')
        self.rounding = int(rounding)
        self.discard_zero_mape = discard_zero_mape
        self.epsilon = epsilon
        zero_percentage_values = self.calculate_discarded_percentage()
        print(
            f'Percentage of zero values in y_test: {zero_percentage_values:.2f}%')
        if zero_percentage_values > 0:
            self.calculate_mape_with_handling_zero_values = True

        else:
            self.calculate_mape_with_handling_zero_values = False

    def calculate_discarded_percentage(self):
        """
        Calculate the percentage of data points with zero values in y_test.

        Returns
        -------
        discarded_percentage : float
            The percentage of data points in y_test that are zero.
        """
        zero_mask = self.y_test == 0  # Identify where y_test is zero
        num_zeros = np.sum(zero_mask)  # Count zeros
        total_points = len(self.y_test)  # Total number of points
        discarded_percentage = (num_zeros / total_points) * \
            100  # Calculate percentage
        return discarded_percentage

    def load_model_from_path(self, model_path):
        """
        Load a model from the given path. 
        If .h5, load a Keras model.
        If .pkl, load a pickle-serialized model (e.g., scikit-learn, XGBoost).
        """
        if model_path.endswith('.h5'):
            # Keras model
            return load_model(model_path)
        elif model_path.endswith('.pkl'):
            # Pickled scikit-learn model
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(
                f"Unknown model file extension for {model_path}. Supported: .h5 (Keras), .pkl (pickle)")

    def reconstruct_y(self, y):
        """
        Reconstruct the y_test and y_pred values by adding the value column from X_test, in order to have the speed 
        instead of the delta-speed.
        """
        return y + self.df_for_ML['target_gman_prediction']

    def get_predictions(self, model_path):

        model = self.load_model_from_path(model_path)
        if 'neural' in model_path.lower():
            print('Normalizing data because error is being calculated for ANN.')
            _, X_test_normalized = normalize_data(X_test=self.X_test)
            y_pred = model.predict(X_test_normalized).flatten()
        else:
            y_pred = model.predict(self.X_test)

        y_pred_before_reconstruction = y_pred.copy()
        y_pred = self.reconstruct_y(y_pred)
        print(f'Mean of y_pred is: {np.mean(round(y_pred,2))}')

        return y_pred, y_pred_before_reconstruction

    def evaluate_model_from_path(self, model_path, print_results=True):
        """
        Evaluate a single model (loaded from model_path) on test data and return four dictionaries:
        1. metrics: A dictionary of error metrics: MAE, Median Absolute Error, RMSE, MAPE.
        2. metrics_std: A dictionary of the standard deviations of the per-sample errors related to these metrics.
        3. naive_metrics: A dictionary of naive error metrics: Naive MAE, Naive Median AE, Naive RMSE, Naive MAPE.
        4. naive_metrics_std: A dictionary of the standard deviations of the naive errors.
        """

        # Get predictions (y_pred-->prediction of the total speed, y_pred_before_reconstruction --> prediction of the gman error)
        y_pred, y_pred_before_reconstruction = self.get_predictions(model_path)

        # Compute per-sample errors˝ (y_test is the actual total speed, y_pred is the predicted total speed)
        errors = self.y_test - y_pred
        abs_errors = np.abs(errors)

        # Model metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        print(f"mae from numpy for {model_path} is: {np.mean(abs_errors)}")
        print(f'mae std from numpy is {np.std(abs_errors)}')
        median_ae = median_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        if not self.calculate_mape_with_handling_zero_values:  # self.calculate_mape_with_handling_zero_values
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
            print(f'mape from sklearn: {mape}')
            mape_manual = np.mean(abs_errors / self.y_test)
            print(f'mape_manual: {mape_manual}')

        # Standard deviation calculations
        mae_std = np.std(abs_errors)
        # not sure if this is correct (rmse std should be same as mae std)
        rmse_std = np.std(errors**2)
        mape_std = np.std(abs_errors / np.abs(self.y_test))

        # Naive metrics
        # use y_test_before_reconstruction for naive (because y_test is originally the speed difference)
        naive_error = np.abs(self.df_for_ML['target_speed_delta'])
        naive_mae = np.mean(naive_error)
        naive_mae_std = np.std(naive_error)
        naive_median_ae = np.median(naive_error)
        naive_rmse = np.sqrt(np.mean(naive_error**2))
        # not sure if this is correct (rmse std should be same as mae std)
        naive_rmse_std = np.std(naive_error**2)
        if not self.calculate_mape_with_handling_zero_values:
            print('Calculating mape without handling zero values')
            naive_mape = np.mean(naive_error / self.y_test)
            naive_mape_std = np.std(naive_error / self.y_test)
            naive_mape = np.mean(
                naive_error/np.abs((naive_error + self.X_test['value'])))
            naive_mape_std = np.std(
                naive_error/np.abs((naive_error + self.X_test['value'])))

        if self.calculate_mape_with_handling_zero_values:
            print('Calculating mape with handling zero values')
            mape, mape_std, naive_mape, naive_mape_std = self.calculate_mape_in_case_of_zero_values(
                y_pred)

        # Collect all metrics
        metrics = {
            'MAE': mae,
            'Median_AE': median_ae,
            'RMSE': rmse,
            'MAPE': mape*100
        }

        metrics_std = {
            'MAE_std': mae_std,
            'Median_AE_std': mae_std,  # Same as MAE (same distribution)
            # Doesn't make sense (std is of squared error)
            'RMSE_std': rmse_std,
            'MAPE_std': mape_std*100
        }

        naive_metrics = {
            'Naive_MAE': naive_mae,
            'Naive_Median_AE': naive_median_ae,
            'Naive_RMSE': naive_rmse,
            'Naive_MAPE': naive_mape*100
        }

        naive_metrics_std = {
            'Naive_MAE_std': naive_mae_std,
            'Naive_Median_AE_std': naive_mae_std,
            'Naive_RMSE_std': naive_rmse_std,
            'Naive_MAPE_std': naive_mape_std*100
        }

        # Apply rounding if specified
        if self.rounding is not None:
            metrics = {key: round(value, self.rounding)
                       for key, value in metrics.items()}
            metrics_std = {key: round(value, self.rounding)
                           for key, value in metrics_std.items()}
            naive_metrics = {key: round(value, self.rounding)
                             for key, value in naive_metrics.items()}
            naive_metrics_std = {key: round(
                value, self.rounding) for key, value in naive_metrics_std.items()}

        # If print_results is True, print the evaluation results starting with the naive metrics
        if print_results:
            self.print_evaluation_results(
                metrics, metrics_std, naive_metrics, naive_metrics_std)

        results = {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "naive_metrics": naive_metrics,
            "naive_metrics_std": naive_metrics_std
        }

        # return metrics, metrics_std, naive_metrics, naive_metrics_std
        return results

    def print_evaluation_results(self, metrics, metrics_std, naive_metrics, naive_metrics_std):
        """
        Print the evaluation results in a structured manner,
        starting with the naive metrics.
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
        Calculates mape in case of zero values in y_test (reconstructed so that it represents speed)
        """

        print('Calculating MAPE with handling of zero values in y_test.')
        # Handle MAPE calculation
        if self.discard_zero_mape:
            zero_mask = self.y_test == 0
            print(
                f"Discarding {np.sum(zero_mask)} zero values in y_test for MAPE calculation.")
            y_test_non_zero = self.y_test[~zero_mask]
            y_test_non_zero_before_reconstruction = self.y_test_before_reconstruction[
                ~zero_mask]
            X_test_non_zero = self.X_test[~zero_mask]
            y_pred_non_zero = y_pred[~zero_mask]
            ape = np.abs((y_test_non_zero - y_pred_non_zero) /
                         (y_test_non_zero))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_error_non_zero = np.abs(
                y_test_non_zero_before_reconstruction)
            naive_ape = np.abs(
                naive_error_non_zero / (y_test_non_zero_before_reconstruction + X_test_non_zero['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)
        else:
            y_test_safe = np.where(self.y_test == 0, self.epsilon, self.y_test)
            y_test_safe_before_reconstruction = np.where(
                self.y_test_before_reconstruction == 0, self.epsilon, self.y_test_before_reconstruction)
            ape = np.abs((y_test_safe - y_pred) / (y_test_safe))
            mape = np.mean(ape)
            mape_std = np.std(ape)

            naive_ape = np.abs((y_test_safe_before_reconstruction) /
                               (y_test_safe_before_reconstruction + self.X_test['value']))
            naive_mape = np.mean(naive_ape)
            naive_mape_std = np.std(naive_ape)

            return mape, mape_std, naive_mape, naive_mape_std

    def evaluate_all_models_from_paths(self, models_dict):
        """
        Evaluate multiple models given by their file paths on the test data and return four dictionaries:
        1. A dictionary of error metrics for each model.
        2. A dictionary of standard deviations for these metrics for each model.
        3. A dictionary of naive error metrics for each model.
        4. A dictionary of standard deviations for the naive metrics for each model.

        Parameters
        ----------
        models_dict : dict
            A dictionary where keys are model names and values are model file paths.

        Returns
        -------
        errors_all : dict
            errors_all[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_std : dict
            errors_all_std[model_name] = { 'MAE': ..., 'Median Absolute Error': ..., 'RMSE': ..., 'MAPE': ... }
        errors_all_naive : dict
            errors_all_naive[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        errors_all_naive_std : dict
            errors_all_naive_std[model_name] = { 'Naive MAE': ..., 'Naive Median AE': ..., 'Naive RMSE': ..., 'Naive MAPE': ... }
        """
        errors_all = {}
        errors_all_std = {}
        errors_all_naive = {}
        errors_all_naive_std = {}

        for model_name, model_path in models_dict.items():
            # Evaluate the model
            metrics, metrics_std, naive_metrics, naive_metrics_std = self.evaluate_model_from_path(
                model_path)

            # Store results
            errors_all[model_name] = metrics
            errors_all_std[model_name] = metrics_std
            errors_all_naive[model_name] = naive_metrics
            errors_all_naive_std[model_name] = naive_metrics_std

        return errors_all, errors_all_std, errors_all_naive, errors_all_naive_std




class ModelComparisons:
    """
    A class to compare model performance using various error metrics and visualization techniques.
    This includes loading pre-trained models, calculating errors on test data, and displaying
    comparison plots of model performance.
    """

    def __init__(self, data_path='../data', models_path='../models', xgb_filename='best_xgboost_model.pkl', rf_filename='best_random_forest_model.pkl',
                 ann_filename='best_neural_network_model.h5', data_file_name='estimated_average_speed_selected_timestamps-edited-new.parquet',
                 load_xgb=True, load_ann=False, load_rf=False,
                 random_state=69):
        """
        Initializes ModelComparisons with model file paths, data file path, and sample size for evaluation.

        Parameters:
        - xgb_filename (str): File path to the saved XGBoost model.
        - rf_filename (str): File path to the saved Random Forest model.
        - ann_filename (str): File path to the saved Neural Network model.
        - data_file_path (str): Path to the CSV file for test data preparation.
        - random_state (int): Seed for reproducibility in random sampling.
        """
        # Model file paths
        self.model_filenames = {
            'XGBoost': os.path.join(models_path, xgb_filename),
            'Random Forest': os.path.join(models_path, rf_filename),
            'Neural Network': os.path.join(models_path, ann_filename)
        }
        self.data_path = data_path
        self.data_file_name = data_file_name
        self.random_state = random_state

        # Whether to load each model
        self.load_xgb = load_xgb
        self.load_ann = load_ann
        self.load_rf = load_rf
        # Dictionaries to store loaded models and error metrics
        self.models = {}
        self.errors = {}

        # Colors for plotting each model's results
        self.model_colors = {
            'XGBoost': '#6495ED',      # Cornflower Blue
            'Random Forest': '#FFA07A',  # Light Salmon
            'Neural Network': '#90EE90'  # Light Green
        }

        # Colors for plotting different horizons
        self.horizon_colors = {
            15: '#6495ED',      # Cornflower Blue
            30: '#FFA07A',      # Light Salmon
            60: '#90EE90'       # Light Green
        }

        # Test data (initialized later)
        self.X_test = None
        self.y_test = None
        self.X_test_normalized = None

    def load_models(self):
        """Loads each model from the specified file paths. Issues warnings if any files are missing."""

        # Define a mapping between model names and the flags
        model_load_flags = {
            'XGBoost': self.load_xgb,
            'Random Forest': self.load_rf,
            'Neural Network': self.load_ann
        }
        for model_name, filename in self.model_filenames.items():
            # Check if the model should be loaded based on its flag
            if not model_load_flags.get(model_name, False):
                print(
                    f"Skipping {model_name} as its loading flag is set to False.")
                continue
            try:
                # Load model based on file type (.pkl for scikit-learn, .h5 for Keras)
                if filename.endswith('.pkl'):
                    with open(filename, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                elif filename.endswith('.h5'):
                    self.models[model_name] = load_model(filename)
                print(f"{model_name} model loaded successfully.")
            except (FileNotFoundError, OSError):
                warnings.warn(
                    f"File '{filename}' not found. Skipping {model_name} model.")

    def prepare_test_data(self, horizon=15, test_size=0.3):
        """
        Prepares test data for evaluation by loading, processing, and sampling data from the specified file.
        Assumes data processing class/methods exist for loading and preparing the data.
        """
        # Here, you would call your data processing pipeline to load and prepare test data
        # This example assumes a method or class exists to handle data loading and returns test splits
        data_processor = TrafficFlowDataProcessing(data_path=self.data_path,
                                                   file_name=self.data_file_name, random_state=self.random_state,
                                                   )
        X_train, X_test, y_train, y_test = data_processor.get_clean_train_test_split(test_size=test_size, horizon=horizon, add_train_test_flag=True,
                                                                                     add_spatial_lags=False, reset_index=True, use_weekend_var=False)

        # Store prepared test data
        self.X_train = X_train
        self.X_test = X_test
        # self.y_test = y_test
        self.y_test = y_test + self.X_test['value'].values
        self.X_train_normalized, self.X_test_normalized = normalize_data(
            X_train, X_test)

    def calculate_errors(self):
        """
        Calculates various error metrics for each loaded model using the sampled test data.
        Metrics calculated: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and
        Mean Absolute Percentage Error (MAPE).
        """
        if not self.models:
            print("No models loaded. Cannot calculate errors.")
            return

        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            print(
                "Test data is not prepared. Please run prepare_test_data() before calculating errors.")
            return

        # Loop over each model and calculate error metrics
        for model_name, model in self.models.items():
            # Generate predictions for the test set
            if 'Neural' in model_name:
                print('Normalizing X_test because error is being calculated for ANN.')
                y_pred = model.predict(self.X_test_normalized).flatten()
            else:
                y_pred = model.predict(self.X_test).flatten()

            # Adjust predictions (speed delta) to actual speeds if needed
            y_pred += self.X_test['value'].values

            # Calculate error metrics and store them
            mae = mean_absolute_error(self.y_test, y_pred)
            median_ae = median_absolute_error(self.y_test, y_pred)
            rmse = mean_squared_error(self.y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(self.y_test, y_pred)

            # Store calculated errors
            self.errors[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            print(
                f"Calculated errors for {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")

            self.errors_all[model_name] = {
                'MAE': mae,
                'Median Absolute Error': median_ae,
                'RMSE': rmse,
                'MAPE': mape
            }

    def _use_before_plotting(self):
        """
        Runs all functions required before generating the plots.
        Used at the start of each plotting function.
        """
        self.load_models()
        self.prepare_test_data()
        self.calculate_errors()

    def plot_error_metrics(self):
        """
        Generates a bar plot comparing MAE, RMSE, and MAPE across all loaded models.
        Displays error values on each bar for clarity.
        """
        self._use_before_plotting()

        if not self.errors:
            print("No error metrics calculated. Run calculate_errors() before plotting.")
            return

        # Define metrics to plot
        metrics = ['MAE', 'RMSE', 'MAPE']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot each metric in a separate subplot
        for i, metric in enumerate(metrics):
            metric_values = [self.errors[model][metric]
                             for model in self.errors]
            model_names = list(self.errors.keys())
            # Default: Light Gray
            colors = [self.model_colors.get(model, '#D3D3D3')
                      for model in model_names]

            # Create bar plot for each metric
            bars = axes[i].bar(model_names, metric_values,
                               color=colors, edgecolor='black', linewidth=0.7)
            axes[i].set_title(f"{metric} Comparison", fontsize=14)
            axes[i].set_ylabel(metric)
            axes[i].yaxis.grid(True, linestyle='--',
                               linewidth=0.5, color='gray', alpha=0.7)

            # Add text labels on bars
            for bar, value in zip(bars, metric_values):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(metric_values),
                             f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self):
        """
        Creates scatter plots of actual vs. predicted speed values for each model.
        Displays a reference diagonal line (y=x) to show perfect predictions.
        """

        self._use_before_plotting()

        if not self.models:
            print(
                "No models loaded. Please load models before plotting actual vs. predicted values.")
            return

        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            print(
                "Test data is not prepared. Please run prepare_test_data() before plotting.")
            return

        # Create a scatter plot for each model's predictions
        fig, axes = plt.subplots(1, len(self.models), figsize=(15, 5))
        if len(self.models) == 1:
            axes = [axes]  # Ensure axes is a list even if only one model

        # Loop over each model to create a scatter plot
        for ax, (model_name, model) in zip(axes, self.models.items()):
            # Generate predictions and adjust to actual speed values
            print(f"MODEL NAME IS: {model_name} AND MODEL IS: {model}")
            if 'Neural' in model_name:
                y_pred = model.predict(self.X_test_normalized).flatten()
            else:
                y_pred = model.predict(self.X_test).flatten()
            # Adjust deltas to actual speeds
            y_pred += self.X_test['value'].values

            # Scatter plot of actual vs predicted values
            ax.scatter(self.y_test, y_pred, alpha=0.1,
                       color=self.model_colors.get(model_name, '#D3D3D3'))
            ax.set_title(f"{model_name} - Actual vs Predicted", fontsize=14)
            ax.set_xlabel("Actual Speed")
            ax.set_ylabel("Predicted Speed")

            # Add a diagonal reference line (y=x) for perfect predictions
            max_val = max(max(self.y_test), max(y_pred))
            min_val = min(min(self.y_test), min(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.tight_layout()
        plt.show()

    def plot_model_error_details(self, model_name):
        """
        Creates a 2x2 grid plot showing MAE, Median Absolute Error, RMSE, and MAPE
        for a specific model.

        Parameters:
        - model_name (str): The name of the model to plot error details for.
        """

        self._use_before_plotting()

        if model_name not in self.errors:
            print(f"No error data available for model '{model_name}'.")
            return

        # Define metrics to plot
        metrics = ['MAE', 'Median Absolute Error', 'RMSE', 'MAPE']
        metric_values = [
            self.errors_all[model_name]['MAE'],
            self.errors[model_name].get('Median Absolute Error', None),
            self.errors[model_name]['RMSE'],
            self.errors[model_name]['MAPE']
        ]

        # If Median Absolute Error is not calculated, assign a placeholder value
        if metric_values[1] is None:
            warnings.warn(
                f"Median Absolute Error is not available for model '{model_name}'.")
            metric_values[1] = 0

        # Create 2x2 grid plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 6))
        axes = axes.flatten()
        colors = self.model_colors.get(
            model_name, '#D3D3D3')  # Default: Light Gray

        titles = [
            "Mean Absolute Error (kph)",
            "Median Absolute Error (kph)",
            "Root Mean Squared Error (kph)",
            "Mean Absolute Percentage Error (%)"
        ]

        for ax, metric, value, title in zip(axes, metrics, metric_values, titles):
            # Bar plot for each metric
            ax.bar([model_name], [value], color=colors,
                   edgecolor='black', linewidth=0.7)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Error")
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5,
                          color='gray', alpha=0.7)

            # Add text label on the bar
            ax.text(0, value + 0.01 * max(metric_values),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()
