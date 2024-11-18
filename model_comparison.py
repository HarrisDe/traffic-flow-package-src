import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import numpy as np
from keras.models import load_model
import matplotlib.patheffects as PathEffects
from .data_processing import TrafficFlowDataProcessing
from .helper_functions import normalize_data
import seaborn as sns
sns.set_style('darkgrid')


class ModelComparisons:
    """
    A class to compare model performance using various error metrics and visualization techniques.
    This includes loading pre-trained models, calculating errors on test data, and displaying
    comparison plots of model performance.
    """

    def __init__(self, xgb_filename='../models/best_xgboost_model.pkl', rf_filename='../models/best_random_forest_model.pkl', 
                 ann_filename='../models/best_neural_network_model.h5', data_file_path='../data/estimated_average_speed_selected_timestamps-edited-new.csv', 
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
            'XGBoost': xgb_filename,
            'Random Forest': rf_filename,
            'Neural Network': ann_filename
        }
        self.data_file_path = data_file_path
        self.random_state = random_state

        # Dictionaries to store loaded models and error metrics
        self.models = {}
        self.errors = {}

        # Colors for plotting each model's results
        self.model_colors = {
            'XGBoost': '#6495ED',      # Cornflower Blue
            'Random Forest': '#FFA07A', # Light Salmon
            'Neural Network': '#90EE90' # Light Green
        }

           # Test data (initialized later)
        self.X_test = None
        self.y_test = None
        self.X_test_normalized = None

    def load_models(self):
        """Loads each model from the specified file paths. Issues warnings if any files are missing."""
        for model_name, filename in self.model_filenames.items():
            try:
                # Load model based on file type (.pkl for scikit-learn, .h5 for Keras)
                if filename.endswith('.pkl'):
                    with open(filename, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                elif filename.endswith('.h5'):
                    self.models[model_name] = load_model(filename)
                print(f"{model_name} model loaded successfully.")
            except (FileNotFoundError, OSError):
                warnings.warn(f"File '{filename}' not found. Skipping {model_name} model.")

    def prepare_test_data(self):
        """
        Prepares test data for evaluation by loading, processing, and sampling data from the specified file.
        Assumes data processing class/methods exist for loading and preparing the data.
        """
        # Here, you would call your data processing pipeline to load and prepare test data
        # This example assumes a method or class exists to handle data loading and returns test splits
        data_processor = TrafficFlowDataProcessing(file_path=self.data_file_path, random_state=self.random_state)
        X_train, X_test, y_train, y_test = data_processor.prepare_data()


        # Store prepared test data
        self.X_train = X_train
        self.X_test = X_test
        #self.y_test = y_test
        self.y_test = y_test + self.X_test['value'].values
        self.X_train_normalized, self.X_test_normalized = normalize_data(X_train,X_test)

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
            print("Test data is not prepared. Please run prepare_test_data() before calculating errors.")
            return

        # Loop over each model and calculate error metrics
        for model_name, model in self.models.items():
            # Generate predictions for the test set
            if 'Neural' in model_name:
                print('Normalizing X_test because error is being calculated for ANN.')
                y_pred = model.predict(self.X_test_normalized).flatten()
            else:
                y_pred = model.predict(self.X_test).flatten()
            
            y_pred += self.X_test['value'].values  # Adjust predictions (speed delta) to actual speeds if needed

            # Calculate error metrics and store them
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = root_mean_squared_error(self.y_test, y_pred)
            mape = mean_absolute_percentage_error(self.y_test, y_pred)

            # Store calculated errors
            self.errors[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            print(f"Calculated errors for {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")

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
            metric_values = [self.errors[model][metric] for model in self.errors]
            model_names = list(self.errors.keys())
            colors = [self.model_colors.get(model, '#D3D3D3') for model in model_names]  # Default: Light Gray

            # Create bar plot for each metric
            bars = axes[i].bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=0.7)
            axes[i].set_title(f"{metric} Comparison", fontsize=14)
            axes[i].set_ylabel(metric)
            axes[i].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

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
            print("No models loaded. Please load models before plotting actual vs. predicted values.")
            return

        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            print("Test data is not prepared. Please run prepare_test_data() before plotting.")
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
            y_pred += self.X_test['value'].values  # Adjust deltas to actual speeds

            # Scatter plot of actual vs predicted values
            ax.scatter(self.y_test, y_pred, alpha=0.1, color=self.model_colors.get(model_name, '#D3D3D3'))
            ax.set_title(f"{model_name} - Actual vs Predicted", fontsize=14)
            ax.set_xlabel("Actual Speed")
            ax.set_ylabel("Predicted Speed")

            # Add a diagonal reference line (y=x) for perfect predictions
            max_val = max(max(self.y_test), max(y_pred))
            min_val = min(min(self.y_test), min(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.tight_layout()
        plt.show()
