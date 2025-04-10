import os
import sys
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from .helper_utils import normalize_data
import pickle
import warnings
import numpy as np


class ModelTuner:
    """
    A class to perform hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
    using either TimeSeriesSplit or standard K-Fold cross-validation.
    """

    def __init__(self, X_train, X_test, y_train, y_test, random_state=69, use_ts_split=True, n_splits=3, use_min_max_norm=False,
                 best_model_name_string_start='best_model_', XGBoost_model_name=None, Random_Forest_model_name=None, ann_model_name=None):
        """
        Initializes ModelTuner with training and test data splits.

        Parameters:
        - X_train, X_test, y_train, y_test: Training and testing data splits.
        - random_state (int): Random seed for reproducibility.
        - use_ts_split (bool): If True, use TimeSeriesSplit; if False, use standard cross-validation.
        - n_splits (int): Number of splits for TimeSeriesSplit or KFold.
        - use_min_max_norm (bool): If True, applies MinMaxScaler for normalization in ANN; otherwise, uses StandardScaler.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_models = {}
        self.random_state = random_state
        self.use_ts_split = use_ts_split
        self.n_splits = n_splits
        self.use_min_max_norm = use_min_max_norm
        self.scaler = None  # Will be initialized when tuning ANN
        # the start of the name of the best model
        self.best_model_name_string_start = best_model_name_string_start
        # Assign model names if not provided
        self.XGBoost_model_name = XGBoost_model_name if XGBoost_model_name is not None else 'XGBoost'
        self.Random_Forest_model_name = Random_Forest_model_name if Random_Forest_model_name is not None else 'Random_Forest'
        self.ann_model_name = ann_model_name if ann_model_name is not None else 'Neural_Network'
        # Normalize data for the Neural Network
        self.X_train_normalized, self.X_test_normalized = normalize_data(
            self.X_train, self.X_test, use_minmax_norm=self.use_min_max_norm)

    def get_cv_splitter(self):
        """
        Returns the appropriate cross-validation splitter based on the use_ts_split parameter.
        """
        if self.use_ts_split:
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def create_ann(self, optimizer='adam', neurons=64, activation='relu', learning_rate=0.001):
        """Builds a Keras sequential model with two dense layers for neural network tuning."""
        if optimizer == 'adam':
            from keras.optimizers import Adam
            optimizer_instance = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            from keras.optimizers import RMSprop
            optimizer_instance = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        model = Sequential()
        model.add(
            Dense(neurons, input_dim=self.X_train.shape[1], activation=activation))
        model.add(Dense(neurons, activation=activation))
        model.add(Dense(1))
        model.compile(optimizer=optimizer_instance, loss='mean_absolute_error')
        return model


    
    # def tune_xgboost(self, model_name=None, params=None, use_gpu=True, objective=None):
    #     model_name = model_name or self.XGBoost_model_name
    #     objective = objective or 'reg:squarederror'
    #     default_params = {'max_depth': [10, 8, 6], 'learning_rate': [
    #         0.1, 0.01], 'n_estimators': [1000, 500, 200]}
    #     # Use provided params if available, else use default
    #     grid_params = params if params is not None else default_params

    #     if use_gpu:
    #         # Instead of updating grid_params, set GPU parameters in the model instantiation:
    #         xgb_model = xgb.XGBRegressor(
    #             objective=objective,
    #             tree_method='gpu_hist',
    #             predictor='gpu_predictor',
    #             n_jobs=-1,
    #             random_state=self.random_state
    #         )
    #     else:
    #         xgb_model = xgb.XGBRegressor(
    #             objective='reg:squarederror',
    #             n_jobs=-1,
    #             random_state=self.random_state
    #         )
    def tune_xgboost(self, model_name=None, params=None, use_gpu=True, objective=None):
        model_name = model_name or self.XGBoost_model_name
        objective = objective or 'reg:squarederror'
        default_params = {
            'max_depth': [10, 8, 6],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [1000, 500, 200]
        }
        grid_params = params if params is not None else default_params

        if use_gpu:
            xgb_model = xgb.XGBRegressor(
                objective=objective,
                tree_method="hist",  # Corrected
                device="cuda",       # Corrected
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            xgb_model = xgb.XGBRegressor(
                objective=objective,
                tree_method="hist",  # CPU mode still uses "hist"
                device="cpu",        # Explicitly set CPU
                n_jobs=-1,
                random_state=self.random_state
            )

        cv_splitter = self.get_cv_splitter()
        print(f'XGBoost objective: {objective}')
        grid = GridSearchCV(
            xgb_model, grid_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        grid.fit(self.X_train, self.y_train)
        best_model_path, best_params_ = self._save_best_grid_model_and_get_errors(
            grid, model_name)
        return best_model_path, best_params_

        cv_splitter = self.get_cv_splitter()
        print(f'XGBoost objective: {objective}')
        grid = GridSearchCV(
            xgb_model, grid_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        grid.fit(self.X_train, self.y_train)
        best_model_path, best_params_ = self._save_best_grid_model_and_get_errors(
            grid, model_name)
        return best_model_path, best_params_

    def tune_random_forest(self, model_name=None, params=None):
        """Perform grid search hyperparameter tuning for Random Forest."""
        if model_name is not None:
            if model_name != self.Random_Forest_model_name:
                warnings.warn(
                    f"The original model name for Random Forest ({self.Random_Forest_model_name}) has been overwritten by the new name: {model_name} ")
                self.Random_Forest_model_name = model_name
        else:
            model_name = self.Random_Forest_model_name
        default_params = {'n_estimators': [100, 200], 'max_depth': [
            10, 20, None], 'min_samples_split': [2, 5]}
        rf_params = params if params else default_params

        rf_model = RandomForestRegressor(
            random_state=self.random_state, n_jobs=-1)
        cv_splitter = self.get_cv_splitter()
        rf_grid = GridSearchCV(
            rf_model, rf_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        rf_grid.fit(self.X_train, self.y_train)

        self._save_best_grid_model_and_get_errors(
            grid_models=rf_grid, model_name=model_name)

    def tune_ann(self,  model_name=None, params=None, use_random=False, n_iter=30):
        """
        Perform tuning for ANN using grid search or random search based on specified parameters.

        Parameters:
        - use_random (bool): If True, use RandomizedSearchCV; otherwise, use GridSearchCV.
        - n_iter (int): Number of iterations for RandomizedSearchCV.
        """
        if model_name is not None:
            if model_name != self.ann_model_name:
                warnings.warn(
                    f"The original model name for Neural Network ({self.ann_model_name}) has been overwritten by the new name: {model_name} ")
                self.ann_model_name = model_name
        else:
            model_name = self.ann_model_name
        default_params = {
            'batch_size': [32, 64, 128],
            'epochs': [2, 50, 100],
            'optimizer': ['adam'],
            'neurons': [16, 32, 64, 128],
            'activation': ['tanh', 'relu'],
            'learning_rate': [0.001, 0.01],
        }
        nn_params = params if params else default_params

        nn_model = KerasRegressor(build_fn=self.create_ann, verbose=0)
        cv_splitter = self.get_cv_splitter()
        if use_random:
            nn_grid = RandomizedSearchCV(
                nn_model, nn_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3, n_iter=n_iter)
        else:
            nn_grid = GridSearchCV(
                nn_model, nn_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        nn_grid.fit(self.X_train_normalized, self.y_train)

        self._save_best_grid_model_and_get_errors(
            grid_models=nn_grid, model_name=model_name)

    def _save_best_grid_model_and_get_errors(self, grid_models, model_name):
        """
        Save the best model from grid search, print the best parameters, and print evaluation metrics.
        """
        best_model = grid_models.best_estimator_
        self.best_models[model_name] = best_model

        # Get best parameters and score
        best_params = grid_models.best_params_
        best_score = -grid_models.best_score_  # Convert back from negative MAE

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best cross-validation MAE for {model_name}: {best_score:.4f}")

        # Evaluate on test set
        # If ann, then use normalized values
        if model_name == self.ann_model_name:
            print('Predicting y for X_test_normalized...')
            y_pred = best_model.predict(self.X_test_normalized)
        else:
            y_pred = best_model.predict(self.X_test)
        test_mae = abs(self.y_test - y_pred).mean()

        # Evaluate on test set
        y_pred = best_model.predict(self.X_test)
        print(f"First 10 values of y_test: {self.y_test[:10]}")
        print(f"First 10 values of y_pred: {y_pred[:10]}")
        print(f"NaNs in y_test: {np.isnan(self.y_test).any()}")
        print(f"NaNs in y_pred: {np.isnan(y_pred).any()}")
        print(f"Zeros in y_test: {(self.y_test == 0).any()}")
        print(f"Zeros in y_pred: {(y_pred == 0).any()}")

        absolute_errors = np.abs(self.y_test - y_pred)
        mae = np.mean(absolute_errors)
        mae_std = np.std(absolute_errors)
        median_ae = np.median(absolute_errors)
        median_ae_std = np.std(absolute_errors)
        rmse = np.sqrt(np.mean(absolute_errors**2))
        rmse_std = np.std(np.sqrt((self.y_test - y_pred)**2))
        mape = np.mean(absolute_errors / self.y_test) * 100
        mape_std = np.std(absolute_errors / self.y_test) * 100
        safe_mape = np.where(self.y_test == 0, np.nan,
                             absolute_errors / self.y_test) * 100
        mape = np.nanmean(safe_mape)
        mape_std = np.nanstd(safe_mape)

        print(f"MAE: {mae:.2f} ± {mae_std:.2f}")
        print(f"Median Absolute Error: {median_ae:.2f} ± {median_ae_std:.2f}")
        print(f"RMSE: {rmse:.2f} ± {rmse_std:.2f}")
        print(f"MAPE: {mape:.2f}% ± {mape_std:.2f}%")

        # Naive model: Predict the last value (last observation before the prediction)
        naive_predictions = np.abs(
            self.X_test['value'] - (self.y_test+self.X_test['value']))
        mae_naive = np.mean(naive_predictions)
        mae_naive_std = np.std(naive_predictions)
        median_ae_naive = np.median(naive_predictions)
        median_ae_naive_std = np.std(naive_predictions)
        rmse_naive = np.sqrt(np.mean(naive_predictions**2))
        rmse_naive_std = np.std(np.sqrt(naive_predictions**2))
        mape_naive = np.mean(naive_predictions / self.X_test['value']) * 100
        mape_naive_std = np.std(naive_predictions / self.X_test['value']) * 100
        print(f"Naive MAE: {mae_naive:.2f} ± {mae_naive_std:.2f}")
        print(
            f"Naive Median Absolute Error: {median_ae_naive:.2f} ± {median_ae_naive_std:.2f}")
        print(f"Naive RMSE: {rmse_naive:.2f} ± {rmse_naive_std:.2f}")
        print(f"Naive MAPE: {mape_naive:.2f}% ± {mape_naive_std:.2f}%")

        print('FINAL COMPARISON:')
        print(f"Test MAE for {model_name}: {test_mae:.2f}")
        print(f"Naive Model MAE: {mae_naive:.2f}")

        self.save_best_model(model_name, best_model)

    def save_best_model(self, model_name, model):
        """Save a single best model based on its name and type."""

        # Ensure the directory exists
        os.makedirs('./models', exist_ok=True)
        best_model_name_string = self.best_model_name_string_start + model_name
        if model_name == self.ann_model_name:
            model.model.save(f'./models/{best_model_name_string}.h5')
            print(f"{model_name} model saved to {best_model_name_string}.h5")
        else:
            with open(f'./models/{best_model_name_string}.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"{model_name} model saved to {best_model_name_string}.pkl")
            
            
            
    # def tune_xgboost(self, model_name=None, params=None, use_gpu=True):
    #     """Perform grid search hyperparameter tuning for XGBoost."""
    #     if model_name is not None:
    #         if model_name != self.XGBoost_model_name:
    #             warnings.warn(
    #                 f"The original model name for XGBoost ({self.XGBoost_model_name}) has been overwritten by the new name: {model_name} ")
    #             self.XGBoost_model_name = model_name
    #     else:
    #         model_name = self.XGBoost_model_name
    #     default_params = {'max_depth': [10, 8, 6], 'learning_rate': [
    #         0.1, 0.01], 'n_estimators': [1000, 500, 200]}
    #     xgb_params = params if params else default_params

    #     # Choose GPU or CPU settings
    #     if use_gpu:
    #         xgb_model = xgb.XGBRegressor(
    #             objective=objective,
    #             tree_method='gpu_hist',  # GPU-optimized tree method
    #             predictor='gpu_predictor',  # Use GPU predictor
    #             n_jobs=-1,
    #             random_state=self.random_state
    #         )
    #         print("Using GPU for XGBoost training.")
    #     else:
    #         xgb_model = xgb.XGBRegressor(
    #             objective=objective,
    #             tree_method='hist',  # CPU-optimized tree method
    #             predictor='cpu_predictor',  # Use CPU predictor
    #             n_jobs=os.cpu_count(),  # Utilize all CPU cores
    #             random_state=self.random_state
    #         )
    #         print(f"Using CPU for XGBoost training with {os.cpu_count()} cores.")
    #     cv_splitter = self.get_cv_splitter()
    #     print(f"XGBoost params: {xgb_params}")
    #     xgb_grid = GridSearchCV(
    #         xgb_model, xgb_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
    #     xgb_grid.fit(self.X_train, self.y_train)

    #     self._save_best_grid_model_and_get_errors(
    #         grid_models=xgb_grid, model_name=model_name)
    

################################################################################################


# class ModelTuner_:
#     """
#     A class for performing hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
#     using either TimeSeriesSplit or standard K-Fold cross-validation.
#     """

#     def __init__(self, X_train, X_test, y_train, y_test, random_state=69, use_ts_split=True, n_splits=3,
#                  use_min_max_norm=False, best_model_name_string_start='best_model_', model_path=None,
#                  XGBoost_model_name=None, Random_Forest_model_name=None, ann_model_name=None):
#         """
#         Initializes ModelTuner with training and test data splits.
#         """
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.best_models = {}
#         self.random_state = random_state
#         self.use_ts_split = use_ts_split
#         self.n_splits = n_splits
#         self.use_min_max_norm = use_min_max_norm
#         self.scaler = None  # Will be initialized when tuning ANN
#         self.best_model_name_string_start = best_model_name_string_start
#         self.model_path = model_path if model_path else './models'
#         os.makedirs(self.model_path, exist_ok=True)
#         self.XGBoost_model_name = XGBoost_model_name if XGBoost_model_name else 'XGBoost'
#         self.Random_Forest_model_name = Random_Forest_model_name if Random_Forest_model_name else 'Random_Forest'
#         self.ann_model_name = ann_model_name if ann_model_name else 'Neural_Network'
#         self.X_train_normalized, self.X_test_normalized = normalize_data(self.X_train, self.X_test, use_minmax_norm=self.use_min_max_norm)

#     def get_cv_splitter(self):
#         """Returns the appropriate cross-validation splitter."""
#         return TimeSeriesSplit(n_splits=self.n_splits) if self.use_ts_split else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)


#     def _get_errors(self, y_true, y_pred):
#         """Calculates and prints different error metrics."""
#         absolute_errors = np.abs(y_true - y_pred)
#         mae = np.mean(absolute_errors)
#         mae_std = np.std(absolute_errors)
#         median_ae = np.median(absolute_errors)
#         median_ae_std = np.std(absolute_errors)
#         rmse = np.sqrt(np.mean(absolute_errors**2))
#         rmse_std = np.std(np.sqrt((y_true - y_pred)**2))
#         safe_mape = np.where(y_true == 0, np.nan, absolute_errors / y_true) * 100
#         mape = np.nanmean(safe_mape)
#         mape_std = np.nanstd(safe_mape)
#         print(f"MAE: {mae:.2f} ± {mae_std:.2f}")
#         print(f"Median Absolute Error: {median_ae:.2f} ± {median_ae_std:.2f}")
#         print(f"RMSE: {rmse:.2f} ± {rmse_std:.2f}")
#         print(f"MAPE: {mape:.2f}% ± {mape_std:.2f}%")

#     def _save_best_grid_model_and_get_errors(self, grid_models, model_name):
#         """Saves the best model and returns its file path, while printing evaluation errors."""
#         best_model = grid_models.best_estimator_

#         if model_name == self.ann_model_name:
#             print('Predicting y for X_test_normalized...')
#             y_pred = best_model.predict(self.X_test_normalized)
#         else:
#             y_pred = best_model.predict(self.X_test)

#         print(f"Evaluation results for {model_name}:")
#         self._get_errors(self.y_test, y_pred)

#         # Compute naive baseline errors
#         naive_predictions = np.abs(self.X_test['value'] - (self.y_test + self.X_test['value']))
#         print("Naive model evaluation:")
#         self._get_errors(self.y_test, naive_predictions)

#         best_model_path = self.save_best_model(model_name, best_model)
#         return best_model_path

#     def save_best_model(self, model_name, model):
#         """Saves the best model to the specified directory and returns the file path."""
#         best_model_name_string = f"{self.best_model_name_string_start}{model_name}"
#         model_file_path = os.path.join(self.model_path, f"{best_model_name_string}.{'h5' if model_name == self.ann_model_name else 'pkl'}")
#         if model_name == self.ann_model_name:
#             model.model.save(model_file_path)
#         else:
#             with open(model_file_path, 'wb') as f:
#                 pickle.dump(model, f)
#         print(f"{model_name} model saved to {model_file_path}")
#         return model_file_path
################################################################################################


# import os
# import sys
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from .helper_functions import normalize_data
# import pickle
# import warnings
# import numpy as np

# class ModelTuner_:
#     """
#     A class for performing hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
#     using either TimeSeriesSplit or standard K-Fold cross-validation.
#     """

#     def __init__(self, X_train, X_test, y_train, y_test, random_state=69, use_ts_split=True, n_splits=3,
#                  use_min_max_norm=False, best_model_name_string_start='best_model_', model_path=None,
#                  XGBoost_model_name=None, Random_Forest_model_name=None, ann_model_name=None):
#         """
#         Initializes ModelTuner with training and test data splits.
#         """
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.best_models = {}
#         self.random_state = random_state
#         self.use_ts_split = use_ts_split
#         self.n_splits = n_splits
#         self.use_min_max_norm = use_min_max_norm
#         self.scaler = None  # Will be initialized when tuning ANN
#         self.best_model_name_string_start = best_model_name_string_start
#         self.model_path = model_path if model_path else './models'
#         os.makedirs(self.model_path, exist_ok=True)
#         self.XGBoost_model_name = XGBoost_model_name if XGBoost_model_name else 'XGBoost'
#         self.Random_Forest_model_name = Random_Forest_model_name if Random_Forest_model_name else 'Random_Forest'
#         self.ann_model_name = ann_model_name if ann_model_name else 'Neural_Network'
#         self.X_train_normalized, self.X_test_normalized = normalize_data(self.X_train, self.X_test, use_minmax_norm=self.use_min_max_norm)

#     def get_cv_splitter(self):
#         """Returns the appropriate cross-validation splitter."""
#         return TimeSeriesSplit(n_splits=self.n_splits) if self.use_ts_split else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

#     def create_ann(self, optimizer='adam', neurons=64, activation='relu', learning_rate=0.001):
#         """Builds a Keras sequential model for ANN tuning."""
#         optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate) if optimizer == 'adam' else keras.optimizers.RMSprop(learning_rate=learning_rate)
#         model = Sequential([
#             Dense(neurons, input_dim=self.X_train.shape[1], activation=activation),
#             Dense(neurons, activation=activation),
#             Dense(1)
#         ])
#         model.compile(optimizer=optimizer_instance, loss='mean_absolute_error')
#         return model

#     def tune_ann(self, model_name=None, params=None, use_random=False, n_iter=30):
#         """Performs hyperparameter tuning for an Artificial Neural Network (ANN) using GridSearchCV or RandomizedSearchCV."""
#         model_name = model_name or self.ann_model_name
#         default_params = {'batch_size': [32, 64, 128], 'epochs': [2, 50, 100], 'optimizer': ['adam'], 'neurons': [16, 32, 64, 128], 'activation': ['tanh', 'relu'], 'learning_rate': [0.001, 0.01]}
#         nn_model = KerasRegressor(build_fn=self.create_ann, verbose=0)
#         search_cls = RandomizedSearchCV if use_random else GridSearchCV
#         grid = search_cls(nn_model, params or default_params, scoring='neg_mean_absolute_error', cv=self.get_cv_splitter(), verbose=3, n_iter=n_iter)
#         grid.fit(self.X_train_normalized, self.y_train)
#         return self._save_best_grid_model_and_get_errors(grid, model_name)

#     def _get_errors(self, y_true, y_pred):
#         """Calculates and prints different error metrics."""
#         absolute_errors = np.abs(y_true - y_pred)
#         mae = np.mean(absolute_errors)
#         mae_std = np.std(absolute_errors)
#         median_ae = np.median(absolute_errors)
#         median_ae_std = np.std(absolute_errors)
#         rmse = np.sqrt(np.mean(absolute_errors**2))
#         rmse_std = np.std(np.sqrt((y_true - y_pred)**2))
#         safe_mape = np.where(y_true == 0, np.nan, absolute_errors / y_true) * 100
#         mape = np.nanmean(safe_mape)
#         mape_std = np.nanstd(safe_mape)
#         print(f"MAE: {mae:.2f} ± {mae_std:.2f}")
#         print(f"Median Absolute Error: {median_ae:.2f} ± {median_ae_std:.2f}")
#         print(f"RMSE: {rmse:.2f} ± {rmse_std:.2f}")
#         print(f"MAPE: {mape:.2f}% ± {mape_std:.2f}%")

#     def _save_best_grid_model_and_get_errors(self, grid_models, model_name):
#         """Saves the best model and returns its file path, while printing evaluation errors."""
#         best_model = grid_models.best_estimator_

#         if model_name == self.ann_model_name:
#             print('Predicting y for X_test_normalized...')
#             y_pred = best_model.predict(self.X_test_normalized)
#         else:
#             y_pred = best_model.predict(self.X_test)

#         print(f"Evaluation results for {model_name}:")
#         self._get_errors(self.y_test, y_pred)


#         # Compute naive baseline errors
#         naive_predictions = np.abs(self.X_test['value'] - (self.y_test + self.X_test['value']))
#         print("Naive model evaluation:")
#         self._get_errors(self.y_test, naive_predictions)

#         best_model_path = self.save_best_model(model_name, best_model)
#         return best_model_path

#     def save_best_model(self, model_name, model):
#         """Saves the best model to the specified directory and returns the file path."""
#         best_model_name_string = f"{self.best_model_name_string_start}{model_name}"
#         model_file_path = os.path.join(self.model_path, f"{best_model_name_string}.{'h5' if model_name == self.ann_model_name else 'pkl'}")
#         if model_name == self.ann_model_name:
#             model.model.save(model_file_path)
#         else:
#             with open(model_file_path, 'wb') as f:
#                 pickle.dump(model, f)
#         print(f"{model_name} model saved to {model_file_path}")
#         return model_file_path


class ModelTuner_:
    """
    A class for performing hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
    using either TimeSeriesSplit or standard K-Fold cross-validation.
    """

    def __init__(self, X_train, X_test, y_train, y_test, random_state=69, use_ts_split=True, n_splits=3,
                 use_min_max_norm=False, best_model_name_string_start='best_model_', model_path=None,
                 XGBoost_model_name=None, Random_Forest_model_name=None, ann_model_name=None):
        """
        Initializes ModelTuner with training and test data splits.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_models = {}
        self.random_state = random_state
        self.use_ts_split = use_ts_split
        self.n_splits = n_splits
        self.use_min_max_norm = use_min_max_norm
        self.scaler = None  # Will be initialized when tuning ANN
        self.best_model_name_string_start = best_model_name_string_start
        self.model_path = model_path if model_path else './models'
        os.makedirs(self.model_path, exist_ok=True)
        self.XGBoost_model_name = XGBoost_model_name if XGBoost_model_name else 'XGBoost'
        self.Random_Forest_model_name = Random_Forest_model_name if Random_Forest_model_name else 'Random_Forest'
        self.ann_model_name = ann_model_name if ann_model_name else 'Neural_Network'
        self.X_train_normalized, self.X_test_normalized = normalize_data(
            self.X_train, self.X_test, use_minmax_norm=self.use_min_max_norm)

    def get_cv_splitter(self):
        """Returns the appropriate cross-validation splitter."""
        return TimeSeriesSplit(n_splits=self.n_splits) if self.use_ts_split else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def _get_errors(self, y_true, y_pred):
        """Calculates and prints different error metrics."""
        absolute_errors = np.abs(y_true - y_pred)
        mae = np.mean(absolute_errors)
        mae_std = np.std(absolute_errors)
        median_ae = np.median(absolute_errors)
        median_ae_std = np.std(absolute_errors)
        rmse = np.sqrt(np.mean(absolute_errors**2))
        safe_mape = np.where(y_true == 0, np.nan,
                             absolute_errors / y_true) * 100
        mape = np.nanmean(safe_mape)
        mape_std = np.nanstd(safe_mape)
        print(f"MAE: {mae:.2f} ± {mae_std:.2f}")
        print(f"Median Absolute Error: {median_ae:.2f} ± {median_ae_std:.2f}")
        print(f"MAPE: {mape:.2f}% ± {mape_std:.2f}%")

    
    def tune_xgboost(self, model_name=None, params=None, use_gpu=True, objective=None):
        model_name = model_name or self.XGBoost_model_name
        objective = objective or 'reg:squarederror'
        default_params = {'max_depth': [10, 8, 6, 4], 'learning_rate': [
            0.1, 0.01], 'n_estimators': [1000, 750, 500, 250]}
        # Use provided params if available, else use default
        grid_params = params if params is not None else default_params

        if use_gpu:
            # Instead of updating grid_params, set GPU parameters in the model instantiation:
            xgb_model = xgb.XGBRegressor(
                objective=objective,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            xgb_model = xgb.XGBRegressor(
                objective=objective,
                n_jobs=-1,
                random_state=self.random_state
            )

        cv_splitter = self.get_cv_splitter()
        print(f'XGBoost objective: {objective}')
        grid = GridSearchCV(
            xgb_model, grid_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        grid.fit(self.X_train, self.y_train)
        best_model_path, best_params_ = self._save_best_grid_model_and_get_errors(
            grid, model_name)
        return best_model_path, best_params_
    
    
    def tune_xgboost_fixed_split_with_gridsearch(self, model_name=None, params=None, use_gpu=True, objective=None, train_val_ratio=2/3):
        """Tunes XGBoost using a fixed train/validation split with GridSearchCV and PredefinedSplit."""
        from sklearn.model_selection import PredefinedSplit, GridSearchCV
        from sklearn.utils import indexable
        import numpy as np

        model_name = model_name or self.XGBoost_model_name
        objective = objective or 'reg:squarederror'

        default_params = {
            'max_depth': [10, 8, 6, 4],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [1000, 750, 500, 250]
        }
        grid_params = params if params is not None else default_params

        # Step 1: Create fixed train/val split using PredefinedSplit
        n_samples = len(self.X_train)
        n_train = int(n_samples * train_val_ratio)

        # test_fold: -1 for training samples, 0 for validation samples
        test_fold = np.concatenate([
            np.full(n_train, -1),
            np.zeros(n_samples - n_train)
        ])
        ps = PredefinedSplit(test_fold)

        # Make sure X and y are indexable in the same way
        X_all, y_all = indexable(self.X_train, self.y_train)

        # Step 2: Define the model
        xgb_model = xgb.XGBRegressor(
            objective=objective,
            tree_method='gpu_hist' if use_gpu else 'auto',
            predictor='gpu_predictor' if use_gpu else 'auto',
            n_jobs=-1,
            random_state=self.random_state
        )

        # Step 3: Perform grid search
        print(f'Using fixed split: {train_val_ratio:.2f} train / {1-train_val_ratio:.2f} val')
        print(f'XGBoost objective: {objective}')
        grid = GridSearchCV(
            xgb_model,
            param_grid=grid_params,
            scoring='neg_mean_absolute_error',
            cv=ps,
            verbose=3
        )
        grid.fit(X_all, y_all)

        # Step 4: Save best model and report metrics
        best_model_path, best_params_ = self._save_best_grid_model_and_get_errors(
            grid, model_name)
        return best_model_path, best_params_
    def tune_xgboost_fixed_split(self, model_name=None, params=None, use_gpu=True, objective=None, train_val_ratio=1/2):
        """Tunes XGBoost using a fixed train/validation split instead of CV."""
        model_name = model_name or self.XGBoost_model_name
        objective = objective or 'reg:squarederror'

        default_params = {
            'max_depth': [10, 8, 6, 4],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [1000, 750, 500, 250]
        }
        grid_params = params if params is not None else default_params

        # 1. Manual train/val split
        num_train_samples = int(len(self.X_train) * train_val_ratio)
        X_train_sub = self.X_train[:num_train_samples]
        y_train_sub = self.y_train[:num_train_samples]
        X_val_sub = self.X_train[num_train_samples:]
        y_val_sub = self.y_train[num_train_samples:]

        # 2. Create all hyperparam combinations
        from itertools import product
        all_combos = list(product(*grid_params.values()))
        param_names = list(grid_params.keys())

        best_model = None
        best_mae = float('inf')
        best_params_ = None

        print(f"Using fixed train/val split ({train_val_ratio:.2f} train / {1-train_val_ratio:.2f} val)")

        for combo in all_combos:
            param_dict = dict(zip(param_names, combo))
            if use_gpu:
                model = xgb.XGBRegressor(
                    **param_dict,
                    objective=objective,
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    n_jobs=-1,
                    random_state=self.random_state
                )
            else:
                model = xgb.XGBRegressor(
                    **param_dict,
                    objective=objective,
                    n_jobs=-1,
                    random_state=self.random_state
                )

            model.fit(X_train_sub, y_train_sub)
            y_pred = model.predict(X_val_sub)
            mae = np.mean(np.abs(y_val_sub - y_pred))
            print(f"Params: {param_dict}, Val MAE: {mae:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_params_ = param_dict

        # Final test set evaluation
        print(f"\nBest fixed-split model ({model_name}) performance on test set:")
        y_test_pred = best_model.predict(self.X_test)
        self._get_errors(self.y_test, y_test_pred)

        # Save the model
        best_model_path = self.save_best_model(model_name, best_model)
        print(f"Best parameters for {model_name}: {best_params_}")
        return best_model_path, best_params_
    

    def tune_random_forest(self, model_name=None, params=None):
        """Performs hyperparameter tuning for Random Forest using GridSearchCV."""
        model_name = model_name or self.Random_Forest_model_name
        default_params = {'n_estimators': [100, 200], 'max_depth': [
            10, 20, None], 'min_samples_split': [2, 5]}
        rf_model = RandomForestRegressor(
            random_state=self.random_state, n_jobs=-1)
        grid = GridSearchCV(rf_model, params or default_params,
                            scoring='neg_mean_absolute_error', cv=self.get_cv_splitter(), verbose=3)
        grid.fit(self.X_train, self.y_train)
        best_model_path, grid_models.best_params_ = self._save_best_grid_model_and_get_errors(
            grid, model_name)
        return best_model_path, grid_models.best_params_
    
    
    

    def _save_best_grid_model_and_get_errors(self, grid_models, model_name):
        """Saves the best model and returns its file path, while printing evaluation errors."""
        best_model = grid_models.best_estimator_
        y_pred = best_model.predict(self.X_test)
        print(f"Evaluation results for {model_name}:")
        self._get_errors(self.y_test, y_pred)

        # Compute naive baseline errors
        naive_predictions = np.abs(
            self.X_test['value'] - (self.y_test + self.X_test['value']))
        print("Naive model evaluation:")
        self._get_errors(self.y_test, naive_predictions)

        # Print best parameters explicitly
        best_params_ = grid_models.best_params_
        print(f"\nBest parameters for {model_name}: {best_params_}")

        best_model_path = self.save_best_model(model_name, best_model)
        return best_model_path, grid_models.best_params_

    def save_best_model(self, model_name, model):
        """Saves the best model to the specified directory and returns the file path."""
        best_model_name_string = f"{self.best_model_name_string_start}{model_name}"
        model_file_path = os.path.join(
            self.model_path, f"{best_model_name_string}.{'h5' if model_name == self.ann_model_name else 'pkl'}")
        if model_name == self.ann_model_name:
            model.model.save(model_file_path)
        else:
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)
        print(f"{model_name} model saved to {model_file_path}")
        return model_file_path
