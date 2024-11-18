import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from .helper_functions import normalize_data
import pickle
import warnings

class ModelTuner:
    """
    A class to perform hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
    using either TimeSeriesSplit or standard K-Fold cross-validation.
    """

    def __init__(self, X_train, X_test, y_train, y_test, random_state=69, use_ts_split=True, n_splits=3,use_min_max_norm = False,best_model_name_string_start = 'best_model_'):
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
        self.best_model_name_string_start = best_model_name_string_start #the start of the name of the best model
        self.XGBoost_model_name = 'XGBoost'
        self.Random_Forest_model_name = 'Random_Forest'
        self.ann_model_name = 'Neural_Network'
        self.X_train_normalized, self.X_test_normalized = normalize_data(self.X_train, self.X_test, use_minmax_norm=self.use_min_max_norm)

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
        model.add(Dense(neurons, input_dim=self.X_train.shape[1], activation=activation))
        model.add(Dense(neurons, activation=activation))
        model.add(Dense(1))
        model.compile(optimizer=optimizer_instance, loss='mean_absolute_error')
        return model


    def tune_xgboost(self, model_name=None, params=None):
        """Perform grid search hyperparameter tuning for XGBoost."""
        if model_name is not None:
            if model_name != self.XGBoost_model_name:
                warnings.warn(f"The original model name for XGBoost ({self.XGBoost_model_name}) has been overwritten by the new name: {model_name} ")
                self.XGBoost_model_name = model_name
        else:
            model_name = self.XGBoost_model_name
        default_params = {'max_depth': [6, 8, 10], 'learning_rate': [0.1, 0.01], 'n_estimators': [500, 1000]}
        xgb_params = params if params else default_params

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=self.random_state)
        cv_splitter = self.get_cv_splitter()
        xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        xgb_grid.fit(self.X_train, self.y_train)

        self._save_best_grid_model_and_get_errors(grid_models=xgb_grid, model_name = model_name)

    def tune_random_forest(self, model_name = None, params=None):
        """Perform grid search hyperparameter tuning for Random Forest."""
        if model_name is not None:
            if model_name != self.Random_Forest_model_name:
                warnings.warn(f"The original model name for Random Forest ({self.Random_Forest_model_name}) has been overwritten by the new name: {model_name} ")
                self.Random_Forest_model_name= model_name
        else:
            model_name = self.Random_Forest_model_name
        default_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
        rf_params = params if params else default_params

        rf_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        cv_splitter = self.get_cv_splitter()
        rf_grid = GridSearchCV(rf_model, rf_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        rf_grid.fit(self.X_train, self.y_train)

        self._save_best_grid_model_and_get_errors(grid_models=rf_grid, model_name=model_name)

    def tune_ann(self,  model_name=None, params=None, use_random=False, n_iter=30):
        """
        Perform tuning for ANN using grid search or random search based on specified parameters.
        
        Parameters:
        - use_random (bool): If True, use RandomizedSearchCV; otherwise, use GridSearchCV.
        - n_iter (int): Number of iterations for RandomizedSearchCV.
        """
        if model_name is not None:
            if model_name != self.ann_model_name:
                warnings.warn(f"The original model name for Neural Network ({self.ann_model_name}) has been overwritten by the new name: {model_name} ")
                self.ann_model_name= model_name
        else: 
            model_name = self.ann_model_name
        default_params = {
            'batch_size': [32, 64, 128],
            'epochs': [50, 100],
            'optimizer': ['adam'],
            'neurons': [16, 32, 64, 128],
            'activation': ['relu', 'tanh'],
            'learning_rate': [0.001, 0.01],
        }
        nn_params = params if params else default_params

        nn_model = KerasRegressor(build_fn=self.create_ann, verbose=0)
        cv_splitter = self.get_cv_splitter()
        if use_random:
            nn_grid = RandomizedSearchCV(nn_model, nn_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3, n_iter=n_iter)
        else:
            nn_grid = GridSearchCV(nn_model, nn_params, scoring='neg_mean_absolute_error', cv=cv_splitter, verbose=3)
        nn_grid.fit(self.X_train_normalized, self.y_train)

        self._save_best_grid_model_and_get_errors(grid_models=nn_grid, model_name=model_name)

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

        print(f"Test MAE for {model_name}: {test_mae:.4f}")

        self.save_best_model(model_name, best_model)

    def save_best_model(self, model_name, model):
        """Save a single best model based on its name and type."""

        best_model_name_string = self.best_model_name_string_start  + model_name
        if model_name == self.ann_model_name:
            model.model.save(f'../models/{best_model_name_string}.h5')
            print(f"{model_name} model saved to {best_model_name_string}.h5")
        else:
            with open(f'../models/{best_model_name_string}.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"{model_name} model saved to {best_model_name_string}.pkl")






# class ModelTuner:
#     """
#     A class to perform hyperparameter tuning for different regression models (XGBoost, Random Forest, Neural Network)
#     using grid and randomized search methods.
#     """

#     def __init__(self, X_train, X_test, y_train, y_test, random_state=69):
#         """
#         Initializes ModelTuner with training and test data splits.

#         Parameters:
#         - X_train, X_test, y_train, y_test: Training and testing data splits.
#         - random_state (int): Random seed for reproducibility.
#         """
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.best_models = {}
#         self.random_state = random_state

#     def create_ann(self, optimizer='adam', neurons=64, activation='relu'):
#         """Builds a Keras sequential model with two dense layers for neural network tuning."""
#         model = Sequential()
#         model.add(Dense(neurons, input_dim=self.X_train.shape[1], activation=activation))
#         model.add(Dense(neurons, activation=activation))
#         model.add(Dense(1))
#         model.compile(optimizer=optimizer, loss='mean_absolute_error')
#         return model

#     def tune_xgboost(self, params=None):
#         """Perform grid search hyperparameter tuning for XGBoost."""
#         default_params = {'max_depth': [6, 8, 10], 'learning_rate': [0.1, 0.01], 'n_estimators': [500, 1000]}
#         xgb_params = params if params else default_params
        
#         xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=self.random_state)
#         xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring='neg_mean_absolute_error', cv=3, verbose=3)
#         xgb_grid.fit(self.X_train, self.y_train)
        
#         self._save_best_grid_model_and_get_errors(grid_models=xgb_grid, model_name='XGBoost')

#     def tune_random_forest(self, params=None):
#         """Perform grid search hyperparameter tuning for Random Forest."""
#         default_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
#         rf_params = params if params else default_params
        
#         rf_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
#         rf_grid = GridSearchCV(rf_model, rf_params, scoring='neg_mean_absolute_error', cv=3, verbose=3)
#         rf_grid.fit(self.X_train, self.y_train)
        
#         self._save_best_grid_model_and_get_errors(grid_models=rf_grid, model_name='Random Forest')
    
#     def tune_ann(self, params=None, use_random=False, n_iter=30):
#         """Perform tuning for ANN using grid search or random search based on specified parameters."""
#         default_params = {
#             'batch_size': [32, 64],
#             'epochs': [50, 100],
#             'optimizer': ['adam', 'rmsprop'],
#             'neurons': [32, 64],
#             'activation': ['relu', 'tanh']
#         }
#         nn_params = params if params else default_params

#         nn_model = KerasRegressor(build_fn=self.create_ann, verbose=0)
#         if use_random:
#             nn_grid = RandomizedSearchCV(nn_model, nn_params, scoring='neg_mean_absolute_error', cv=3, verbose=3, n_iter=n_iter)
#         else:
#             nn_grid = GridSearchCV(nn_model, nn_params, scoring='neg_mean_absolute_error', cv=3, verbose=3)
#         nn_grid.fit(self.X_train, self.y_train)
        
#         self._save_best_grid_model_and_get_errors(grid_models=nn_grid, model_name='Neural Network')

#     def _save_best_grid_model_and_get_errors(self, grid_models, model_name):
#         """Save the best model from grid search and print evaluation metrics."""
#         best_model = grid_models.best_estimator_
#         self.best_models[model_name] = best_model
#         self.save_best_model(model_name, best_model)

#     def save_best_model(self, model_name, model):
#         """Save a single best model based on its name and type."""
#         if model_name == 'Neural Network':
#             model.model.save(f'best_{model_name.lower().replace(" ", "_")}_model.h5')
#             print(f"{model_name} model saved to best_{model_name.lower().replace(' ', '_')}_model.h5")
#         else:
#             with open(f'best_{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
#                 pickle.dump(model, f)
#             print(f"{model_name} model saved to best_{model_name.lower().replace(' ', '_')}_model.pkl")
