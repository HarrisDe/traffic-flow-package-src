from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np




def normalize_data(X_train=None, X_test=None, use_minmax_norm=False, use_full_data=False):
    """
    Normalizes training and/or testing data using StandardScaler or MinMaxScaler.

    Parameters:
    - X_train (array-like, optional): Training dataset. If None, only `X_test` will be normalized.
    - X_test (array-like, optional): Testing dataset. If None, only `X_train` will be normalized.
    - use_minmax_norm (bool): If True, uses MinMaxScaler; otherwise, uses StandardScaler.
    - use_full_data (bool): If True, normalizes using both training and testing data combined.
        WARNING: Using this option introduces data leakage because the test data influences 
        the scaling applied to the training data. This approach may lead to overly optimistic 
        performance metrics and is not recommended for real-world scenarios where the test 
        set must remain unseen until final evaluation.

    Returns:
    - X_train_normalized (array-like, optional): Normalized training dataset (if `X_train` is provided).
    - X_test_normalized (array-like, optional): Normalized testing dataset (if `X_test` is provided).

    Notes:
    - Default behavior (`use_full_data=False`) ensures that scaling is based solely on the training data, 
      which is a best practice to avoid data leakage.
    - Use `use_full_data=True` only in controlled experiments where you need consistent scaling across 
      both training and test sets and are aware of the potential risks.
    - At least one of `X_train` or `X_test` must be provided.

    Example Usage:
    - Normalize with StandardScaler (default):
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test)
    - Normalize only `X_train`:
        X_train_normalized, _ = normalize_data(X_train=X_train)
    - Normalize with MinMaxScaler using both datasets:
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test, use_minmax_norm=True, use_full_data=True)
    """
    if X_train is None and X_test is None:
        raise ValueError("At least one of X_train or X_test must be provided.")

    scaler = MinMaxScaler() if use_minmax_norm else StandardScaler()

    if use_full_data:
        if X_train is not None and X_test is not None:
            # Concatenate training and test data for joint scaling
            full_data = np.concatenate([X_train, X_test], axis=0)
            full_data_normalized = scaler.fit_transform(full_data)
            # Split back into training and test sets
            X_train_normalized = full_data_normalized[:X_train.shape[0], :]
            X_test_normalized = full_data_normalized[X_train.shape[0]:, :]
        elif X_train is not None:
            X_train_normalized = scaler.fit_transform(X_train)
            X_test_normalized = None
        elif X_test is not None:
            X_test_normalized = scaler.fit_transform(X_test)
            X_train_normalized = None
    else:
        if X_train is not None:
            X_train_normalized = scaler.fit_transform(X_train)
            if X_test is not None:
                X_test_normalized = scaler.transform(X_test)
            else:
                X_test_normalized = None
        elif X_test is not None:
            # Scale only test data if training data is not provided
            X_test_normalized = scaler.fit_transform(X_test)
            X_train_normalized = None

    return X_train_normalized, X_test_normalized


def load_and_evaluate_a_model(model_name, X_train, X_test, y_train, y_test, horizon, return_csv=True):
    """
    Load the best model by name, make predictions on a new dataset, and calculate prediction errors
    along with naive predictions.

    Parameters:
    - model_name (str): The name of the model to load (e.g., 'XGBoost', 'Random_Forest', 'Neural_Network').
    - new_X (pd.DataFrame): New input features for prediction.
    - new_y (pd.Series): True target values for the new dataset.
    - use_normalized (bool): If True, normalize the input features for ANN.

    Returns:
    - dict: A dictionary containing model MAE and naive MAE for comparison.
    """
    # Load the model
    model_path = f'../models/{model_name}'
    if 'neural' in model_name.lower():
        from keras.models import load_model
        best_model = load_model(f'{model_path}.h5')
        print(f"{model_name} model loaded from {model_path}.h5")
        print("Normalizing new dataset for ANN.")
        X_train_normalized, X_test_normalized = normalize_data(
            X_train, X_test, use_minmax_norm=False, use_full_data=False)
        y_pred = best_model.predict(X_test_normalized)

    else:
        with open(f'{model_path}.pkl', 'rb') as f:
            best_model = pickle.load(f)
        print(f"{model_name} model loaded from {model_path}.pkl")
        y_pred = best_model.predict(X_test)

    test_mae = abs(y_test - y_pred).mean()

    # Naive model: Predict the last value (last observation before the prediction)
    naive_predictions = np.abs(y_test)  # (naive(same as last) - actual)
    naive_mae = np.mean(naive_predictions)

    print(f'Test MAE (horizon: {horizon}min):{test_mae:.4f}')
    print(f'Test naive MAE (horizon: {horizon}min:{naive_mae:.4f}')

    if return_csv:
        df = {'incremental_id': X_test['incremental_id'],
              f'y_test_{horizon}': y_test, f'y_pred_{horizon}': y_pred}
        df = pd.DataFrame(df)
        df.to_csv(f'results_horizon_{horizon}_min.csv')


def optimize_dtypes(df):
    """
    Optimize the data types of numeric columns in a DataFrame to reduce memory usage.

    This function identifies numeric columns (`int` and `float`) in the DataFrame and
    downcasts their data types to the smallest possible type (`float32`, `int32`, etc.)
    without losing precision. This can help reduce memory usage for large DataFrames.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to optimize.

    Returns:
    - pd.DataFrame: The DataFrame with optimized data types for numeric columns.
    """

    # Iterate over numeric columns (int and float types)
    for col in df.select_dtypes(include=['int', 'float']).columns:
        # Check and downcast float columns
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        # Check and downcast integer columns
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')

    return df


def get_adjacent_sensors_dict(adj_sensors_csv_loc, nr_of_adj_sensors, delete_duplicate_rows=True):
    """
    Create a dictionary of sensor IDs with their adjacent sensors and distances.

    Args:
        adj_sensors_csv_loc (str): File path to the CSV containing sensor data.
        nr_of_adj_sensors (int): Number of adjacent sensors to retrieve.
        delete_duplicate_rows (bool): Whether to remove rows where the first and second columns are the same.

    Returns:
        dict: A dictionary where each key is a sensor ID and the value is a dictionary
              containing adjacent sensors and distances.
    """
    # Load the sensor data from the CSV
    sensor_data = pd.read_csv(adj_sensors_csv_loc, delimiter=';')
    sensor_data.columns = sensor_data.columns.str.strip()  # Ensure no extra whitespace
    sensor_data['distance'] = pd.to_numeric(
        sensor_data['distance'], errors='coerce')  # Ensure numeric distance

    # Remove rows where the first and second columns are the same if delete_duplicate_rows is True
    if delete_duplicate_rows:
        duplicate_rows = sensor_data[sensor_data['point_dgl_loc']
                                     == sensor_data['conn_points_dgl_loc']]
        print("Rows where the first and second columns are the same:")
        print(duplicate_rows)

        sensor_data = sensor_data[sensor_data['point_dgl_loc']
                                  != sensor_data['conn_points_dgl_loc']]

    # Extract unique sensor IDs
    sensor_ids = pd.concat(
        [sensor_data['point_dgl_loc'], sensor_data['conn_points_dgl_loc']]).unique()

    # Create a dictionary to hold the result
    sensor_dict = {sensor_id: {"previous_sensors": [], "distance": []}
                   for sensor_id in sensor_ids}

    # Build a lookup for previous sensor and distance
    for sensor_id in sensor_ids:
        # Temporary lists to store previous sensors and distances
        prev_sensors = []
        distances = []

        current_sensor = sensor_id

        # Traverse backwards up to the number of required previous sensors
        while len(prev_sensors) < nr_of_adj_sensors:
            # Find the row where current_sensor appears in conn_points_dgl_loc
            row = sensor_data[sensor_data['conn_points_dgl_loc']
                              == current_sensor]

            if row.empty:
                # If no previous sensor exists, append None for the remaining slots
                prev_sensors.extend(
                    [None] * (nr_of_adj_sensors - len(prev_sensors)))
                distances.extend([None] * (nr_of_adj_sensors - len(distances)))
                break
            else:
                # Get the previous sensor and distance
                previous_sensor = row.iloc[0]['point_dgl_loc']
                distance = row.iloc[0]['distance']

                prev_sensors.append(previous_sensor)
                distances.append(distance)

                # Move to the previous sensor
                current_sensor = previous_sensor

        # Ensure the lengths of the lists match nr_of_adj_sensors
        while len(prev_sensors) < nr_of_adj_sensors:
            prev_sensors.append(None)
            distances.append(None)

        # Update the dictionary
        sensor_dict[sensor_id]['previous_sensors'] = prev_sensors
        sensor_dict[sensor_id]['distance'] = distances

    return sensor_dict


def get_surrounding_sensors(sensor_dict, sensor_id, n):
    """
    Get N sensors before and N sensors after a specific sensor ID.

    Args:
        sensor_dict (dict): Dictionary generated by `get_adjacent_sensors_dict`.
        sensor_id (str): The sensor ID for which to retrieve surrounding sensors.
        n (int): The number of sensors before and after to retrieve.

    Returns:
        dict: A dictionary with two keys: 'previous_sensors' and 'next_sensors'.
              Each key contains a list of N sensors. If there are fewer than N sensors,
              the remaining entries are filled with None.
    """
    # Initialize the result dictionary
    result = {
        "previous_sensors": [],
        "next_sensors": []
    }

    # Get the previous sensors
    previous_sensors = sensor_dict[sensor_id]['previous_sensors']
    result["previous_sensors"] = previous_sensors[:n] + \
        [None] * (n - len(previous_sensors))

    # Find the next sensors
    next_sensors = []
    current_sensor = sensor_id
    while len(next_sensors) < n:
        # Find the row where the current_sensor is the starting point
        next_sensor_data = [
            key for key, value in sensor_dict.items()
            if value['previous_sensors'] and value['previous_sensors'][0] == current_sensor
        ]

        if not next_sensor_data:
            # If no next sensor is found, fill with None
            next_sensors.extend([None] * (n - len(next_sensors)))
            break
        else:
            next_sensor = next_sensor_data[0]
            next_sensors.append(next_sensor)
            current_sensor = next_sensor

    # Ensure the list has exactly N elements
    result["next_sensors"] = next_sensors[:n] + \
        [None] * (n - len(next_sensors))

    return result
