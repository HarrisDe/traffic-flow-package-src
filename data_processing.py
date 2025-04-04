
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from .constants import colnames
import random
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm
from .helper_functions import get_adjacent_sensors_dict
import pickle
import time
import json
import re 

class InitialTrafficDataLoader:
    """
    Lightweight preprocessing class for traffic data prior to ML ingestion.
    Includes loading, train-test split tagging, and optional filtering.
    """

    def __init__(self, file_path, datetime_cols=['datetime', 'date'], sensor_col='sensor_id', value_col='value'):
        """
        Initialize the loader with path to parquet file.

        Args:
            file_path (str): Path to the parquet data file.
            datetime_col (str): Name of the datetime column.
        """
        self.file_path = file_path
        self.df = None
        self.df_orig = None
        self.datetime_cols = datetime_cols
        self.datetime_col = None
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.first_test_timestamp = None

    def load_data_parquet(self,df=None):
        """Loads parquet data and ensures datetime format."""
        if df is None:
            self.df = pd.read_parquet(self.file_path, engine='pyarrow')
        else:
            self.df = df
        
        
        col_not_found=0
        for col in self.datetime_cols:
            
            if col in self.df.columns:
                self.datetime_col = col
                self.df[col] = pd.to_datetime(self.df[col])
            else:
                col_not_found+=1
            if col_not_found==len(self.datetime_cols):
                raise ValueError(
                f"Missing expected datetime column(s): {self.datetime_cols}")

        self.df_orig = self.df.copy()
        self.df = self.df.sort_values([self.datetime_col,self.sensor_col]).reset_index(drop=True)
        logging.info(f"Loaded {len(self.df)} rows from {self.file_path}. Columns: {self.df.columns}")

    def add_test_set_column(self, test_size=1/3,return_test_first_timestamp=False):
        """Adds a 'test_set' column by splitting based on timestamp."""
        unique_timestamps = self.df[self.datetime_col].sort_values().unique()
        split_index = int(len(unique_timestamps) * (1 - test_size))
        split_time = unique_timestamps[split_index]

        self.df['test_set'] = self.df[self.datetime_col] >= split_time
        logging.info(f"Added test set column with split time (first test set timestamp): {split_time}")
        self.first_test_timestamp = split_time

    

    def filter_extreme_changes(self, value_col='value', relative_threshold=0.7):
        """
        Filters out extreme relative changes in value (speed), replacing them with NaN and interpolating.
        """
        self.df['relative_change'] = self.df.groupby(
            'sensor_id')[value_col].pct_change().abs()
        outliers = self.df['relative_change'] > relative_threshold
        self.df.loc[outliers, value_col] = np.nan
        self.df[value_col] = self.df.groupby('sensor_id')[value_col].transform(
            lambda x: x.interpolate().ffill().bfill()
        )
        self.df.drop(columns=['relative_change'], inplace=True)

    def smooth_speeds(self, value_col='value', window_size=3, train_only=True):
        """
        Smooths speed using rolling median. Can be applied to full data or just train set.

        Args:
            value_col (str): Column with speed values.
            window_size (int): Window size for smoothing.
            train_only (bool): If True, apply only to training data.
        """
        if 'test_set' not in self.df.columns:
            logging.info(
                "Adding test set column because smooth_speeds() was called.")
            self.add_test_set_column()
        if train_only:
            mask = ~self.df['test_set']
        else:
            mask = self.df.index == self.df.index  # All rows

        smoothed = (
            self.df.loc[mask]
            .groupby('sensor_id')[value_col]
            .transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).median())
        )

        self.df.loc[mask, value_col] = smoothed.ffill().bfill()

    def get_data(self, window_size=3, train_only=True, filter_extreme_changes=True, smooth_speeds=True, relative_threshold=0.7, test_size=1/3):
        """Returns the processed dataframe."""
        self.load_data_parquet()
        self.add_test_set_column(test_size=test_size)
        if filter_extreme_changes:
            self.filter_extreme_changes(relative_threshold=relative_threshold)
        if smooth_speeds:
            self.smooth_speeds(window_size=window_size, train_only=train_only)

        return self.df.copy()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # You can set this to DEBUG, WARNING, etc. as needed
)


class AdjacentSensorFeatureAdder:
    """
    Adds downstream and upstream sensor features to a traffic dataset.
    """

    def __init__(self,
                sensor_dict_path='../data',
                 spatial_adj=5,
                 normalize_by_distance=False,
                 fill_nans_value=-1,
                 disable_logs=False):
        self.sensor_dict_path = sensor_dict_path
        with open(os.path.join(self.sensor_dict_path,'downstream_dict.json'), 'r') as f:
            self.downstream_sensor_dict = json.load(f)
            
        with open(os.path.join(self.sensor_dict_path,'upstream_dict.json'), 'r') as f:    
            self.upstream_sensor_dict = json.load(f)
        self.spatial_adj = spatial_adj
        self.normalize_by_distance = normalize_by_distance
        self.fill_nans_value = fill_nans_value
        self.disable_logs = disable_logs
        self.new_columns = []

    def _log(self, msg):
        if not self.disable_logs:
            logging.info(msg)

    def transform(self, df, sensor_col='sensor_id', time_col='date', value_col='value'):
        """
        Adds adjacent sensor features to the input DataFrame.

        Returns:
            pd.DataFrame, list[str]: Modified DataFrame, list of new feature columns
        """
        self._log("Adding adjacent sensor features.")
        pivot = df.pivot(index=time_col, columns=sensor_col, values=value_col)
        sensors = df[sensor_col].unique()

        for i in range(self.spatial_adj):
            down_col, up_col = f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}'
            down_map, down_dist = {}, {}
            up_map, up_dist = {}, {}

            for s in sensors:
                ds = self.downstream_sensor_dict.get(s, {})
                us = self.upstream_sensor_dict.get(s, {})

                down_map[s] = ds.get('downstream_sensor', [None]*self.spatial_adj)[i]
                down_dist[s] = ds.get('downstream_distance', [np.nan]*self.spatial_adj)[i]
                up_map[s] = us.get('upstream_sensor', [None]*self.spatial_adj)[i]
                up_dist[s] = us.get('upstream_distance', [np.nan]*self.spatial_adj)[i]

            # Vectorized lookup
            df[f'{down_col}_id'] = df[sensor_col].map(down_map)
            df[f'{up_col}_id'] = df[sensor_col].map(up_map)

            def safe_lookup(date, sid):
                try:
                    return np.nan if pd.isna(sid) else pivot.at[date, sid]
                except KeyError:
                    return np.nan

            df[down_col] = [safe_lookup(d, s) for d, s in zip(df[time_col], df[f'{down_col}_id'])]
            df[up_col] = [safe_lookup(d, s) for d, s in zip(df[time_col], df[f'{up_col}_id'])]

            if self.normalize_by_distance:
                df[down_col] = df[down_col] / 3.6 / df[sensor_col].map(down_dist)
                df[up_col] = df[up_col] / 3.6 / df[sensor_col].map(up_dist)

            df.drop(columns=[f'{down_col}_id', f'{up_col}_id'], inplace=True)
            df[down_col].fillna(self.fill_nans_value, inplace=True)
            df[up_col].fillna(self.fill_nans_value, inplace=True)
            self.new_columns += [down_col, up_col]
            self._log(f"Added {down_col}, {up_col}")

        return df, self.new_columns


class TemporalLagFeatureAdder:
    """
    Adds temporal lag features (absolute or relative) to a traffic dataset.
    """

    def __init__(self, lags=3, relative=False, fill_nans_value=-1, disable_logs=False):
        self.lags = lags
        self.relative = relative
        self.fill_nans_value = fill_nans_value
        self.disable_logs = disable_logs
        self.new_columns = []

    def _log(self, msg):
        if not self.disable_logs:
            logging.info(msg)

    def transform(self, df, sensor_col='sensor_id', value_col='value'):
        """
        Adds lag or relative lag features to the DataFrame.

        Returns:
            pd.DataFrame, list[str]: Modified DataFrame, list of new lag columns
        """
        epsilon = 1e-5
        self._log(f"Adding {'relative' if self.relative else 'absolute'} lags (lags={self.lags})")

        for i in range(1, self.lags + 1):
            col_name = f"{'relative_diff_lag' if self.relative else 'lag'}{i}"
            shifted = df.groupby(sensor_col)[value_col].shift(i)

            if self.relative:
                df[col_name] = (df[value_col] - shifted) / (shifted + epsilon)
            else:
                df[col_name] = shifted - df[value_col]

            df[col_name].fillna(self.fill_nans_value, inplace=True)
            self.new_columns.append(col_name)

        return df, self.new_columns

class TrafficFlowDataProcessing:

    """
    A class to process and prepare traffic flow data for time-series prediction.
    This includes methods for loading, cleaning, feature engineering, and data splitting.
    """

    def __init__(self, data_path='../data', file_name='estimated_average_speed_selected_timestamps-edited-new.csv', adj_sensors_file_path=None,
                 column_names=None, lags=20, spatial_adj=None, horizon=15,
                 correlation_threshold=0.01, columns_to_use=None, lags_are_relative=False,
                 time_col_name='sensor_time_min', sensor_id_col_name='sensor_uid', gman_correction_as_target=False,smoothing_on_train_set_only = False,
                 df_gman=None, random_state=69,use_weather_features=True,flag_outliers=False,adj_sensor_columns_exist=False,lag_columns_exist = False):
        """
        Initialize data processing parameters.

        Parameters:
        - file_path (str): Path to the CSV file with traffic data.
        - column_names (list): Column names for the data file. If None, defaults to colnames.
        - lags (int): Number of temporal lag features to generate.
        - spatial_adj (int): Number of spatial adjucent sensors to include.
        - correlation_threshold (float): Minimum correlation for feature selection.
        - random_state (int): Seed for reproducible train-test split.
        """
        self.data_path = data_path
        self.file_name = file_name
        self.horizon = horizon
        self.file_path = os.path.join(self.data_path, self.file_name)
        self.csv_column_names = column_names if column_names else colnames  # Use default if None
        self.lags = lags
        self.smoothing_on_train_set_only = smoothing_on_train_set_only
        self.spatial_adj = spatial_adj
        self.adj_sensor_columns_exist = adj_sensor_columns_exist
        self.lag_columns_exist = lag_columns_exist
        if self.spatial_adj is not None:
            with open(os.path.join(data_path,'downstream_dict.json'), 'r') as f:
                self.downstream_sensor_dict = json.load(f)
            with open(os.path.join(data_path,'upstream_dict.json'), 'r') as f:
                self.upstream_sensor_dict = json.load(f)
                
        self.use_weather_features = use_weather_features

        self.correlation_threshold = correlation_threshold
        self.lags_are_relative = lags_are_relative
        self.random_state = random_state
        self.flag_outliers = flag_outliers
        self.adj_sensors_file_path = adj_sensors_file_path
        # The col name after transforming the column sensor_uid
        self.sensor_id_col_name = sensor_id_col_name
        self.time_col_name = time_col_name
        self.previous_plotted_sensors = set()
        self.gman_correction_as_target = gman_correction_as_target
        self.df = None
        self.df_orig = None
        self.test_size = None
        self.df_gman = df_gman
        self.cache = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.columns_weather = ['Storm_relative_helicity_height_above_ground_layer',
                                'U-Component_Storm_Motion_height_above_ground_layer',
                                'Wind_speed_gust_surface', 'u-component_of_wind_maximum_wind',
                                'v-component_of_wind_height_above_ground']
        if columns_to_use is None:
            # Define features dynamically based on existing columns (if features are not selected based on correlation)
            self.columns_to_use = [
                self.sensor_id_col_name, 'value', 'longitude', 'latitude',
                'Storm_relative_helicity_height_above_ground_layer',
                'U-Component_Storm_Motion_height_above_ground_layer',
                'Wind_speed_gust_surface', 'u-component_of_wind_maximum_wind',
                'v-component_of_wind_height_above_ground',
                       'Per_cent_frozen_precipitation_surface',
       'Precipitable_water_entire_atmosphere_single_layer',
       'Precipitation_rate_surface_3_Hour_Average',
       'Total_precipitation_surface_3_Hour_Accumulation',
       'Categorical_Rain_surface_3_Hour_Average',
       'Categorical_Freezing_Rain_surface_3_Hour_Average',
       'Categorical_Ice_Pellets_surface_3_Hour_Average',
       'Categorical_Snow_surface_3_Hour_Average',
       'Convective_Precipitation_Rate_surface_3_Hour_Average',
       'Convective_precipitation_surface_3_Hour_Accumulation',
       'V-Component_Storm_Motion_height_above_ground_layer',
       'Geopotential_height_highest_tropospheric_freezing',
       'Relative_humidity_highest_tropospheric_freezing', 'Ice_cover_surface',
       'Snow_depth_surface',
       'Water_equivalent_of_accumulated_snow_depth_surface',
       'u-component_of_wind_height_above_ground',
            ]
            self.columns_to_use = set(self.columns_to_use)
            self.columns_to_use = list(self.columns_to_use)
            # self.columns_to_use.remove('incremental_id')
            self.columns_to_use.append('date')

        else:
            self.columns_to_use = columns_to_use

    def load_data(self, nrows=None, sort_by_datetime=True):
        """Loads and preprocesses raw data, converting 'date' column to datetime and sorting by it."""

        # load df with pyarrow for faster loading, if nrows is specified, then you can't use pyarrow
        if nrows is None:
            self.df = pd.read_csv(self.file_path, names=self.csv_column_names)
        else:
            logging.info(f"selecting df with {nrows} nrows.")
            self.df = pd.read_csv(
                self.file_path, names=self.csv_column_names, nrows=nrows)
        self.df_orig = self.df.copy()
        rows_that_target_var_is_nan = self.df['value'].isna().sum()
        if rows_that_target_var_is_nan > 0:
            warnings.warn(
                f'Target variable (column "value") is Nan {rows_that_target_var_is_nan} times.')
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        if sort_by_datetime:
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            logging.info(f"df got sorted by datetime.")

        if self.df_gman is not None:
            self.add_gman_predictions()
            
            
    def drop_weather_features(self):
        logging.info("Dropping weather-related features.")
        
        # Drop from columns_to_use FIRST
        before = set(self.columns_to_use)
        self.columns_to_use = [col for col in self.columns_to_use if col not in self.columns_weather]
        after = set(self.columns_to_use)
        dropped = list(before - after)
        logging.info(f"Updated columns_to_use. Dropped: {dropped}")

        # Now drop from X_train / X_test
        if self.X_train is not None and self.X_test is not None:
            self.X_train.drop(columns=dropped, inplace=True, errors='ignore')
            self.X_test.drop(columns=dropped, inplace=True, errors='ignore')
            logging.info(f"Also dropped from X_train/X_test: {dropped}")
        else:
            logging.warning("X_train/X_test not defined at time of drop_weather_features().")

    def add_gman_predictions(self):
        """Merges gman data with the main DataFrame and drops rows with missing gman_predictions.

        Logs the total number of rows dropped, and the minimum and maximum dates for these dropped rows.
        """
        logging.info("Merging gman data.")

        # Ensure sensor_id is of categorical type for both DataFrames
        self.df['sensor_id'] = self.df['sensor_id'].astype('category')
        self.df_gman['sensor_id'] = self.df_gman['sensor_id'].astype(
            'category')

        # Set a multi-index on both DataFrames
        self.df = self.df.set_index(['date', 'sensor_id'])
        self.df_gman = self.df_gman.set_index(['date', 'sensor_id'])

        # Join the DataFrames (faster with indexed DataFrames)
        self.df = self.df.join(self.df_gman, how='left').reset_index()

        # Identify rows with missing gman_predictions
        missing_rows = self.df[self.df['gman_prediction'].isna()]
        if not missing_rows.empty:
            dropped_count = missing_rows.shape[0]
            min_date = missing_rows['date'].min()
            max_date = missing_rows['date'].max()
            logging.info(
                f"Dropping {dropped_count} rows with missing 'gman_prediction ({dropped_count/self.df['sensor_id'].nunique()} rows per sensor)'. "
                f"Date range: {min_date} to {max_date}."
            )
            # Drop rows where gman_predictions is NaN
            self.df = self.df.dropna(subset=['gman_prediction'])
        else:
            logging.info("No rows dropped for missing 'gman_predictions'.")

    def diagnose_extreme_changes(self, relative_threshold=0.7):
        """
        Identifies and logs extreme changes in speed between consecutive timestamps for each sensor.

        Args:
            relative_threshold (float): Threshold for flagging changes as extreme.
        """
        logging.info(
            f"Diagnosing extreme speed changes (for relative threshold: {relative_threshold*100}% of immediate change)...")

        self.df['relative_speed_change'] = self.df.groupby('sensor_id')[
            'value'].pct_change().abs()

        extreme_changes = self.df[self.df['relative_speed_change']
                                  > relative_threshold]

        num_extremes = extreme_changes.shape[0]
        total_points = self.df.shape[0]
        perc_extremes = (num_extremes / total_points) * 100

        logging.info(
            f"Total extreme changes: {num_extremes} ({perc_extremes:.2f}% of data)")

        # Cleanup
        self.df.drop(columns=['relative_speed_change'], inplace=True)

    def filter_extreme_changes(self, relative_threshold=0.7):
        """
        Filters out extreme relative changes between consecutive speeds.

        Args:
            relative_threshold (float): Threshold to flag changes as abnormal (default: 0.7).
        """

        logging.info(
            f"Filtering extreme speed changes (> {relative_threshold*100:.0f}% change).")

        self.df['relative_speed_change'] = self.df.groupby('sensor_id')[
            'value'].pct_change().abs()
        extreme_changes = self.df['relative_speed_change'] > relative_threshold

        logging.info(f"Extreme changes detected: {extreme_changes.sum()}")

        # Set outliers to NaN and interpolate
        self.df.loc[extreme_changes, 'value'] = np.nan
        self.df['value'] = self.df.groupby('sensor_id')['value'].transform(
            lambda x: x.interpolate().ffill().bfill())

        self.df.drop(columns=['relative_speed_change'], inplace=True)

    
   
    def smooth_speeds(self, window_size=3):
        logging.info("Applying rolling median smoothing to speed values.")
        
        if self.smoothing_on_train_set_only:
            print("Applying smoothing only on training set rows.")
            train_set = self.df.loc[self.df['test_set'] == False].copy()
            print(f'in train set,dates are from {train_set["date"].min()} to {train_set["date"].max()}')
            print(f'in train set, nr of sensors are {train_set["sensor_id"].nunique()}')
            self._apply_smoothing(self.df.loc[self.df['test_set'] == False],window_size)
            self._apply_smoothing(self.df_for_ML.loc[self.df['test_set'] == False],window_size)
        else:
            self._apply_smoothing(self.df,window_size)
            self._apply_smoothing(self.df_for_ML,window_size)
            
            
    def _apply_smoothing(self, df, window_size):
        
        print(f"Applying smoothing on dates from {df['date'].min()} to {df['date'].max()}")
        df['value'] = df.groupby('sensor_id')['value'].transform(
                lambda x: x.rolling(window=window_size, center=False, min_periods=1).median())
    
        
    

    def load_data_parquet(self,df=None, nrows=None):
        """Reads the parquet file and selects nr of rows (it's already sorted by datetime)."""
        
        if df is None:
            self.df = pd.read_parquet(self.file_path, engine='pyarrow')

            logging.info(f"Reading parquet file: {self.file_path}")
            self.df = pd.read_parquet(self.file_path, engine='pyarrow')
        else:
            self.df = df

        # Ensure that date and datetime columns are present and converted to datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['datetime'] = pd.to_datetime(self.df['date'])
        elif 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df['date'] = self.df['datetime']

        else:
            raise ValueError(
                "The 'date' or 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")

        logging.info(f"df['date'] converted to datetime.")
        self.df = self.df.sort_values(
            by=['date','sensor_id']).reset_index(drop=True)
        logging.info(f"df got sorted by datetime.")
        
        self.df['value_original'] = self.df['value']  

        if nrows is not None:
            nrows = int(nrows)
            self.df = self.df.iloc[:nrows]
        rows_that_target_var_is_nan = self.df['value'].isna().sum()
        if rows_that_target_var_is_nan > 0:
            warnings.warn(
                f'Target variable (column "value") is Nan {rows_that_target_var_is_nan} times.')

        if nrows is not None:
            nrows = int(nrows)
            self.df = self.df.iloc[:nrows]

        if self.df_gman is not None:
            self.add_gman_predictions()

    def preprocess_data(self, select_relevant_cols=False):
        """Run the data preprocessing pipeline to clean and prepare the data."""
        logging.info("Starting preprocessing of data.")
        self._map_sensor_ids()
        self._discard_sensor_uids_w_uncommon_nrows()
        # self._discard_misformatted_values()
        self._apply_clean_and_convert_to_float32()
        if select_relevant_cols:
            self._select_relevant_columns()
        else:
            self._select_fixed_columns()
        logging.info("Preprocessing completed.")

    def _map_sensor_ids(self):
        """Map unique sensor IDs to integers for categorical feature conversion."""
        logging.info("Mapping sensor IDs to integers.")

        nr_unique_sensors = self.df['sensor_id'].nunique()

        self.df[self.sensor_id_col_name] = self.df['sensor_id'].map(
            {s: i for i, s in enumerate(set(self.df['sensor_id']))})
        # Log the number of sensors enumerated
        logging.info(
            f"Enumerated {nr_unique_sensors} sensors, starting from 1.")
        # Ensure `sensor_uid` is still numeric
        self.df[self.sensor_id_col_name] = self.df[self.sensor_id_col_name].astype(
            int)

    def _discard_misformatted_values(self):
        """DEPRECATED. Use _clean_and_convert_to_float16() instead.
        This method is deprecated so that strings of type eg '101 . 315' are converted to 101.315 instead of being dropped."""
        """Remove rows with misformatted values and convert 'value' column to float. """
        logging.info("Discarding misformatted values in 'value' column.")

        # Count the number of rows before filtering
        initial_row_count = len(self.df)
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
        self.df = self.df.dropna(subset=['value'])
        # Count the number of rows after filtering
        final_row_count = len(self.df)

        # Log the number of rows dropped
        rows_dropped = initial_row_count - final_row_count
        logging.info(
            f"Discarded {rows_dropped} rows with misformatted values in 'value' column.")

    def _clean_and_convert_to_float32(self, value):
        """
        Cleans and converts a single value to float16 format.

        This method performs the following steps:
        - Checks if the input is already a float; if so, rounds it to two decimal places and converts it to float32.
        - If the input is a string, replaces spaces with dots and attempts to convert it to float16.
        - Handles misformatted or invalid values by returning None.

        Parameters:
        - value: The input value to be cleaned and converted (can be a float or a string).

        Returns:
        - np.float16: The cleaned and converted value.
        - None: If the value cannot be converted to float32.
        """

        try:
            # Check if value is already a float
            if isinstance(value, float):
                return np.float32(round(value, 2))
            # Replace space with a dot and convert to float16
            return np.float32(round(float(value.replace(' ', '.')), 2))
        except (ValueError, AttributeError):
            # If conversion fails, return None or NaN
            return None

    def _apply_clean_and_convert_to_float32(self):
        """
        Applies the `_clean_and_convert_to_float32` method to the 'value' column in the DataFrame.

        This method iterates over the 'value' column in `self.df`, applying the cleaning
        and conversion logic to ensure all values are properly formatted and converted
        to float16 format. Invalid or misformatted values are replaced with NaN.

        Raises:
        - AttributeError: If `self.df` does not have a 'value' column.

        Notes:
        - Ensure the DataFrame is loaded into `self.df` before calling this method.
        """
        logging.info("Cleaning and converting 'value' column to float32.")
        len_df_prev = len(self.df)
        self.df = self.df.dropna(subset=['value'])
        len_df_after = len(self.df)
        logging.info(
            f"Discarded {len_df_prev - len_df_after} rows with NaN values in 'value' column. (method _clean_and_convert_to_float32())")

        self.df['value'] = self.df['value'].apply(
            self._clean_and_convert_to_float32)

    def align_sensors_to_common_timeframe(self):
        """
        Align all sensors to the same timeframe based on the sensor with the fewest recordings.
        This ensures all sensors have data for the same timestamps.
        This method should be used if only a subset of the dataframe is to be read.
        """
        logging.info(
            "Aligning sensors to a common timeframe based on the sensor with the fewest recordings (becase a subset of ).")

        # Count the number of recordings for each sensor
        sensor_counts = self.df.groupby('sensor_id').size()

        # Identify the sensor with the fewest recordings
        min_recording_sensor = sensor_counts.idxmin()
        min_recording_count = sensor_counts.min()

        # Get the timeframe for the sensor with the fewest recordings
        common_timeframe = self.df[self.df['sensor_id']
                                   == min_recording_sensor]['datetime']
        min_time = common_timeframe.min()
        max_time = common_timeframe.max()

        # Filter the DataFrame to include only data within the common timeframe for all sensors
        original_row_count = len(self.df)
        self.df = self.df[(self.df['datetime'] >= min_time)
                          & (self.df['datetime'] <= max_time)]
        filtered_row_count = len(self.df)

        logging.info(
            f"Aligned all sensors to the common timeframe: {min_time} to {max_time}.")
        logging.info(
            f"Rows before alignment: {original_row_count}, Rows after alignment: {filtered_row_count}.")

    def _select_relevant_columns(self, method=None):
        """Filter out columns with low correlation to the target variable 'value'."""
        logging.info(
            f"Selecting relevant columns using correlation method: {method or 'default'}.")
        accepted_corr_method = ['pearson', 'spearman', 'kendall']
        df_dropped = self.df.drop(['sensor_id', 'date'], axis=1).dropna()
        df_dropped_clean = df_dropped.dropna()
        if method is None:
            correlations = df_dropped_clean.corr()['value'].abs()
        else:
            assert method in accepted_corr_method, f'The correlation method {method} (input variable) must be one of the following: {accepted_corr_method}.'
            correlations = df_dropped_clean.corr(method)['value'].abs()

        relevant_columns = correlations[correlations >=
                                        self.correlation_threshold].index
        logging.info(
            f'Selected relevant columns based on {method} correlation are now : {relevant_columns}')
        new_columns_to_use = [self.sensor_id_col_name,
                              'date'] + list(relevant_columns)
        self.columns_to_use = new_columns_to_use

    def _select_fixed_columns(self):
        """Select fixed columns based on the original notebook provided."""
        logging.info("Selecting fixed columns for modeling.")

   

    def convert_datetime(self):
        """Extract hour, day, and month from the datetime column and create new columns."""
        logging.info(
            "Extracting hour, day, and month from the datetime column.")
        if 'datetime' not in self.df.columns:
            raise ValueError(
                "The 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")

        # Create new columns
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day'] = self.df['datetime'].dt.dayofweek
        # self.df['month'] = self.df['datetime'].dt.month

        # Add new columns to `columns_to_use`
        # self.columns_to_use += ['hour', 'day', 'month']
        self.columns_to_use += ['hour', 'day']

    def create_target_variable_common_part(self):
        """Generate initial code for target variable."""
        logging.info("Creating target variable.")
        

        # Initial target is the delta speed prediction
        self.df_for_ML['target'] = self.df_for_ML.groupby(self.sensor_id_col_name)[
            'value'].shift(-self.horizon) - self.df_for_ML['value']
        self.df_for_ML['target_total_speed'] = self.df_for_ML.groupby(self.sensor_id_col_name)[
            'value'].shift(-self.horizon)
        self.df_for_ML['target_speed_delta'] = self.df_for_ML['target']

        # Add the gman prediction in the prediction horizon
        if self.df_gman is not None:
            self.df_for_ML['target_gman_prediction'] = self.df_for_ML.groupby(
                self.sensor_id_col_name)['gman_prediction'].shift(-self.horizon)
            # Update the columns to use with the new feature
            # if not self.gman_correction_as_target:
            self.columns_to_use += ['target_gman_prediction']

    def create_target_variable(self):
        """Generate target variable as speed delta based on specified horizon."""

        self.create_target_variable_common_part()
        self.df_for_ML = self.df_for_ML.dropna(subset=['target'])

    def create_target_variable_as_gman_error(self):
        """Generate target variable as gman error prediction. You need to add the error prediction to the gman prediction in order to get the new target prediction."""
        self.create_target_variable_common_part()
        self.df_for_ML['target'] = self.df_for_ML['target_total_speed'] - \
            self.df_for_ML['target_gman_prediction']
        self.df_for_ML = self.df_for_ML.dropna(subset=['target'])
        # check that the target is indeed the gman error
        target_is_gman_error_check = (self.df_for_ML['target_total_speed'] - (
            self.df_for_ML['target'] + self.df_for_ML['target_gman_prediction'])).sum()
        if target_is_gman_error_check != 0:
            raise ValueError(
                "The target variable is not the gman error. Please check the code.")
    
    def check_temporal_lags(self):
        """
        Verifies that only the expected temporal lag columns (standard or relative) are present.
        Drops any excess lag columns beyond self.lags in both self.df and self.df_for_ML.
        """
        logging.info("Checking temporal lag columns.")

        if self.lags_are_relative:
            prefix = "relative_diff_lag"
        else:
            prefix = "lag"

        expected_lag_cols = {f"{prefix}{i+1}" for i in range(self.lags)}

        for df_name in ['df', 'df_for_ML']:
            df_obj = getattr(self, df_name)
            if df_obj is None:
                continue

            existing_lag_cols = {col for col in df_obj.columns if col.startswith(prefix)}
            cols_to_drop = existing_lag_cols - expected_lag_cols

            if cols_to_drop:
                df_obj.drop(columns=list(cols_to_drop), inplace=True)
                setattr(self, df_name, df_obj)
                logging.info(f"{df_name}: Dropped extra lag columns: {sorted(cols_to_drop)}")

                # Clean from columns_to_use as well
                self.columns_to_use = [col for col in self.columns_to_use if col not in cols_to_drop]
            else:
                logging.info(f"{df_name}: No extra lag columns found.")
                logging.info("No extra lag columns found to drop.")    
            
    def check_adjucent_sensors(self):
        """
        Checks and drops excess adjacent sensor columns beyond self.spatial_adj.

        For example, if self.spatial_adj = 2, then 'downstream_sensor_3', 'upstream_sensor_3', etc.
        will be dropped from self.df_for_ML and removed from self.columns_to_use if present.
        """
        if not hasattr(self, 'spatial_adj') or self.spatial_adj is None:
            logging.warning("self.spatial_adj is not defined. Skipping check.")
            return

        max_allowed = self.spatial_adj
        columns_to_drop = []

        for col in self.df_for_ML.columns:
            match = re.match(r'(upstream|downstream)_sensor_(\d+)', col)
            if match:
                idx = int(match.group(2))
                if idx > max_allowed:
                    columns_to_drop.append(col)

        # Drop from DataFrame
        self.df_for_ML.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Also drop from columns_to_use if present
        self.columns_to_use = [col for col in self.columns_to_use if col not in columns_to_drop]

        logging.info(f"Dropped excess adjacent sensor columns: {columns_to_drop}")
    
    def find_outliers(self, lower_bound=0.01, upper_bound=0.99):
        """
        Adds a binary 'is_outlier' column marking extreme outliers in 'value'.

        Outliers are detected using percentiles from the training set.
        This column is added to df_for_ML, and the split X_train/X_test is updated to include it.
        """
        logging.info("Flagging outliers in training set.")

        # Compute percentiles from training data
        lower_bound_value = self.X_train['value'].quantile(lower_bound)
        upper_bound_value = self.X_train['value'].quantile(upper_bound)
        logging.info(f"Using outlier thresholds: lower={lower_bound_value}, upper={upper_bound_value}")

        # Add the is_outlier flag to df_for_ML
        self.df_for_ML['is_outlier'] = ((self.df_for_ML['value'] < lower_bound_value) | 
                                        (self.df_for_ML['value'] > upper_bound_value)).astype(int)

        # Add to feature list if not already
        if 'is_outlier' not in self.columns_to_use:
            self.columns_to_use.append('is_outlier')

        # Ensure X_train and X_test are updated with this column
        if 'test_set' in self.df_for_ML.columns:
            self.X_train['is_outlier'] = self.df_for_ML.loc[self.X_train.index, 'is_outlier'].values
            self.X_test['is_outlier'] = self.df_for_ML.loc[self.X_test.index, 'is_outlier'].values
        else:
            raise RuntimeError("Missing 'test_set' column in df_for_ML. You must split data before calling find_outliers().")

        logging.info(f"Flagged {self.df_for_ML['is_outlier'].sum()} outliers ({self.df_for_ML['is_outlier'].mean()*100:.2f}%) in total.")
        
    def prepare_data(self, select_relevant_cols=False, nrows=None, sort_by_datetime=True, use_weekend_var=True):
        """Run the full data preparation pipeline without splitting."""
        logging.info('Preparing the dataset')

        self.preprocess_data(select_relevant_cols=select_relevant_cols)
        self.convert_datetime()
        if use_weekend_var:
            self.add_weekend_columns()
        self.df_for_ML = self.df.copy()



    def add_bottleneck_columns(self, hour_start=6, hour_end=19, quantile_threshold=0.9, quantile_percentage=0.65):
        """
        Adds a binary 'is_congested' column indicating traffic congestion status.

        The congestion threshold is computed only using training data to avoid data leakage.
        Congestion is defined as speed below 65% of the 90th percentile of speeds between 6 AM and 7 PM.
            """
        logging.info(
            "Adding congestion ('is_congested') column based on training set thresholds.")
        congestion_thr = self.df_for_ML[(self.df_for_ML['hour'] >= hour_start) & (
            self.df_for_ML['hour'] <= hour_end) & (~self.df_for_ML['test_set'])].groupby('sensor_id')['value'].quantile(quantile_threshold)*quantile_percentage

        # Map the thresholds to each row in the original DataFrame based on sensor_id
        thresholds = self.df_for_ML['sensor_id'].map(congestion_thr)

        # Create a new column 'is_below_threshold' that is True if 'value' is less than the threshold for that sensor_id, False otherwise
        self.df_for_ML['is_congested'] = (
            self.df_for_ML['value'] < thresholds).astype(int)
        self.columns_to_use += ['is_congested']

    def split_data(self, test_size):
        """Split the data into training and testing sets based on defined features."""
        logging.info('Splitting the dataset.')
        logging.info(
            f"columns to use are: {self.columns_to_use}, in split_data method.")
        X = self.df_for_ML[self.columns_to_use]
        y = self.df_for_ML['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=self.random_state
        )

    

    def split_data_by_timestamps(self):
        """
        Splits the dataset into train and test sets strictly based on the 'test_set' flag.
        
        This ensures:
        - The train/test split is **exactly the same** as the one defined in `flag_train_test_set()`.
        - No new timestamp-based logic is introduced.
        """
        logging.info('Splitting dataset based on pre-flagged "test_set" column.')

        # Use the existing test_set flag
        train_mask = self.df_for_ML['test_set'] == False
        test_mask = self.df_for_ML['test_set'] == True

        # Apply the train-test split
        self.X_train = self.df_for_ML.loc[train_mask, self.columns_to_use].copy()
        self.X_test = self.df_for_ML.loc[test_mask, self.columns_to_use].copy()
        self.y_train = self.df_for_ML.loc[train_mask, 'target'].copy()
        self.y_test = self.df_for_ML.loc[test_mask, 'target'].copy()

        # Ensure consistency
        assert self.df.loc[self.X_test.index, 'test_set'].all(), "Mismatch: Some test rows were not flagged."
        assert not self.df.loc[self.X_train.index, 'test_set'].any(), "Mismatch: Some train rows were incorrectly flagged."
        assert self.df_for_ML.loc[self.X_test.index, 'test_set'].all(), "Mismatch in df_for_ML: Some test rows were not flagged."
        assert not self.df_for_ML.loc[self.X_train.index, 'test_set'].any(), "Mismatch in df_for_ML: Some train rows were incorrectly flagged."

        logging.info(
            f"Train set: {self.df_for_ML.loc[train_mask, 'date'].min()} to {self.df_for_ML.loc[train_mask, 'date'].max()}")
        logging.info(
            f"Test set: {self.df_for_ML.loc[test_mask, 'date'].min()} to {self.df_for_ML.loc[test_mask, 'date'].max()}")

    def add_time_column(self, sort_by_datetime=False):
        """Adds time column for the sensors."""
        self.df = self.df.sort_values(
            ['sensor_id', 'datetime']).reset_index(drop=True)

        # Add the 'time' column as the difference in minutes (dt)
        self.df[self.time_col_name] = self.df.groupby(
            'sensor_id')['datetime'].diff().dt.total_seconds() / 60
        # ensure the first value of each sensor is 0
        self.df[self.time_col_name].fillna(0, inplace=True)
        self.df[self.time_col_name] = self.df.groupby(
            'sensor_id')[self.time_col_name].cumsum()
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

    def add_weekend_columns(self):
        """Add columns indicating whether the date is a Saturday or Sunday."""
        logging.info("Adding weekend columns for Saturday and Sunday.")
        self.df['is_saturday'] = self.df['datetime'].dt.dayofweek == 5  # Saturday
        self.df['is_sunday'] = self.df['datetime'].dt.dayofweek == 6  # Sunday
        self.df['is_saturday'] = self.df['is_saturday'].astype(int)
        self.df['is_sunday'] = self.df['is_sunday'].astype(int)
        self.columns_to_use += ['is_saturday']
        self.columns_to_use += ['is_sunday']
        # logging.info(f"columns to use are: {self.columns_to_use}, after adding weekend")

    def add_hour_column(self):
        """Add a column with the hour of the day extracted from the datetime column."""
        logging.info("Adding hour column to the DataFrame.")
        if 'datetime' not in self.df.columns:
            raise ValueError(
                "The 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")

        self.df['hour'] = self.df['datetime'].dt.hour
        self.columns_to_use += ['hour']

    def _discard_sensor_uids_w_uncommon_nrows(self):
        """Remove rows where the sensor_uid group size is not equal to the most common size."""
        logging.info('Discarding sensor ids with uncommon nr of rows.')
        # Calculate the group sizes
        group_sizes = self.df.groupby(self.sensor_id_col_name).size()

        # Find the most common group size
        most_common_size = group_sizes.mode().iloc[0]

        # Identify sensor_uids with the most common size
        valid_sensors = group_sizes[group_sizes == most_common_size].index

        # Count sensors
        total_sensors = len(group_sizes)
        valid_sensor_count = len(valid_sensors)
        discarded_sensor_count = total_sensors - valid_sensor_count

        # Print details
        logging.info(f"Most common group size (mode): {most_common_size}")
        logging.info(f"Number of sensors with this size: {valid_sensor_count}")
        logging.info(f"Number of sensors discarded: {discarded_sensor_count}")

        # Filter the DataFrame to include only rows with valid sensor_uids
        initial_row_count = len(self.df)
        self.df = self.df[self.df[self.sensor_id_col_name].isin(valid_sensors)]
        final_row_count = len(self.df)

        logging.info(
            f"Discarded {initial_row_count - final_row_count} rows with uncommon sensor_uid group sizes.")

    def get_train_test_split(self, reset_index=False):
        """Return train and test data splits."""

        if reset_index:
            for dataset in [self.X_train, self.X_test, self.y_train, self.y_test]:
                dataset.reset_index(drop=True, inplace=True)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def assert_valid_filter_method_and_params(self, filter_method, filter_params):
        """
        Validates the filtering method and parameters for preprocessing.

        Args:
            filter_method (str or None): The filtering method ('threshold', 'smoothing', or None).
            filter_params (dict or None): Parameters for the filtering method.

        Raises:
            AssertionError: If provided method or parameters are invalid.
        """

        valid_methods = [None, 'threshold', 'smoothing']
        assert filter_method in valid_methods, \
            f"Invalid filter_method '{filter_method}'. Choose from {valid_methods}."

        if filter_method is None:
            assert filter_params is None or filter_params == {}, \
                "filter_params must be None or empty if filter_method is None."

        elif filter_method == 'threshold':
            assert filter_params is not None and 'relative_threshold' in filter_params and len(filter_params) == 1, \
                "For 'threshold' method, filter_params must be {'relative_threshold': float}."
            assert isinstance(filter_params['relative_threshold'], float), \
                "relative_threshold must be a float (e.g., 0.7)."

        elif filter_method == 'smoothing':
            assert filter_params is not None and 'window_size' in filter_params and len(filter_params) == 1, \
                "For smoothing method, filter_params must be {'window_size': int}."
            assert isinstance(filter_params['window_size'], int), \
                "window_size must be an integer."
                
                
    
    def add_adjacent_sensors(self, normalize_by_distance=False,fill_nans_value=-1):
        """
        Adds features for adjacent sensors to the DataFrame in a vectorized manner.
        
        For each sensor and for each spatial adjacency (up to self.spatial_adj), this method:
        - Creates mapping dictionaries to look up the downstream/upstream sensor IDs (and distances)
        - Pivots the DataFrame so that each row (date) allows fast lookup of a sensor's value
        - Uses a vectorized lookup to assign the adjacent sensor's value for each row.
        
        Parameters:
        normalize_by_distance (bool): If True, divides the adjacent sensor's value by its corresponding distance.
        """
        logging.info("Starting to add adjacent sensor features (optimized).")
        
        
        
        # Initialize empty columns for each spatial adjacency and update the feature list.
        for i in range(self.spatial_adj):
            self.df_for_ML[f'downstream_sensor_{i+1}'] = np.nan
            self.df_for_ML[f'upstream_sensor_{i+1}'] = np.nan
            self.columns_to_use += [f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}']
        
        logging.info("Pivoting the DataFrame for fast lookups by date and sensor_id.")
        # Pivot the DataFrame for fast date-sensor lookups.
        pivot = self.df_for_ML.pivot(index='date', columns='sensor_id', values='value')
        logging.info(f"Pivot table shape: {pivot.shape}")
        
        unique_sensors = self.df_for_ML['sensor_id'].unique()
        logging.info(f"Number of unique sensors: {len(unique_sensors)}")
        
        # Process each spatial index (e.g. 1st, 2nd, etc. adjacent sensor)
        for i in range(self.spatial_adj):
            
            logging.info(f"Processing adjacent sensor index: {i+1}")
            # Build mapping dictionaries for this spatial level.
            downstream_map = {}
            downstream_dist_map = {}
            upstream_map = {}
            upstream_dist_map = {}
            
            for sensor in unique_sensors:
                # Downstream mapping
                if sensor in self.downstream_sensor_dict:
                    ds_list = self.downstream_sensor_dict[sensor]['downstream_sensor']
                    ds_dist_list = self.downstream_sensor_dict[sensor]['downstream_distance']
                    if i < len(ds_list) and ds_list[i] is not None:
                        downstream_map[sensor] = ds_list[i]
                        downstream_dist_map[sensor] = ds_dist_list[i]
                    else:
                        downstream_map[sensor] = None
                        downstream_dist_map[sensor] = np.nan
                else:
                    downstream_map[sensor] = None
                    downstream_dist_map[sensor] = np.nan

                # Upstream mapping
                if sensor in self.upstream_sensor_dict:
                    us_list = self.upstream_sensor_dict[sensor]['upstream_sensor']
                    us_dist_list = self.upstream_sensor_dict[sensor]['upstream_distance']
                    if i < len(us_list) and us_list[i] is not None:
                        upstream_map[sensor] = us_list[i]
                        upstream_dist_map[sensor] = us_dist_list[i]
                    else:
                        upstream_map[sensor] = None
                        upstream_dist_map[sensor] = np.nan
                else:
                    upstream_map[sensor] = None
                    upstream_dist_map[sensor] = np.nan
            
            logging.info(f"Mapping dictionaries built for adjacent sensor index {i+1}.")
            
            # Map the adjacent sensor IDs to a temporary column using dictionary mapping via a lambda.
            self.df_for_ML[f'downstream_sensor_id_{i+1}'] = self.df_for_ML['sensor_id'].map(
                lambda x: downstream_map.get(x, None)
            )
            self.df_for_ML[f'upstream_sensor_id_{i+1}'] = self.df_for_ML['sensor_id'].map(
                lambda x: upstream_map.get(x, None)
            )
            
            # Prepare arrays for vectorized lookup.
            dates = self.df_for_ML['date'].values
            ds_ids = self.df_for_ML[f'downstream_sensor_id_{i+1}'].values
            us_ids = self.df_for_ML[f'upstream_sensor_id_{i+1}'].values
            
            # Define a safe lookup function to get the adjacent sensor's value from the pivot table.
            def safe_lookup(date, sensor_id):
                try:
                    if pd.isna(sensor_id):
                        return np.nan
                    return pivot.at[date, sensor_id]
                except KeyError:
                    return np.nan
            
            logging.info(f"Performing vectorized lookup for adjacent sensor index {i+1}.")
            # Lookup downstream and upstream values row-wise.
            self.df_for_ML[f'downstream_sensor_{i+1}'] = [safe_lookup(d, s) for d, s in zip(dates, ds_ids)]
            self.df_for_ML[f'upstream_sensor_{i+1}'] = [safe_lookup(d, s) for d, s in zip(dates, us_ids)]
            
            # Normalize by distance if requested.
            if normalize_by_distance:
                self.df_for_ML[f'downstream_sensor_{i+1}'] = self.df_for_ML[f'downstream_sensor_{i+1}'] / 3.6
                logging.info(f"Normalizing values by distance for adjacent sensor index {i+1}.")
                self.df_for_ML[f'downstream_sensor_{i+1}'] = (
                    self.df_for_ML[f'downstream_sensor_{i+1}'] /
                    self.df_for_ML['sensor_id'].map(lambda x: downstream_dist_map.get(x, np.nan))
                )
                self.df_for_ML[f'upstream_sensor_{i+1}'] = self.df_for_ML[f'upstream_sensor_{i+1}'] / 3.6
                self.df_for_ML[f'upstream_sensor_{i+1}'] = (
                    self.df_for_ML[f'upstream_sensor_{i+1}'] /
                    self.df_for_ML['sensor_id'].map(lambda x: upstream_dist_map.get(x, np.nan))
                )
            
            # Drop the temporary adjacent sensor ID columns.
            self.df_for_ML.drop(columns=[f'downstream_sensor_id_{i+1}', f'upstream_sensor_id_{i+1}'], inplace=True)
            logging.info(f"Finished processing adjacent sensor index {i+1}.")
            self.df_for_ML[f'downstream_sensor_{i+1}'].fillna(fill_nans_value, inplace=True)
            self.df_for_ML[f'upstream_sensor_{i+1}'].fillna(fill_nans_value, inplace=True)
        
        logging.info("Finished adding all adjacent sensor features (optimized).")
        
    


    def flag_train_test_set(self, test_size):
        """
        Flags each row as part of the train or test set **per sensor** based on timestamps.
        This ensures the last X% of timepoints per sensor are marked as test.
        """
        logging.info("Flagging train/test set based on per-sensor timestamps.")

        def flag_group(df_sensor):
            timestamps = df_sensor['date'].sort_values().unique()
            split_index = int(len(timestamps) * (1 - test_size))
            test_start = timestamps[split_index]
            return df_sensor['date'] >= test_start

        for df in [self.df, self.df_for_ML]:
            if df is not None:
                df['test_set'] = df.groupby('sensor_id', group_keys=False).apply(flag_group).reset_index(drop=True)

        logging.info("Train/test flags assigned per sensor.")
        
        
        
    def _add_lags_to_df(self, df, kind='absolute', fill_nans_value=-1):
        """
        Adds temporal lags (absolute or relative) to the given dataframe.
        
        Parameters:
        - df (pd.DataFrame): DataFrame to modify (either self.df or self.df_for_ML)
        - kind (str): 'absolute' or 'relative'
        - fill_nans_value (float): value to use for NaNs
        """
        epsilon = 1e-5  # for relative division

        for i in range(1, self.lags + 1):
            if kind == 'absolute':
                lag_col = f'lag{i}'
                df[lag_col] = df.groupby(self.sensor_id_col_name)['value'].shift(i) - df['value']
            elif kind == 'relative':
                lag_col = f'relative_diff_lag{i}'
                shifted = df.groupby(self.sensor_id_col_name)['value'].shift(i)
                df[lag_col] = (df['value'] - shifted) / (shifted + epsilon)
            else:
                raise ValueError("Invalid kind: choose 'absolute' or 'relative'")

            df[lag_col].fillna(fill_nans_value, inplace=True)
            if lag_col not in self.columns_to_use:
                self.columns_to_use.append(lag_col)
    
    def add_temporal_lags(self, fill_nans_value=-1):
        logging.info("Adding absolute temporal lags to both df and df_for_ML.")
        for df in [self.df, self.df_for_ML]:
            self._add_lags_to_df(df, kind='absolute', fill_nans_value=fill_nans_value)
    
    def add_relative_temporal_lags(self, fill_nans_value=-1):
        logging.info("Adding relative temporal lags to both df and df_for_ML.")
        for df in [self.df, self.df_for_ML]:
            self._add_lags_to_df(df, kind='relative', fill_nans_value=fill_nans_value)
        
        
    def get_clean_train_test_split(self, df=None,test_size=0.5, nrows=None, select_relevant_cols=False, add_congestion=True,normalize_by_distance=False,
                                   hour_start=6, hour_end=19, quantile_threshold=0.9, quantile_percentage=0.65, lower_bound=0.01, upper_bound=0.99,
                                   use_weekend_var=True, reset_index=False, print_nans=True, filter_method=None, filter_params=None):
        """Split the data into train and test sets and optionally add a 'train_set' flag."""
        # Validate filtering method and parameters
        
        self.assert_valid_filter_method_and_params(
            filter_method, filter_params)
        self.test_size = test_size
        self.load_data_parquet(nrows=nrows,df=df)
        print('Data loaded as parquet')
        if filter_params is not None and 'relative_threshold' in filter_params:
            relative_threshold = filter_params['relative_threshold']

        else:
            relative_threshold = 0.7
        self.diagnose_extreme_changes(relative_threshold)
        # Apply preprocessing


        self.prepare_data(nrows=nrows, select_relevant_cols=select_relevant_cols,
                          use_weekend_var=use_weekend_var)

        # self.split_data_by_timestamps(test_size=test_size)
        # self.df['test_set'] = False
        # self.df_for_ML['test_set'] = False
        # self.df.loc[self.X_test.index, 'test_set'] = True
        # self.df_for_ML.loc[self.X_test.index, 'test_set'] = True
        if 'test_set' not in self.df.columns:
            self.flag_train_test_set(test_size)
        else:
            logging.info("Using existing 'test_set' column from input DataFrame.")

        # Apply filtering if specified
        if filter_method:
            if filter_params is None:
                filter_params = {}

            if filter_method == 'threshold':
                self.filter_extreme_changes(**filter_params)
            elif filter_method == 'smoothing':
                self.smooth_speeds(**filter_params)
            else:
                raise ValueError(
                    f"filter_method '{filter_method}' is invalid. Use 'threshold', 'smoothing', or None.")
        
        
        if self.gman_correction_as_target:
            self.create_target_variable_as_gman_error()
        else:
            self.create_target_variable()
        
        if self.lag_columns_exist:
            self.check_temporal_lags()
        else:   
            if self.lags_are_relative:
                self.add_relative_temporal_lags()
            else:
                self.add_temporal_lags() 
         
        if self.check_adjucent_sensors:
            self.check_adjucent_sensors()
        else:   
            if self.spatial_adj is not None:    
                self.add_adjacent_sensors(normalize_by_distance=normalize_by_distance)
            
                
        if add_congestion:
            self.add_bottleneck_columns(hour_start=hour_start, hour_end=hour_end,
                                        quantile_threshold=quantile_threshold, quantile_percentage=quantile_percentage)
            # After adding congestion, you must split again to ensure it’s included
            
        if 'test_set' in self.df_for_ML.columns:
            logging.info("Splitting data based on 'test_set' flag.")
            self.X_train = self.df_for_ML[~self.df_for_ML['test_set']]
            self.X_test = self.df_for_ML[self.df_for_ML['test_set']]
            self.y_train = self.df_for_ML[~self.df_for_ML['test_set']]['target']
            self.y_test = self.df_for_ML[self.df_for_ML['test_set']]['target']
        else:
            logging.info("Splitting data based on timestamps using test_size parameter of get_clean_train_test_split.")
            self.split_data_by_timestamps()
            
        # Add outliers now — adds 'is_outlier' to columns_to_use
        if self.flag_outliers:
            self.find_outliers(lower_bound=lower_bound, upper_bound=upper_bound)

        # Now remove weather cols AFTER outliers were added to columns_to_use
        if not self.use_weather_features:
            self.drop_weather_features()

        # Reassign the feature sets using FINAL columns_to_use
        self.X_train = self.X_train[self.columns_to_use]
        self.X_test = self.X_test[self.columns_to_use]

                
        # else:
        #     # If no filtering is specified, at least filter extreme changes with a relative threshold of 0.7
        #     self.filter_extreme_changes(relative_threshold=0.7)
         
        

        if print_nans:
            logging.info(
                f"number of nans in X_train: {self.X_train.isna().sum().sum()}")
            logging.info(
                f"number of nans in X_test: {self.X_test.isna().sum().sum()}")
            logging.info(
                f"number of nans in y_train: {self.y_train.isna().sum().sum()}")
            logging.info(
                f"number of nans in y_test: {self.y_test.isna().sum().sum()}")

        return self.get_train_test_split(reset_index=reset_index)

    def validate_target_variable(self):
        """Check if the target variable was computed correctly."""
        logging.info("Validating target variable...")

        # Copy the dataframe to avoid modifying the original
        df_test = self.df_for_ML.copy()

        # Ensure sorting is correct
        df_test = df_test.sort_values(by=[self.sensor_id_col_name, 'datetime'])

        # Compute expected target values based on target type
        if self.gman_correction_as_target:
            # Expected target should be (true future speed - GMAN-predicted speed)
            df_test['expected_target'] = (
                df_test.groupby(self.sensor_id_col_name)[
                    'value'].shift(-self.horizon)
                - df_test.groupby(self.sensor_id_col_name)['gman_prediction'].shift(-self.horizon)
            )
        else:
            # Expected target should be (true future speed - current speed)
            df_test['expected_target'] = (
                df_test.groupby(self.sensor_id_col_name)[
                    'value'].shift(-self.horizon)
                - df_test['value']
            )

        # Compare expected vs actual target
        df_test['target_correct'] = df_test['target'] == df_test['expected_target']

        # If all values match, return success
        if df_test['target_correct'].all():
            logging.info("All target values are correct!")
            return True
        else:
            logging.warning(
                "Some target values are incorrect! Inspecting incorrect rows...")

            incorrect_rows = df_test[df_test['target_correct'] == False]

            return False

    def plot_sensor_train_test_split(self, test_size=0.5):
        """
        Randomly sample a sensor and plot its train-test split.
        Ensures a previously plotted sensor is not plotted again consecutively.
        """
        if self.df is None:
            raise ValueError(
                "Dataframe is empty. Please load and preprocess the data first.")
        if self.test_size is None:
            raise ValueError(
                "No test size specified. Please load and preprocess the data first.")

        # Add time column
        self.add_time_column(sort_by_datetime=False)

        # Get unique sensors and ensure no duplicate consecutive plotting
        unique_sensors = set(self.df[self.sensor_id_col_name].unique())
        available_sensors = unique_sensors - self.previous_plotted_sensors
        if not available_sensors:
            self.previous_plotted_sensors.clear()  # Reset once all sensors are plotted
            available_sensors = unique_sensors

        # Randomly sample a sensor
        sensor_to_plot = random.choice(list(available_sensors))
        self.previous_plotted_sensors.add(sensor_to_plot)

        # Filter data for the selected sensor
        sensor_data = self.df[self.df[self.sensor_id_col_name]
                              == sensor_to_plot].copy()
        sensor_to_plot_name = sensor_to_plot

        # Assign train/test flags chronologically
        sensor_data = sensor_data.sort_values(self.time_col_name)
        cutoff = int(len(sensor_data) * (1 - test_size))
        sensor_data['train_set'] = False
        sensor_data.iloc[:cutoff,
                         sensor_data.columns.get_loc('train_set')] = True

        # Plot the train-test split
        plt.figure(figsize=(12, 6))
        plt.scatter(
            sensor_data[self.time_col_name], sensor_data['value'],
            c=sensor_data['train_set'].map({True: 'blue', False: 'red'}),
            alpha=0.6
        )
        plt.title(f"Train-Test Split Visualization for {sensor_to_plot_name}")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Speed [kph]")
        plt.grid()
        plt.show()
   
   
   
   # def split_data_by_timestamps(self, test_size):
    #     """
    #     Splits the dataset into train and test sets based on timestamps 
    #     to ensure consistency across different models.
    #     """
    #     logging.info('Splitting the dataset based on timestamps.')

    #     # Get the total number of unique timestamps
    #     unique_timestamps = self.df_for_ML['date'].sort_values().unique()
    #     num_timestamps = len(unique_timestamps)

    #     # Determine the split index based on the test_size ratio
    #     split_index = int(num_timestamps * (1 - test_size))

    #     # Get the first test timestamp
    #     first_test_timestamp = unique_timestamps[split_index]
    #     last_test_timestamp = unique_timestamps[-1]
    #     first_train_timestamp = unique_timestamps[0]
    #     last_train_timestamp = unique_timestamps[split_index - 1]

    #     # Create the train and test sets
    #     train_mask = self.df_for_ML['date'] < first_test_timestamp
    #     test_mask = self.df_for_ML['date'] >= first_test_timestamp

    #     self.X_train = self.df_for_ML.loc[train_mask,
    #                                       self.columns_to_use].copy()
    #     self.X_test = self.df_for_ML.loc[test_mask, self.columns_to_use].copy()
    #     self.y_train = self.df_for_ML.loc[train_mask, 'target'].copy()
    #     self.y_test = self.df_for_ML.loc[test_mask, 'target'].copy()

    #     logging.info(
    #         f"Train set timestamps: {first_train_timestamp} to {last_train_timestamp}")
    #     logging.info(
    #         f"Test set timestamps: {first_test_timestamp} to {last_test_timestamp}")
        
   
 # def smooth_speeds(self, window_size=3):
    #     logging.info("Applying rolling median smoothing to speed values.")
        
    #     # Ensure original values are stored before modifying 'value'

    #     if self.smoothing_on_train_set_only:

    #         # Apply smoothing only on training set rows
    #         train_mask_df = self.df['test_set'] == False
    #         train_mask_df_ml = self.df_for_ML['test_set'] == False

    #         logging.info("Applying smoothing only on training set rows.")
    #         self._apply_smoothing(self.df, train_mask_df, window_size)
    #         self._apply_smoothing(self.df_for_ML, train_mask_df_ml, window_size)
    #     else:
    #         logging.info("Applying smoothing to the entire dataset.")
    #         self._apply_smoothing(self.df, None, window_size)
    #         self._apply_smoothing(self.df_for_ML, None, window_size)

    #     logging.info("Smoothing completed.")
        
    # def _apply_smoothing(self, df, mask, window_size):
    #     """
    #     Applies rolling median smoothing to the 'value' column.

    #     Args:
    #         df (pd.DataFrame): The dataframe on which smoothing is to be applied.
    #         mask (pd.Series or None): A boolean mask for selecting rows. If None, applies to entire df.
    #         window_size (int): The size of the rolling window.
    #     """
    #     if mask is not None:
    #         print(f'Applying smoothing on training only (len: {len(df[mask])})')
    #         print(f'Applying smoothing on nr of sensors: {df[mask]["sensor_id"].nunique()}')
    #         print(f"Applying smoothing on dates from {df.loc[mask,'date'].min()} to {df.loc[mask,'date'].max()}")
    #         df.loc[mask, 'value'] = df.loc[mask].groupby('sensor_id')['value'].transform(
    #             lambda x: x.rolling(window=window_size, center=False, min_periods=1).median())
    #     else:
    #         print(f'Applying smoothing on entire dataset (len: {len(df)})')
    #         print(f'Applying smoothing on nr of sensors: {df["sensor_id"].nunique()}')
    #         print(f"Applying smoothing on dates from {df['date'].min()} to {df['date'].max()}")
    #         df['value'] = df.groupby('sensor_id')['value'].transform(
    #             lambda x: x.rolling(window=window_size, center=False, min_periods=1).median())

        #df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
        
   
 # def add_temporal_lags(self, fill_nans_value=-1):
    #     """Add temporal lag features, creating 'lag1', 'lag2', ..., 'lagN' features for model input."""
    #     logging.info("Adding temporal lags.")
    #     for i in range(self.lags):
    #         lag_col_name = f'lag{i+1}'
    #         # self.df_for_ML[lag_col_name] = self.df_for_ML.groupby(self.sensor_id_col_name)['value'].shift(i+1) - self.df['value']
    #         self.df[lag_col_name] = self.df.groupby(self.sensor_id_col_name)[
    #             'value'].shift(i+1) - self.df['value']
    #         self.df[lag_col_name].fillna(fill_nans_value, inplace=True)
    #         self.columns_to_use += [lag_col_name]

    # def add_relative_temporal_lags(self, fill_nans_value=-1):
    #     """
    #     Add temporal lag features as relative percentage differences compared to past values.

    #     This method generates features representing how much the current speed has changed
    #     relative to the speeds at previous timestamps (lags).

    #     Formula:
    #         relative_diff_lag_i = (value_t - value_{t-i}) / (value_{t-i} + epsilon)

    #     Parameters:
    #         fill_nans_value (float, optional): Value to fill NaNs introduced by lagging. Defaults to -1.

    #     Adds columns:
    #         relative_diff_lag1, relative_diff_lag2, ..., relative_diff_lagN
    #     """
    #     logging.info("Adding relative percentage temporal lag features.")

    #     epsilon = 1e-5  # Small number to avoid division by zero

    #     for i in range(1, self.lags + 1):
    #         lag_col_name = f'relative_diff_lag{i}'

    #         # Shift values by i to get past speeds
    #         shifted_values = self.df.groupby(self.sensor_id_col_name)[
    #             'value'].shift(i)

    #         # Compute relative difference
    #         self.df[lag_col_name] = (
    #             self.df['value'] - shifted_values) / (shifted_values + epsilon)

    #         # Handle NaNs introduced by shifting
    #         self.df[lag_col_name].fillna(fill_nans_value, inplace=True)

    #         # Append new column to the list of features to use
    #         self.columns_to_use.append(lag_col_name)     
        
# def add_adjacent_sensors(self, normalize_by_distance=False):
    #     """
    #     Optimized function to add adjacent sensor features without using nested loops.

    #     Instead of iterating through every sensor, we vectorize operations by using 
    #     pandas `.merge()` and `.map()`.
    #     """
    #     logging.info("Adding adjacent sensor features efficiently.")

    #     # Convert dictionary into DataFrame for fast lookup
    #     downstream_df = pd.DataFrame.from_dict(self.downstream_sensor_dict, orient="index")
    #     upstream_df = pd.DataFrame.from_dict(self.upstream_sensor_dict, orient="index")

    #     # Expand downstream/upstream sensor lists into separate columns
    #     downstream_df = downstream_df.apply(pd.Series.explode)
    #     upstream_df = upstream_df.apply(pd.Series.explode)

    #     # Merge downstream/upstream sensor speeds into main DataFrame
    #     for i in range(self.spatial_adj):
    #         logging.info(f"Processing adjacent sensor {i+1}")

    #         # Downstream Mapping
    #         self.df_for_ML[f'downstream_sensor_{i+1}'] = self.df_for_ML['sensor_id'].map(
    #             downstream_df['downstream_sensor']
    #         )
            
    #         # Upstream Mapping
    #         self.df_for_ML[f'upstream_sensor_{i+1}'] = self.df_for_ML['sensor_id'].map(
    #             upstream_df['upstream_sensor']
    #         )

    #         # Merge speed values using a left join on `sensor_id` and `date`
    #         self.df_for_ML = self.df_for_ML.merge(
    #             self.df_for_ML[['sensor_id', 'date', 'value']],
    #             left_on=[f'downstream_sensor_{i+1}', 'date'],
    #             right_on=['sensor_id', 'date'],
    #             how='left',
    #             suffixes=('', f'_downstream_{i+1}')
    #         )

    #         self.df_for_ML = self.df_for_ML.merge(
    #             self.df_for_ML[['sensor_id', 'date', 'value']],
    #             left_on=[f'upstream_sensor_{i+1}', 'date'],
    #             right_on=['sensor_id', 'date'],
    #             how='left',
    #             suffixes=('', f'_upstream_{i+1}')
    #         )

    #         # Rename merged columns
    #         self.df_for_ML.rename(
    #             columns={
    #                 'value_downstream_' + str(i+1): f'downstream_sensor_{i+1}_speed',
    #                 'value_upstream_' + str(i+1): f'upstream_sensor_{i+1}_speed',
    #             },
    #             inplace=True
    #         )

    #         # Drop unnecessary columns
    #         self.df_for_ML.drop(columns=[f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}'], inplace=True)

    #         # Normalize by distance if needed
    #         if normalize_by_distance:
    #             # Convert speed from kph to m/s
    #             #self.df_for_ML[f'downstream_sensor_{i+1}_speed'] /= 3.6

    #             # Normalize by distance
    #             #self.df_for_ML[f'downstream_sensor_{i+1}_speed'] /= downstream_df['downstream_distance']
                
                
    #             # Convert speed from kph to m/s
    #             #self.df_for_ML[f'upstream_sensor_{i+1}_speed'] /= 3.6

    #             # Normalize by distance
    #             #self.df_for_ML[f'upstream_sensor_{i+1}_speed'] /= downstream_df['upstream_distance']
    #             self.df_for_ML[f'downstream_sensor_{i+1}_speed'] /= downstream_df['downstream_distance']
    #             self.df_for_ML[f'upstream_sensor_{i+1}_speed']/3.6 /= upstream_df['upstream_distance']

    #         self.columns_to_use.extend([f'downstream_sensor_{i+1}_speed', f'upstream_sensor_{i+1}_speed'])

    #     logging.info("Efficiently added adjacent sensor features.")
   
    # def add_adjacent_sensors(self, normalize_by_distance=False):
    #     """
    #     Adds features for adjacent sensors to the DataFrame in a vectorized manner.
        
    #     For each sensor and for each spatial adjacency (up to self.spatial_adj), this method:
    #     - Creates mapping dictionaries to look up the downstream/upstream sensor IDs (and distances)
    #     - Pivots the DataFrame so that each row (date) allows fast lookup of a sensor's value
    #     - Uses a vectorized lookup to assign the adjacent sensor's value for each row.
        
    #     Parameters:
    #     - normalize_by_distance (bool): If True, the adjacent sensor's value is divided by its
    #     corresponding distance.
    #     """
    #     logging.info("Starting to add adjacent sensor features (optimized).")
        
    #     # Initialize empty columns for each spatial adjacency and update the feature list.
    #     for i in range(self.spatial_adj):
    #         self.df_for_ML[f'downstream_sensor_{i+1}'] = np.nan
    #         self.df_for_ML[f'upstream_sensor_{i+1}'] = np.nan
    #         self.columns_to_use += [f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}']
        
    #     logging.info("Pivoting the DataFrame for fast lookups by date and sensor_id.")
    #     # Pivot the DataFrame for fast date-sensor lookups.
    #     pivot = self.df_for_ML.pivot(index='date', columns='sensor_id', values='value')
    #     logging.info(f"Pivot table shape: {pivot.shape}")
        
    #     # Process each spatial index (e.g. 1st, 2nd, … adjacent sensor)
    #     unique_sensors = self.df_for_ML['sensor_id'].unique()
    #     logging.info(f"Number of unique sensors: {len(unique_sensors)}")
        
    #     for i in range(self.spatial_adj):
    #         logging.info(f"Processing adjacent sensor index: {i+1}")
    #         # Build mapping dictionaries for this spatial level.
    #         downstream_map = {}
    #         downstream_dist_map = {}
    #         upstream_map = {}
    #         upstream_dist_map = {}
            
    #         for sensor in unique_sensors:
    #             # Downstream mapping
    #             if sensor in self.downstream_sensor_dict:
    #                 ds_list = self.downstream_sensor_dict[sensor]['downstream_sensor']
    #                 ds_dist_list = self.downstream_sensor_dict[sensor]['downstream_distance']
    #                 if i < len(ds_list) and ds_list[i] is not None:
    #                     downstream_map[sensor] = ds_list[i]
    #                     downstream_dist_map[sensor] = ds_dist_list[i]
    #                 else:
    #                     downstream_map[sensor] = None
    #                     downstream_dist_map[sensor] = np.nan
    #             else:
    #                 downstream_map[sensor] = None
    #                 downstream_dist_map[sensor] = np.nan

    #             # Upstream mapping
    #             if sensor in self.upstream_sensor_dict:
    #                 us_list = self.upstream_sensor_dict[sensor]['upstream_sensor']
    #                 us_dist_list = self.upstream_sensor_dict[sensor]['upstream_distance']
    #                 if i < len(us_list) and us_list[i] is not None:
    #                     upstream_map[sensor] = us_list[i]
    #                     upstream_dist_map[sensor] = us_dist_list[i]
    #                 else:
    #                     upstream_map[sensor] = None
    #                     upstream_dist_map[sensor] = np.nan
    #             else:
    #                 upstream_map[sensor] = None
    #                 upstream_dist_map[sensor] = np.nan
            
    #         logging.info(f"Mapping dictionaries built for adjacent index {i+1}.")
            
    #         # Map the adjacent sensor IDs to the rows in df_for_ML.
    #         self.df_for_ML[f'downstream_sensor_id_{i+1}'] = self.df_for_ML['sensor_id'].map(downstream_map)
    #         self.df_for_ML[f'upstream_sensor_id_{i+1}'] = self.df_for_ML['sensor_id'].map(upstream_map)
            
    #         # Prepare arrays for vectorized lookup.
    #         dates = self.df_for_ML['date'].values
    #         ds_ids = self.df_for_ML[f'downstream_sensor_id_{i+1}'].values
    #         us_ids = self.df_for_ML[f'upstream_sensor_id_{i+1}'].values
            
    #         # Define a safe lookup function to get the adjacent sensor's value.
    #         def safe_lookup(date, sensor_id):
    #             try:
    #                 if pd.isna(sensor_id):
    #                     return np.nan
    #                 return pivot.at[date, sensor_id]
    #             except KeyError:
    #                 return np.nan
            
        #     logging.info(f"Performing vectorized lookup for downstream and upstream values at adjacent index {i+1}.")
        #     # Lookup downstream and upstream values row-wise.
        #     self.df_for_ML[f'downstream_sensor_{i+1}'] = [safe_lookup(d, s) for d, s in zip(dates, ds_ids)]
        #     self.df_for_ML[f'upstream_sensor_{i+1}'] = [safe_lookup(d, s) for d, s in zip(dates, us_ids)]
            
        #     # Normalize by distance if requested.
        #     if normalize_by_distance:
        #         logging.info(f"Normalizing values by distance for adjacent index {i+1}.")
        #         self.df_for_ML[f'downstream_sensor_{i+1}'] = (
        #             (self.df_for_ML[f'downstream_sensor_{i+1}']/3.6) /
        #             self.df_for_ML['sensor_id'].map(downstream_dist_map)
        #         )
        #         self.df_for_ML[f'upstream_sensor_{i+1}'] = (
        #             (self.df_for_ML[f'upstream_sensor_{i+1}']/3.6) /
        #             self.df_for_ML['sensor_id'].map(upstream_dist_map)
        #         )
            
        #     # Drop the temporary adjacent sensor ID columns.
        #     self.df_for_ML.drop(columns=[f'downstream_sensor_id_{i+1}', f'upstream_sensor_id_{i+1}'], inplace=True)
        #     logging.info(f"Finished processing adjacent sensor index {i+1}.")
        
        # self.df_for_ML[f'downstream_sensor_{i+1}'].fillna(-1, inplace=True)
        # self.df_for_ML[f'upstream_sensor_{i+1}'].fillna(-1, inplace=True)
        
        # logging.info("Finished adding all adjacent sensor features (optimized).")
    
    
    # def flag_train_test_set(self, test_size):
    #     """
    #     Flags each row as being part of the train or test set without splitting the dataset.

    #     This ensures that:
    #     - The test set is determined based on timestamps before any transformations.
    #     - The 'test_set' column can be used for filtering/smoothing before splitting.

    #     Parameters:
    #     - test_size (float): The proportion of data to be used as the test set.
    #     """
    #     logging.info("Flagging train/test set based on timestamps.")

    #     # Get unique timestamps, sorted
    #     unique_timestamps = self.df['date'].sort_values().unique()
    #     num_timestamps = len(unique_timestamps)

    #     # Determine the split index for test set
    #     split_index = int(num_timestamps * (1 - test_size))

    #     # Identify the first test timestamp
    #     first_test_timestamp = unique_timestamps[split_index]

    #     # Assign 'test_set' flag in both dataframes
    #     for df in [self.df, self.df_for_ML]:
    #         if df is not None:
    #             df['test_set'] = df['date'] >= first_test_timestamp

    #     logging.info(f"Test set starts from timestamp: {first_test_timestamp}.")
        
# def add_adjacent_sensors(self, normalize_by_distance=False):
#         """
#         Adds features for adjacent sensors to the DataFrame.

#         The method adds as features the speed of adjacent sensors for each timestamp.

#         Parameters:
#         - normalize_by_distance (bool): If True, the speed is normalized by the (total) distance between sensors.
#         """
#         logging.info("Adding adjacent sensor features.")
        
#         # Create adjacency columns
#         for i in range(0, self.spatial_adj):
#             self.df_for_ML[f'downstream_sensor_{i+1}'] = np.nan
#             self.columns_to_use += [f'downstream_sensor_{i+1}']
#             self.df_for_ML[f'upstream_sensor_{i+1}'] = np.nan
#             self.columns_to_use += [f'upstream_sensor_{i+1}']
            
#         logging.info(f"Initialized adjacent sensor features.")
        
#         for sensor in self.df_for_ML['sensor_id'].unique():
#             logging.info(f"Processing sensor adjucency of: {sensor}.")
#             downstream_sensors = self.downstream_sensor_dict[sensor]['downstream_sensor']
#             downstream_distances = self.downstream_sensor_dict[sensor]['downstream_distance']
#             upstream_sensors = self.upstream_sensor_dict[sensor]['upstream_sensor']
#             upstream_distances = self.upstream_sensor_dict[sensor]['upstream_distance']
            
#             for i in range(0, self.spatial_adj):
#                 logging.info(f"Processing sensor adjucency of: {sensor}, downstream sensor: {i}.")
#                 for date in self.df_for_ML['date'].unique():
#                     downsteam_value = self.df_for_ML[(self.df_for_ML['sensor_id'] == downstream_sensors[i]) & (self.df_for_ML['date'] == date)]['value'].values
#                     downstream_distance = downstream_distances[i]
#                     upsteam_value = self.df_for_ML[(self.df_for_ML['sensor_id'] == upstream_sensors[i]) & (self.df_for_ML['date'] == date)]['value'].values
#                     upstream_distance = upstream_distances[i]
#                     if normalize_by_distance:
#                         downsteam_value = downsteam_value / downstream_distance
#                         upsteam_value = upsteam_value / upstream_distance
#                     self.df_for_ML.loc[(self.df_for_ML['sensor_id'] == sensor) & (self.df_for_ML['date'] == date), f'downstream_sensor_{i+1}'] = downsteam_value
#                     self.df_for_ML.loc[(self.df_for_ML['sensor_id'] == sensor) & (self.df_for_ML['date'] == date), f'upstream_sensor_{i+1}'] = upsteam_value






        
        
    ####### Second Version of add_adjacent_sensors ####### (infinite loop maybe)
    # def add_adjacent_sensors_old(self, nr_of_adj_sensors=2, value_if_no_adjacent=-1, normalize_by_distance=True):
    #     """
    #     Adds adjacent sensor speeds as new features.
    #     If normalize_by_distance is True, speeds are divided by distance.
        
    #     Parameters:
    #     - nr_of_adj_sensors (int): Number of adjacent sensors to consider upstream and downstream.
    #     - value_if_no_adjacent (float): Value to use if an adjacent sensor is missing (default -1).
    #     - normalize_by_distance (bool): If True, speeds are divided by distance; otherwise, raw speed values are used.
    #     """
    #     logging.info(f"Adding {nr_of_adj_sensors} adjacent sensors' speeds as features (Old Version).")
        
    #     # Load adjacency data
    #     adj_df = pd.read_csv(self.adj_sensors_file_path, sep=";")
        
    #     # Build adjacency graph (chaining connections correctly)
    #     adjacency_dict = {}
    #     for _, row in adj_df.iterrows():
    #         sensor = row['point_dgl_loc']
    #         connected_sensor = row['conn_points_dgl_loc']
    #         distance = row['distance']
            
    #         if sensor not in adjacency_dict:
    #             adjacency_dict[sensor] = {'upstream': [], 'downstream': []}
            
    #         adjacency_dict[sensor]['downstream'].append((connected_sensor, distance))
    #         adjacency_dict[connected_sensor] = adjacency_dict.get(connected_sensor, {'upstream': [], 'downstream': []})
    #         adjacency_dict[connected_sensor]['upstream'].append((sensor, distance))
        
    #     logging.info("Finished building adjacency graph.")
        
    #     def find_nth_adjacent(sensor_id, direction, depth=1):
    #         """
    #         Recursively finds the nth adjacent sensor by following adjacency links.
    #         """
    #         if depth > nr_of_adj_sensors or sensor_id not in adjacency_dict:
    #             return None, None
            
    #         if adjacency_dict[sensor_id][direction]:
    #             next_sensor, distance = adjacency_dict[sensor_id][direction][0]
    #             if depth == 1:
    #                 return next_sensor, distance
    #             return find_nth_adjacent(next_sensor, direction, depth - 1)
    #         return None, None
        
    #     def get_adjacent_speed(row, direction):
    #         """
    #         Retrieves the speed values of adjacent sensors at the same timestamp.
    #         """
    #         sensor_id = row['sensor_id']
    #         timestamp = row['date']
            
    #         values = []
    #         for depth in range(1, nr_of_adj_sensors + 1):
    #             adj_sensor, dist = find_nth_adjacent(sensor_id, direction, depth)
    #             if adj_sensor is None:
    #                 values.append(value_if_no_adjacent)
    #             else:
    #                 adj_value = self.df_for_ML.loc[
    #                     (self.df_for_ML['sensor_id'] == adj_sensor) & (self.df_for_ML['date'] == timestamp), 'value']
    #                 if not adj_value.empty:
    #                     value = adj_value.values[0] / dist if normalize_by_distance else adj_value.values[0]
    #                     values.append(value)
    #                 else:
    #                     values.append(value_if_no_adjacent)
            
    #         return values
        
    #     logging.info("Starting computation of adjacent sensor speeds.")
        
    #     self.df_for_ML[[f'upstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speed(row, 'upstream')), axis=1
    #     )
    #     self.df_for_ML[[f'downstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speed(row, 'downstream')), axis=1
    #     )
        
    #     logging.info("Finished adding adjacent sensor speeds (Old Version).")
    
    # def add_adjacent_sensors(self, nr_of_adj_sensors=2, value_if_no_adjacent=-1, normalize_by_distance=True):
    #     """
    #     Adds adjacent sensor speeds as new features.
    #     If normalize_by_distance is True, speeds are divided by distance.
        
    #     Parameters:
    #     - nr_of_adj_sensors (int): Number of adjacent sensors to consider upstream and downstream.
    #     - value_if_no_adjacent (float): Value to use if an adjacent sensor is missing (default -1).
    #     - normalize_by_distance (bool): If True, speeds are divided by distance; otherwise, raw speed values are used.
    #     """
    #     logging.info(f"Adding {nr_of_adj_sensors} adjacent sensors' speeds as features.")
        
    #     # Load adjacency data
    #     adj_df = pd.read_csv(self.adj_sensors_file_path, sep=";")
        
    #     adjacency_dict = {}
    #     for _, row in adj_df.iterrows():
    #         sensor = row['point_dgl_loc']
    #         connected_sensor = row['conn_points_dgl_loc']
    #         distance = row['distance']
            
    #         if sensor not in adjacency_dict:
    #             adjacency_dict[sensor] = {'upstream': [], 'downstream': []}
            
    #         adjacency_dict[sensor]['downstream'].append((connected_sensor, distance))
    #         adjacency_dict[connected_sensor] = adjacency_dict.get(connected_sensor, {'upstream': [], 'downstream': []})
    #         adjacency_dict[connected_sensor]['upstream'].append((sensor, distance))
        
    #     logging.info("Finished building adjacency graph.")
        
    #     def find_nth_adjacent(sensor_id, direction, depth=1):
    #         if depth > nr_of_adj_sensors or sensor_id not in adjacency_dict:
    #             return None, None
            
    #         if adjacency_dict[sensor_id][direction]:
    #             next_sensor, distance = adjacency_dict[sensor_id][direction][0]
    #             if depth == 1:
    #                 return next_sensor, distance
    #             return find_nth_adjacent(next_sensor, direction, depth - 1)
    #         return None, None
        
    #     def get_adjacent_speeds(sensor_id, timestamp, direction):
    #         values = []
    #         for depth in range(1, nr_of_adj_sensors + 1):
    #             adj_sensor, dist = find_nth_adjacent(sensor_id, direction, depth)
    #             if adj_sensor is None:
    #                 values.append(value_if_no_adjacent)
    #             else:
    #                 speed = self.df_for_ML.loc[
    #                     (self.df_for_ML['sensor_id'] == adj_sensor) & (self.df_for_ML['date'] == timestamp), 'value']
    #                 if not speed.empty:
    #                     values.append(speed.values[0] / dist if normalize_by_distance else speed.values[0])
    #                 else:
    #                     values.append(value_if_no_adjacent)
    #         return values
        
    #     logging.info("Starting computation of adjacent sensor speeds.")
        
    #     self.df_for_ML[[f'upstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speeds(row['sensor_id'], row['date'], 'upstream')), axis=1
    #     )
    #     self.df_for_ML[[f'downstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speeds(row['sensor_id'], row['date'], 'downstream')), axis=1
    #     )
        
    #     logging.info("Finished adding adjacent sensor speeds.")

        
####### First Version of add_adjacent_sensors ####### (works only for one adjacent sensor)
        
    # def add_adjacent_sensors_old(self, nr_of_adj_sensors=2, value_if_no_adjacent=-1, normalize_by_distance=True):
    #     """
    #     Adds adjacent sensor speeds as new features.
    #     If normalize_by_distance is True, speeds are divided by distance.
        
    #     Parameters:
    #     - nr_of_adj_sensors (int): Number of adjacent sensors to consider upstream and downstream.
    #     - value_if_no_adjacent (float): Value to use if an adjacent sensor is missing (default -1).
    #     - normalize_by_distance (bool): If True, speeds are divided by distance; otherwise, raw speed values are used.
    #     """
    #     logging.info(f"Adding {nr_of_adj_sensors} adjacent sensors' speeds as features (Old Version).")
        
    #     # Load adjacency data
    #     adj_df = pd.read_csv(self.adj_sensors_file_path, sep=";")
        
    #     # Build adjacency dictionary
    #     adjacency_dict = {}
    #     for _, row in adj_df.iterrows():
    #         sensor = row['point_dgl_loc']
    #         connected_sensor = row['conn_points_dgl_loc']
    #         distance = row['distance']
            
    #         if sensor not in adjacency_dict:
    #             adjacency_dict[sensor] = {'upstream': [], 'downstream': []}
    #         if connected_sensor not in adjacency_dict:
    #             adjacency_dict[connected_sensor] = {'upstream': [], 'downstream': []}
            
    #         # Assign upstream and downstream relationships
    #         adjacency_dict[sensor]['downstream'].append((connected_sensor, distance))
    #         adjacency_dict[connected_sensor]['upstream'].append((sensor, distance))
        
    #     # Sort by distance (ensures closest sensors are considered first)
    #     for sensor in adjacency_dict:
    #         adjacency_dict[sensor]['upstream'].sort(key=lambda x: x[1])
    #         adjacency_dict[sensor]['downstream'].sort(key=lambda x: x[1])
        
    #     # Function to get adjacent sensor speeds
    #     def get_adjacent_speed(row, direction):
    #         sensor_id = row['sensor_id']
    #         timestamp = row['date']
            
    #         if sensor_id not in adjacency_dict:
    #             return [value_if_no_adjacent] * nr_of_adj_sensors
            
    #         adjacent_sensors = adjacency_dict[sensor_id][direction][:nr_of_adj_sensors]
    #         values = []
    #         for adj_sensor, dist in adjacent_sensors:
    #             adj_value = self.df_for_ML.loc[
    #                 (self.df_for_ML['sensor_id'] == adj_sensor) & (self.df_for_ML['date'] == timestamp), 'value'
    #             ]
    #             if not adj_value.empty:
    #                 value = adj_value.values[0] / dist if normalize_by_distance else adj_value.values[0]
    #                 values.append(value)
    #             else:
    #                 values.append(value_if_no_adjacent)
            
    #         values += [value_if_no_adjacent] * (nr_of_adj_sensors - len(values))
    #         return values
        
    #     # Apply function
    #     self.df_for_ML[[f'upstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speed(row, 'upstream')), axis=1
    #     )
    #     self.df_for_ML[[f'downstream_{i+1}' for i in range(nr_of_adj_sensors)]] = self.df_for_ML.apply(
    #         lambda row: pd.Series(get_adjacent_speed(row, 'downstream')), axis=1
    #     )
        
    #     logging.info("Finished adding adjacent sensor speeds (Old Version).")
    
    # def add_adjacent_sensors(self, nr_of_adj_sensors=2, value_if_no_adjacent=-1, normalize_by_distance=True):
    #     """
    #     Adds adjacent sensor speeds as new features.
    #     If normalize_by_distance is True, speeds are divided by distance.
        
    #     Parameters:
    #     - nr_of_adj_sensors (int): Number of adjacent sensors to consider upstream and downstream.
    #     - value_if_no_adjacent (float): Value to use if an adjacent sensor is missing (default -1).
    #     - normalize_by_distance (bool): If True, speeds are divided by distance; otherwise, raw speed values are used.
    #     """
    #     logging.info(f"Adding {nr_of_adj_sensors} adjacent sensors' speeds as features.")
        
    #     # Load adjacency data
    #     adj_df = pd.read_csv(self.adj_sensors_file_path, sep=";")
        
    #     # Build adjacency dictionary
    #     adjacency_dict = {}
    #     for _, row in adj_df.iterrows():
    #         sensor = row['point_dgl_loc']
    #         connected_sensor = row['conn_points_dgl_loc']
    #         distance = row['distance']
            
    #         if sensor not in adjacency_dict:
    #             adjacency_dict[sensor] = {'upstream': [], 'downstream': []}
    #         if connected_sensor not in adjacency_dict:
    #             adjacency_dict[connected_sensor] = {'upstream': [], 'downstream': []}
            
    #         adjacency_dict[sensor]['downstream'].append((connected_sensor, distance))
    #         adjacency_dict[connected_sensor]['upstream'].append((sensor, distance))
    #     logging.info("Finished building adjacency dictionary.")
    #     for sensor in adjacency_dict:
    #         adjacency_dict[sensor]['upstream'].sort(key=lambda x: x[1])
    #         adjacency_dict[sensor]['downstream'].sort(key=lambda x: x[1])
        
    #     speed_lookup = self.df_for_ML.set_index(['sensor_id', 'date'])['value'].to_dict()
    #     logging.info("Finished building speed lookup table.")
    #     def get_adjacent_speeds_optimized(sensor_id, timestamp, direction):
    #         if sensor_id not in adjacency_dict:
    #             return [value_if_no_adjacent] * nr_of_adj_sensors
            
    #         adjacent_sensors = adjacency_dict[sensor_id][direction][:nr_of_adj_sensors]
    #         values = []
    #         for adj_sensor, dist in adjacent_sensors:
    #             speed = speed_lookup.get((adj_sensor, timestamp), None)
    #             if speed is None:
    #                 values.append(value_if_no_adjacent)
    #             else:
    #                 values.append(speed / dist if normalize_by_distance else speed)
            
    #         values += [value_if_no_adjacent] * (nr_of_adj_sensors - len(values))
    #         return values
        
    #     self.df_for_ML[[f'upstream_{i+1}' for i in range(nr_of_adj_sensors)]] = pd.DataFrame(
    #         self.df_for_ML.apply(lambda row: get_adjacent_speeds_optimized(row['sensor_id'], row['date'], 'upstream'), axis=1).tolist(),
    #         index=self.df_for_ML.index
    #     )
    #     logging.info("Finished adding upstream speeds.")
    #     self.df_for_ML[[f'downstream_{i+1}' for i in range(nr_of_adj_sensors)]] = pd.DataFrame(
    #         self.df_for_ML.apply(lambda row: get_adjacent_speeds_optimized(row['sensor_id'], row['date'], 'downstream'), axis=1).tolist(),
    #         index=self.df_for_ML.index
    #     )
        
    #     logging.info("Finished adding adjacent sensor speeds.")

    
                
   