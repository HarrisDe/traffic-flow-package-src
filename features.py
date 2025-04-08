
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
from .helper_utils import *
import pickle
import time
import json
import re 
# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # You can set this to DEBUG, WARNING, etc. as needed
)




class AdjacentSensorFeatureAdder(LoggingMixin):
    def __init__(self,
                 sensor_dict_path='../data',
                 spatial_adj=5,
                 normalize_by_distance=True,
                 datetime_col='datetime',
                 value_col='value',
                 sensor_col='sensor_id',
                 fill_nans_value=-1,
                 disable_logs=False):
        super().__init__(disable_logs)
        self.sensor_dict_path = sensor_dict_path
        self.downstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'downstream_dict.json')))
        self.upstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'upstream_dict.json')))
        self.spatial_adj = spatial_adj
        self.normalize_by_distance = normalize_by_distance
        self.fill_nans_value = fill_nans_value
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.sensor_col = sensor_col
        self.new_columns = []

    def transform(self, df, current_smoothing=None, prev_smoothing=None):
        self._log("Adding adjacent sensor features.")
        
        if self.spatial_adj < 1:
            self._log("No adjacent sensors to add. Skipping.")
            return df, []
    
        
        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        sensors = df[self.sensor_col].unique()
        
        
        # Drop any excess previously computed adjacent features
        for direction in ['upstream', 'downstream']:
            existing_cols = [col for col in df.columns if col.startswith(f'{direction}_sensor_') and not col.endswith('_id')]
            expected_cols = [f'{direction}_sensor_{i+1}' for i in range(self.spatial_adj)]
            to_drop = list(set(existing_cols) - set(expected_cols))
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
                self._log(f"Dropped excess {direction} columns: {to_drop}")

        for i in range(self.spatial_adj):
            down_col, up_col = f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}'
            if down_col in df.columns and up_col in df.columns and current_smoothing == prev_smoothing:
                self._log(f"Skipping {down_col} and {up_col}, they already exist in the df (max value of downstream: {df[down_col].max()}).")
                self.new_columns += [down_col, up_col]
                continue
            down_map, down_dist = {}, {}
            up_map, up_dist = {}, {}

            for s in sensors:
                ds = self.downstream_sensor_dict.get(s, {})
                us = self.upstream_sensor_dict.get(s, {})

                down_map[s] = ds.get('downstream_sensor', [None]*self.spatial_adj)[i]
                down_dist[s] = ds.get('downstream_distance', [np.nan]*self.spatial_adj)[i]
                up_map[s] = us.get('upstream_sensor', [None]*self.spatial_adj)[i]
                up_dist[s] = us.get('upstream_distance', [np.nan]*self.spatial_adj)[i]

            df[f'{down_col}_id'] = df[self.sensor_col].map(down_map)
            df[f'{up_col}_id'] = df[self.sensor_col].map(up_map)

            def safe_lookup(date, sid):
                try:
                    return np.nan if pd.isna(sid) else pivot.at[date, sid]
                except KeyError:
                    return np.nan

            df[down_col] = [safe_lookup(d, s) for d, s in zip(df[self.datetime_col], df[f'{down_col}_id'])]
            df[up_col] = [safe_lookup(d, s) for d, s in zip(df[self.datetime_col], df[f'{up_col}_id'])]

            if self.normalize_by_distance:
                df[down_col] = df[down_col] / 3.6 / df[self.sensor_col].map(down_dist)
                df[up_col] = df[up_col] / 3.6 / df[self.sensor_col].map(up_dist)

            df.drop(columns=[f'{down_col}_id', f'{up_col}_id'], inplace=True)
            df[down_col].fillna(self.fill_nans_value, inplace=True)
            df[up_col].fillna(self.fill_nans_value, inplace=True)
            self.new_columns += [down_col, up_col]
            self._log(f"Added {down_col}, {up_col}")

        return df, self.new_columns
    
    
class AdjacentSensorFeatureAdderOptimal(LoggingMixin):
    def __init__(self,
                 sensor_dict_path='../data',
                 spatial_adj=5,
                 normalize_by_distance=True,
                 datetime_col='datetime',
                 value_col='value',
                 sensor_col='sensor_id',
                 fill_nans_value=-1,
                 disable_logs=False):
        super().__init__(disable_logs)
        self.sensor_dict_path = sensor_dict_path
        self.downstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'downstream_dict.json')))
        self.upstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'upstream_dict.json')))
        self.spatial_adj = spatial_adj
        self.normalize_by_distance = normalize_by_distance
        self.fill_nans_value = fill_nans_value
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.sensor_col = sensor_col
        self.new_columns = []

    def transform(self, df, current_smoothing=None, prev_smoothing=None):
        self._log("Adding adjacent sensor features.")

        if self.spatial_adj is None:
            self._log("No adjacent sensors to add. Skipping.")
            return df, []
        if self.spatial_adj < 1:
            self._log("No adjacent sensors to add. Skipping.")
            return df, []

        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        pivot_stacked = pivot.stack().to_frame(self.value_col).rename_axis([self.datetime_col, self.sensor_col]).sort_index()

        # Drop excess previously computed adjacent features
        for direction in ['upstream', 'downstream']:
            existing_cols = [col for col in df.columns if col.startswith(f'{direction}_sensor_') and not col.endswith('_id')]
            expected_cols = [f'{direction}_sensor_{i+1}' for i in range(self.spatial_adj)]
            to_drop = list(set(existing_cols) - set(expected_cols))
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
                self._log(f"Dropped excess {direction} columns: {to_drop}")

        for i in range(self.spatial_adj):
            down_col, up_col = f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}'
            if down_col in df.columns and up_col in df.columns and current_smoothing == prev_smoothing:
                self._log(f"Skipping {down_col} and {up_col}, they already exist in the df (max value of downstream: {df[down_col].max()}).")
                self.new_columns += [down_col, up_col]
                continue

            # Generate maps
            down_map = {
                s: self.downstream_sensor_dict.get(s, {}).get('downstream_sensor', [None] * self.spatial_adj)[i]
                for s in df[self.sensor_col].unique()
            }
            up_map = {
                s: self.upstream_sensor_dict.get(s, {}).get('upstream_sensor', [None] * self.spatial_adj)[i]
                for s in df[self.sensor_col].unique()
            }

            # Map to get sensor ids
            df[f'{down_col}_id'] = df[self.sensor_col].map(down_map)
            df[f'{up_col}_id'] = df[self.sensor_col].map(up_map)

            # Create tuples for lookup and perform reindexing
            down_values = pivot_stacked[self.value_col].reindex(
                list(zip(df[self.datetime_col], df[f'{down_col}_id']))).values
            up_values = pivot_stacked[self.value_col].reindex(
                list(zip(df[self.datetime_col], df[f'{up_col}_id']))).values

            df[down_col] = down_values
            df[up_col] = up_values

            # Normalize
            if self.normalize_by_distance:
                down_dist_map = {
                    s: self.downstream_sensor_dict.get(s, {}).get('downstream_distance', [np.nan] * self.spatial_adj)[i]
                    for s in df[self.sensor_col].unique()
                }
                up_dist_map = {
                    s: self.upstream_sensor_dict.get(s, {}).get('upstream_distance', [np.nan] * self.spatial_adj)[i]
                    for s in df[self.sensor_col].unique()
                }
                df[down_col] = df[down_col] / 3.6 / df[self.sensor_col].map(down_dist_map)
                df[up_col] = df[up_col] / 3.6 / df[self.sensor_col].map(up_dist_map)

            # Cleanup
            df.drop(columns=[f'{down_col}_id', f'{up_col}_id'], inplace=True)
            df[down_col].fillna(self.fill_nans_value, inplace=True)
            df[up_col].fillna(self.fill_nans_value, inplace=True)
            self.new_columns += [down_col, up_col]
            self._log(f"Added {down_col}, {up_col}")

        return df, self.new_columns

# class AdjacentSensorFeatureAdder(LoggingMixin):
#     def __init__(self,
#                  sensor_dict_path='../data',
#                  spatial_adj=5,
#                  normalize_by_distance=True,
#                  datetime_col='datetime',
#                  value_col='value',
#                  sensor_col='sensor_id',
#                  fill_nans_value=-1,
#                  disable_logs=False):
#         super().__init__(disable_logs)
#         self.sensor_dict_path = sensor_dict_path
#         self.downstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'downstream_dict.json')))
#         self.upstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, 'upstream_dict.json')))
#         self.spatial_adj = spatial_adj
#         self.normalize_by_distance = normalize_by_distance
#         self.fill_nans_value = fill_nans_value
#         self.datetime_col = datetime_col
#         self.value_col = value_col
#         self.sensor_col = sensor_col
#         self.new_columns = []

#     def transform(self, df, current_smoothing=None, prev_smoothing=None):
#         self._log("Adding adjacent sensor features.")

#         if self.spatial_adj < 1:
#             self._log("No adjacent sensors to add. Skipping.")
#             return df, []

#         pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
#         pivot_stacked = pivot.stack().to_frame(self.value_col).rename_axis([self.datetime_col, self.sensor_col]).sort_index()

#         # Drop excess previously computed adjacent features
#         for direction in ['upstream', 'downstream']:
#             existing_cols = [col for col in df.columns if col.startswith(f'{direction}_sensor_') and not col.endswith('_id')]
#             expected_cols = [f'{direction}_sensor_{i+1}' for i in range(self.spatial_adj)]
#             to_drop = list(set(existing_cols) - set(expected_cols))
#             if to_drop:
#                 df.drop(columns=to_drop, inplace=True)
#                 self._log(f"Dropped excess {direction} columns: {to_drop}")

#         for i in range(self.spatial_adj):
#             down_col, up_col = f'downstream_sensor_{i+1}', f'upstream_sensor_{i+1}'
#             if down_col in df.columns and up_col in df.columns and current_smoothing == prev_smoothing:
#                 self._log(f"Skipping {down_col} and {up_col}, they already exist in the df (max value of downstream: {df[down_col].max()}).")
#                 self.new_columns += [down_col, up_col]
#                 continue

#             # Generate maps
#             down_map = {
#                 s: self.downstream_sensor_dict.get(s, {}).get('downstream_sensor', [None] * self.spatial_adj)[i]
#                 for s in df[self.sensor_col].unique()
#             }
#             up_map = {
#                 s: self.upstream_sensor_dict.get(s, {}).get('upstream_sensor', [None] * self.spatial_adj)[i]
#                 for s in df[self.sensor_col].unique()
#             }

#             # Map to get sensor ids
#             df[f'{down_col}_id'] = df[self.sensor_col].map(down_map)
#             df[f'{up_col}_id'] = df[self.sensor_col].map(up_map)

#             # Create tuples for lookup and perform reindexing
#             down_values = pivot_stacked[self.value_col].reindex(
#                 list(zip(df[self.datetime_col], df[f'{down_col}_id']))).values
#             up_values = pivot_stacked[self.value_col].reindex(
#                 list(zip(df[self.datetime_col], df[f'{up_col}_id']))).values

#             df[down_col] = down_values
#             df[up_col] = up_values

#             # Normalize
#             if self.normalize_by_distance:
#                 down_dist_map = {
#                     s: self.downstream_sensor_dict.get(s, {}).get('downstream_distance', [np.nan] * self.spatial_adj)[i]
#                     for s in df[self.sensor_col].unique()
#                 }
#                 up_dist_map = {
#                     s: self.upstream_sensor_dict.get(s, {}).get('upstream_distance', [np.nan] * self.spatial_adj)[i]
#                     for s in df[self.sensor_col].unique()
#                 }
#                 df[down_col] = df[down_col] / 3.6 / df[self.sensor_col].map(down_dist_map)
#                 df[up_col] = df[up_col] / 3.6 / df[self.sensor_col].map(up_dist_map)

#             # Cleanup
#             df.drop(columns=[f'{down_col}_id', f'{up_col}_id'], inplace=True)
#             df[down_col].fillna(self.fill_nans_value, inplace=True)
#             df[up_col].fillna(self.fill_nans_value, inplace=True)
#             self.new_columns += [down_col, up_col]
#             self._log(f"Added {down_col}, {up_col}")

#         return df, self.new_columns

class TemporalLagFeatureAdder(LoggingMixin):
    def __init__(self, lags=3, relative=False, fill_nans_value=-1, disable_logs=False,
                 sensor_col='sensor_id', value_col='value', datetime_col='datetime'):
        super().__init__(disable_logs)
        self.lags = lags
        self.relative = relative
        self.fill_nans_value = fill_nans_value
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.new_columns = []

    def transform(self, df, current_smoothing=None, prev_smoothing=None):
        epsilon = 1e-5
        self._log(f"Adding {'relative' if self.relative else 'absolute'} lags (lags={self.lags})")
        col_name_start = f"{'relative_diff_lag' if self.relative else 'lag'}"
        existing_cols = [col for col in df.columns if col.startswith(col_name_start)]
        expected_cols = [f'{col_name_start}_{i+1}' for i in range(self.lags)]
        to_drop = list(set(existing_cols) - set(expected_cols))
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            self._log(f"Dropped excess lag columns: {to_drop}")

        for i in range(1, self.lags + 1):
            col_name = f"{'relative_diff_lag' if self.relative else 'lag'}{i}"
            if col_name in df.columns and current_smoothing == prev_smoothing:
                self._log(f"Skipping already existing column: {col_name}")
                self.new_columns.append(col_name)
                continue
            shifted = df.groupby(self.sensor_col)[self.value_col].shift(i)

            if self.relative:
                df[col_name] = (df[self.value_col] - shifted) / (shifted + epsilon)
            else:
                df[col_name] = shifted - df[self.value_col]

            df[col_name].fillna(self.fill_nans_value, inplace=True)
            self.new_columns.append(col_name)

        return df, self.new_columns

    

class MiscellaneousFeatureEngineer(LoggingMixin):
    """
    Adds miscellaneous non-temporal, non-spatial features to the DataFrame.
    """

    def __init__(self, sensor_col='sensor_id', new_sensor_id_col='sensor_uid', weather_cols=None,disable_logs=False):
        self.sensor_col = sensor_col
        self.new_sensor_id_col = new_sensor_id_col
        self.weather_cols = weather_cols or []
        super().__init__(disable_logs)

    def map_sensor_ids(self, df):
        """Assigns unique integer IDs to each sensor."""
        logging.info("Mapping sensor IDs to integers.")
        sensor_ids = {sid: i for i, sid in enumerate(sorted(df[self.sensor_col].unique()))}
        df[self.new_sensor_id_col] = df[self.sensor_col].map(sensor_ids).astype(int)
        logging.info(f"Mapped {len(sensor_ids)} unique sensor IDs.")
        return df, [self.new_sensor_id_col]

    def drop_weather_features(self, df):
        """Drops weather-related columns if present."""
        logging.info("Dropping weather-related features.")
        before_cols = set(df.columns)
        df = df.drop(columns=[col for col in self.weather_cols if col in df.columns], errors='ignore')
        dropped = list(before_cols - set(df.columns))
        logging.info(f"Dropped columns: {dropped}")
        return df, dropped

    def transform(self, df):
        """Applies all miscellaneous transformations in one step."""
        df, id_cols = self.map_sensor_ids(df)
        df, dropped_cols = self.drop_weather_features(df)
        return df, id_cols + dropped_cols


class DateTimeFeatureEngineer(LoggingMixin):
    """
    Adds temporal features derived from a datetime column.
    """

    def __init__(self, datetime_col='datetime',disable_logs=False):
        self.datetime_col = datetime_col
        super().__init__(disable_logs)

    def add_weekend_columns(self, df):
        """Adds weekend indicators to the DataFrame."""
        self._log("Adding weekend indicator columns.")
        df['is_saturday'] = (df[self.datetime_col].dt.dayofweek == 5).astype(int)
        df['is_sunday'] = (df[self.datetime_col].dt.dayofweek == 6).astype(int)
        return df, ['is_saturday', 'is_sunday']

    def add_hour_column(self, df):
        """Adds the hour-of-day column."""
        self._log("Adding hour column.")
        df['hour'] = df[self.datetime_col].dt.hour
        
    def add_day_column(self, df):
        """Adds the day-of-week column."""
        self._log("Adding day column.")
        df['day'] = df[self.datetime_col].dt.dayofweek
        return df, ['day']

    def transform(self,df):
        """Applies all transformations in one step."""
        self._log("Adding datetime features.")
        self.add_hour_column(df)
        self.add_day_column(df)
        self.add_weekend_columns(df)
        return df, ['hour', 'day', 'is_saturday', 'is_sunday']
        

class CongestionFeatureEngineer(LoggingMixin):
    """
    Adds congestion-related features including binary congestion flags and outlier detection.
    """

    def __init__(self, hour_start=6, hour_end=19, quantile_threshold=0.9, quantile_percentage=0.65,lower_bound=0.01,upper_bound=0.99,disable_logs=False):
        self.hour_start = hour_start
        self.hour_end = hour_end
        self.quantile_threshold = quantile_threshold
        self.quantile_percentage = quantile_percentage
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(disable_logs)

    def transform_congestion(self, df):
        self._log("Adding congestion feature based on thresholds.")
        if 'hour' not in df.columns:
            raise ValueError("Column 'hour' is missing. Run DateTimeFeatureEngineer first.")
        if 'test_set' not in df.columns:
            raise ValueError("Column 'test_set' is missing. Ensure data has been split.")

        mask = (
            (df['hour'] >= self.hour_start) &
            (df['hour'] <= self.hour_end) &
            (~df['test_set'])
        )
        congestion_thr = df[mask].groupby('sensor_id')['value'].quantile(self.quantile_threshold)
        thresholds = df['sensor_id'].map(congestion_thr) * self.quantile_percentage

        df['is_congested'] = (df['value'] < thresholds).astype(int)
        return df, ['is_congested']

    def add_outlier_flags(self, df):
        """
        Adds a binary 'is_outlier' column marking extreme outliers in 'value'.

        Outliers are detected using percentiles from the training set.
        """
        self._log("Flagging outliers in training set.")

        if 'test_set' not in df.columns:
            raise ValueError("Missing 'test_set' column. Split your data first.")

        train_df = df[~df['test_set']]
        lower_val = train_df['value'].quantile(self.lower_bound)
        upper_val = train_df['value'].quantile(self.upper_bound)

        self._log(f"Using outlier thresholds: lower={lower_val}, upper={upper_val}")

        df['is_outlier'] = ((df['value'] < lower_val) | (df['value'] > upper_val)).astype(int)
        return df, ['is_outlier']

    def transform(self, df):
        """Applies both congestion and outlier detection features."""
        df, c_cols = self.transform_congestion(df)
        df, o_cols = self.add_outlier_flags(df)
        return df, c_cols + o_cols


class TargetVariableCreator(LoggingMixin):
    """
    Adds a target variable for prediction (delta or GMAN correction).
    """

    def __init__(
        self,
        horizon=15,
        sensor_col='sensor_id',
        datetime_col='date',
        value_col='value',
        gman_col='gman_prediction',
        use_gman=False,
        disable_logs=False
    ):
        super().__init__(disable_logs)
        self.horizon = horizon
        self.sensor_col = sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.gman_col = gman_col
        self.use_gman = use_gman

    def transform(self, df):
        self._log("Creating target variables.")
        df = df.sort_values(by=[self.sensor_col, self.datetime_col]).copy()

        df['target_total_speed'] = df.groupby(self.sensor_col)[self.value_col].shift(-self.horizon)
        df['target_speed_delta'] = df['target_total_speed'] - df[self.value_col]
        self._log("Computed 'target_total_speed' and 'target_speed_delta'.")

        if self.use_gman:
            df['target_gman_prediction'] = df.groupby(self.sensor_col)[self.gman_col].shift(-self.horizon)
            df['target'] = df['target_total_speed'] - df['target_gman_prediction']

            check = df['target_total_speed'] - (df['target'] + df['target_gman_prediction'])
            if not np.allclose(check.fillna(0), 0):
                raise ValueError("Target variable is not a valid GMAN correction.")

            self._log("GMAN correction target validated.")
            used_cols = ['target_total_speed', 'target_speed_delta', 'target_gman_prediction', 'target']
        else:
            df['target'] = df['target_speed_delta']
            used_cols = ['target_total_speed', 'target_speed_delta', 'target']

        df = df.dropna(subset=['target'])
        self._log(f"Final target column ready. {df.shape[0]} rows retained after dropping NaNs.")
        return df, used_cols

    
    


