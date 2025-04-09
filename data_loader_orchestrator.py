
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


class InitialTrafficDataLoader(LoggingMixin):
    def __init__(
        self,
        file_path,
        datetime_cols=['datetime', 'date'],
        sensor_col='sensor_id',
        value_col='value',
        disable_logs=False
    ):
        super().__init__(disable_logs=disable_logs)
        self.file_path = file_path
        self.df = None
        self.df_orig = None
        self.datetime_cols = datetime_cols
        self.datetime_col = None
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.df_gman = None
        self.first_test_timestamp = None

    def _clean_and_convert_to_float32(self, value):
        try:
            if isinstance(value, float):
                return np.float32(round(value, 2))
            return np.float32(round(float(value.replace(' ', '.')), 2))
        except (ValueError, AttributeError):
            return None

    def _apply_clean_and_convert_to_float32(self):
        self._log("Cleaning and converting 'value' column to float32.")
        len_df_prev = len(self.df)
        self.df = self.df.dropna(subset=[self.value_col])
        len_df_after = len(self.df)
        self._log(f"Discarded {len_df_prev - len_df_after} rows with NaN values in 'value' column. (method _apply_clean_and_convert_to_float32())")
        self.df[self.value_col] = self.df[self.value_col].apply(self._clean_and_convert_to_float32)
        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)

    def load_data_parquet(self):
        self.df = pd.read_parquet(self.file_path, engine='pyarrow')

        if self.sensor_col not in self.df.columns:
            raise ValueError(f"Missing expected sensor column: {self.sensor_col}. Columns: {self.df.columns}. Change sensor_col parameter.")

        for col in self.datetime_cols:
            if col in self.df.columns:
                self.datetime_col = col
                self.df[col] = pd.to_datetime(self.df[col])
                break

        if not self.datetime_col:
            raise ValueError(f"Missing expected datetime column(s): {self.datetime_cols}")

        self.df_orig = self.df.copy()
        self.df = self.df.sort_values([self.datetime_col, self.sensor_col]).reset_index(drop=True)
        self._log(f"Loaded {len(self.df)} rows from {self.file_path}. Columns: {self.df.columns.tolist()}")
        self._apply_clean_and_convert_to_float32()

    def align_sensors_to_common_timeframe(self):
        self._log("Aligning sensors to a common timeframe based on the sensor with the fewest recordings.")
        sensor_counts = self.df.groupby(self.sensor_col).size()
        min_recording_sensor = sensor_counts.idxmin()
        common_timeframe = self.df[self.df[self.sensor_col] == min_recording_sensor][self.datetime_col]
        min_time = common_timeframe.min()
        max_time = common_timeframe.max()
        original_row_count = len(self.df)
        self.df = self.df[(self.df[self.datetime_col] >= min_time) & (self.df[self.datetime_col] <= max_time)]
        filtered_row_count = len(self.df)
        self._log(f"Aligned all sensors to the common timeframe: {min_time} to {max_time}.")
        self._log(f"Rows before alignment: {original_row_count}, Rows after alignment: {filtered_row_count}. {original_row_count - filtered_row_count} rows have been dropped")

    def add_test_set_column(self, test_size=1/3):
        unique_times = self.df[self.datetime_col].sort_values().unique()
        split_index = int(len(unique_times) * (1 - test_size))
        split_time = unique_times[split_index]
        self.df['test_set'] = self.df[self.datetime_col] >= split_time
        self._log(f"'test_set' column added. Split time (first test set timestamp): {split_time}")
        self.first_test_timestamp = split_time

    def _compute_relative_change_mask(self, threshold):
        col_temp = '__temp_rel_change__'
        self.df[col_temp] = self.df.groupby(self.sensor_col)[self.value_col].pct_change().abs()
        mask = self.df[col_temp] > threshold
        self.df.drop(columns=[col_temp], inplace=True)
        return mask

    def diagnose_extreme_changes(self, relative_threshold=0.7):
        self._log(f"Diagnosing changes > {relative_threshold * 100:.1f}%")
        mask = self._compute_relative_change_mask(relative_threshold)
        count_extremes = mask.sum()
        total = len(self.df)
        self._log(f"Extreme changes: {count_extremes} of {total} rows ({100 * count_extremes / total:.2f}%)")

    def filter_extreme_changes(self, relative_threshold=0.7):
        mask = self._compute_relative_change_mask(relative_threshold)
        self.df.loc[mask, self.value_col] = np.nan
        self.df[self.value_col] = self.df.groupby(self.sensor_col)[self.value_col].transform(lambda x: x.interpolate().ffill().bfill())
        self._log(f"Filtered and interpolated {mask.sum()} extreme changes.")

    def smooth_speeds(self, window_size=3, filter_on_train_only=True, use_median_instead_of_mean=True):
        if 'test_set' not in self.df.columns:
            self._log("Test set column not found. Automatically adding it.")
            self.add_test_set_column()

        mask = ~self.df['test_set'] if filter_on_train_only else self.df.index == self.df.index
        if use_median_instead_of_mean:
            smoothed = self.df.loc[mask].groupby(self.sensor_col)[self.value_col].transform(lambda x: x.rolling(window=window_size, center=False, min_periods=1).median())
        else:
            smoothed = self.df.loc[mask].groupby(self.sensor_col)[self.value_col].transform(lambda x: x.rolling(window=window_size, center=False, min_periods=1).mean())
        self.df.loc[mask, self.value_col] = smoothed.ffill().bfill()
        self._log(f"Applied smoothing (window={window_size}, train_only={filter_on_train_only}, use_median_instead_of_mean={use_median_instead_of_mean}).")

    def add_gman_predictions(self):
        self._log("Merging gman data.")
        assert self.df_gman is not None, "gman DataFrame is not provided. Please set df_gman in the constructor."
        self.df[self.sensor_col] = self.df[self.sensor_col].astype('category')
        self.df_gman[self.sensor_col] = self.df_gman[self.sensor_col].astype('category')
        self.df = self.df.set_index([self.datetime_col, self.sensor_col])
        self.df_gman = self.df_gman.set_index([self.datetime_col, self.sensor_col])
        self.df = self.df.join(self.df_gman, how='left').reset_index()
        missing_rows = self.df[self.df['gman_prediction'].isna()]
        if not missing_rows.empty:
            dropped_count = missing_rows.shape[0]
            min_date = missing_rows[self.datetime_col].min()
            max_date = missing_rows[self.datetime_col].max()
            self._log(f"Dropping {dropped_count} rows with missing 'gman_prediction'. Date range: {min_date} to {max_date}.")
            self.df = self.df.dropna(subset=['gman_prediction'])
        else:
            self._log("No rows dropped for missing 'gman_predictions'.")

    def get_data(self,
                 window_size=3,
                 filter_on_train_only=True,
                 filter_extreme_changes=True,
                 smooth_speeds=True,
                 use_median_instead_of_mean=True,
                 relative_threshold=0.7,
                 test_size=1/3,
                 diagnose_extreme_changes=False,
                 add_gman_predictions=False):
        self.load_data_parquet()
        self.align_sensors_to_common_timeframe()
        self.add_test_set_column(test_size=test_size)
        if diagnose_extreme_changes:
            self.diagnose_extreme_changes(relative_threshold=relative_threshold)
        if filter_extreme_changes:
            self.filter_extreme_changes(relative_threshold=relative_threshold)
        if smooth_speeds:
            self.smooth_speeds(window_size=window_size, filter_on_train_only=filter_on_train_only,use_median_instead_of_mean=use_median_instead_of_mean)
        if add_gman_predictions:
            self.add_gman_predictions()
        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)
        return self.df.copy()




    
    


