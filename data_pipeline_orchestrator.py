
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from .features import *
from .data_loader_orchestrator import InitialTrafficDataLoader
from .constants import colnames, WEATHER_COLUMNS
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


class TrafficDataPipelineOrchestrator(LoggingMixin):
    def __init__(
        self,
        file_path,
        sensor_col='sensor_id',
        datetime_col='date',
        value_col='value',
        new_sensor_id_col='sensor_uid',
        weather_cols=WEATHER_COLUMNS,
        disable_logs=False,
        df_gman=None
    ):
        super().__init__(disable_logs=disable_logs)
        self.file_path = file_path
        self.sensor_dict_path = os.path.dirname(file_path)
        self.sensor_col = sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.new_sensor_id_col = new_sensor_id_col
        self.weather_cols = weather_cols
        self.df = None
        self.df_orig = None
        self.df_gman = df_gman
        self.first_test_timestamp = None
        self.feature_log = {}  # Track added features
        self.smoothing = None
        self.smoothing_prev = None
        self.upstream_sensor_dict = None
        self.downstream_sensor_dict = None

    def run_pipeline(
        self,
        test_size=1/3,
        filter_extreme_changes=True,
        smooth_speeds=True,
        relative_threshold=0.7,
        diagnose_extreme_changes=False,
        add_gman_predictions=False,
        use_gman_target=False,
        convert_gman_prediction_to_delta_speed=True,
        window_size=3,
        spatial_adj=1,
        adj_are_relative = False,
        normalize_by_distance=True,
        lag_steps=20,
        relative_lags=True,
        horizon=15,
        filter_on_train_only=True,
        hour_start=6,
        hour_end=19,
        quantile_threshold=0.9,
        quantile_percentage=0.65,
        lower_bound=0.01,
        upper_bound=0.99,
        use_median_instead_of_mean_smoothing=True,
        drop_weather=True,
        add_previous_weekday_feature=True,
        strict_weekday_match=True
    ):
        
        if not add_previous_weekday_feature and strict_weekday_match is not None:
            warnings.warn("'strict_weekday_match' has no effect since 'add_previous_weekday_feature' is False")


        # Determine current smoothing strategy ID
        smoothing_id = (
            f"smoothing_{window_size}_{'train_only' if filter_on_train_only else 'all'}"
            if smooth_speeds else "no_smoothing"
        )

        # Step 1: Initial data loading and cleaning
        loader = InitialTrafficDataLoader(
            file_path=self.file_path,
            datetime_cols=[self.datetime_col],
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            disable_logs=self.disable_logs,
            df_gman=self.df_gman
        )

        df = loader.get_data(
            window_size=window_size,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            relative_threshold=relative_threshold,
            test_size=test_size,
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=add_gman_predictions,
            use_median_instead_of_mean=use_median_instead_of_mean_smoothing,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed
        )
        self.df_orig = loader.df_orig
        self.first_test_timestamp = loader.first_test_timestamp
        self.smoothing_prev = self.smoothing
        self.smoothing = smoothing_id

        # Step 2: DateTime Features
        dt_features = DateTimeFeatureEngineer(datetime_col=self.datetime_col)
        df, dt_cols = dt_features.transform(df)
        self.feature_log['datetime_features'] = dt_cols

        # Step 3: Spatial Features
        spatial = AdjacentSensorFeatureAdderOptimal(
            sensor_dict_path=self.sensor_dict_path,
            spatial_adj=spatial_adj,
            normalize_by_distance=normalize_by_distance,
            adj_are_relative= adj_are_relative,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            sensor_col=self.sensor_col,
            disable_logs=self.disable_logs,
        )
        self.upstream_sensor_dict = spatial.upstream_sensor_dict
        self.downstream_sensor_dict = spatial.downstream_sensor_dict
        df, spatial_cols = spatial.transform(
            df, smoothing_id, self.smoothing_prev)
        self.feature_log['spatial_features'] = spatial_cols

        # Step 4: Temporal Lag Features
        lagger = TemporalLagFeatureAdder(
            lags=lag_steps,
            relative=relative_lags,
            fill_nans_value=-1,
            disable_logs=self.disable_logs,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
          
        )
        df, lag_cols = lagger.transform(df, smoothing_id, self.smoothing_prev)
        self.feature_log['lag_features'] = lag_cols

        # Step 5: Congestion and Outlier Features
        congestion = CongestionFeatureEngineer(hour_start=hour_start, hour_end=hour_end,
                                               quantile_threshold=quantile_threshold, quantile_percentage=quantile_percentage,
                                               lower_bound=lower_bound, upper_bound=upper_bound)
        df, congestion_cols = congestion.transform(df)
        self.feature_log['congestion_features'] = congestion_cols
        
        
        if add_previous_weekday_feature:
            prevday = PreviousWeekdayValueFeatureEngineer(
                datetime_col=self.datetime_col,
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                horizon_minutes=horizon,
                strict_weekday_match=strict_weekday_match,
                disable_logs=self.disable_logs
            )
            df, prevday_cols = prevday.transform(df)
            self.feature_log['previous_day_features'] = prevday_cols

        # Step 6: Miscellaneous Features
        misc = MiscellaneousFeatureEngineer(
            sensor_col=self.sensor_col,
            new_sensor_id_col=self.new_sensor_id_col,
            weather_cols=self.weather_cols,
        )
        df, misc_cols = misc.transform(df, drop_weather=drop_weather)
        self.feature_log['miscellaneous_features'] = misc_cols

        # Step 7: Target Variable
        target_creator = TargetVariableCreator(
            horizon=horizon,
            sensor_col=self.new_sensor_id_col,
            value_col=self.value_col,
            datetime_col=self.datetime_col,
            gman_col='gman_prediction',
            use_gman=use_gman_target,
        )
        df, target_cols = target_creator.transform(df)
        self.feature_log['target_variables'] = target_cols

        # Store outputs
        self.df = df
        self.all_added_features = list(
            set(col for cols in self.feature_log.values() for col in cols))

        # Train/test split
        train_df = df[~df['test_set']].copy()
        test_df = df[df['test_set']].copy()

        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']

        cols_to_drop = ['sensor_id', 'target_total_speed',
                        'target_speed_delta', 'date', 'sensor_id', 'test_set', 'gman_prediction_date', 'gman_target_date']

        # Drop unwanted columns
        for df in [X_train, X_test]:

            df = df.drop(
                columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def validate_target_computation(self, use_gman_target=False, horizon=15):
        self._log("Validating target variable...")

        df_test = self.df.copy().sort_values(
            by=[self.sensor_col, self.datetime_col])

        if use_gman_target:
            df_test['expected_target'] = (
                df_test.groupby(self.sensor_col)[
                    self.value_col].shift(-horizon)
                - df_test.groupby(self.sensor_col)['gman_prediction'].shift(-horizon)
            )
        else:
            df_test['expected_target'] = (
                df_test.groupby(self.sensor_col)[
                    self.value_col].shift(-horizon)
                - df_test[self.value_col]
            )

        df_test['target_correct'] = df_test['target'] == df_test['expected_target']

        if df_test['target_correct'].all():
            self._log("All target values are correct!")
            return True
        else:
            incorrect_rows = df_test[df_test['target_correct'] == False]
            self._log(
                f"{len(incorrect_rows)} rows have incorrect target values.")
            return False


# class TrafficMLPipelineOrchestrator(LoggingMixin):
#     def __init__(
#         self,
#         file_path,
#         datetime_col="datetime",
#         sensor_col="sensor_id",
#         value_col="value",
#         test_size=1 / 3,
#         gman_df=None,
#         gman_correction_as_target=False,
#         horizon=15,
#         disable_logs=False,
#     ):
#         self.file_path = file_path
#         self.datetime_col = datetime_col
#         self.sensor_col = sensor_col
#         self.value_col = value_col

#         self.gman_df = gman_df
#         self.gman_correction_as_target = gman_correction_as_target
#         self.horizon = horizon
#         self.disable_logs = disable_logs

#         self.df = None
#         self.df_orig = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None

#     def run_pipeline(self):
#         loader = InitialTrafficDataLoader(
#             file_path=self.file_path,
#             datetime_cols=[self.datetime_col],
#             sensor_col=self.sensor_col,
#             value_col=self.value_col,
#             disable_logs=self.disable_logs,

#         )
#         loader.df_gman = self.gman_df
#         df = loader.get_data(add_gman_predictions=self.gman_df is not None)
#         self.df_orig = loader.df_orig.copy()

#         df = df.sort_values(by=[self.sensor_col, self.datetime_col])

#         df, _ = MiscellaneousFeatureEngineer(
#             sensor_col=self.sensor_col,
#             new_sensor_id_col="sensor_uid",
#             weather_cols=[],
#         ).transform(df)

#         df, _ = DateTimeFeatureEngineer(datetime_col=self.datetime_col).convert_datetime(df)
#         df, _ = DateTimeFeatureEngineer(datetime_col=self.datetime_col).add_weekend_columns(df)

#         df, _ = AdjacentSensorFeatureAdder(
#             sensor_col="sensor_uid",
#             datetime_col=self.datetime_col,
#             value_col=self.value_col,
#             disable_logs=self.disable_logs,
#         ).transform(df)

#         df, _ = TemporalLagFeatureAdder(
#             sensor_col="sensor_uid",
#             value_col=self.value_col,
#             disable_logs=self.disable_logs,
#         ).transform(df)

#         df, _ = CongestionFeatureEngineer().transform(df)

#         df, _ = TargetVariableCreator(
#             sensor_col="sensor_uid",
#             value_col=self.value_col,
#             gman_col="gman_prediction",
#             use_gman=self.gman_correction_as_target,
#             horizon=self.horizon,
#         ).transform(df)

#         # Train-test split
#         if "test_set" not in df.columns:
#             raise ValueError("Expected 'test_set' column not found in DataFrame.")

#         self.df = df.copy()
#         self.X_train = df.loc[~df.test_set].drop(columns=["target"])
#         self.X_test = df.loc[df.test_set].drop(columns=["target"])
#         self.y_train = df.loc[~df.test_set, "target"]
#         self.y_test = df.loc[df.test_set, "target"]

#         return self.X_train, self.X_test, self.y_train, self.y_test

#     def validate_target_variable(self):
#         self._log("Validating target variable...")
#         df_test = self.df.copy().sort_values(by=["sensor_uid", "datetime"])

#         if self.gman_correction_as_target:
#             df_test["expected_target"] = (
#                 df_test.groupby("sensor_uid")["value"].shift(-self.horizon)
#                 - df_test.groupby("sensor_uid")["gman_prediction"].shift(-self.horizon)
#             )
#         else:
#             df_test["expected_target"] = (
#                 df_test.groupby("sensor_uid")["value"].shift(-self.horizon)
#                 - df_test["value"]
#             )

#         df_test["target_correct"] = df_test["target"] == df_test["expected_target"]

#         if df_test["target_correct"].all():
#             self._log("All target values are correct!")
#             return True
#         else:
#             self._log("Some target values are incorrect!")
#             return False
