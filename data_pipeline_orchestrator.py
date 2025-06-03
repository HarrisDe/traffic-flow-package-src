from .features.sensor_encoder import MeanSensorEncoder, OrdinalSensorEncoder, OneHotSensorEncoder
from .features.calendar_features import DateTimeFeatureEngineer
from .features.temporal_features import TemporalLagFeatureAdder
from .features.congestion_features import CongestionFeatureEngineer
from .features.historical_reference_features import PreviousWeekdayValueFeatureEngineer
from .features.adjacent_features import AdjacentSensorFeatureAdder
from .features.target_variable_feature import TargetVariableCreator
from .features.misc_features import WeatherFeatureDropper
from .data_loader_orchestrator import InitialTrafficDataLoader
from .constants import colnames, WEATHER_COLUMNS
from .helper_utils import LoggingMixin
from typing import Optional
import os
import warnings
import pandas as pd

class TrafficDataPipelineOrchestrator(LoggingMixin):
    def __init__(
        self,
        file_path: str,
        sensor_col: str = 'sensor_id',
        datetime_col: str = 'date',
        value_col: str = 'value',
        new_sensor_col: str = 'sensor_uid',
        weather_cols: Optional[list] = WEATHER_COLUMNS,
        disable_logs: bool = False,
        df_gman: Optional[pd.DataFrame] = None,
        sensor_encoding_type: str = "ordinal"  # "ordinal", "mean", or "onehot"
    ):
        super().__init__(disable_logs=disable_logs)
        self.file_path = file_path
        self.sensor_dict_path = os.path.dirname(file_path)
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        
        self.weather_cols = weather_cols or []
        self.df_gman = df_gman
        self.sensor_encoding_type = sensor_encoding_type

        # Runtime attributes
        self.df = None
        self.df_orig = None
        self.first_test_timestamp = None
        self.feature_log = {}
        self.smoothing = None
        self.smoothing_prev = None
        self.upstream_sensor_dict = None
        self.downstream_sensor_dict = None

    def _get_sensor_encoder(self):
        if self.sensor_encoding_type == "ordinal":
            return OrdinalSensorEncoder(sensor_col=self.sensor_col, new_sensor_col=self.new_sensor_col, disable_logs=self.disable_logs)
        elif self.sensor_encoding_type == "mean":
            return MeanSensorEncoder(sensor_col=self.sensor_col,new_sensor_col=self.new_sensor_col, disable_logs=self.disable_logs)
        elif self.sensor_encoding_type == "onehot":
            return OneHotSensorEncoder(sensor_col=self.sensor_col,new_sensor_col=self.new_sensor_col, disable_logs=self.disable_logs)
        else:
            raise ValueError(f"Unsupported sensor encoding type: {self.sensor_encoding_type},you must choose 'ordinal', 'mean', or 'onehot'.")

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
        adj_are_relative=False,
        normalize_by_distance=True,
        lag_steps=25,
        relative_lags=True,
        horizon=15,
        filter_on_train_only=True,
        hour_start=6,
        hour_end=19,
        quantile_threshold=0.9,
        quantile_percentage=0.65,
        lower_bound=0.01,
        upper_bound=0.99,
        use_median_instead_of_mean_smoothing=False,
        drop_weather=True,
        add_previous_weekday_feature=True,
        strict_weekday_match=True
    ):
        if not add_previous_weekday_feature and strict_weekday_match is not None:
            warnings.warn("'strict_weekday_match' has no effect since 'add_previous_weekday_feature' is False")

        smoothing_id = (
            f"smoothing_{window_size}_{'train_only' if filter_on_train_only else 'all'}"
            if smooth_speeds else "no_smoothing"
        )

        # Step 1: Load data
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
        
        
        # Step 2: Encode Sensor IDs
        self.df = df
        encoder = self._get_sensor_encoder()
        df = encoder.encode(df)

        # Step 2: DateTime Features
        dt_features = DateTimeFeatureEngineer(datetime_col=self.datetime_col)
        df, dt_cols = dt_features.transform(df)
        self.feature_log['datetime_features'] = dt_cols

        # Step 3: Spatial Features
        spatial = AdjacentSensorFeatureAdder(
            sensor_dict_path=self.sensor_dict_path,
            spatial_adj=spatial_adj,
            normalize_by_distance=normalize_by_distance,
            adj_are_relative=adj_are_relative,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            sensor_col=self.sensor_col,
            disable_logs=self.disable_logs,
        )
        self.upstream_sensor_dict = spatial.upstream_sensor_dict
        self.downstream_sensor_dict = spatial.downstream_sensor_dict
        df, spatial_cols = spatial.transform(df, smoothing_id, self.smoothing_prev)
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

        # Step 5: Congestion & Outliers
        congestion = CongestionFeatureEngineer(
            hour_start=hour_start,
            hour_end=hour_end,
            quantile_threshold=quantile_threshold,
            quantile_percentage=quantile_percentage,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        df, congestion_cols = congestion.transform(df)
        self.feature_log['congestion_features'] = congestion_cols

        # Step 6: Previous Weekday
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

        # Step 7: Drop Weather Features
        if drop_weather:
            dropper = WeatherFeatureDropper(weather_cols=self.weather_cols)
            df, dropped_cols = dropper.transform(df)
            self.feature_log['weather_dropped'] = dropped_cols

        

        # Step 9: Target Variable
        target_creator = TargetVariableCreator(
            horizon=horizon,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            datetime_col=self.datetime_col,
            gman_col='gman_prediction',
            use_gman=use_gman_target
        )
        df, target_cols = target_creator.transform(df)
        self.feature_log['target_variables'] = target_cols

        self.df = df
        self.df['date_of_prediction'] = self.df.groupby(self.sensor_col)[self.datetime_col].shift(-horizon)
        self.all_added_features = list(set(col for cols in self.feature_log.values() for col in cols))

        train_df = df[~df['test_set']].copy()
        test_df = df[df['test_set']].copy()

        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']

        cols_to_drop = [
            'sensor_id', 'target_total_speed', 'target_speed_delta', 'date',
            'test_set', 'gman_prediction_date', 'gman_target_date', 'date_of_prediction'
        ]

        for df_ in [X_train, X_test]:
            df_.drop(columns=[col for col in cols_to_drop if col in df_.columns], inplace=True)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test