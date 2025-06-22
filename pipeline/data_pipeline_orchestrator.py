from typing import Optional, Union, Dict, List, Set
import warnings
import os
import pandas as pd

from ..features.sensor_encoder import (
    MeanSensorEncoder,
    OrdinalSensorEncoder,
    OneHotSensorEncoder,
)
from ..features.calendar_features import DateTimeFeatureEngineer
from ..features.temporal_features import TemporalLagFeatureAdder
from ..features.congestion_features import CongestionFeatureEngineer
from ..features.historical_reference_features import (
    PreviousWeekdayWindowFeatureEngineer,
)
from ..features.adjacent_features import AdjacentSensorFeatureAdder
from ..features.target_variable_feature import TargetVariableCreator
from ..features.misc_features import WeatherFeatureDropper
from ..data_loading.data_loader_orchestrator import InitialTrafficDataLoader
from ..constants.constants import WEATHER_COLUMNS
from ..utils.helper_utils import LoggingMixin


class TrafficDataPipelineOrchestrator(LoggingMixin):
    """
    End-to-end feature-engineering pipeline for traffic-sensor data.

    New in v2
    ----------
    • ``prepare_base_features``:  runs steps 0-6 (horizon-independent) **once**  
    • ``finalise_for_horizon``  :  adds steps 7-10 for any forecast horizon  
    • Legacy ``run_pipeline`` still works by delegating to the two new methods.
    """

    # ---------------------------------------------------------------------#
    # Construction
    # ---------------------------------------------------------------------#
    def __init__(
        self,
        file_path: str,
        *,
        sensor_col: str = "sensor_id",
        datetime_col: str = "date",
        value_col: str = "value",
        new_sensor_col: str = "sensor_uid",
        weather_cols: Optional[list] = WEATHER_COLUMNS,
        disable_logs: bool = False,
        df_gman: Optional[pd.DataFrame] = None,
        sensor_encoding_type: str = "ordinal",
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        # Immutable parameters
        self.file_path = file_path
        self.sensor_dict_path = os.path.dirname(file_path)
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.weather_cols = weather_cols or []
        self.df_gman = df_gman
        self.sensor_encoding_type = sensor_encoding_type

        # Runtime state
        self.df_orig: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.base_df: Optional[pd.DataFrame] = None  # cached horizon-independent df
        self.first_test_timestamp: Optional[pd.Timestamp] = None
        self.feature_log: Dict[str, List[str]] = {}
        self.smoothing: Optional[str] = None
        self.smoothing_prev: Optional[str] = None
        self.upstream_sensor_dict: Optional[dict] = None
        self.downstream_sensor_dict: Optional[dict] = None

        # Exposed train/test splits (filled after finalise_for_horizon / run_pipeline)
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ---------------------------------------------------------------------#
    # Helper – choose sensor encoder
    # ---------------------------------------------------------------------#
    def _get_sensor_encoder(self):
        if self.sensor_encoding_type == "ordinal":
            return OrdinalSensorEncoder(
                sensor_col=self.sensor_col,
                new_sensor_col=self.new_sensor_col,
                disable_logs=self.disable_logs,
            )
        if self.sensor_encoding_type == "mean":
            return MeanSensorEncoder(
                sensor_col=self.sensor_col,
                new_sensor_col=self.new_sensor_col,
                disable_logs=self.disable_logs,
            )
        if self.sensor_encoding_type == "onehot":
            return OneHotSensorEncoder(
                sensor_col=self.sensor_col,
                new_sensor_col=self.new_sensor_col,
                disable_logs=self.disable_logs,
            )
        raise ValueError(
            f"Unsupported sensor_encoding_type {self.sensor_encoding_type!r}; "
            "choose 'ordinal', 'mean', or 'onehot'."
        )

    # =====================================================================#
    #  NEW -- step 0-6 (horizon-independent)                               #
    # =====================================================================#
    def prepare_base_features(
        self,
        *,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
        filter_on_train_only: bool = True,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        relative_threshold: float = 0.7,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
        window_size: int = 5,
        spatial_adj: int = 1,
        adj_are_relative: bool = False,
        normalize_by_distance: bool = True,
        lag_steps: int = 25,
        relative_lags: bool = True,
        hour_start: int = 6,
        hour_end: int = 19,
        quantile_threshold: float = 0.9,
        quantile_percentage: float = 0.65,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        use_median_instead_of_mean_smoothing: bool = False,
    ) -> pd.DataFrame:
        """
        Run the pipeline **up to** congestion / outlier flags (inclusive) and
        cache the resulting dataframe in ``self.base_df``.

        Returns
        -------
        pd.DataFrame
            The cached horizon-independent dataframe (copy).
        """
        # --------------------------------------------------------------#
        # 0) Decide on train/test split strategy
        # --------------------------------------------------------------#
        if test_start_time is not None and test_size != 1 / 3:
            warnings.warn(
                "Both 'test_start_time' and 'test_size' were provided; "
                "defaulting to the explicit date split."
            )
        effective_test_size = test_size if test_start_time is None else 1 / 3

        # --------------------------------------------------------------#
        # 1) Raw loading & basic preprocessing
        # --------------------------------------------------------------#
        smoothing_id = (
            f"smoothing_{window_size}_{'train_only' if filter_on_train_only else 'all'}"
            if smooth_speeds
            else "no_smoothing"
        )

        loader = InitialTrafficDataLoader(
            file_path=self.file_path,
            datetime_cols=[self.datetime_col],
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            disable_logs=self.disable_logs,
            df_gman=self.df_gman,
        )

        df = loader.get_data(
            window_size=window_size,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            relative_threshold=relative_threshold,
            test_size=effective_test_size,
            test_start_time=test_start_time,
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=add_gman_predictions,
            use_median_instead_of_mean=use_median_instead_of_mean_smoothing,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed,
        )

        # Explicit date split → log the realised ratio
        if test_start_time is not None:
            test_ratio = df["test_set"].mean()
            self._log(
                f"Explicit test_start_time={pd.to_datetime(test_start_time)} → "
                f"test set ratio: {test_ratio:.2%}"
            )

        # Cache loader artefacts
        self.df_orig = loader.df_orig
        self.first_test_timestamp = loader.first_test_timestamp
        self.smoothing_prev, self.smoothing = self.smoothing, smoothing_id

        # --------------------------------------------------------------#
        # 2) Sensor encoding
        # --------------------------------------------------------------#
        encoder = self._get_sensor_encoder()
        df = encoder.encode(df)

        # --------------------------------------------------------------#
        # 3) Date-time features
        # --------------------------------------------------------------#
        dt_fe = DateTimeFeatureEngineer(datetime_col=self.datetime_col)
        df, dt_cols = dt_fe.transform(df)
        self.feature_log["datetime_features"] = dt_cols

        # --------------------------------------------------------------#
        # 4) Spatial adjacency features
        # --------------------------------------------------------------#
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
        self.feature_log["spatial_features"] = spatial_cols

        # --------------------------------------------------------------#
        # 5) Temporal lags
        # --------------------------------------------------------------#
        lagger = TemporalLagFeatureAdder(
            lags=lag_steps,
            relative=relative_lags,
            fill_nans_value=-1,
            disable_logs=self.disable_logs,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
        )
        df, lag_cols = lagger.transform(df, smoothing_id, self.smoothing_prev)
        self.feature_log["lag_features"] = lag_cols

        # --------------------------------------------------------------#
        # 6) Congestion / outlier flags
        # --------------------------------------------------------------#
        congestion = CongestionFeatureEngineer(
            hour_start=hour_start,
            hour_end=hour_end,
            quantile_threshold=quantile_threshold,
            quantile_percentage=quantile_percentage,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        df, congestion_cols = congestion.transform(df)
        self.feature_log["congestion_features"] = congestion_cols

        # --------------------------------------------------------------#
        # Cache & return
        # --------------------------------------------------------------#
        self.base_df = df
        return df.copy()

    # =====================================================================#
    #  NEW -- step 7-10 (horizon-specific)                                 #
    # =====================================================================#
    def finalise_for_horizon(
        self,
        *,
        horizon: int,
        base_df: Optional[pd.DataFrame] = None,
        add_previous_weekday_feature: bool = True,
        previous_weekday_window_min: int = 5,
        drop_weather: bool = True,
        use_gman_target: bool = False,
    ):
        """
        Add horizon-specific features to the cached dataframe and return
        `X_train, X_test, y_train, y_test`.

        Parameters
        ----------
        horizon
            Forecast horizon in minutes.
        base_df
            Provide explicitly or let the method use the cached ``self.base_df``.
        """
        if base_df is None:
            if self.base_df is None:
                raise RuntimeError("Call `prepare_base_features()` first.")
            base_df = self.base_df

        df = base_df.copy()  # avoid mutating the cached frame

        # --------------------------------------------------------------#
        # 7) Historical reference (depends on horizon)                   #
        # --------------------------------------------------------------#
        if add_previous_weekday_feature:
            prevday = PreviousWeekdayWindowFeatureEngineer(
                datetime_col=self.datetime_col,
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                horizon_min=horizon,
                window_before_min=previous_weekday_window_min,
                window_after_min=previous_weekday_window_min,
                step_min=1,
                aggs=[],
                disable_logs=self.disable_logs,
            )
            df, prevday_cols = prevday.transform(df)
            self.feature_log.setdefault("previous_day_features", []).extend(prevday_cols)

        # --------------------------------------------------------------#
        # 8) Optional drop weather columns (kept here due to legacy & compatibility with older scripts)                              
        # --------------------------------------------------------------#
        if drop_weather:
            dropper = WeatherFeatureDropper(weather_cols=self.weather_cols)
            df, dropped_cols = dropper.transform(df)
            self.feature_log["weather_dropped"] = dropped_cols

        # --------------------------------------------------------------#
        # 9) Target variable & prediction date                           #
        # --------------------------------------------------------------#
        df["date_of_prediction"] = df.groupby(self.sensor_col)[self.datetime_col].shift(
            -horizon
        )

        target_creator = TargetVariableCreator(
            horizon=horizon,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            datetime_col=self.datetime_col,
            gman_col="gman_prediction",
            use_gman=use_gman_target,
        )
        df, target_cols = target_creator.transform(df)
        self.feature_log["target_variables"] = target_cols

        # --------------------------------------------------------------#
        # 10) Split train / test & drop helper columns                   #
        # --------------------------------------------------------------#
        train_df = df[~df["test_set"]].copy()
        test_df = df[df["test_set"]].copy()

        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

        cols_to_drop: Set[str] = {
            "sensor_id",
            "target_total_speed",
            "target_speed_delta",
            "date",
            "test_set",
            "gman_prediction_date",
            "gman_target_date",
            "date_of_prediction",
        }
        for subset in (X_train, X_test):
            subset.drop(columns=[c for c in cols_to_drop if c in subset.columns], inplace=True)

        # Store for external access / debugging
        self.df = df
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test

    # =====================================================================#
    #  Legacy façade – still returns the same as before                    #
    # =====================================================================#
    def run_pipeline(
        self,
        *,
        # ---- args that affect *only* the base ---------------------------
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
        filter_on_train_only: bool = True,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        relative_threshold: float = 0.7,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
        window_size: int = 3,
        spatial_adj: int = 1,
        adj_are_relative: bool = False,
        normalize_by_distance: bool = True,
        lag_steps: int = 25,
        relative_lags: bool = True,
        hour_start: int = 6,
        hour_end: int = 19,
        quantile_threshold: float = 0.9,
        quantile_percentage: float = 0.65,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        use_median_instead_of_mean_smoothing: bool = False,
        # ---- args that are truly horizon-specific -----------------------
        horizon: int = 15,
        drop_weather: bool = True,
        add_previous_weekday_feature: bool = True,
        previous_weekday_window_min: int = 5,
        use_gman_target: bool = False,
    ):
        """
        Legacy single-call API – behaviour is **identical** to the original
        implementation but is now internally composed of the two new methods.
        """
        # 1) horizon-independent part (cached on self)
        self.prepare_base_features(
            test_size=test_size,
            test_start_time=test_start_time,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            relative_threshold=relative_threshold,
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=add_gman_predictions,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed,
            window_size=window_size,
            spatial_adj=spatial_adj,
            adj_are_relative=adj_are_relative,
            normalize_by_distance=normalize_by_distance,
            lag_steps=lag_steps,
            relative_lags=relative_lags,
            hour_start=hour_start,
            hour_end=hour_end,
            quantile_threshold=quantile_threshold,
            quantile_percentage=quantile_percentage,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_median_instead_of_mean_smoothing=use_median_instead_of_mean_smoothing,
        )

        # 2) horizon-specific augmentation
        return self.finalise_for_horizon(
            horizon=horizon,
            add_previous_weekday_feature=add_previous_weekday_feature,
            previous_weekday_window_min=previous_weekday_window_min,
            drop_weather=drop_weather,
            use_gman_target=use_gman_target,
        )

class TrafficDataPipelineOrchestrator_deprecated(LoggingMixin):
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
        previous_weekday_window_min= 5
    ):

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
            # prevday = PreviousWeekdayValueFeatureEngineer(
            #     datetime_col=self.datetime_col,
            #     sensor_col=self.sensor_col,
            #     value_col=self.value_col,
            #     horizon_minutes=horizon,
            #     strict_weekday_match=strict_weekday_match,
            #     disable_logs=self.disable_logs
            # )
            

            prevday = PreviousWeekdayWindowFeatureEngineer(
                datetime_col=self.datetime_col,
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                horizon_min=horizon,
                window_before_min = previous_weekday_window_min,
                window_after_min = previous_weekday_window_min,
                step_min=1,
                aggs=[],
                disable_logs=self.disable_logs

            )
            df, prevday_cols = prevday.transform(df)
            self.feature_log['previous_day_features'] = prevday_cols

        # Step 7: Drop Weather Features
        if drop_weather:
            dropper = WeatherFeatureDropper(weather_cols=self.weather_cols)
            df, dropped_cols = dropper.transform(df)
            self.feature_log['weather_dropped'] = dropped_cols

        
        df['date_of_prediction'] = df.groupby(self.sensor_col)[self.datetime_col].shift(-horizon)
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
        #self.df['date_of_prediction'] = self.df.groupby(self.sensor_col)[self.datetime_col].shift(-horizon)
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