
import os
import pandas as pd
import numpy as np
import warnings
from ..features import *
from ..data_loading.data_loader_orchestrator import InitialTrafficDataLoader
from ..constants.constants import colnames, WEATHER_COLUMNS
import logging
from ..utils.helper_utils import *
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
from ..features.gman_features import GMANPredictionAdder
from ..data_loading.data_loader_orchestrator import InitialTrafficDataLoader
from ..constants.constants import WEATHER_COLUMNS
from ..utils.helper_utils import LoggingMixin

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # You can set this to DEBUG, WARNING, etc. as needed
)


class TrafficDataPipelineOrchestrator_deprecated(LoggingMixin):
    """
    End‑to‑end feature‑engineering pipeline for traffic‑sensor data.

    The orchestrator wraps together:
    1. Loading & basic cleansing   – via :class:`InitialTrafficDataLoader`
    2. Categorical encoding        – ordinal / mean‑target / one‑hot
    3. DateTime features           – hour‑of‑day, weekday, etc.
    4. Spatial adjacency features  – upstream / downstream sensors
    5. Temporal lag windows        – autoregressive lags of `value`
    6. Congestion / outlier flags  – quantile & time‑of‑day rules
    7. Historical reference feats  – previous‑weekday windows
    8. Target variable creation    – n‑step horizon forecast
    9. Optional GMAN predictions   – merged from external model

    Parameters
    ----------
    file_path
        Parquet file containing raw sensor records.
    sensor_col, datetime_col, value_col
        Column names in the raw data.
    new_sensor_col
        Column that will hold encoded sensor IDs.
    weather_cols
        Optional list of weather columns to eventually drop.
    df_gman
        Optional GMAN prediction dataframe.
    sensor_encoding_type
        One of ``'ordinal'``, ``'mean'`` or ``'onehot'``.

    Notes
    -----
    *New functionality*: the pipeline now accepts **`test_start_time`**.
    If both `test_size` *and* `test_start_time` are supplied, a warning is
    issued and the date‑based split takes precedence.
    """
    # ------------------------------------------------------------------#
    # Construction
    # ------------------------------------------------------------------#
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

        # Immutable parameters --------------------------------------------
        self.file_path = file_path
        self.sensor_dict_path = os.path.dirname(file_path)
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.weather_cols = weather_cols or []
        self.df_gman = df_gman
        self.sensor_encoding_type = sensor_encoding_type

        # Runtime state ----------------------------------------------------
        self.df: Optional[pd.DataFrame] = None
        self.df_orig: Optional[pd.DataFrame] = None
        self.first_test_timestamp: Optional[pd.Timestamp] = None
        self.feature_log: dict[str, list[str]] = {}
        self.smoothing: Optional[str] = None
        self.smoothing_prev: Optional[str] = None
        self.upstream_sensor_dict: Optional[dict] = None
        self.downstream_sensor_dict: Optional[dict] = None

    # ------------------------------------------------------------------#
    # Helper – choose sensor encoder
    # ------------------------------------------------------------------#
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
            "Unsupported sensor_encoding_type "
            f"{self.sensor_encoding_type!r}; choose 'ordinal', 'mean', or 'onehot'."
        )

    # ------------------------------------------------------------------#
    # Main pipeline
    # ------------------------------------------------------------------#
    def run_pipeline(
        self,
        *,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp, None]] = None,  # str | pd.Timestamp | None = None,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        relative_threshold: float = 0.7,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        use_gman_target: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
        window_size: int = 3,
        spatial_adj: int = 1,
        adj_are_relative: bool = False,
        normalize_by_distance: bool = True,
        lag_steps: int = 25,
        relative_lags: bool = True,
        horizon: int = 15,
        filter_on_train_only: bool = True,
        hour_start: int = 6,
        hour_end: int = 19,
        quantile_threshold: float = 0.9,
        quantile_percentage: float = 0.65,
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
        use_median_instead_of_mean_smoothing: bool = False,
        drop_weather: bool = True,
        add_previous_weekday_feature: bool = True,
        previous_weekday_window_min: int = 5,
    ):
        """Execute the full feature‑engineering pipeline and return train/test splits."""
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
            test_start_time=test_start_time,  # <-- new
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=add_gman_predictions,
            use_median_instead_of_mean=use_median_instead_of_mean_smoothing,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed,
        )

        # Log actual test‑set ratio if explicit date was used
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
        self.df = df
        encoder = self._get_sensor_encoder()
        df = encoder.encode(df)

        # --------------------------------------------------------------#
        # 3) Date‑time features
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
        # 7) Historical reference (previous weekday)
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
            self.feature_log["previous_day_features"] = prevday_cols

        # --------------------------------------------------------------#
        # 8) Optional drop weather columns
        # --------------------------------------------------------------#
        if drop_weather:
            dropper = WeatherFeatureDropper(weather_cols=self.weather_cols)
            df, dropped_cols = dropper.transform(df)
            self.feature_log["weather_dropped"] = dropped_cols

        # --------------------------------------------------------------#
        # 9) Target variable & prediction date
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
        # 10) Final bookkeeping & return
        # --------------------------------------------------------------#
        self.df = df
        self.all_added_features = sorted(
            {col for cols in self.feature_log.values() for col in cols}
        )

        train_df = df[~df["test_set"]].copy()
        test_df = df[df["test_set"]].copy()

        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

        # Drop helper columns that should not feed the model
        cols_to_drop = {
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

        # Store for debugging / external use
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        return X_train, X_test, y_train, y_test

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
