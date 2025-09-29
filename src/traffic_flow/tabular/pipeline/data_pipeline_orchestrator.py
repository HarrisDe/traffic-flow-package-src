from typing import Any, Dict, Optional, Sequence, List, Tuple, Union
import warnings
import os
import pandas as pd

from ..features.sensor_encoder import (
    MeanSensorEncoder,
    OrdinalSensorEncoder,
)
from ..features.calendar_features import DateTimeFeatureEngineer
from ..features.temporal_features import TemporalLagFeatureAdder
from ..features.congestion_features import CongestionFeatureEngineer
from ..features.congestion_threshold import PerSensorCongestionFlagger
from ..features.congestion_outlier_features    import GlobalOutlierFlagger
from ..features.calendar_cyclical_features import PredictionTimeCyclicalFeatureEngineer
from ..features.adjacent_features_congestion import AdjacentSensorFeatureAdderCongestion
from ..features.upstream_shifted_features import UpstreamTravelTimeShiftedFeatures
from ..features.historical_reference_features import (
    PreviousWeekdayWindowFeatureEngineer,
)
from ..features.momentum_features import MomentumFeatureEngineer
from ..features.base import SensorEncodingStrategy
from ..features.adjacent_features import AdjacentSensorFeatureAdder
from ..features.target_variable_feature import TargetVariableCreator
from ..features.misc_features import WeatherFeatureDropper
from ..features.gman_features import GMANPredictionAdder
from ...data_loading.data_loader_orchestrator import InitialTrafficDataLoader
from ..constants.constants import WEATHER_COLUMNS
from ...utils.helper_utils import LoggingMixin
from ...preprocessing.dtypes import build_dtype_schema, enforce_dtypes



class TrafficDataPipelineOrchestrator(LoggingMixin):
    """
    End-to-end, horizon-aware feature engineering orchestrator for traffic flow forecasting.

    High-level flow:
    ----------------
    1) prepare_base_features(...)
       - Load & clean data (smoothing, filtering, train/test split anchoring)
       - Sensor encoding (ordinal/mean)
       - Calendar/time features (year/month/day/hour, etc.)
       - Spatial adjacency features (neighbor speeds/distances)
       - Temporal lag features (relative or absolute lags)
       - Congestion & outlier flags (per-sensor quantiles + global bounds)
       - [Optional] Adjacency congestion features
       - [Optional] Upstream travel-time–shifted features (physics-inspired τ shift)
       - [Optional] Momentum features (slopes, EWM, volatility, minmax)

       Caches a "base_df" that is horizon-agnostic.

    2) finalise_for_horizon(...)
       - [Optional] Merge GMAN predictions (and place them before lag features)
       - Previous weekday window feature (raw previous weekday at target time)
       - Prediction-time cyclical encodings (target time and optionally current time)
       - [Optional] Drop weather columns
       - Create target columns (total speed / delta speed vs GMAN)
       - Enforce compact dtypes
       - Train/test split into X/y with final column drops for modeling

    3) run_pipeline(...)
       - Thin façade that exposes *all* arguments of both stages and runs them in order.

    Notes on internal state:
    ------------------------
    - `self.feature_log` tracks emitted feature groups and their column names.
    - `self._lag_anchor_col` stores the *name* of the first lag feature so we can
      compute its *index* later (prevents staleness when new columns are added).
    - `self.upstream_dict` and `self.downstream_dict` are the canonical attribute
      names; legacy mirrors (`upstream_sensor_dict`, `downstream_sensor_dict`) are
      maintained for backward compatibility.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        file_path: str,
        *,
        sensor_col: str = "sensor_id",
        datetime_col: str = "date",
        value_col: str = "value",
        new_sensor_col: str = "sensor_uid",
        weather_cols: Optional[List[str]] = WEATHER_COLUMNS,
        disable_logs: bool = False,
        df_gman: Optional[pd.DataFrame] = None,       # kept for legacy run_pipeline
        sensor_encoding_type: str = "mean",
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        # --- Core schema & IO
        self.file_path = file_path
        self.sensor_dict_path = os.path.dirname(file_path)

        # --- Column names
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.weather_cols = weather_cols or []

        # --- Legacy GMAN compatibility
        self.df_gman_legacy = df_gman

        # --- Config
        self.sensor_encoding_type = sensor_encoding_type

        # --- Run-time states (set during pipeline)
        self.df_orig: Optional[pd.DataFrame] = None
        self.df_orig_smoothed: Optional[pd.DataFrame] = None
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_raw_test: Optional[pd.DataFrame] = None

        self.df: Optional[pd.DataFrame] = None
        self.base_df: Optional[pd.DataFrame] = None
        self.first_test_timestamp: Optional[pd.Timestamp] = None
        self.last_test_timestamp: Optional[pd.Timestamp] = None

        self.feature_log: Dict[str, List[str]] = {}
        self.clean_state: Optional[dict] = None

        self.base_features_prepared = False
        self.horizon_finalized = False

        # Anchor to insert GMAN predictions before lag features. We store the
        # *column name* of the first lag to avoid staleness if columns are added later.
        self._lag_anchor_col: Optional[str] = None

        # Spatial neighbor graphs (canonical + legacy mirrors)
        self.upstream_dict: Optional[dict] = None
        self.downstream_dict: Optional[dict] = None
        # Legacy names kept for code that still references them
        self.upstream_sensor_dict: Optional[dict] = None
        self.downstream_sensor_dict: Optional[dict] = None

        # Feature transformers (set during fit/transform)
        self.sensor_encoder: Optional[SensorEncodingStrategy] = None
        self.datetime_fe: Optional[DateTimeFeatureEngineer] = None
        self.adjacency_fe: Optional[AdjacentSensorFeatureAdder] = None
        self.lag_fe: Optional[TemporalLagFeatureAdder] = None
        self.congestion_flagger: PerSensorCongestionFlagger | None = None
        self.outlier_flagger: GlobalOutlierFlagger | None = None
        self.momentum_fe: MomentumFeatureEngineer | None = None
        self.pred_time_cyc_fe: PredictionTimeCyclicalFeatureEngineer | None = None
        self.adjacency_fe_congested: Optional[AdjacentSensorFeatureAdderCongestion] = None
        self.upstream_shifted_fe: Optional[UpstreamTravelTimeShiftedFeatures] = None
        self.previous_day_fe: Optional[PreviousWeekdayWindowFeatureEngineer] = None
        self.weather_dropper: Optional[WeatherFeatureDropper] = None
        self.gman_adder: Optional[GMANPredictionAdder] = None

        # Train/test matrices (finalise_for_horizon)
        self.X_train = self.X_test = self.y_train = self.y_test = None

        # Misc
        self.smoothing_id: Optional[str] = None
        self.smoothing_id_prev: Optional[str] = None
        self.dtype_schema: Optional[dict] = None
        self.row_order: List[str] = [self.datetime_col, self.sensor_col]

    # ------------------------------------------------------------------ #
    # Helpers (defaults computed at call-time to pick up current attrs)
    # ------------------------------------------------------------------ #
    def _get_sensor_encoder(self):
        """Factory to obtain the configured sensor encoder strategy."""
        self._log(f"Using sensor encoding type: {self.sensor_encoding_type}")
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
                value_col=self.value_col,
                disable_logs=self.disable_logs,
            )
        raise ValueError(f"Unsupported sensor_encoding_type {self.sensor_encoding_type}")

    def _default_momentum_params(self) -> Dict[str, Any]:
        """Reasonable defaults for momentum-style features."""
        return {
            "sensor_col": getattr(self, "sensor_col", "sensor_id"),
            "value_col": getattr(self, "value_col", "value"),
            "datetime_col": getattr(self, "datetime_col", "date"),
            "slope_windows": (5, 10, 15, 30),
            "ewm_halflives": (5.0, 10.0),
            "vol_windows": (10, 30),
            "minmax_windows": (15, 30),
            "thresholds_kph": (70.0, 80.0, 90.0),
            "minutes_per_row": float(getattr(self, "minutes_per_row", 1.0)),
            "drop_fast_flag": True,
            "fast_flag_window": 5,
            "fast_flag_thresh": -1.0,
            "fill_nans_value": -1.0,
            "epsilon": 1e-6,
            "disable_logs": getattr(self, "disable_logs", False),
        }

    def _default_upstream_shifted_params(self) -> Dict[str, Any]:
        """
        Defaults for UpstreamTravelTimeShiftedFeatures. We try to reuse the
        neighbor graph discovered by AdjacentSensorFeatureAdder (if present),
        otherwise we fall back to sensor_dict_path on disk.

        Key intuition:
        --------------
        For an upstream sensor U and downstream sensor S, we align U's speed
        at time t-τ to S's time t, where τ approximates travel time between U->S
        under free-flow conditions. This can supply early signals for S.
        """
        return {
            # Prefer the in-memory upstream dict built earlier in prepare_base_features
            "upstream_dict": getattr(self, "upstream_dict", None),
            # Fallback to on-disk JSONs if upstream_dict is None
            "sensor_dict_path": getattr(self, "sensor_dict_path", None),

            "spatial_adj": 1,                 # first upstream neighbor by default

            # Column names
            "datetime_col": self.datetime_col,
            "value_col": self.value_col,
            "sensor_col": self.sensor_col,

            # τ selection = distance / freeflow_speed
            "freeflow_percentile": 0.95,      # estimate free-flow from 95th pct
            "use_upstream_freeflow": True,    # use upstream sensor’s free-flow
            "fallback_freeflow_kph": 100.0,   # used if quantile is missing/0
            "minutes_per_row": float(getattr(self, "minutes_per_row", 1.0)),
            "cap_minutes": 30.0,              # clamp τ to avoid huge shifts

            # Emitted features for each upstream neighbor k
            "add_speed": True,                # U[k] speed aligned to S at t
            "add_delta1": True,               # Δ speed vs. S(t)
            "add_slope": True,                # slope over U[k](t-τ..t-τ-w)
            "slope_windows": (3, 5),

            # Fill & logging
            "fill_nans_value": -1.0,
            "disable_logs": getattr(self, "disable_logs", False),
        }

    # ================================================================== #
    # 0-6  Horizon-independent part
    # ================================================================== #
    def prepare_base_features(
        self,
        *,
        test_size: float = 1 / 3,
        test_start_time: Optional[str | pd.Timestamp] = None,
        filter_on_train_only: bool = False,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        relative_threshold: float = 0.7,
        diagnose_extreme_changes: bool = False,
        window_size: int = 5,
        spatial_adj: int = 1,
        adj_are_relative: bool = True,
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
        add_momentum_features: bool = False,
        momentum_params: dict | None = None,
        add_adjacency_congestion_features: bool = False,
        normalize_by_distance_congested: bool = False,
        add_upstream_shifted_features: bool = False,
        upstream_shifted_params: dict | None = None,
    ) -> pd.DataFrame:
        """
        Build the horizon-agnostic feature table.

        Parameters mirror the original implementation; the ones added at the end
        are optional feature toggles with sensible defaults.

        Returns
        -------
        pd.DataFrame
            The full horizon-agnostic feature frame (including `test_set` flags).
        """
        if test_start_time is not None and test_size != 1 / 3:
            warnings.warn(
                "Both 'test_start_time' and 'test_size' supplied; "
                "explicit date split takes precedence (test_size is ignored by loader)."
            )

        self.smoothing_id = f"win{window_size}_{'med' if use_median_instead_of_mean_smoothing else 'mean'}"

        # 1) Load & basic cleaning
        loader = InitialTrafficDataLoader(
            self.file_path,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            disable_logs=self.disable_logs,
        )
        df = loader.get_data(
            window_size=window_size,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            use_median_instead_of_mean=use_median_instead_of_mean_smoothing,
            relative_threshold=relative_threshold,
            test_size=test_size,
            test_start_time=test_start_time,
            diagnose_extreme_changes=diagnose_extreme_changes,
        )
        df.sort_values(by=[self.datetime_col, self.sensor_col], inplace=True)

        # Save raw/clean references and metadata
        self.df_orig = loader.df_orig
        self.df_orig_smoothed = loader.df_orig_smoothed
        self.df_raw = loader.df_raw
        self.df_raw_test = loader.df_raw_test
        self.first_test_timestamp = loader.first_test_timestamp
        self.last_test_timestamp = loader.last_test_timestamp
        df.attrs["smoothing_id"] = self.smoothing_id
        self.clean_state = {
            "relative_threshold": relative_threshold,
            "smoothing_window": window_size,
            "use_median": use_median_instead_of_mean_smoothing,
            "filter_extreme_changes": bool(filter_extreme_changes),
            "filter_on_train_only": bool(filter_on_train_only),
            "smooth_speeds": bool(smooth_speeds),
        }

        # 2) Sensor encoding
        encoder = self._get_sensor_encoder()
        encoder.fit(df)
        df = encoder.transform(df)
        self.sensor_encoder = encoder
        # Store mapping if your encoders provide it
        self.sensor_encoder_mapping = getattr(encoder, "mapping_", None)

        # 3) Date-time features
        dt_fe = DateTimeFeatureEngineer(datetime_col=self.datetime_col)
        dt_fe.fit(df)
        df = dt_fe.transform(df)
        self.datetime_fe = dt_fe
        self.feature_log["datetime_features"] = dt_fe.feature_names_out_

        # 4) Spatial adjacency (fit -> read dicts -> transform)
        spatial = AdjacentSensorFeatureAdder(
            sensor_dict_path=self.sensor_dict_path,
            spatial_adj=spatial_adj,
            normalize_by_distance=normalize_by_distance,
            adj_are_relative=adj_are_relative,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            sensor_col=self.sensor_col,
            disable_logs=self.disable_logs,
            smoothing_id=self.smoothing_id,
        )
        spatial.fit(df)
        # Cache neighbor graphs (canonical + legacy mirror names)
        self.downstream_dict = spatial.downstream_dict_
        self.upstream_dict = spatial.upstream_dict_
        self.downstream_sensor_dict = self.downstream_dict
        self.upstream_sensor_dict = self.upstream_dict

        df = spatial.transform(df)
        self.adjacency_fe = spatial
        self.feature_log["spatial_features"] = spatial.feature_names_out_

        # 5) Temporal lags
        lag_fe = TemporalLagFeatureAdder(
            lags=lag_steps,
            relative=relative_lags,
            fill_nans_value=-1,
            disable_logs=self.disable_logs,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            smoothing_id=self.smoothing_id,
        )
        lag_fe.fit(df)
        df = lag_fe.transform(df)
        self.lag_fe = lag_fe
        self.feature_log["lag_features"] = lag_fe.feature_names_out_

        # Anchor the *name* of the first lag column to prevent staleness.
        if self.feature_log["lag_features"]:
            self._lag_anchor_col = self.feature_log["lag_features"][0]
        else:
            self._lag_anchor_col = None

        # 6) Congestion & outlier flags
        cong_flagger = PerSensorCongestionFlagger(
            hour_start=hour_start,
            hour_end=hour_end,
            quantile_threshold=quantile_threshold,
            quantile_percentage=quantile_percentage,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            hour_col="hour",
            disable_logs=self.disable_logs,
        )
        outlier_flagger = GlobalOutlierFlagger(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            value_col=self.value_col,
            sensor_col=self.sensor_col,
            disable_logs=self.disable_logs,
        )
        cong_flagger.fit(df)
        outlier_flagger.fit(df)
        df = cong_flagger.transform(df)
        df = outlier_flagger.transform(df)
        self.congestion_flagger = cong_flagger
        self.outlier_flagger = outlier_flagger
        self.feature_log["congestion_features"] = (
            cong_flagger.feature_names_out_ + outlier_flagger.feature_names_out_
        )

        # 6.5) Optional adjacency congestion features
        if add_adjacency_congestion_features:
            spatial_congested = AdjacentSensorFeatureAdderCongestion(
                sensor_dict_path=self.sensor_dict_path,
                spatial_adj=spatial_adj,
                normalize_by_distance=normalize_by_distance_congested,
                datetime_col=self.datetime_col,
                value_col="is_congested",
                sensor_col=self.sensor_col,
                disable_logs=self.disable_logs,
                smoothing_id=self.smoothing_id,
            )
            spatial_congested.fit(df)
            df = spatial_congested.transform(df)
            self.adjacency_fe_congested = spatial_congested
            self.feature_log["spatial_features_congested"] = spatial_congested.feature_names_out_

        # 7) Optional upstream travel-time–shifted features (horizon-agnostic)
        if add_upstream_shifted_features:
            df = df.sort_values([self.sensor_col, self.datetime_col])  # ensure order for rolling
            us_kwargs = self._default_upstream_shifted_params()
            if upstream_shifted_params is not None:
                # Merge *without* mutating caller; preserve defaults and override with user
                us_kwargs = {**us_kwargs, **upstream_shifted_params}
            upshift_fe = UpstreamTravelTimeShiftedFeatures(**us_kwargs)
            upshift_fe.fit(df)
            df = upshift_fe.transform(df)
            self.upstream_shifted_fe = upshift_fe
            self.feature_log["upstream_shifted_features"] = list(upshift_fe.feature_names_out_)
            df = df.sort_values([self.datetime_col, self.sensor_col])

        # 8) Optional momentum features (still horizon-agnostic)
        if add_momentum_features:
            df = df.sort_values([self.sensor_col, self.datetime_col])
            mm_kwargs = self._default_momentum_params()
            if momentum_params is not None:
                mm_kwargs = {**mm_kwargs, **momentum_params}
            mom_fe = MomentumFeatureEngineer(**mm_kwargs)
            mom_fe.fit(df)
            df = mom_fe.transform(df)
            self.momentum_fe = mom_fe
            self.feature_log["momentum_features"] = list(mom_fe.feature_names_out_)
            df = df.sort_values([self.datetime_col, self.sensor_col])

        # Cache and mark completion (once)
        self.base_df = df
        self.smoothing_id_prev = self.smoothing_id
        self.base_features_prepared = True

        return df.copy()

    # ================================================================== #
    # 7-10  Horizon-specific part
    # ================================================================== #
    def finalise_for_horizon(
        self,
        *,
        horizon: int = 15,
        df_gman: Optional[pd.DataFrame] = None,
        convert_gman_prediction_to_delta_speed: bool = True,
        add_previous_weekday_feature: bool = True,
        previous_weekday_window_min: int = 0,
        drop_weather: bool = True,
        use_gman_target: bool = False,
        drop_missing_gman_rows: bool = False,
        drop_datetime: bool = True,
        drop_sensor_id: bool = True,
        add_prediction_time_cyclical_features: bool = True,
        include_current_time_cyclical: bool = True,  # aligned default with run_pipeline
    ):
        """
        Add horizon-specific signals, targets, and split into X/y.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        if not self.base_features_prepared:
            raise RuntimeError("Call prepare_base_features() first.")

        if not (drop_datetime and drop_sensor_id):
            warnings.warn(
                "[finalise_for_horizon] drop_datetime/drop_sensor_id are typically True "
                "for modeling; set to True or drop those columns manually before training."
            )

        df = self.base_df.copy()
        df = df.sort_values(self.row_order, kind="mergesort").reset_index(drop=True)

        # --- GMAN merge (horizon-specific)
        if df_gman is not None:
            adder = GMANPredictionAdder(
                df_gman,
                sensor_col=self.sensor_col,
                datetime_col=self.datetime_col,
                convert_to_delta=convert_gman_prediction_to_delta_speed,
                drop_missing=drop_missing_gman_rows,
            )
            df = adder.transform(df, value_col=self.value_col)
            gman_cols = [adder.prediction_col]
            if getattr(adder, "keep_target_date", False) and "gman_target_date" in df.columns:
                gman_cols.append("gman_target_date")

            # Keep legacy ordering: insert GMAN cols *before* first lag feature.
            cols_current = list(df.columns)
            for c in gman_cols:
                cols_current.remove(c)

            if self._lag_anchor_col and self._lag_anchor_col in df.columns:
                anchor = df.columns.get_loc(self._lag_anchor_col)
            else:
                anchor = len(cols_current)  # no lags; append to the end

            cols_current[anchor:anchor] = gman_cols
            df = df[cols_current]
            self.gman_adder = adder
            self.feature_log["gman_predictions"] = gman_cols

        # --- previous weekday (raw lookups around previous weekday target time)
        if add_previous_weekday_feature:
            prev = PreviousWeekdayWindowFeatureEngineer(
                datetime_col=self.datetime_col,
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                horizon_min=horizon,
                window_before_min=previous_weekday_window_min,
                window_after_min=previous_weekday_window_min,
                step_min=1,
                aggs=[],  # if your transformer expects an explicit raw feature, keep as designed
                disable_logs=self.disable_logs,
            )
            prev.fit(df)
            df = prev.transform(df)
            self.previous_day_fe = prev
            self.feature_log["previous_day_features"] = prev.feature_names_out_

        # --- prediction-time cyclical encodings (target-time; optionally current time)
        if add_prediction_time_cyclical_features:
            ptfe = PredictionTimeCyclicalFeatureEngineer(
                datetime_col=self.datetime_col,
                horizon_min=horizon,
                add_day=True,
                add_hour=True,
                add_minute=True,
                include_current_time=include_current_time_cyclical,
                include_forecast_time=True,   # always include encodings for target time
                disable_logs=self.disable_logs,
            )
            ptfe.fit(df)  # stateless, but keeps parity with other components
            df = ptfe.transform(df)
            self.pred_time_cyc_fe = ptfe
            self.feature_log["prediction_time_cyclical"] = list(ptfe.feature_names_out_)

        # --- optionally drop weather columns
        if drop_weather:
            weather_dropper = WeatherFeatureDropper(
                weather_cols=self.weather_cols, disable_logs=self.disable_logs
            )
            weather_dropper.fit(df)
            df = weather_dropper.transform(df)
            self.feature_log["weather_dropped"] = weather_dropper.cols_to_drop_
            self.weather_dropper = weather_dropper

        # --- target & prediction timestamps
        df["prediction_time"] = df.groupby(self.sensor_col)[self.datetime_col].shift(-horizon)

        target_creator = TargetVariableCreator(
            horizon=horizon,
            sensor_col=self.sensor_col,
            value_col=self.value_col,
            datetime_col=self.datetime_col,
            gman_col="gman_prediction",
            use_gman=use_gman_target,
        )
        target_creator.fit(df)
        df = target_creator.transform(df)
        self.target_creator = target_creator
        self.feature_log["target_variables"] = target_creator.feature_names_out_

        # --- enforce compact dtypes (features + targets + bookkeeping)
        dtype_schema = build_dtype_schema(df)
        df = enforce_dtypes(df, dtype_schema)
        self.dtype_schema = dtype_schema

        # --- final split
        self.df = df.sort_values(by=[self.datetime_col, self.sensor_col])
        train_df = self.df[~self.df["test_set"]].copy()
        test_df = self.df[self.df["test_set"]].copy()

        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

        # Columns to drop for modeling (if present)
        cols_to_drop: List[str] = [
            "target_total_speed",
            "target_speed_delta",
            "test_set",
            "gman_prediction_date",
            "gman_target_date",
            "prediction_time",
        ]
        if drop_datetime:
            cols_to_drop.append(self.datetime_col)
        if drop_sensor_id:
            cols_to_drop.append(self.sensor_col)

        for subset in (X_train, X_test):
            subset.drop(columns=[c for c in cols_to_drop if c in subset.columns], inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.horizon_finalized = True
        return X_train, X_test, y_train, y_test

    # ================================================================== #
    # Legacy façade – same signature / behaviour as before, but complete #
    # ================================================================== #
    def run_pipeline(
        self,
        *,
        # --- base args (complete set) ---
        test_size: float = 1 / 3,
        test_start_time: Optional[str | pd.Timestamp] = None,
        filter_on_train_only: bool = False,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        relative_threshold: float = 0.7,
        diagnose_extreme_changes: bool = False,
        window_size: int = 5,
        spatial_adj: int = 1,
        adj_are_relative: bool = True,
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
        add_momentum_features: bool = False,
        momentum_params: dict | None = None,
        add_adjacency_congestion_features: bool = False,
        normalize_by_distance_congested: bool = False,
        add_upstream_shifted_features: bool = False,
        upstream_shifted_params: dict | None = None,

        # --- horizon-specific args (complete set) ---
        horizon: int = 15,
        drop_weather: bool = True,
        add_previous_weekday_feature: bool = True,
        previous_weekday_window_min: int = 0,
        use_gman_target: bool = False,
        drop_missing_gman_rows: bool = False,
        drop_datetime: bool = True,
        drop_sensor_id: bool = True,
        add_prediction_time_cyclical_features: bool = True,
        include_current_time_cyclical: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
    ):
        """
        Convenience façade that runs the full pipeline with a single call, exposing
        all arguments of `prepare_base_features` and `finalise_for_horizon`.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        # 1) base (horizon-independent) part
        self.prepare_base_features(
            test_size=test_size,
            test_start_time=test_start_time,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            relative_threshold=relative_threshold,
            diagnose_extreme_changes=diagnose_extreme_changes,
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
            add_momentum_features=add_momentum_features,
            momentum_params=momentum_params,
            add_adjacency_congestion_features=add_adjacency_congestion_features,
            normalize_by_distance_congested=normalize_by_distance_congested,
            add_upstream_shifted_features=add_upstream_shifted_features,
            upstream_shifted_params=upstream_shifted_params,
        )

        # 2) horizon-specific part
        return self.finalise_for_horizon(
            horizon=horizon,
            df_gman=self.df_gman_legacy if add_gman_predictions else None,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed,
            drop_weather=drop_weather,
            add_previous_weekday_feature=add_previous_weekday_feature,
            previous_weekday_window_min=previous_weekday_window_min,
            use_gman_target=use_gman_target,
            drop_missing_gman_rows=drop_missing_gman_rows,
            drop_datetime=drop_datetime,
            drop_sensor_id=drop_sensor_id,
            add_prediction_time_cyclical_features=add_prediction_time_cyclical_features,
            include_current_time_cyclical=include_current_time_cyclical,
        )

    # ================================================================== #
    # Export
    # ================================================================== #
    def export_states(self, strict: bool = False) -> dict:
        """
        Collect everything an inference pipeline will need.
        strict=False  -> never raises for missing components; absent parts export as None.
        strict=True   -> raises if typical training artifacts are missing (optional).
        """
        if not self.base_features_prepared:
            raise RuntimeError("Base features not prepared. Run prepare_base_features() first.")
        if not self.horizon_finalized:
            raise RuntimeError("Horizon not finalized. Run finalise_for_horizon() first.")

        def safe_export(obj: Optional[object]) -> Optional[dict]:
            """Safely export state of an object, or None if not available."""
            return obj.export_state() if obj is not None else None

        states = {
            # Schema/cleaning metadata
            "schema_state": {
                "sensor_col": getattr(self, "sensor_col", None),
                "datetime_col": getattr(self, "datetime_col", None),
                "value_col": getattr(self, "value_col", None),
                "row_order": getattr(self, "row_order", None),
            },
            "clean_state": getattr(self, "clean_state", None),

            # Core transformers
            "sensor_encoder_state": safe_export(self.sensor_encoder),
            "datetime_state": safe_export(self.datetime_fe),
            "adjacency_state": safe_export(self.adjacency_fe),
            "lag_state": safe_export(self.lag_fe),
            "congestion_state": safe_export(self.congestion_flagger),
            "outlier_state": safe_export(self.outlier_flagger),
            "adjacent_congestion_state": safe_export(self.adjacency_fe_congested),
            "momentum_state": safe_export(self.momentum_fe),
            "upstream_shifted_state": safe_export(self.upstream_shifted_fe),

            "prediction_time_cyc_state": safe_export(self.pred_time_cyc_fe),

            # Targets
            "target_state": safe_export(self.target_creator),

            # Horizon-specific / optional
            "previous_day_state": safe_export(self.previous_day_fe),
            "weather_state": safe_export(self.weather_dropper),
            "gman_state": safe_export(self.gman_adder),

            # Model wiring
            "feature_cols": list(self.X_train.columns) if self.X_train is not None else None,
            "dtype_schema": getattr(self, "dtype_schema", None),
        }

        if strict:
            must = [
                "schema_state", "clean_state", "sensor_encoder_state",
                "datetime_state", "adjacency_state", "lag_state",
                "congestion_state", "outlier_state", "target_state",
                "feature_cols", "dtype_schema",
            ]
            missing = [k for k in must if states.get(k) in (None, {}, [])]
            if missing:
                raise RuntimeError(f"export_states(strict=True): missing pieces: {missing}")

        return states


