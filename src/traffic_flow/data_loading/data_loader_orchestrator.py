# traffic_flow_package_src/data_loading/data_loader_orchestrator.py
"""
InitialTrafficDataLoader  - now free of horizon-dependent logic.
Python 3.6 compatible.
"""
from typing import List, Optional, Union, Any
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from ..tabular.features.gman_features import GMANPredictionAdder
from ..utils.helper_utils import LoggingMixin
from ..preprocessing.cleaning import (
    clean_and_cast,
    filter_and_interpolate_extremes,
    smooth_speeds,
)


Float = np.float32  # alias only for typing comments


class InitialTrafficDataLoader(LoggingMixin):  # type: ignore[misc]
    """Load, sanitise and prepare traffic-sensor speed data."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        file_path: Union[str, os.PathLike],
        *,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        datetime_cols: Optional[List[str]] = None,
        disable_logs: bool = False,
        df_gman: Optional[pd.DataFrame] = None,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        self.file_path = file_path
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.datetime_cols = datetime_cols or ["datetime", "date"]
        self.df_gman = df_gman                   # kept for legacy calls
        self.test_start_time = (
            pd.to_datetime(test_start_time) if test_start_time is not None else None
        )

        # mutable state
        self.df: Optional[pd.DataFrame] = None
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_orig: Optional[pd.DataFrame] = None
        self.df_orig_smoothed: Optional[pd.DataFrame] = None
        self.datetime_col: Optional[str] = None
        self.first_test_timestamp: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_clean_and_convert_to_float32(self) -> None:
        assert self.df is not None
        self.df = clean_and_cast(self.df, value_col=self.value_col)

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    
    
    def load_data_parquet(self) -> None:
        
        path = str(self.file_path)
        print(f"path: {path}")
        if not (os.path.isfile(path) or os.path.isdir(path)):
            raise FileNotFoundError(f"Parquet path not found: {path}")

        #self.df = self._read_parquet_robust(self.file_path)
        self.df = pd.read_parquet(self.file_path)
        self.df_raw = self.df.copy()  # keep the raw data for reference

        # resolve datetime column
        for col in self.datetime_cols:
            if col in self.df.columns:
                self.datetime_col = col
                self.df[col] = pd.to_datetime(self.df[col])
                break
        else:
            raise ValueError(
                "None of the datetime columns {} were found.".format(self.datetime_cols)
            )

        if self.sensor_col not in self.df.columns:
            raise ValueError(
                "Missing sensor column '{}'. Available: {}".format(
                    self.sensor_col, self.df.columns.tolist()
                )
            )

        self.df_orig = self.df.copy()
        self.df = (
            self.df.sort_values([self.datetime_col, self.sensor_col])
            .reset_index(drop=True)
        )
        self._log("Loaded {} rows from '{}'.".format(len(self.df), self.file_path))

        self._apply_clean_and_convert_to_float32()

    # ------------------------------------------------------------------ #
    # Alignment
    # ------------------------------------------------------------------ #
    def align_sensors_to_common_timeframe(self) -> None:
        assert self.df is not None and self.datetime_col is not None
        anchor = self.df.groupby(self.sensor_col).size().idxmin()
        timeframe = self.df[self.df[self.sensor_col] == anchor][self.datetime_col]
        min_t, max_t = timeframe.min(), timeframe.max()

        pre = len(self.df)
        self.df = self.df[
            (self.df[self.datetime_col] >= min_t) & (self.df[self.datetime_col] <= max_t)
        ]
        self._log(
            "Aligned sensors to {}-{}. Dropped {} rows.".format(
                min_t, max_t, pre - len(self.df)
            )
        )

    # ------------------------------------------------------------------ #
    # Train / test split
    # ------------------------------------------------------------------ #
    def add_test_set_column(
        self,
        *,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        assert self.df is not None and self.datetime_col is not None
        explicit = test_start_time or self.test_start_time
        if explicit is not None:
            split_time = pd.to_datetime(explicit)
            self._log("Using explicit `test_start_time` = {}".format(split_time))
        else:
            unique_times = self.df[self.datetime_col].sort_values().unique()
            split_idx = max(0, min(int(len(unique_times) * (1 - test_size)), len(unique_times) - 1))
            split_time = pd.Timestamp(unique_times[split_idx])
            self._log("Proportional split (test_size={}) at {}".format(test_size, split_time))

        self.df["test_set"] = self.df[self.datetime_col] >= split_time
        self.first_test_timestamp = split_time
        self.last_test_timestamp = self.df.loc[self.df['test_set'],self.datetime_col].max()
        
        if self.df_raw is not None and self.datetime_col is not None:
            raw_dt = pd.to_datetime(self.df_raw[self.datetime_col], errors="coerce")
            self.df_raw_test = self.df_raw[raw_dt >= split_time].copy()

    # ------------------------------------------------------------------ #
    # Public method
    # ------------------------------------------------------------------ #
    def get_data(
        self,
        *,
        window_size: int = 5,
        filter_on_train_only: bool = False,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        use_median_instead_of_mean: bool = False,
        relative_threshold: float = 0.7,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
    ) -> pd.DataFrame:
        """
        Return a cleaned dataframe.

        *add_gman_predictions* is kept for backward compatibility – a
        DeprecationWarning is raised and the merge is delegated to
        ``GMANPredictionAdder`` (requires ``df_gman``).
        """
        self.load_data_parquet()
        self.align_sensors_to_common_timeframe()
        self.add_test_set_column(test_size=test_size, test_start_time=test_start_time)

        if diagnose_extreme_changes:
            mask = self._compute_relative_change_mask(relative_threshold)
            self._log(
                "Large changes (> {:.0%}): {} / {} rows ({:.2%})".format(
                    relative_threshold, mask.sum(), len(mask), mask.mean()
                )
            )
        if filter_extreme_changes:
            if filter_on_train_only and "test_set" in self.df.columns:
                mask = ~self.df['test_set']
                self.df.loc[mask] = filter_and_interpolate_extremes(
                    self.df.loc[mask], sensor_col=self.sensor_col,
                    value_col=self.value_col, threshold=relative_threshold
                )
            else:
                self.df = filter_and_interpolate_extremes(
                self.df, sensor_col=self.sensor_col,
                value_col=self.value_col, threshold=relative_threshold
                )
                #self.filter_extreme_changes(relative_threshold=relative_threshold)
        if smooth_speeds:
            self.smooth_speeds(
                window_size=window_size,
                filter_on_train_only=filter_on_train_only,
                use_median_instead_of_mean=use_median_instead_of_mean,
            )
            self.df_orig_smoothed = self.df

        if add_gman_predictions:
            warnings.warn(
                "add_gman_predictions=True is deprecated in "
                "InitialTrafficDataLoader.  Merge GMAN data via "
                "TrafficDataPipelineOrchestrator.finalise_for_horizon() instead.",
                DeprecationWarning,
            )
            if self.df_gman is None:
                raise ValueError("df_gman must be provided for GMAN merge.")
            adder = GMANPredictionAdder(
                self.df_gman,
                sensor_col=self.sensor_col,
                datetime_col=self.datetime_col,  # type: ignore[arg-type]
                convert_to_delta=convert_gman_prediction_to_delta_speed,
            )
            self.df = adder.transform(self.df, value_col=self.value_col)

        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)
        return self.df.copy()

    # ------------------------------------------------------------------ #
    # (Private) helpers reused from earlier code
    # ------------------------------------------------------------------ #
    def _compute_relative_change_mask(self, threshold: float) -> pd.Series:
        assert self.df is not None
        tmp = "__pct_change__"
        self.df[tmp] = (
            self.df.groupby(self.sensor_col)[self.value_col].pct_change().abs()
        )
        mask = self.df[tmp] > threshold
        self.df.drop(columns=[tmp], inplace=True)
        return mask

    # def filter_extreme_changes(self, *, relative_threshold: float = 0.7) -> None:
    #     assert self.df is not None
    #     self.df = filter_and_interpolate_extremes(
    #     self.df,
    #     sensor_col=self.sensor_col,
    #     value_col=self.value_col,
    #     threshold=relative_threshold,
    # )


    
    def smooth_speeds(
        self,
        *,
        window_size: int = 5,
        filter_on_train_only: bool = True,
        use_median_instead_of_mean: bool = True,
        smooth_on_train_and_test_separately: bool = True,
        ) -> None:
        assert self.df is not None

        if filter_on_train_only and smooth_on_train_and_test_separately:
            warnings.warn(
                "`filter_on_train_only=True` implies test is untouched, "
                "so `smooth_on_train_and_test_separately=True` has no effect.",
                RuntimeWarning,
            )

        if filter_on_train_only and "test_set" in self.df.columns:
            # Smooth only training set
            mask = ~self.df["test_set"]
            smoothed_part = smooth_speeds(
                self.df.loc[mask],
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                window_size=window_size,
                use_median=use_median_instead_of_mean,
            )
            self.df.loc[mask] = smoothed_part
            self.df.loc[mask, self.value_col] = smoothed_part[self.value_col].astype(np.float32).to_numpy()

        elif smooth_on_train_and_test_separately and "test_set" in self.df.columns:
            # Smooth train and test independently
            for subset in [True, False]:
                mask = self.df["test_set"] == subset
                smoothed = smooth_speeds(
                    self.df.loc[mask],
                    sensor_col=self.sensor_col,
                    value_col=self.value_col,
                    window_size=window_size,
                    use_median=use_median_instead_of_mean,
                )
                self.df.loc[mask, self.value_col] = smoothed[self.value_col].astype(np.float32).to_numpy()

        else:
            # Smooth full dataset
            self.df = smooth_speeds(
                self.df,
                sensor_col=self.sensor_col,
                value_col=self.value_col,
                window_size=window_size,
                use_median=use_median_instead_of_mean,
            )
            
            self.df[self.value_col] = self.df[self.value_col].astype(np.float32).to_numpy()


    def convert_to_ts_dataframe(self,
        *,
        window_size: int = 3,
        filter_on_train_only: bool = False,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        use_median_instead_of_mean: bool = False,
        relative_threshold: float = 0.7,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
    ) -> pd.DataFrame:
        
        df = self.get_data(window_size=window_size,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            use_median_instead_of_mean=use_median_instead_of_mean,
            relative_threshold=relative_threshold,
            test_size=test_size,
            test_start_time=test_start_time,
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=add_gman_predictions,
            convert_gman_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed,
        )
        
        # wide values: date index, one column per sensor
        values_wide = (
            df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
            .sort_index()
        )

        # single test flag per date (identical across sensors)
        test_per_date = (
            df[[self.datetime_col, "test_set"]]
            .drop_duplicates(subset=self.datetime_col, keep="last")
            .set_index(self.datetime_col)
        )

        out = values_wide.join(test_per_date).reset_index()
        out.columns.name = None

        # ensure 'test_set' is the last column
        out = out[[c for c in out.columns if c != "test_set"] + ["test_set"]]

        return out