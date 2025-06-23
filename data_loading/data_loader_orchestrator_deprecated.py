from typing import Any, List, Union, Optional
from ..utils.helper_utils import LoggingMixin
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------------#
#  Helper type alias
# -----------------------------------------------------------------------------#
Float = np.float32

# -----------------------------------------------------------------------------#
#  Modern implementation
# -----------------------------------------------------------------------------#


class InitialTrafficDataLoader(LoggingMixin):  # type: ignore[misc]
    """Load, sanitise and prepare traffic-sensor speed data.

    Parameters
    ----------
    file_path
        Path to the **Parquet** file containing the raw measurements.
    datetime_cols
        Candidate column names that may hold the timestamp.  The first match
        becomes the canonical date-time column.  Default: ``["datetime", "date"]``.
    sensor_col
        Column with the sensor ID.  Default: ``"sensor_id"``.
    value_col
        Column holding the primary measurement (e.g. *speed*).  Default: ``"value"``.
    disable_logs
        Suppress messages emitted via :class:`LoggingMixin`.
    df_gman
        Optional GMAN-prediction dataframe (needed only if
        :pymeth:`add_gman_predictions` is called).
    test_start_time
        **New** – explicit, *inclusive* timestamp that marks the beginning of the
        test set.  Overrides the fractional `test_size` split when supplied.
    """

    # ------------------------------------------------------------------#
    # Construction / immutable state
    # ------------------------------------------------------------------#
    def __init__(
        self,
        file_path: Union[str, os.PathLike],
        *,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        datetime_cols: Optional[List[str]] = ['date'],
        disable_logs: bool = False,
        df_gman: Optional[pd.DataFrame] = None,
        test_start_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        super().__init__(disable_logs=disable_logs)

        if datetime_cols is None:
            datetime_cols = ["datetime", "date"]

        self.file_path = file_path
        self.datetime_cols: List[str] = datetime_cols
        self.sensor_col: str = sensor_col
        self.value_col= value_col
        self.df_gman = df_gman
        self.test_start_time = (
            pd.to_datetime(
                test_start_time) if test_start_time is not None else None
        )

        # Mutable state populated during processing
        self.df: Optional[pd.DataFrame] = None               # working copy
        self.df_orig: Optional[pd.DataFrame] = None          # pristine clone
        self.datetime_col: Optional[str] = None              # resolved name
        self.first_test_timestamp: Optional[pd.Timestamp] = None
        self.df_as_gman_input: Optional[pd.DataFrame] = None
        self.df_as_gman_input_orig: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------#
    # Utilities
    # ------------------------------------------------------------------#
    def _clean_and_convert_to_float32(self, value: Any) -> Optional[Float]:
        """Coerce numeric/string to **float32** rounded to two decimals."""
        try:
            if isinstance(value, float):
                return np.float32(round(value, 2))
            return np.float32(round(float(str(value).replace(" ", ".")), 2))
        except (ValueError, AttributeError):
            return None  # will be dropped later

    def _apply_clean_and_convert_to_float32(self) -> None:
        """Vector-ised cleansing of the *value* column with logging."""
        assert self.df is not None
        prev_len = len(self.df)
        self.df = self.df.dropna(subset=[self.value_col])
        dropped = prev_len - len(self.df)
        if dropped:
            self._log(f"Discarded {dropped} rows with NaN '{self.value_col}'.")
        self.df[self.value_col] = (
            self.df[self.value_col]
            .apply(self._clean_and_convert_to_float32)
            .astype(np.float32)
        )

    # ------------------------------------------------------------------#
    # I/O
    # ------------------------------------------------------------------#
    def load_data_parquet(self) -> None:
        """Load parquet file and perform minimal validation."""
        self.df = pd.read_parquet(self.file_path, engine="pyarrow")

        # sensor column
        if self.sensor_col not in self.df.columns:
            raise ValueError(
                f"Missing sensor column '{self.sensor_col}'. "
                f"Available: {self.df.columns.tolist()}"
            )

        # detect datetime column
        for col in self.datetime_cols:
            if col in self.df.columns:
                self.datetime_col = col
                self.df[col] = pd.to_datetime(self.df[col])
                break
        else:
            raise ValueError(
                f"None of the datetime columns {self.datetime_cols} found."
            )

        # housekeeping
        self.df_orig = self.df.copy()
        self.df = (
            self.df.sort_values([self.datetime_col, self.sensor_col])
            .reset_index(drop=True)
        )
        self._log(f"Loaded {len(self.df)} rows from '{self.file_path}'.")

        self._apply_clean_and_convert_to_float32()

    # ------------------------------------------------------------------#
    # Alignment
    # ------------------------------------------------------------------#
    def align_sensors_to_common_timeframe(self) -> None:
        """Restrict data to the shared time-range of all sensors."""
        assert self.df is not None and self.datetime_col is not None
        anchor = self.df.groupby(self.sensor_col).size().idxmin()
        timeframe = self.df[self.df[self.sensor_col]
                            == anchor][self.datetime_col]
        min_t, max_t = timeframe.min(), timeframe.max()

        pre = len(self.df)
        self.df = self.df[
            (self.df[self.datetime_col] >= min_t)
            & (self.df[self.datetime_col] <= max_t)
        ]
        self._log(
            f"Aligned sensors to {min_t}–{max_t}.  Dropped {pre - len(self.df)} rows."
        )

    # ------------------------------------------------------------------#
    # Train / test split
    # ------------------------------------------------------------------#
    def add_test_set_column(
        self,
        *,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp, None]]= None,
    ) -> None:
        """Add **test_set** boolean column using size or explicit timestamp."""
        assert self.df is not None and self.datetime_col is not None

        explicit = test_start_time or self.test_start_time
        if explicit is not None:
            split_time = pd.to_datetime(explicit)
            self._log(f"Using explicit `test_start_time` = {split_time}")
        else:
            unique_times = self.df[self.datetime_col].sort_values().unique()
            split_idx = max(
                0, min(int(len(unique_times) * (1 - test_size)),
                       len(unique_times) - 1)
            )
            split_time = pd.Timestamp(unique_times[split_idx])
            self._log(
                f"Proportional split (test_size={test_size}) at {split_time}")

        self.df["test_set"] = self.df[self.datetime_col] >= split_time
        self.first_test_timestamp = split_time

    # ------------------------------------------------------------------#
    # Anomaly detection / smoothing
    # ------------------------------------------------------------------#
    def _compute_relative_change_mask(self, threshold: float) -> pd.Series:
        """Mask where absolute pct-change exceeds *threshold*."""
        assert self.df is not None
        tmp = "__pct_change__"
        self.df[tmp] = (
            self.df.groupby(self.sensor_col)[self.value_col].pct_change().abs()
        )
        mask = self.df[tmp] > threshold
        self.df.drop(columns=[tmp], inplace=True)
        return mask

    def diagnose_extreme_changes(self, *, relative_threshold: float = 0.7) -> None:
        mask = self._compute_relative_change_mask(relative_threshold)
        self._log(
            f"Large changes (> {relative_threshold:.0%}): "
            f"{mask.sum()} / {len(mask)} rows ({mask.mean():.2%})"
        )

    def filter_extreme_changes(self, *, relative_threshold: float = 0.7) -> None:
        assert self.df is not None
        mask = self._compute_relative_change_mask(relative_threshold)
        self.df.loc[mask, self.value_col] = np.nan
        self.df[self.value_col] = (
            self.df.groupby(self.sensor_col)[self.value_col]
            .transform(lambda s: s.interpolate().ffill().bfill())
        )
        self._log(f"Interpolated {mask.sum()} extreme changes.")

    def smooth_speeds(
        self,
        *,
        window_size: int = 3,
        filter_on_train_only: bool = True,
        use_median_instead_of_mean: bool = True,
    ) -> None:
        """Apply rolling median/mean smoothing (optionally train-only)."""
        assert self.df is not None
        if "test_set" not in self.df.columns:
            self.add_test_set_column()

        mask = (
            ~self.df["test_set"] if filter_on_train_only else self.df.index == self.df.index
        )

        # decide once which rolling statistic to apply
        if use_median_instead_of_mean:
            def roll_fn(s): return s.rolling(window=window_size,
                                             min_periods=1, center=False).median()
        else:
            def roll_fn(s): return s.rolling(window=window_size,
                                             min_periods=1, center=False).mean()

        smoothed = (
            self.df.loc[mask]
            .groupby(self.sensor_col)[self.value_col]
            .transform(roll_fn)
        )

        self.df.loc[mask, self.value_col] = smoothed.ffill().bfill()

        self._log(
            f"Smoothing applied (window={window_size}, "
            f"{'median' if use_median_instead_of_mean else 'mean'}, "
            f"{'train-only' if filter_on_train_only else 'all rows'})."
        )

    # ------------------------------------------------------------------#
    # GMAN merge utilities (legacy + modern)
    # ------------------------------------------------------------------#
    def add_gman_predictions_deprecated(
        self,
        *,
        model_prediction_col: str = "gman_prediction",
        model_prediction_date_col: str = "gman_prediction_date",
    ) -> None:
        """Legacy GMAN merge (kept untouched for reproducibility)."""
        assert self.df is not None and self.df_gman is not None

        self._log("Merging GMAN data (deprecated) …")
        self.df[self.sensor_col] = self.df[self.sensor_col].astype("category")
        self.df_gman[self.sensor_col] = self.df_gman[self.sensor_col].astype(
            "category")
        self.df_gman[model_prediction_date_col] = pd.to_datetime(
            self.df_gman[model_prediction_date_col]
        )
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

        min_d, max_d = self.df[self.datetime_col].min(
        ), self.df[self.datetime_col].max()
        self.df_gman = self.df_gman[
            (self.df_gman[model_prediction_date_col] >= min_d)
            & (self.df_gman[model_prediction_date_col] <= max_d)
        ]

        self.df = self.df.set_index([self.datetime_col, self.sensor_col])
        self.df_gman = self.df_gman.set_index(
            [model_prediction_date_col, self.sensor_col]
        )
        self.df = self.df.join(self.df_gman, how="left").reset_index()

        missing = self.df[self.df[model_prediction_col].isna()]
        if not missing.empty:
            self._log(
                f"Dropping {len(missing)} rows with missing '{model_prediction_col}'."
            )
            self.df = self.df.dropna(subset=[model_prediction_col])

    def add_gman_predictions(
        self,
        *,
        convert_prediction_to_delta_speed: bool = True,
        model_prediction_col: str = "gman_prediction",
        model_prediction_date_col: str = "gman_prediction_date",
        keep_target_date: bool = True,
    ) -> None:
        """Modern, efficient GMAN merge."""
        assert self.df is not None and self.df_gman is not None and self.datetime_col

        self._log("Merging GMAN data …")

        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        self.df_gman[model_prediction_date_col] = pd.to_datetime(
            self.df_gman[model_prediction_date_col]
        )

        min_d, max_d = self.df[self.datetime_col].min(
        ), self.df[self.datetime_col].max()
        gman_trim = self.df_gman[
            (self.df_gman[model_prediction_date_col] >= min_d)
            & (self.df_gman[model_prediction_date_col] <= max_d)
        ]

        cols = [self.sensor_col, model_prediction_date_col, model_prediction_col]
        if keep_target_date and "gman_target_date" in gman_trim.columns:
            cols.append("gman_target_date")

        gman_trim = gman_trim[cols]

        self.df = pd.merge(
            self.df,
            gman_trim,
            how="left",
            left_on=[self.datetime_col, self.sensor_col],
            right_on=[model_prediction_date_col, self.sensor_col],
        )

        if not keep_target_date:
            self.df.drop(columns=[model_prediction_date_col], inplace=True)

        missing = self.df[self.df[model_prediction_col].isna()]
        if not missing.empty:
            self._log(
                f"Dropping {len(missing)} rows without '{model_prediction_col}'."
            )
            self.df = self.df.dropna(subset=[model_prediction_col])

        if convert_prediction_to_delta_speed:
            self.df[f"{model_prediction_col}_orig"] = self.df[model_prediction_col]
            self.df[model_prediction_col] -= self.df[self.value_col]

    # ------------------------------------------------------------------#
    # Public data getters
    # ------------------------------------------------------------------#
    def get_data(
        self,
        *,
        window_size: int = 3,
        filter_on_train_only: bool = False,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        use_median_instead_of_mean: bool = False,
        relative_threshold: float = 0.7,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp, None]] = None,  # str | pd.Timestamp | None = None,
        diagnose_extreme_changes: bool = False,
        add_gman_predictions: bool = False,
        convert_gman_prediction_to_delta_speed: bool = True,
    ) -> pd.DataFrame:
        """Return cleaned dataframe ready for modelling."""
        # Core pipeline
        self.load_data_parquet()
        self.align_sensors_to_common_timeframe()
        self.add_test_set_column(
            test_size=test_size, test_start_time=test_start_time
        )

        if diagnose_extreme_changes:
            self.diagnose_extreme_changes(
                relative_threshold=relative_threshold)
        if filter_extreme_changes:
            self.filter_extreme_changes(relative_threshold=relative_threshold)
        if smooth_speeds:
            self.smooth_speeds(
                window_size=window_size,
                filter_on_train_only=filter_on_train_only,
                use_median_instead_of_mean=use_median_instead_of_mean,
            )
        if add_gman_predictions:
            self.add_gman_predictions(
                convert_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed
            )

        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)
        return self.df.copy()

    # ------------------------------------------------------------------#
    # GMAN input helper
    # ------------------------------------------------------------------#
    def _pivot_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot to (time-index × sensor-columns) layout."""
        pivot = df.pivot(
            index=self.datetime_col, columns=self.sensor_col, values=self.value_col
        )
        pivot.columns.name = None
        return pivot

    def get_data_as_gman_input(
        self,
        *,
        window_size: int = 3,
        filter_on_train_only: bool = True,
        filter_extreme_changes: bool = True,
        smooth_speeds: bool = True,
        use_median_instead_of_mean: bool = True,
        relative_threshold: float = 0.7,
        test_size: float = 1 / 3,
        test_start_time: Optional[Union[str, pd.Timestamp, None]] = None,  # str | pd.Timestamp | None = None,
        diagnose_extreme_changes: bool = False,
    ) -> pd.DataFrame:
        """Return wide-format dataframe suitable for GMAN training."""
        self.get_data(
            window_size=window_size,
            filter_on_train_only=filter_on_train_only,
            filter_extreme_changes=filter_extreme_changes,
            smooth_speeds=smooth_speeds,
            use_median_instead_of_mean=use_median_instead_of_mean,
            relative_threshold=relative_threshold,
            test_size=test_size,
            test_start_time=test_start_time,
            diagnose_extreme_changes=diagnose_extreme_changes,
            add_gman_predictions=False,
        )

        self.df_as_gman_input = self.df.copy()

        pivot = self._pivot_dataframe(self.df)
        pivot_orig = self._pivot_dataframe(self.df_orig)

        # Mark test rows
        pivot["test_set"] = pivot.index >= self.first_test_timestamp

        self.df_as_gman_input = pivot
        self.df_as_gman_input_orig = pivot_orig
        return pivot


# -----------------------------------------------------------------------------#
#  Legacy implementation
# -----------------------------------------------------------------------------#

class InitialTrafficDataLoader_deprecated(LoggingMixin):
    def __init__(
        self,
        file_path,
        datetime_cols=['datetime', 'date'],
        sensor_col='sensor_id',
        value_col='value',
        disable_logs=False,
        df_gman=None
    ):
        super().__init__(disable_logs=disable_logs)
        self.file_path = file_path
        self.df = None
        self.df_orig = None
        self.datetime_cols = datetime_cols
        self.datetime_col = None
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.df_gman = df_gman
        self.first_test_timestamp = None
        self.df_as_gman_input = None
        self.df_as_gman_input_orig = None

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
        self._log(
            f"Discarded {len_df_prev - len_df_after} rows with NaN values in 'value' column. (method _apply_clean_and_convert_to_float32())")
        self.df[self.value_col] = self.df[self.value_col].apply(
            self._clean_and_convert_to_float32)
        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)

    def load_data_parquet(self):
        self.df = pd.read_parquet(self.file_path, engine='pyarrow')

        if self.sensor_col not in self.df.columns:
            raise ValueError(
                f"Missing expected sensor column: {self.sensor_col}. Columns: {self.df.columns}. Change sensor_col parameter.")

        for col in self.datetime_cols:
            if col in self.df.columns:
                self.datetime_col = col
                self.df[col] = pd.to_datetime(self.df[col])
                break

        if not self.datetime_col:
            raise ValueError(
                f"Missing expected datetime column(s): {self.datetime_cols}")

        self.df_orig = self.df.copy()
        self.df = self.df.sort_values(
            [self.datetime_col, self.sensor_col]).reset_index(drop=True)
        self._log(
            f"Loaded {len(self.df)} rows from {self.file_path}. Columns: {self.df.columns.tolist()}")
        self._apply_clean_and_convert_to_float32()

    def align_sensors_to_common_timeframe(self):
        self._log(
            "Aligning sensors to a common timeframe based on the sensor with the fewest recordings.")
        sensor_counts = self.df.groupby(self.sensor_col).size()
        min_recording_sensor = sensor_counts.idxmin()
        common_timeframe = self.df[self.df[self.sensor_col]
                                   == min_recording_sensor][self.datetime_col]
        min_time = common_timeframe.min()
        max_time = common_timeframe.max()
        original_row_count = len(self.df)
        self.df = self.df[(self.df[self.datetime_col] >= min_time) & (
            self.df[self.datetime_col] <= max_time)]
        filtered_row_count = len(self.df)
        self._log(
            f"Aligned all sensors to the common timeframe: {min_time} to {max_time}.")
        self._log(
            f"Rows before alignment: {original_row_count}, Rows after alignment: {filtered_row_count}. {original_row_count - filtered_row_count} rows have been dropped")

    def add_test_set_column(self, test_size=1/3):
        unique_times = self.df[self.datetime_col].sort_values().unique()
        split_index = int(len(unique_times) * (1 - test_size))
        split_time = unique_times[split_index]
        self.df['test_set'] = self.df[self.datetime_col] >= split_time
        self._log(
            f"'test_set' column added. Split time (first test set timestamp): {split_time}")
        self.first_test_timestamp = pd.to_datetime(split_time)

    def _compute_relative_change_mask(self, threshold):
        col_temp = '__temp_rel_change__'
        self.df[col_temp] = self.df.groupby(self.sensor_col)[
            self.value_col].pct_change().abs()
        mask = self.df[col_temp] > threshold
        self.df.drop(columns=[col_temp], inplace=True)
        return mask

    def diagnose_extreme_changes(self, relative_threshold=0.7):
        self._log(f"Diagnosing changes > {relative_threshold * 100:.1f}%")
        mask = self._compute_relative_change_mask(relative_threshold)
        count_extremes = mask.sum()
        total = len(self.df)
        self._log(
            f"Extreme changes: {count_extremes} of {total} rows ({100 * count_extremes / total:.2f}%)")

    def filter_extreme_changes(self, relative_threshold=0.7):
        mask = self._compute_relative_change_mask(relative_threshold)
        self.df.loc[mask, self.value_col] = np.nan
        self.df[self.value_col] = self.df.groupby(self.sensor_col)[
            self.value_col].transform(lambda x: x.interpolate().ffill().bfill())
        self._log(f"Filtered and interpolated {mask.sum()} extreme changes.")

    def smooth_speeds(self, window_size=3, filter_on_train_only=True, use_median_instead_of_mean=True):
        if 'test_set' not in self.df.columns:
            self._log("Test set column not found. Automatically adding it.")
            self.add_test_set_column()

        mask = ~self.df['test_set'] if filter_on_train_only else self.df.index == self.df.index
        if use_median_instead_of_mean:
            smoothed = self.df.loc[mask].groupby(self.sensor_col)[self.value_col].transform(
                lambda x: x.rolling(window=window_size, center=False, min_periods=1).median())
        else:
            smoothed = self.df.loc[mask].groupby(self.sensor_col)[self.value_col].transform(
                lambda x: x.rolling(window=window_size, center=False, min_periods=1).mean())
        self.df.loc[mask, self.value_col] = smoothed.ffill().bfill()
        self._log(
            f"Applied smoothing (window={window_size}, train_only={filter_on_train_only}, use_median_instead_of_mean={use_median_instead_of_mean}).")

    def add_gman_predictions_deprecated(self, model_prediction_col='gman_prediction', model_prediction_date_col='gman_prediction_date'):
        self._log("Merging gman data.")
        assert self.df_gman is not None, "gman DataFrame is not provided. Please set df_gman in the constructor."
        self.df[self.sensor_col] = self.df[self.sensor_col].astype('category')
        self.df_gman[self.sensor_col] = self.df_gman[self.sensor_col].astype(
            'category')
        self.df_gman[model_prediction_date_col] = pd.to_datetime(
            self.df_gman[model_prediction_date_col])
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        # Limit df_gman to the datetime range of df
        min_date = self.df[self.datetime_col].min()
        max_date = self.df[self.datetime_col].max()
        self.df_gman = self.df_gman[
            (self.df_gman[model_prediction_date_col] >= min_date) &
            (self.df_gman[model_prediction_date_col] <= max_date)
        ]
        self.df = self.df.set_index([self.datetime_col, self.sensor_col])
        self.df_gman = self.df_gman.set_index(
            [model_prediction_date_col, self.sensor_col])
        self.df = self.df.join(self.df_gman, how='left').reset_index()
        missing_rows = self.df[self.df[model_prediction_col].isna()]
        if not missing_rows.empty:
            dropped_count = missing_rows.shape[0]
            min_date = missing_rows[self.datetime_col].min()
            max_date = missing_rows[self.datetime_col].max()
            self._log(
                f"Dropping {dropped_count} rows with missing 'gman_prediction'. Date range: {min_date} to {max_date}.")
            self.df = self.df.dropna(subset=[model_prediction_col])
        else:
            self._log("No rows dropped for missing 'gman_predictions'.")

    def add_gman_predictions(self,
                             convert_prediction_to_delta_speed=True,
                             model_prediction_col='gman_prediction',
                             model_prediction_date_col='gman_prediction_date',
                             keep_target_date=True):
        self._log("Merging gman data.")
        assert self.df_gman is not None, "gman DataFrame is not provided. Please set df_gman in the constructor."

        # Ensure datetime columns are datetime
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        self.df_gman[model_prediction_date_col] = pd.to_datetime(
            self.df_gman[model_prediction_date_col])

        # Restrict df_gman to the date range in df
        min_date = self.df[self.datetime_col].min()
        max_date = self.df[self.datetime_col].max()
        self.df_gman = self.df_gman[
            (self.df_gman[model_prediction_date_col] >= min_date) &
            (self.df_gman[model_prediction_date_col] <= max_date)
        ]

        # Columns to keep: sensor_id, prediction_date, gman_prediction (+ optionally target_date)
        merge_cols = [self.sensor_col,
                      model_prediction_date_col, model_prediction_col]
        if keep_target_date and 'gman_target_date' in self.df_gman.columns:
            merge_cols.append('gman_target_date')

        df_gman_trimmed = self.df_gman[merge_cols].copy()

        # Merge using prediction_date + sensor_id as keys
        self.df = pd.merge(
            self.df,
            df_gman_trimmed,
            how='left',
            left_on=[self.datetime_col, self.sensor_col],
            right_on=[model_prediction_date_col, self.sensor_col]
        )

        # Optionally drop the join key (prediction_date)
        if not keep_target_date:
            self.df.drop(columns=[model_prediction_date_col], inplace=True)

        # Log dropped rows (missing predictions)
        missing_rows = self.df[self.df[model_prediction_col].isna()]
        if not missing_rows.empty:
            dropped_count = missing_rows.shape[0]
            min_date = missing_rows[self.datetime_col].min()
            max_date = missing_rows[self.datetime_col].max()
            self._log(f"Dropping {dropped_count} rows with missing '{model_prediction_col}'. "
                      f"Date range: {min_date} to {max_date}.")
            self.df = self.df.dropna(subset=[model_prediction_col])
        else:
            self._log("No rows dropped for missing gman predictions.")

        if convert_prediction_to_delta_speed:
            model_prediction_col_orig = model_prediction_col + '_orig'
            self.df[model_prediction_col_orig] = self.df[model_prediction_col]
            self.df[model_prediction_col] = self.df[model_prediction_col] - \
                self.df[self.value_col]

    def get_data(self,
                 window_size=3,
                 filter_on_train_only=False,
                 filter_extreme_changes=True,
                 smooth_speeds=True,
                 use_median_instead_of_mean=False,
                 relative_threshold=0.7,
                 test_size=1/3,
                 diagnose_extreme_changes=False,
                 add_gman_predictions=False,
                 convert_gman_prediction_to_delta_speed=True,
                 ):
        self.load_data_parquet()
        self.align_sensors_to_common_timeframe()
        self.add_test_set_column(test_size=test_size)
        if diagnose_extreme_changes:
            self.diagnose_extreme_changes(
                relative_threshold=relative_threshold)
        if filter_extreme_changes:
            self.filter_extreme_changes(relative_threshold=relative_threshold)
        if smooth_speeds:
            self.smooth_speeds(window_size=window_size, filter_on_train_only=filter_on_train_only,
                               use_median_instead_of_mean=use_median_instead_of_mean)
        if add_gman_predictions:
            self.add_gman_predictions(
                convert_prediction_to_delta_speed=convert_gman_prediction_to_delta_speed)
        self.df[self.value_col] = self.df[self.value_col].astype(np.float32)
        return self.df.copy()

    def _pivot_dataframe(self, df):
        pivoted = df.pivot(index=self.datetime_col,
                           columns=self.sensor_col, values=self.value_col)
        pivoted.columns.name = None
        return pivoted

    def get_data_as_gman_input(self,
                               window_size=3,
                               filter_on_train_only=True,
                               filter_extreme_changes=True,
                               smooth_speeds=True,
                               use_median_instead_of_mean=True,
                               relative_threshold=0.7,
                               test_size=1/3,
                               diagnose_extreme_changes=False):

        self.get_data(window_size=window_size,
                      filter_on_train_only=filter_on_train_only,
                      filter_extreme_changes=filter_extreme_changes,
                      smooth_speeds=smooth_speeds,
                      use_median_instead_of_mean=use_median_instead_of_mean,
                      relative_threshold=relative_threshold,
                      test_size=test_size,
                      diagnose_extreme_changes=diagnose_extreme_changes,
                      add_gman_predictions=False)

        self.df_as_gman_input = self.df.copy()

        pivoted_df = self._pivot_dataframe(self.df)
        pivoted_df_orig = self._pivot_dataframe(self.df_orig)
        # Add test_set column based on self.first_test_timestamp
        pivoted_df['test_set'] = pivoted_df.index >= self.first_test_timestamp
        self.df_as_gman_input = pivoted_df
        self.df_as_gman_input_orig = pivoted_df_orig

        return pivoted_df
