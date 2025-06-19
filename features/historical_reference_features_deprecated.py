# features/historical_reference_window_features.py
import pandas as pd
from datetime import timedelta
from typing import List, Sequence, Tuple
from .base import FeatureTransformer          # unchanged

# ──────────────────────────────────────────────────────────────────────
_VALID_AGGS: set[str] = {"mean", "min", "max", "std", "median"}

class PreviousWeekdayWindowFeatureEngineer(FeatureTransformer):
    """
    Add raw & aggregated historical-reference features one weekday back.

    Row-to-lookup rules
    ───────────────────
    Tue-Fri → previous calendar day  
    Mon     → previous Friday (-3 d)  
    Sat     → previous Saturday (-7 d)  
    Sun     → previous Sunday  (-7 d),  
              fallback to Saturday (-1 d) *only* if that Sunday slot absent.

    If the final lookup timestamp does not exist in the data, NaNs are kept.

    Parameters
    ----------
    datetime_col, sensor_col, value_col : str
    horizon_min  : int
    window_before_min, window_after_min : int   inclusive
    step_min     : int   sampling inside the window
    aggs         : sequence[str]  subset of {"mean","min","max","std","median"}
    disable_logs : bool
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        datetime_col: str = "date",
        sensor_col:   str = "sensor_uid",
        value_col:    str = "value",
        horizon_min:  int = 15,
        window_before_min: int = 5,
        window_after_min:  int = 5,
        step_min:          int = 1,
        aggs: Sequence[str] | None = None,
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)

        # basic config
        self.dt  = datetime_col
        self.sid = sensor_col
        self.val = value_col

        self.horizon     = timedelta(minutes=horizon_min)
        self.win_before  = timedelta(minutes=window_before_min)
        self.win_after   = timedelta(minutes=window_after_min)
        self.step        = timedelta(minutes=step_min)

        self.aggs = list(aggs or [])
        bad = set(self.aggs) - _VALID_AGGS
        if bad:
            raise ValueError(
                f"{self.__class__.__name__}: unsupported aggs {bad}. "
                f"Allowed: {sorted(_VALID_AGGS)}"
            )

        self._log(
            f"[{self.__class__.__name__}] horizon={horizon_min}′  "
            f"window=[-{window_before_min},+{window_after_min}]′  "
            f"step={step_min}′  aggs={self.aggs or '-'}",
            level="debug",
        )

    # ──────────────────────────────────────────────────────────────────
    def transform(self, df_in: pd.DataFrame):
        self._log(f"transform(): incoming shape {df_in.shape}", level="debug")
        df = df_in.copy()
        df[self.dt] = pd.to_datetime(df[self.dt])

        # fast LUT  (time, sensor) → value
        lut = df.set_index([self.dt, self.sid])[self.val]

        # ── 1. primary lookup_time vector  -----------------------------
        dow = df[self.dt].dt.weekday

        df["lookup_time"] = df[self.dt] + self.horizon                # look ahead
        df.loc[dow.isin([1, 2, 3, 4]), "lookup_time"] -= timedelta(days=1)  # Tue-Fri
        df.loc[dow == 0,                    "lookup_time"] -= timedelta(days=3)  # Mon
        df.loc[dow.isin([5, 6]),            "lookup_time"] -= timedelta(days=7)  # Sat/Sun

        # ── 2. Sunday → Saturday fallback  ----------------------------
        sunday_mask = dow == 6
        if sunday_mask.any():
            # index positions of the Sunday rows
            sun_idx = df.index[sunday_mask]

            # does the Sunday-minus-7 slot exist?
            lookup_idx = list(
                zip(df.loc[sun_idx, "lookup_time"], df.loc[sun_idx, self.sid])
            )
            missing = lut.reindex(lookup_idx).isna().to_numpy()  # boolean numpy array

            if missing.any():                                    # only if something missing
                df.loc[sun_idx[missing], "lookup_time"] = (
                    df.loc[sun_idx[missing], self.dt] + self.horizon - timedelta(days=1)
                )

        self._log("lookup_time vector ready", level="debug")

        # ── 3. RAW sample columns  ------------------------------------
        new_cols: List[str] = []
        offset = -self.win_before
        while offset <= self.win_after:
            col = f"prev_wd_val_t{int(offset.total_seconds()/60):+d}".replace("+", "")
            new_cols.append(col)
            df[col] = lut.reindex(
                list(zip(df["lookup_time"] + offset, df[self.sid]))
            ).values
            offset += self.step
        self._log(f"raw features added ({len(new_cols)})", level="debug")

        # ── 4. Aggregates  -------------------------------------------
        if self.aggs:
            # group by minute-of-day
            df["__mod__"] = df["lookup_time"].dt.strftime("%H:%M")
            needed_times   = df["lookup_time"].unique()
            needed_sensors = df[self.sid].unique()

            mask = lut.index.get_level_values(0).isin(needed_times) & \
                   lut.index.get_level_values(1).isin(needed_sensors)
            windowed = lut.loc[mask].unstack(self.sid)
            grouped  = windowed.groupby(windowed.index.strftime("%H:%M"))

            for agg in self.aggs:
                col = f"prev_wd_{agg}_{int(self.win_before.total_seconds()/60)}_" \
                      f"{int(self.win_after.total_seconds()/60)}"
                new_cols.append(col)
                df[col] = grouped.transform(agg).stack().reindex(
                    list(zip(df["lookup_time"], df[self.sid]))
                ).values

            df.drop(columns="__mod__", inplace=True)
            self._log(f"aggregate features added {self.aggs}", level="debug")

        # ── 5. cleanup / return  --------------------------------------
        df.drop(columns="lookup_time", inplace=True)
        self._log(f"transform(): finished shape {df.shape}", level="debug")
        return df, new_cols

# class PreviousWeekdayWindowFeatureEngineer(FeatureTransformer):
#     """
#     Adds one or more features that look `horizon_min` into the future on the
#     previous weekday (–1d, skipping weekends).  Inside a user-defined window
#     around that lookup-time it can either return raw samples or aggregates.

#     ─────────────────────────────────────────────────────────────────────
#     Parameters
#     ----------
#     datetime_col : str
#     sensor_col   : str
#     value_col    : str
#     horizon_min  : int          how far *ahead* you predict
#     window_before_min : int     minutes *before* the lookup-time (inclusive)
#     window_after_min  : int     minutes *after*  the lookup-time (inclusive)
#     step_min          : int     granularity inside the window
#     aggs : List[str]            any of {"mean","min","max","std","median"}-VALID_AGGS
#     disable_logs (bool): If True, suppress logging
#     strict_weekday_match : bool if False, still fills weekends             """

#     def __init__(self,
#                  datetime_col: str = "date",
#                  sensor_col:   str = "sensor_uid",
#                  value_col:    str = "value",
#                  horizon_min:  int = 15,
#                  window_before_min: int = 0,
#                  window_after_min:  int = 0,
#                  step_min:          int = 1,
#                  aggs: List[str] = None,
#                  strict_weekday_match: bool = True,
#                 disable_logs: bool = False):

#         super().__init__(disable_logs)
#         self.datetime_col = datetime_col
#         self.sensor_col   = sensor_col
#         self.value_col    = value_col

#         self.horizon       = timedelta(minutes=horizon_min)
#         self.window_before = timedelta(minutes=window_before_min)
#         self.window_after  = timedelta(minutes=window_after_min)
#         self.step          = timedelta(minutes=step_min)
#         self.aggs          = aggs or []
#         self.strict_match  = strict_weekday_match
#         unknown = set(self.aggs) - VALID_AGGS
#         if unknown:
#             raise ValueError(
#                 f"{self.__class__.__name__}: unsupported agg(s) {unknown}. "
#                 f"Allowed: {sorted(VALID_AGGS)}"
#         )


#         self._log(
#             f"[{self.__class__.__name__}] "
#             f"horizon={horizon_min}′, "
#             f"window=[-{window_before_min}, +{window_after_min}]′, "
#             f"step={step_min}′, aggs={self.aggs}, "
#             f"strict_weekday_match={strict_weekday_match}",
#             level="debug"
#         )

#     def transform(self, df):
#         self._log(f"transform() called - incoming shape={df.shape}", level="debug")
#         df = df.copy()
#         df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

#         # ---------- build lookup table ----------------------------------
#         pivot   = df.pivot(index=self.datetime_col,
#                            columns=self.sensor_col,
#                            values=self.value_col)
#         stacked = pivot.stack()          # MultiIndex [time, sensor] → value

#         self._log("pivot/stack completed", level="debug")

#         # ---------- compute lookup times --------------------------------
#         one_day   = pd.Timedelta(days=1)
#         df["lookup_time"] = df[self.datetime_col] + self.horizon

#         # step back one weekday (skip weekend)
#         weekday  = df[self.datetime_col].dt.weekday
#         df["lookup_time"] -= one_day                         # always −1d
#         df.loc[(weekday == 0), "lookup_time"] -= one_day     # Monday → Friday

#         if self.strict_match:
#             wk_ok = (weekday < 5) & (df["lookup_time"].dt.weekday < 5)
#             df.loc[~wk_ok, "lookup_time"] = pd.NaT

#         self._log("lookup_time column calculated", level="debug")

#         # ---------- generate windowed raw samples -----------------------
#         new_cols = []
#         start_off = -self.window_before
#         end_off   =  self.window_after

#         offset = start_off
#         while offset <= end_off:
#             col = f"prev_wd_val_t{int(offset.total_seconds()/60)}"
#             new_cols.append(col)
#             df[col] = stacked.reindex(
#                 list(zip(df["lookup_time"] + offset, df[self.sensor_col]))
#             ).values
#             offset += self.step

#         self._log(f"raw-window features added: {len(new_cols)} cols", level="debug")

#         # ---------- aggregated statistics (optional) --------------------
#         if self.aggs:
#             lo = df["lookup_time"] - self.window_before
#             hi = df["lookup_time"] + self.window_after
#             window_mask = (
#                 (stacked.index.get_level_values(0) >= lo.min()) &
#                 (stacked.index.get_level_values(0) <= hi.max())
#             )
#             windowed = stacked.loc[window_mask].unstack(self.sensor_col)

#             for agg in self.aggs:
#                 col = f"prev_wd_{agg}_{int(self.window_before.total_seconds()/60)}_" \
#                       f"{int(self.window_after.total_seconds()/60)}"
#                 new_cols.append(col)
#                 df[col] = windowed.groupby(
#                     windowed.index.strftime("%H:%M")
#                 ).transform(agg).stack().reindex(
#                     list(zip(df["lookup_time"], df[self.sensor_col]))
#                 ).values

#             self._log(f"aggregate features added: {set(self.aggs)}", level="debug")

#         # ---------- tidy up & return ------------------------------------
#         df.drop(columns=["lookup_time"], inplace=True)
#         self._log(f"transform() finished - new shape={df.shape}", level="debug")
#         return df, new_cols
    
    
    
class PreviousWeekdayValueFeatureEngineer(FeatureTransformer):
    """
    Adds a feature representing the value of each sensor from the previous non-weekend day,
    shifted forward by a specified horizon (in minutes). The feature uses a pivoted lookup table
    and supports optional filtering based on weekday alignment.

    Example use case:
    - At 10:00 AM on Tuesday, this feature can reference the value at 10:15 AM on Monday (horizon = 15).

    Parameters:
        datetime_col (str): Name of datetime column.
        sensor_col (str): Name of sensor identifier column.
        value_col (str): Name of the target value column.
        horizon_minutes (int): Time offset to apply after identifying the previous weekday.
        strict_weekday_match (bool): If True, keeps only Mon–Fri → Mon–Fri pairs.
        disable_logs (bool): Suppress logging.
    """

    def __init__(
        self,
        datetime_col: str = 'date',
        sensor_col: str = 'sensor_id',
        value_col: str = 'value',
        horizon_minutes: int = 15,
        strict_weekday_match: bool = True,
        disable_logs: bool = False
    ):
        super().__init__(disable_logs)
        self.datetime_col = datetime_col
        self.sensor_col = sensor_col
        self.value_col = value_col
        self.horizon_minutes = horizon_minutes
        self.strict_weekday_match = strict_weekday_match
        self.new_column_name = f'prev_weekday_value_h{self.horizon_minutes}'

    def _get_previous_weekdays(self, dates: pd.Series) -> pd.Series:
        """
        Vectorized computation of previous weekday shifted by the given horizon.

        Args:
            dates (pd.Series): Input datetime series.

        Returns:
            pd.Series: Adjusted datetime for value lookup.
        """
        self._log("Computing adjusted timestamps for previous weekdays.")
        prev = dates - pd.Timedelta(days=1)
        prev = prev.mask(prev.dt.weekday == 6, prev - pd.Timedelta(days=2))  # Sunday → Friday
        prev = prev.mask(prev.dt.weekday == 5, prev - pd.Timedelta(days=1))  # Saturday → Friday
        return prev + pd.Timedelta(minutes=self.horizon_minutes)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transforms the dataframe by adding a feature from the previous weekday value.

        Args:
            df (pd.DataFrame): Input dataframe with timestamps and sensor values.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Updated dataframe and list of added feature names.
        """
        self._log(f"Adding historical reference feature: {self.new_column_name}")
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        # Create pivot table for fast multi-index lookup
        self._log("Creating pivot table for fast sensor-time value retrieval.")
        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        stacked = pivot.stack()
        stacked.index.names = [self.datetime_col, self.sensor_col]

        # Create adjusted timestamps for lookup
        df['current_dayofweek'] = df[self.datetime_col].dt.weekday
        df['lookup_time'] = self._get_previous_weekdays(df[self.datetime_col])
        df['lookup_dayofweek'] = df['lookup_time'].dt.weekday

        # Filter to ensure both source and lookup times are weekdays (if requested)
        if self.strict_weekday_match:
            valid_mask = (df['current_dayofweek'] < 5) & (df['lookup_dayofweek'] < 5)
            self._log("Filtering to keep only weekday-to-weekday lookup pairs.")
        else:
            valid_mask = pd.Series(True, index=df.index)

        # Perform lookup from pivoted data using MultiIndex
        self._log("Performing value lookups using reindex.")
        lookup_index = list(zip(df['lookup_time'], df[self.sensor_col]))
        lookup_values = stacked.reindex(lookup_index).values
        df[self.new_column_name] = np.where(valid_mask, lookup_values, np.nan)

        # Clean temporary columns
        df.drop(columns=['current_dayofweek', 'lookup_time', 'lookup_dayofweek'], inplace=True)

        self._log(f"Feature '{self.new_column_name}' successfully added.")
        return df, [self.new_column_name]