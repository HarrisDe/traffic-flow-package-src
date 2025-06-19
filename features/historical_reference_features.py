# features/historical_reference_window_features.py
import pandas as pd
from datetime import timedelta
from typing import List, Sequence, Tuple, Set, Optional
from .base import FeatureTransformer , BaseAggregator          # unchanged



from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
class ClimatologyAggregator(BaseAggregator):
    """
    Aggregate **all historical weeks** for *the same minute-of-day*.

    NaN handling
    ------------
    * A `pandas.Series` is built from `values` (dtype=float), so any missing
      entries are already converted to `np.nan`.
    * We then call the chosen aggregation function with **`skipna=True`**.
      This is the most memory-efficient way to ignore NaNs, because:

        • no extra copy is created (unlike an explicit `.dropna()`),  
        • every pandas reduction (`mean`, `median`, `min`, `max`, `std`) accepts
          the `skipna` flag and is internally vectorised.

    * If **every** value is NaN, the reduction returns `np.nan`, signalling
      “insufficient long-term data”.
    """

    def __init__(self, agg_fn: str = "mean") -> None:
        self.agg_fn = agg_fn  # 'mean', 'median', 'max', 'min', 'std', …

    def aggregate(self, values: Iterable[float]) -> float:
        s = pd.Series(values, dtype=float)
        return getattr(s, self.agg_fn)(skipna=True)


# ------------------------------------------------------------------
class LocalWindowAggregator(BaseAggregator):
    """
    Aggregate **only the raw window** that was fetched for the current row.

    NaN handling
    ------------
    * We **explicitly drop NaNs** with `.dropna()` before computing the
      statistic.  Two reasons:

        1. When the window is small (e.g. 11 samples) every finite datum
           counts—dropping NaNs first prevents them from diluting `std`
           or producing misleading `min/max` behaviour.

        2. Some pandas reductions (e.g. `.median()`) ignore NaNs by default,
           but `.min()` / `.max()` do **not** unless `skipna=True`.  By
           dropping we ensure uniform behaviour across all aggregation
           functions with no extra keyword juggling.

    * If the window contains **no finite values after dropping**, we return
      `np.nan` so downstream code can decide how to handle “fully missing”.
    """

    def __init__(self, agg_fn: str = "mean") -> None:
        self.agg_fn = agg_fn

    def aggregate(self, values: Iterable[float]) -> float:
        s = pd.Series(values, dtype=float).dropna()
        if s.empty:
            return np.nan
        else:
            return getattr(s, self.agg_fn)()  # e.g. s.mean(), s.std(), …
       
_VALID_AGGS: Set[str]  = {"mean", "min", "max", "std", "median"}
_VALID_MODES: Set[str] = {"climatology", "local"}

# ------------------------------------------------------------------
class PreviousWeekdayWindowFeatureEngineer(FeatureTransformer):
    """
    Add raw & aggregated *previous-weekday* reference features.

    Two aggregation modes
    ---------------------
    - **climatology** (default) – statistic over *all* past weeks
      for the same minute-of-day.
    - **local** – statistic over the *instant* look-back window
      just retrieved for the row.

    NaNs are ignored in every statistic.
    """

    # ───────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        datetime_col: str = "date",
        sensor_col:   str = "sensor_uid",
        value_col:    str = "value",
        horizon_min:  int = 15,
        window_before_min: int = 5,
        window_after_min:  int = 5,
        step_min:          int = 1,
        aggs: Optional[Sequence[str]] = None,
        agg_mode: str = "local", # "climatology" or "local"
        agg_fn: str = "mean",                 # for LocalWindowAggregator
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)

        # column aliases
        self.dt, self.sid, self.val = datetime_col, sensor_col, value_col

        # timing config
        self.horizon     = timedelta(minutes=horizon_min)
        self.win_before  = timedelta(minutes=window_before_min)
        self.win_after   = timedelta(minutes=window_after_min)
        self.step        = timedelta(minutes=step_min)

        # aggregation config & validation --------------------------
        self.aggs = list(aggs or [])
        unknown = set(self.aggs) - _VALID_AGGS
        if unknown:
            raise ValueError(f"Unsupported aggs {unknown}. Allowed: {_VALID_AGGS}")

        if agg_mode not in _VALID_MODES:
            raise ValueError(f"agg_mode must be one of {_VALID_MODES}")
        self.agg_mode = agg_mode

        # strategy instance
        if self.agg_mode == "local":
            self.aggregator = LocalWindowAggregator(agg_fn)
        else:
            # agg_fn irrelevant – each statistic in self.aggs is applied
            self.aggregator = ClimatologyAggregator()

        self._log(
            f"horizon={horizon_min}′  window=[-{window_before_min},+{window_after_min}]′ "
            f"step={step_min}′  aggs={self.aggs or '-'}  mode={self.agg_mode}",
            level="debug",
        )

    # ───────────────────────────────────────────────────────────
    def _lookup_shift(self, ts: pd.Timestamp) -> timedelta:
        """Return how many days to subtract to reach 'previous weekday'."""
        wd = ts.weekday()
        if wd in (1, 2, 3, 4):  # Tue-Fri
            return timedelta(days=1)
        if wd == 0:             # Monday
            return timedelta(days=3)
        return timedelta(days=7)  # Sat/Sun

    # ───────────────────────────────────────────────────────────
    def transform(self, df_in: pd.DataFrame):
        """
        Parameters
        ----------
        df_in : pd.DataFrame
            Must contain columns  [datetime_col, sensor_col, value_col]

        Returns
        -------
        df_out : pd.DataFrame
            Original dataframe plus raw-window and aggregate columns.
        new_raw_cols : list[str]
            Names of the raw window columns added.
        """
        # ── 0.  Defensive copy & datetime cast  --------------------
        df = df_in.copy()
        df[self.dt] = pd.to_datetime(df[self.dt])    # ensures timezone-aware ops

        # Fast lookup-table:  (timestamp, sensor_uid) ➜ value
        lut = df.set_index([self.dt, self.sid])[self.val]

        # ── 1.  Build lookup_time for “last same weekday” ----------
        # Start with ‘prediction time + horizon’ (i.e. when we want to know speed)
        df["lookup_time"] = df[self.dt] + self.horizon

        # Subtract days so that lookup_time represents the **previous occurrence**
        # of the same weekday, following the domain rules:
        #   • Tue–Fri  : previous calendar day  (−1)
        #   • Monday   : previous Friday        (−3)
        #   • Sat/Sun  : same weekday last week (−7)
        dow = df[self.dt].dt.weekday
        df.loc[dow.isin([1, 2, 3, 4]), "lookup_time"] -= timedelta(days=1)
        df.loc[dow == 0,               "lookup_time"] -= timedelta(days=3)
        df.loc[dow.isin([5, 6]),       "lookup_time"] -= timedelta(days=7)

        # Sunday special case: if that Sunday slot is empty (all NaN),
        # fall back to previous Saturday.
        sun_idx = df.index[dow == 6]
        if len(sun_idx):
            # vector of (timestamp, sensor) pairs to probe in LUT
            pairs   = list(zip(df.loc[sun_idx, "lookup_time"],
                               df.loc[sun_idx, self.sid]))
            missing = lut.reindex(pairs).isna().to_numpy()
            if missing.any():
                df.loc[sun_idx[missing], "lookup_time"] = (
                    df.loc[sun_idx[missing], self.dt] + self.horizon - timedelta(days=1)
                )

        # ── 2.  Raw window samples around lookup_time  -------------
        raw_cols: List[str] = []
        offset = -self.win_before
        while offset <= self.win_after:
            minutes = int(offset.total_seconds() / 60)
            col     = f"prev_wd_val_t{minutes:+}".replace("+", "")
            raw_cols.append(col)

            # Fetch value for (lookup_time + offset, sensor_uid)
            df[col] = lut.reindex(
                list(zip(df["lookup_time"] + offset, df[self.sid]))
            ).values
            offset += self.step

        # ── 3.  Aggregate features  -------------------------------
        if self.aggs:
            if self.agg_mode == "climatology":
                # --- long-term minute-of-day statistic -------------
                df["__hhmm__"] = df["lookup_time"].dt.strftime("%H:%M")

                # limit LUT to relevant minutes & sensors for speed
                hhmm_needed   = df["__hhmm__"].unique()
                sensors_used  = df[self.sid].unique()
                mask = (lut.index.get_level_values(0).strftime("%H:%M").isin(hhmm_needed) &
                        lut.index.get_level_values(1).isin(sensors_used))

                wide   = lut.loc[mask].unstack(self.sid)                  # 2-D frame
                grouped = wide.groupby(wide.index.strftime("%H:%M"))      # group by HH:MM

                for agg in self.aggs:
                    colname = f"prev_wd_{agg}_{self.win_before.seconds//60}_" \
                              f"{self.win_after.seconds//60}"
                    # Fill via (HH:MM, sensor_uid) lookup
                    df[colname] = grouped.transform(agg, skipna=True).stack().reindex(
                        list(zip(df["lookup_time"], df[self.sid]))
                    ).values

                df.drop(columns="__hhmm__", inplace=True)

            else:  # agg_mode == "local"
                # --- short-window statistic computed row-by-row ----
                def _row_aggs(r):
                    vals = r[raw_cols]                           # window vector
                    return {
                        f"{agg}_local": self.aggregator.aggregate(vals)
                        for agg in self.aggs
                    }

                df = pd.concat([df,
                                df.apply(_row_aggs, axis=1, result_type="expand")],
                               axis=1)

        # ── 4.  Cleanup & return  ---------------------------------
        df.drop(columns="lookup_time", inplace=True, errors="ignore")
        return df, raw_cols


    
# ------------------------------------------------------------------   
    
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