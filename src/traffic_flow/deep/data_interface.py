# traffic_flow/deep/data_interface.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

from ..utils.helper_utils import LoggingMixin


class TrafficDeepSplitInterface(LoggingMixin):
    """
    Build contiguous TRAIN/VAL/TEST splits **by target timestamp** from a
    wide dataframe: [date, sensor1, ..., sensorN, test_set].

    Usage:
        iface = TrafficDeepSplitInterface(datetime_col="date", test_col="test_set")
        iface.attach_frame(df_wide)               # <- already cleaned/sorted by your loader
        splits = iface.build_target_splits(val_fraction_of_train=0.3)
        # splits -> {'train': (i0,i1), 'val': (i0,i1), 'test': (i0,i1)} over row indices
    """

    def __init__(
        self,
        *,
        datetime_col: str = "date",
        test_col: str = "test_set",
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.datetime_col = datetime_col
        self.test_col = test_col

        self.df_: Optional[pd.DataFrame] = None
        self.first_test_timestamp: Optional[pd.Timestamp] = None
        self.last_test_timestamp: Optional[pd.Timestamp] = None
        self.splits_: Optional[Dict[str, Tuple[int, int]]] = None
        self.sensor_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    def attach_frame(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """Store the provided wide frame and run sanity checks."""
        d = df_wide.copy()
        if self.datetime_col not in d.columns:
            raise ValueError(f"Missing datetime column '{self.datetime_col}'.")
        if self.test_col not in d.columns:
            raise ValueError(f"Missing test flag column '{self.test_col}'.")

        # ensure sorted & unique timestamps
        self._assert_time_sorted_unique(d, self.datetime_col)

        # store sensor column list
        self.sensor_cols_ = [c for c in d.columns if c not in (self.datetime_col, self.test_col)]

        self.df_ = d
        mask = d[self.test_col].astype(bool).to_numpy()
        if mask.any():
            first_test = int(np.argmax(mask))
            self.first_test_timestamp = pd.to_datetime(d[self.datetime_col].iloc[first_test])
            self.last_test_timestamp = pd.to_datetime(d[self.datetime_col].iloc[-1])
        else:
            self.first_test_tim
            estamp = None
            self.last_test_timestamp = None

        self._log(
            f"[split] attached frame: {len(d)} rows, sensors={len(self.sensor_cols_)}; "
            f"first_test_ts={self.first_test_timestamp}"
        )
        return d

    # ------------------------------------------------------------------ #
    def build_target_splits(
        self,
        *,
        val_fraction_of_train: float = 0.3,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Return contiguous index ranges **over target timestamps**:
        {'train': (i0,i1), 'val': (i0,i1), 'test': (i0,i1)}.

        - TEST = rows where test_set is True (from first True to end)
        - TRAIN+VAL = all rows strictly before TEST starts.
          VAL is carved as the last `val_fraction_of_train` of TRAIN+VAL.
        """
        if self.df_ is None:
            raise RuntimeError("Call attach_frame(df_wide) first.")

        df = self.df_
        N = len(df)
        mask = df[self.test_col].astype(bool).to_numpy()

        if mask.any():
            first_test = int(np.argmax(mask))
            test = (first_test, N - 1)

            # pre-test region [0 .. first_test-1]
            pre_end = first_test - 1
            if pre_end < 0:
                train = (-1, -1)
                val = (-1, -1)
            else:
                n_pre = pre_end + 1
                n_val = int(round(n_pre * float(val_fraction_of_train)))
                n_val = min(max(n_val, 0), n_pre)
                val_start = n_pre - n_val
                train = (0, val_start - 1) if val_start > 0 else (-1, -1)
                val   = (val_start, pre_end) if n_val > 0 else (-1, -1)
        else:
            # no test region marked; split by fractions 80/20 (train/val) as a fallback
            train_end = int(N * 0.8) - 1
            val_end   = N - 1
            train = (0, max(train_end, -1))
            val   = (train_end + 1, max(val_end, -1))
            test  = (-1, -1)

        splits = {"train": train, "val": val, "test": test}
        self.splits_ = splits
        self._log(f"[split] target-based splits: {splits}")
        return splits

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assert_time_sorted_unique(df: pd.DataFrame, datetime_col: str) -> None:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")
        if not dt.is_monotonic_increasing:
            raise ValueError("Datetime is not strictly increasing.")
        if dt.duplicated().any():
            dup_cnt = int(dt.duplicated().sum())
            raise ValueError(f"Found {dup_cnt} duplicate timestamps; expected unique index.")