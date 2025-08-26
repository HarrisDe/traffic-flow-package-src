# traffic_flow/deep/windowing.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils.helper_utils import LoggingMixin


class TFMultiSeriesSeq2OneBuilder(LoggingMixin):
    """
    Multiseries (204 sensors) → seq2one LSTM dataset builder.

    Assumptions
    -----------
    - Input df is wide: [date, s1, s2, ..., sN, test_set] at 1-minute spacing.
    - horizon_minutes is an integer in {1,5,10,15,30,45,60}.
    - Splits are provided **by target time indices**, e.g. from TrafficDeepSplitInterface.

    Outputs
    -------
    - make_arrays(df) -> X, y, base_times, base_values, target_times, sensor_cols, meta
    - make_datasets(df, target_splits) -> (ds_train, ds_val, ds_test, info)
      where each ds is tf.data.Dataset of (X, y).
    """

    def __init__(
        self,
        *,
        datetime_col: str = "date",
        seq_len: int = 60,
        horizon_minutes: int = 15,
        stride: int = 1,
        target_mode: str = "delta",  # or "absolute"
        batch_size: int = 256,
        dropna: bool = True,
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        if target_mode not in ("delta", "absolute"):
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be a positive integer.")

        self.datetime_col = datetime_col
        self.seq_len = int(seq_len)
        self.H = int(horizon_minutes)      # 1-minute base → steps = minutes
        self.stride = int(stride)
        self.target_mode = target_mode
        self.batch_size = int(batch_size)
        self.dropna = bool(dropna)

    # ------------------------------------------------------------------ #
    def _to_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        d = df.copy()
        dt = pd.to_datetime(d[self.datetime_col])
        sensor_cols = [c for c in d.columns if c not in (self.datetime_col, "test_set")]
        values = d[sensor_cols].to_numpy(dtype=np.float32)  # (T, F)
        times = dt.to_numpy()
        return values, times, d.get("test_set", pd.Series(False, index=d.index)).to_numpy(bool), sensor_cols

    # ------------------------------------------------------------------ #
    def make_arrays(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Build full sliding windows across the entire series.

        Returns
        -------
        X : (N, L, F)
        y : (N, F)
        base_times : (N,)
        base_values : (N, F)
        target_times : (N,)
        sensor_cols : list[str]
        meta : dict with keys:
            - 'endpoints'    : (N,) indices of last input step (t)
            - 'target_idx'   : (N,) indices of target step (t+H)
        """
        values, times, test_flag, sensor_cols = self._to_arrays(df)
        T, F = values.shape
        L, H, S = self.seq_len, self.H, self.stride

        if T < L + H:
            raise ValueError(f"Not enough rows: need at least seq_len+H = {L+H}, got {T}")

        endpoints = np.arange(L - 1, T - H, S, dtype=np.int64)    # t (end of history)
        target_idx = endpoints + H                                # u = t + H

        N = endpoints.shape[0]
        X = np.empty((N, L, F), dtype=np.float32)
        y = np.empty((N, F), dtype=np.float32)
        base_values = np.empty((N, F), dtype=np.float32)

        for i, t in enumerate(endpoints):
            hist = values[t - L + 1 : t + 1]  # (L, F)
            cur  = values[t]
            fut  = values[t + H]

            X[i] = hist
            base_values[i] = cur
            y[i] = (fut - cur) if self.target_mode == "delta" else fut

        base_times   = times[endpoints]
        target_times = times[target_idx]

        if self.dropna:
            mask = ~(
                np.isnan(X).any(axis=(1, 2)) |
                np.isnan(y).any(axis=1)
            )
            X, y, base_times, base_values, target_times, endpoints, target_idx = \
                X[mask], y[mask], base_times[mask], base_values[mask], target_times[mask], endpoints[mask], target_idx[mask]

        meta = {"endpoints": endpoints, "target_idx": target_idx}
        return X, y, base_times, base_values, target_times, sensor_cols, meta

    # ------------------------------------------------------------------ #
    def make_datasets(
        self,
        df: pd.DataFrame,
        target_splits: Dict[str, Tuple[int, int]],
    ) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict]:
        """
        Build tf.data datasets for train/val/test using **target-index** splits.
        """
        X, y, base_times, base_values, target_times, sensor_cols, meta = self.make_arrays(df)
        u = meta["target_idx"]  # (N,)

        def _slice(mask: np.ndarray) -> Optional[tf.data.Dataset]:
            if mask.size == 0 or not mask.any():
                return None
            Xs, ys = X[mask], y[mask]
            ds = tf.data.Dataset.from_tensor_slices((Xs, ys))
            ds = ds.shuffle(min(len(Xs), 2048), reshuffle_each_iteration=True)\
                   .batch(self.batch_size)\
                   .prefetch(tf.data.AUTOTUNE)
            return ds

        def _mk_mask(rng: Tuple[int, int]) -> np.ndarray:
            i0, i1 = rng
            if i0 < 0 or i1 < 0 or i1 < i0:
                return np.zeros_like(u, dtype=bool)
            return (u >= i0) & (u <= i1)

        ds_train = _slice(_mk_mask(target_splits.get("train", (-1, -1))))
        ds_val   = _slice(_mk_mask(target_splits.get("val",   (-1, -1))))
        ds_test  = _slice(_mk_mask(target_splits.get("test",  (-1, -1))))

        info = dict(
            sensor_cols=sensor_cols,
            seq_len=self.seq_len,
            horizon_minutes=self.H,
            target_mode=self.target_mode,
            base_times=base_times,
            target_times=target_times,
            endpoints=meta["endpoints"],
            target_idx=u,
        )
        return ds_train, ds_val, ds_test, info