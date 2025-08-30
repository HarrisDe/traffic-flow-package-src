# traffic_flow/deep/windowing.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils.helper_utils import LoggingMixin

def _time_encodings_from_dt(
    dt: pd.Series,
    components: Iterable[str] = ("minute","hour","dow","mon"),
    *,
    prefix: str = ""
) -> pd.DataFrame:
    """Return sin/cos cyclic encodings for selected calendar components."""
    components = tuple(components)
    out = pd.DataFrame(index=dt.index)

    if "minute" in components:
        m = dt.dt.minute.to_numpy()
        out[f"{prefix}min_sin"] = np.sin(2*np.pi*m/60)
        out[f"{prefix}min_cos"] = np.cos(2*np.pi*m/60)

    if "hour" in components:
        h = dt.dt.hour.to_numpy()
        out[f"{prefix}hour_sin"] = np.sin(2*np.pi*h/24)
        out[f"{prefix}hour_cos"] = np.cos(2*np.pi*h/24)

    if "dow" in components:
        d = dt.dt.dayofweek.to_numpy()
        out[f"{prefix}dow_sin"] = np.sin(2*np.pi*d/7)
        out[f"{prefix}dow_cos"] = np.cos(2*np.pi*d/7)

    if "mon" in components:
        mo = dt.dt.month.to_numpy()
        out[f"{prefix}mon_sin"] = np.sin(2*np.pi*mo/12)
        out[f"{prefix}mon_cos"] = np.cos(2*np.pi*mo/12)

    return out.astype(np.float32)


class TFMultiSeriesSeq2OneBuilder(LoggingMixin):
    """
    Multiseries (e.g. 204 sensors) → seq2one builder.
    - History-time encodings can include minute/hour/dow/mon.
    - **Prediction-time** encodings at t+H are also added (broadcast over the L steps).
    """

    def __init__(
        self,
        *,
        datetime_col: str = "date",
        seq_len: int = 60,
        horizon_minutes: int = 15,
        stride: int = 1,
        target_mode: str = "delta",                  # "delta" | "absolute"
        feature_mode: str = "value_plus_time",       # "value_only" | "value_plus_time"
        custom_feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        # NEW: control which calendar features are used
        time_components: Iterable[str] = ("minute","hour","dow"),
        pred_time_components: Iterable[str] = ("minute","hour","dow"),
        batch_size: int = 256,
        dropna: bool = True,
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        if target_mode not in ("delta", "absolute"):
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        if feature_mode not in ("value_only", "value_plus_time"):
            raise ValueError("feature_mode must be 'value_only' or 'value_plus_time'")
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be a positive integer.")
        self.datetime_col = datetime_col
        self.seq_len = int(seq_len)
        self.H = int(horizon_minutes)       # 1-min sampling ⇒ steps == minutes
        self.stride = int(stride)
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.custom_feature_fn = custom_feature_fn
        self.time_components = tuple(time_components)
        self.pred_time_components = tuple(pred_time_components)
        self.batch_size = int(batch_size)
        self.dropna = bool(dropna)

        # resolved later
        self.sensor_cols_: List[str] = []
        self.time_feature_cols_: List[str] = []
        self.pred_time_feature_cols_: List[str] = []
        self.custom_feature_cols_: List[str] = []
        self.feature_cols_: List[str] = []   # final order in X (history + pred-time + custom)

    # ---------- internals ----------
    def _split_columns(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        dt_col = self.datetime_col
        if dt_col not in df.columns:
            raise ValueError(f"Missing datetime column '{dt_col}'.")
        sensor_cols = [c for c in df.columns if c not in (dt_col, "test_set")]
        if not sensor_cols:
            raise ValueError("No sensor columns found.")
        return sensor_cols, dt_col

    def _build_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        d = df.copy()
        sensor_cols, dt_col = self._split_columns(d)
        self.sensor_cols_ = sensor_cols

        times  = pd.to_datetime(d[dt_col], errors="coerce")
        values = d[sensor_cols].to_numpy(dtype=np.float32)      # (T, F)

        blocks: List[np.ndarray] = [values]
        self.time_feature_cols_ = []
        self.custom_feature_cols_ = []

        if self.feature_mode == "value_plus_time":
            tfe = _time_encodings_from_dt(times, self.time_components, prefix="")
            self.time_feature_cols_ = list(tfe.columns)
            blocks.append(tfe.to_numpy(dtype=np.float32))

        if self.custom_feature_fn is not None:
            extra = self.custom_feature_fn(d)
            if not isinstance(extra, pd.DataFrame):
                raise TypeError("custom_feature_fn must return a pandas DataFrame.")
            extra = extra.select_dtypes(include=[np.number]).astype(np.float32)
            if extra.shape[0] != len(d):
                raise ValueError("custom_feature_fn must return same number of rows as input df.")
            dup = [c for c in extra.columns if c in (sensor_cols + self.time_feature_cols_)]
            if dup:
                raise ValueError(f"custom_feature_fn returned duplicate columns: {dup}")
            self.custom_feature_cols_ = list(extra.columns)
            blocks.append(extra.to_numpy(dtype=np.float32))

        feats = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        # history feature names (sensor + time + custom). pred-time feature names are appended in make_arrays
        self.feature_cols_ = sensor_cols + self.time_feature_cols_ + self.custom_feature_cols_
        return feats, values, times.to_numpy(), sensor_cols

    # ---------- public ----------
    def make_arrays(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Return:
          X: (N, L, P), y: (N, F), base_times: (N,), base_values: (N, F),
          target_times: (N,), sensor_cols, meta
        """
        feats, values, times, sensor_cols = self._build_feature_matrix(df)
        T, P_hist = feats.shape
        F = len(sensor_cols)
        L, H, S = self.seq_len, self.H, self.stride
        if T < L + H:
            raise ValueError(f"Not enough rows: need at least seq_len+H = {L+H}, got {T}")

        endpoints  = np.arange(L - 1, T - H, S, dtype=np.int64)  # t
        target_idx = endpoints + H                               # u = t+H
        N = len(endpoints)

        # Build prediction-time encodings at t+H
        pred_times = pd.to_datetime(times[target_idx])
        pred_tfe_df = _time_encodings_from_dt(pd.Series(pred_times), self.pred_time_components, prefix="pred_")
        self.pred_time_feature_cols_ = list(pred_tfe_df.columns)
        P_pred = pred_tfe_df.shape[1]
        pred_tfe = pred_tfe_df.to_numpy(dtype=np.float32)  # (N, P_pred)

        # Allocate
        X = np.empty((N, L, P_hist + P_pred), dtype=np.float32)
        y = np.empty((N, F), dtype=np.float32)
        base_values = np.empty((N, F), dtype=np.float32)

        # Fill
        for i, t in enumerate(endpoints):
            hist = feats[t - L + 1 : t + 1]                 # (L, P_hist)
            cur  = values[t]                                # (F,)
            fut  = values[t + H]                            # (F,)
            # broadcast pred-time features over L steps and concat as extra channels
            pt   = np.tile(pred_tfe[i][None, :], (L, 1))    # (L, P_pred)
            X[i] = np.concatenate([hist, pt], axis=1)       # (L, P_hist + P_pred)

            base_values[i] = cur
            y[i] = (fut - cur) if self.target_mode == "delta" else fut

        base_times   = times[endpoints]
        target_times = times[target_idx]

        if self.dropna:
            mask = ~(
                np.isnan(X).any(axis=(1, 2)) |
                np.isnan(y).any(axis=1)   |
                np.isnan(base_values).any(axis=1)
            )
            X, y, base_times, base_values, target_times, endpoints, target_idx, pred_tfe = (
                X[mask], y[mask], base_times[mask], base_values[mask],
                target_times[mask], endpoints[mask], target_idx[mask], pred_tfe[mask]
            )

        # Update final feature list to include pred-time encodings (they are in every row of the window)
        self.feature_cols_ = (
            self.sensor_cols_ + self.time_feature_cols_ + self.custom_feature_cols_ + self.pred_time_feature_cols_
        )

        meta = {
            "endpoints": endpoints,
            "target_idx": target_idx,
        }
        return X, y, base_times, base_values, target_times, sensor_cols, meta

    def make_datasets(
        self,
        df: pd.DataFrame,
        target_splits: Dict[str, Tuple[int, int]],
    ) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict]:
        """
        Build ds_train/val/test and a **meta** dict compatible with assemble_results_dataframe.
        """
        X, y, base_times, base_values, target_times, sensor_cols, meta = self.make_arrays(df)
        u = meta["target_idx"]
        h_label = f"{self.H}m"  # single horizon label in minutes

        def _mk_mask(rng: Tuple[int, int]) -> np.ndarray:
            i0, i1 = rng
            if i0 < 0 or i1 < 0 or i1 < i0: return np.zeros_like(u, dtype=bool)
            return (u >= i0) & (u <= i1)

        def _slice(mask: np.ndarray) -> Optional[tf.data.Dataset]:
            if mask.size == 0 or not mask.any(): return None
            Xs, ys = X[mask], y[mask]
            ds = tf.data.Dataset.from_tensor_slices((Xs, ys))
            ds = ds.shuffle(min(len(Xs), 2048), reshuffle_each_iteration=True)\
                   .batch(self.batch_size)\
                   .prefetch(tf.data.AUTOTUNE)
            return ds

        ds_train = _slice(_mk_mask(target_splits.get("train", (-1, -1))))
        ds_val   = _slice(_mk_mask(target_splits.get("val",   (-1, -1))))
        ds_test  = _slice(_mk_mask(target_splits.get("test",  (-1, -1))))

        meta_all = {
            "X": X,
            "y": y,                                   # (N, F)  ← slice per sensor when assembling
            "base_times": base_times,                 # (N,)
            "base_values": base_values,               # (N, F)
            "pred_times": {h_label: target_times},    # dict[str] -> (N,)
            "feature_cols": self.feature_cols_,
            "custom_feature_cols": self.custom_feature_cols_,
            "pred_time_feature_cols": self.pred_time_feature_cols_,
            "time_feature_cols": self.time_feature_cols_,
            "h_labels": [h_label],
            "target_mode": self.target_mode,
            "sensor_cols": sensor_cols,
            "seq_len": self.seq_len,
            "horizon_minutes": self.H,
            "endpoints": meta["endpoints"],
            "target_idx": u,
        }

        self._log(
            f"[windowing] features/step: total={len(self.feature_cols_)} "
            f"(sensors={len(sensor_cols)}, time={len(self.time_feature_cols_)}, "
            f"pred_time={len(self.pred_time_feature_cols_)}, custom={len(self.custom_feature_cols_)})"
        )
        return ds_train, ds_val, ds_test, meta_all