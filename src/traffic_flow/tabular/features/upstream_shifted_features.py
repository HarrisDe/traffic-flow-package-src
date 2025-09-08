# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class UpstreamTravelTimeShiftedFeatures(BaseFeatureTransformer):
    """
    For each sensor S and its upstream neighbor U[i], add features from U shifted by the
    estimated travel time τ(U->S). τ is derived from inter-sensor distance and a per-sensor
    free-flow speed (quantile of historical speed).

    Added features per upstream index i (1-based):
        - up{i}_shift_speed
        - up{i}_shift_delta1        (optional)
        - up{i}_shift_slope_w{W}    (optional, for W in slope_windows)

    Parameters
    ----------
    upstream_dict : dict or None
        Neighbor dictionary (same schema as AdjacentSensorFeatureAdder.upstream_dict_).
        If None, it is loaded from `sensor_dict_path/upstream_dict.json`.
    sensor_dict_path : str or None
        Folder to load upstream_dict when not provided explicitly.
    spatial_adj : int
        How many upstream neighbors to consider (1..N).
    datetime_col, value_col, sensor_col : str
        Column names.
    freeflow_percentile : float (0..1)
        Percentile of speed used as "free-flow" per sensor (e.g., 0.95).
    use_upstream_freeflow : bool
        If True, τ uses the upstream sensor's freeflow; else the target sensor's.
    fallback_freeflow_kph : float
        Used if we cannot compute a sensor's freeflow (missing or 0).
    minutes_per_row : float
        Sampling interval (1 if rows are 1 minute apart).
    cap_minutes : float
        Max τ (minutes) to avoid very large shifts.
    add_speed : bool
    add_delta1 : bool
    add_slope : bool
    slope_windows : Sequence[int]
        Windows (rows) for upstream rolling OLS slope before shifting.
    fill_nans_value : float
        Value to fill NaNs after alignment.
    """

    def __init__(
        self,
        *,
        upstream_dict: Optional[Dict[str, Any]] = None,
        sensor_dict_path: Optional[str] = None,
        spatial_adj: int = 1,
        datetime_col: str = "date",
        value_col: str = "value",
        sensor_col: str = "sensor_id",
        freeflow_percentile: float = 0.95,
        use_upstream_freeflow: bool = True,
        fallback_freeflow_kph: float = 100.0,
        minutes_per_row: float = 1.0,
        cap_minutes: float = 30.0,
        add_speed: bool = True,
        add_delta1: bool = True,
        add_slope: bool = True,
        slope_windows: Sequence[int] = (3, 5),
        fill_nans_value: float = -1.0,
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)

        self.upstream_dict = upstream_dict
        self.sensor_dict_path = sensor_dict_path
        self.spatial_adj = int(spatial_adj)

        self.datetime_col = datetime_col
        self.value_col = value_col
        self.sensor_col = sensor_col

        self.freeflow_percentile = float(freeflow_percentile)
        self.use_upstream_freeflow = bool(use_upstream_freeflow)
        self.fallback_freeflow_kph = float(fallback_freeflow_kph)
        self.minutes_per_row = float(minutes_per_row)
        self.cap_minutes = float(cap_minutes)

        self.add_speed = bool(add_speed)
        self.add_delta1 = bool(add_delta1)
        self.add_slope = bool(add_slope)
        self.slope_windows = tuple(int(w) for w in slope_windows)

        self.fill_nans_value = float(fill_nans_value)

        self.feature_names_out_: List[str] = []
        self.fitted_ = False

        # learned/derived state
        self._upstream_dict_: Dict[str, Any] | None = None
        self._freeflow_map_: Dict[Any, float] = {}
        # maps sensor S -> list of (U_id, tau_rows) for i=0..spatial_adj-1
        self._pair_shift_map_: Dict[Any, List[Tuple[Any, int]]] = {}

    # ------------------------------- helpers --------------------------- #
    @staticmethod
    def _rolling_ols_slope_func(window: int):
        """Return a fast function computing OLS slope over x = [0..window-1]."""
        n = float(window)
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            denom = 1.0

        def f(y: np.ndarray) -> float:
            y = y.astype(np.float64, copy=False)
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            return float((n * sum_xy - sum_x * sum_y) / denom)

        return f

    def _ensure_upstream_dict(self):
        if self._upstream_dict_ is not None:
            return
        if self.upstream_dict is not None:
            self._upstream_dict_ = self.upstream_dict
            return
        if not self.sensor_dict_path:
            raise ValueError("Provide either upstream_dict or sensor_dict_path.")
        path = os.path.join(self.sensor_dict_path, "upstream_dict.json")
        with open(path, "r") as f:
            self._upstream_dict_ = json.load(f)

    # ------------------------------------------------------------------ #
    # sklearn-style API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None):
        """Compute per-sensor freeflow and τ-row shifts based on neighbor distances."""
        self._ensure_upstream_dict()

        if not {self.sensor_col, self.value_col, self.datetime_col}.issubset(X.columns):
            missing = [c for c in (self.sensor_col, self.value_col, self.datetime_col) if c not in X.columns]
            raise ValueError(f"UpstreamTravelTimeShiftedFeatures.fit: missing columns {missing}")

        # Free-flow per sensor (quantile of speed)
        ff = (
            X.groupby(self.sensor_col)[self.value_col]
            .quantile(self.freeflow_percentile)
            .to_dict()
        )
        self._freeflow_map_ = {k: (float(v) if np.isfinite(v) and v > 0 else self.fallback_freeflow_kph) for k, v in ff.items()}

        # Precompute τ_rows per pair (S, U[i])
        self._pair_shift_map_.clear()
        unique_sensors = list(X[self.sensor_col].unique())
        for s in unique_sensors:
            u_entry = self._upstream_dict_.get(str(s), {}) if isinstance(self._upstream_dict_, dict) else {}
            u_ids = u_entry.get("upstream_sensor", [None] * self.spatial_adj)
            u_dists = u_entry.get("upstream_distance", [np.nan] * self.spatial_adj)

            row_list: List[Tuple[Any, int]] = []
            for i in range(self.spatial_adj):
                u_id = u_ids[i] if i < len(u_ids) else None
                dist_km = u_dists[i] if i < len(u_dists) else np.nan

                if (u_id is None) or (not np.isfinite(dist_km)) or (dist_km <= 0):
                    row_list.append((None, 0))
                    continue

                # Choose which sensor's freeflow to use
                ff_kph = self._freeflow_map_.get(u_id if self.use_upstream_freeflow else s, self.fallback_freeflow_kph)
                ff_km_per_min = ff_kph / 60.0
                tau_minutes = dist_km / max(ff_km_per_min, 1e-6)
                tau_minutes = min(max(tau_minutes, 0.0), self.cap_minutes)
                tau_rows = max(1, int(round(tau_minutes / max(self.minutes_per_row, 1e-6))))
                row_list.append((u_id, tau_rows))

            self._pair_shift_map_[s] = row_list

        self.fitted_ = True
        self._log(f"Fitted UpstreamTravelTimeShiftedFeatures: spatial_adj={self.spatial_adj}, "
                  f"freeflow_p={self.freeflow_percentile}, minutes_per_row={self.minutes_per_row}, "
                  f"cap_minutes={self.cap_minutes}")
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() (or from_state()) before transform().")

        df = X.copy()
        # ensure stable ordering
        if self.datetime_col in df.columns:
            df = df.sort_values([self.sensor_col, self.datetime_col])

        # Prepare output columns
        added_cols: List[str] = []

        # Build quick lookups: per-sensor Series indexed by datetime
        # (we will reuse these to avoid slicing df repeatedly)
        series_by_sensor: Dict[Any, pd.Series] = {}
        for sid, sub in df[[self.sensor_col, self.datetime_col, self.value_col]].groupby(self.sensor_col, sort=False):
            s = sub.set_index(self.datetime_col)[self.value_col].astype(float)
            series_by_sensor[sid] = s

        # Process per target sensor
        for s, sub in df.groupby(self.sensor_col, sort=False):
            shifts = self._pair_shift_map_.get(s, [])
            if not shifts:
                continue
            t_index = sub[self.datetime_col].values  # preserve time order for S
            row_idx = sub.index

            for i, (u_id, tau_rows) in enumerate(shifts, start=1):
                # skip if no upstream
                base = f"up{i}_shift"
                col_speed = f"{base}_speed"
                col_delta = f"{base}_delta1"
                col_slopes = [f"{base}_slope_w{w}" for w in self.slope_windows]

                if u_id is None or tau_rows <= 0 or u_id not in series_by_sensor:
                    # fill straight with NaN (later filled with sentinel)
                    df.loc[row_idx, col_speed] = np.nan if self.add_speed else None
                    if self.add_delta1:
                        df.loc[row_idx, col_delta] = np.nan
                    if self.add_slope:
                        for c in col_slopes:
                            df.loc[row_idx, c] = np.nan
                    continue

                su = series_by_sensor[u_id]

                if self.add_speed:
                    s_shift = su.shift(tau_rows)
                    df.loc[row_idx, col_speed] = s_shift.reindex(t_index).values

                if self.add_delta1:
                    s_diff = su.diff()
                    s_diff_shift = s_diff.shift(tau_rows)
                    df.loc[row_idx, col_delta] = s_diff_shift.reindex(t_index).values

                if self.add_slope and len(self.slope_windows) > 0:
                    for w in self.slope_windows:
                        func = self._rolling_ols_slope_func(w)
                        s_slope = su.rolling(w, min_periods=w).apply(func, raw=True)
                        s_slope_shift = s_slope.shift(tau_rows)
                        df.loc[row_idx, f"{base}_slope_w{w}"] = s_slope_shift.reindex(t_index).values

                # keep track of names (only once per column)
                if self.add_speed:
                    added_cols.append(col_speed)
                if self.add_delta1:
                    added_cols.append(col_delta)
                if self.add_slope:
                    added_cols.extend(col_slopes)

        # Fill NaNs and ensure types
        for c in added_cols:
            df[c] = df[c].astype(float).fillna(self.fill_nans_value)

        self.feature_names_out_ = added_cols
        return df

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "upstream_shifted",
            "upstream_dict": self._upstream_dict_,
            "spatial_adj": self.spatial_adj,
            "datetime_col": self.datetime_col,
            "value_col": self.value_col,
            "sensor_col": self.sensor_col,
            "freeflow_percentile": self.freeflow_percentile,
            "use_upstream_freeflow": self.use_upstream_freeflow,
            "fallback_freeflow_kph": self.fallback_freeflow_kph,
            "minutes_per_row": self.minutes_per_row,
            "cap_minutes": self.cap_minutes,
            "add_speed": self.add_speed,
            "add_delta1": self.add_delta1,
            "add_slope": self.add_slope,
            "slope_windows": list(self.slope_windows),
            "fill_nans_value": self.fill_nans_value,
            # learned maps so inference needn't recompute from small batches
            "freeflow_map": self._freeflow_map_,
            "pair_shift_map": self._pair_shift_map_,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "UpstreamTravelTimeShiftedFeatures":
        inst = cls(
            upstream_dict=state.get("upstream_dict"),
            sensor_dict_path=None,
            spatial_adj=state["spatial_adj"],
            datetime_col=state["datetime_col"],
            value_col=state["value_col"],
            sensor_col=state["sensor_col"],
            freeflow_percentile=state["freeflow_percentile"],
            use_upstream_freeflow=state["use_upstream_freeflow"],
            fallback_freeflow_kph=state["fallback_freeflow_kph"],
            minutes_per_row=state["minutes_per_row"],
            cap_minutes=state["cap_minutes"],
            add_speed=state["add_speed"],
            add_delta1=state["add_delta1"],
            add_slope=state["add_slope"],
            slope_windows=tuple(state["slope_windows"]),
            fill_nans_value=state["fill_nans_value"],
        )
        # restore learned maps
        inst._upstream_dict_ = state.get("upstream_dict")
        inst._freeflow_map_ = state.get("freeflow_map", {})
        inst._pair_shift_map_ = state.get("pair_shift_map", {})
        inst.fitted_ = True
        return inst