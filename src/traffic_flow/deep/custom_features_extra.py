# traffic_flow/deep/custom_features_extra.py
from __future__ import annotations
from typing import Iterable
import numpy as np, pandas as pd

_MPS_PER_KPH = 1000.0 / 3600.0  # 1 kph = 0.277777... m/s

def _ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False).mean()

def make_short_term_dynamics_fn(
    *,
    datetime_col: str = "date",
    base_unit: str = "kph",              # set to "kph" for your data
    short_windows: Iterable[int] = (5, 15, 30),   # minutes (for rolling stats)
    diff_windows:  Iterable[int] = (1, 3, 5, 10), # minutes (for finite differences)
    ema_fast: int = 5,
    ema_slow: int = 15,
    z_k: float = 1.5,                     # z-threshold for early drop flag
) -> callable:
    """
    df_wide -> DataFrame of per-sensor short-term dynamics, unit-consistent.

    Assumes regularly sampled data. If cadence isn't exactly 1 minute, we infer
    the typical step in seconds from `datetime_col`.

    Outputs (all causal):
      - Acceleration (m/s^2):      (v_t - v_{t-w}) / (w * Δt)
      - Jerk (m/s^3) [optional]:   (v_t - 2 v_{t-w} + v_{t-2w}) / (w*Δt)^2
      - Short-window z-scores on speed (unitless)
      - Distance to rolling min over short windows (m/s)
      - EMA fast/slow spread on speed (m/s)
      - Binary early-drop warning if any short z < -z_k (0/1)
    """
    sw = tuple(int(w) for w in short_windows)
    dw = tuple(int(w) for w in diff_windows)

    def _fn(df_wide: pd.DataFrame) -> pd.DataFrame:
        sensors = [c for c in df_wide.columns if c not in (datetime_col, "test_set")]
        dt = pd.to_datetime(df_wide[datetime_col], errors="coerce")

        # infer typical step (seconds)
        if dt.notna().any() and dt.size >= 2:
            step_sec = float(pd.Series(dt).diff().dt.total_seconds().median())
            if not np.isfinite(step_sec) or step_sec <= 0:
                step_sec = 60.0
        else:
            step_sec = 60.0

        X = df_wide[sensors].astype(np.float32)

        # convert speed to m/s if original is kph
        if base_unit.lower() == "kph":
            V = X * _MPS_PER_KPH    # m/s
        else:
            V = X.copy()            # already m/s

        out = []

        # --- Acceleration (m/s^2): first derivative of speed ---
        for w in dw:
            dt_sec = w * step_sec
            acc = (V - V.shift(w)) / dt_sec
            acc.columns = [f"{c}__acc_{w}m_mps2" for c in sensors]
            out.append(acc.fillna(0.0).astype(np.float32))

        # --- Jerk (m/s^3): second derivative of speed (optional but useful for sharp onsets) ---
        for w in (3, 5):
            dt_sec = w * step_sec
            jerk = (V - 2*V.shift(w) + V.shift(2*w)) / (dt_sec**2)
            jerk.columns = [f"{c}__jerk_{w}m_mps3" for c in sensors]
            out.append(jerk.fillna(0.0).astype(np.float32))

        # --- Short rolling stats on speed (m/s) ---
        for w in sw:
            mu = V.rolling(window=w, min_periods=w).mean()
            sd = V.rolling(window=w, min_periods=w).std()

            z = (V - mu) / sd.replace(0.0, np.nan)
            z.columns = [f"{c}__z_{w}m" for c in sensors]
            out.append(z.astype(np.float32).fillna(0.0))

            rmin = V.rolling(window=w, min_periods=w).min()
            dist_min = (V - rmin)  # m/s
            dist_min.columns = [f"{c}__distmin_{w}m_mps" for c in sensors]
            out.append(dist_min.astype(np.float32).fillna(0.0))

        # --- EMA fast/slow on speed (m/s) and their spread (m/s) ---
        ema_f = _ema(V, ema_fast); ema_s = _ema(V, ema_slow)
        spread = ema_f - ema_s
        spread.columns = [f"{c}__ema_spread_{ema_fast}_{ema_slow}_mps" for c in sensors]
        out.append(spread.astype(np.float32).fillna(0.0))

        # --- Early drop warning: any short window has z < -z_k ---
        warn_any = None
        for w in sw:
            mu = V.rolling(w, min_periods=w).mean()
            sd = V.rolling(w, min_periods=w).std()
            z = (V - mu) / sd.replace(0.0, np.nan)
            flag = (z < -z_k).astype(np.float32)
            warn_any = flag if warn_any is None else (warn_any.add(flag, fill_value=0.0))
        warn_any = (warn_any > 0).astype(np.float32) if warn_any is not None else pd.DataFrame(0.0, index=V.index, columns=sensors)
        warn_any.columns = [f"{c}__dropwarn" for c in sensors]
        out.append(warn_any.fillna(0.0))

        return pd.concat(out, axis=1).astype(np.float32)

    _fn.__name__ = "short_term_dynamics_features"
    return _fn


def compose_feature_fns(*fns):
    """Compose multiple df_wide->DataFrame feature callables."""
    def _combo(df_wide: pd.DataFrame) -> pd.DataFrame:
        parts = [fn(df_wide) for fn in fns if fn is not None]
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_wide.index)
    _combo.__name__ = "composed_features"
    return _combo