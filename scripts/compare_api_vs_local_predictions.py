# scripts/compare_api_vs_local.py
from __future__ import annotations
import os, json, time, argparse, threading
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
import sys

# ---------- repo import bootstrap ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]   # repo root (parent of 'scripts')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_flow.service.app import create_app
from traffic_flow.service.runtime import InferenceRuntime
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
from traffic_flow.preprocessing.cleaning import clean_and_cast

# Columns to drop immediately from RAW to avoid JSON NaN issues
DROP_RAW_COLS = [
    "Snow_depth_surface",
    "Water_equivalent_of_accumulated_snow_depth_surface",
]

# --------------------- Utilities ---------------------

class ServerThread(threading.Thread):
    """Run the Flask app in-process so you don’t need a second terminal."""
    def __init__(self, artifact_path: str, host: str = "127.0.0.1", port: int = 8080):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.app = create_app(artifact_path=artifact_path)
        from werkzeug.serving import make_server
        self._srv = make_server(self.host, self.port, self.app)
        self._ctx = self.app.app_context()
        self._ctx.push()

    def run(self):
        self._srv.serve_forever()

    def shutdown(self):
        self._srv.shutdown()
        self._ctx.pop()


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size].copy()


def to_json_records(df: pd.DataFrame) -> list[dict]:
    """
    Minimal, fast JSON sanitizer:
    - format 'date' to YYYY-mm-dd HH:MM:SS
    - replace inf/-inf -> NaN
    - convert all NaN/NA -> None (JSON null)
    - convert numpy scalars to Python scalars
    """
    out = df.copy()

    if "date" in out.columns:
        ts = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        out.loc[ts.isna(), "date"] = None

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.astype(object).where(pd.notna(out), None)

    # Ensure numpy scalars become python scalars for JSON
    def _py(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    records = [{k: _py(v) for k, v in row.items()} for row in out.to_dict("records")]
    return records


# --------- Load artifact metadata ---------

def load_artifact_states(artifact_path: str | Path) -> dict:
    bundle = joblib.load(artifact_path)
    return {
        "states": bundle["states"],
        "horizon": int(bundle.get("horizon", 15)),
    }


# --------- Build RAW test rows aligned with artifact cleaning ---------

def make_orchestrator_with_clean_params(raw_path: str | Path, states: dict) -> TrafficDataPipelineOrchestrator:
    clean = states["clean_state"]
    tdp = TrafficDataPipelineOrchestrator(file_path=str(raw_path), sensor_encoding_type="mean")
    tdp.prepare_base_features(
        window_size=clean["smoothing_window"],
        filter_extreme_changes=True,
        smooth_speeds=True,
        relative_threshold=clean["relative_threshold"],
        use_median_instead_of_mean_smoothing=clean["use_median"],
    )
    return tdp


def get_test_raw_with_states(raw_path: str | Path, states: dict) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Return RAW test rows (sorted) and the split timestamp; drop 2 weather cols."""
    tdp = make_orchestrator_with_clean_params(raw_path, states)
    raw = pd.read_parquet(raw_path)
    raw = clean_and_cast(raw, value_col="value")   # numeric safety

    # Drop the two problematic weather columns immediately (leaner and faster)
    drop_now = [c for c in DROP_RAW_COLS if c in raw.columns]
    if drop_now:
        raw = raw.drop(columns=drop_now)

    raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
    raw_test.sort_values(["date", "sensor_id"], kind="mergesort", inplace=True)
    return raw_test, tdp.first_test_timestamp


# --------- Build *smoothed* ground truth (training-style target) ---------

def build_smoothed_y_act(raw_path: str | Path, states: dict, horizon: int) -> pd.DataFrame:
    """Re-run minimal training steps to produce y_act at date_of_prediction."""
    tdp = make_orchestrator_with_clean_params(raw_path, states)
    tdp.finalise_for_horizon(
        horizon=horizon,
        drop_weather=True,
        add_previous_weekday_feature=False,
        drop_datetime=False,
    )
    df = tdp.df.loc[tdp.df["test_set"], ["sensor_id", "date_of_prediction", "target_total_speed"]].copy()
    df.rename(columns={"date_of_prediction": "timestamp", "target_total_speed": "y_act"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# --------- Prediction helpers (API & Local; both batched) ---------

def api_predict_full(base_url: str,
                     raw_df: pd.DataFrame,
                     batch_size: int = 20000,
                     timeout_s: int = 300) -> pd.DataFrame:
    """
    Call /predict in batches. Returns: timestamp, sensor_id, y_pred (total).
    Assumes the service responds with fields: sensor_id, date, prediction_time, y_pred_total, ...
    """
    outs: list[pd.DataFrame] = []

    for i, batch in enumerate(iter_batches(raw_df, batch_size), start=1):
        payload = {"records": to_json_records(batch)}
        r = requests.post(f"{base_url}/predict", json=payload, timeout=timeout_s)
        r.raise_for_status()

        part = pd.DataFrame(r.json()["predictions"])
        # Standardize columns
        if "prediction_time" not in part.columns or "y_pred_total" not in part.columns:
            raise RuntimeError("API response missing expected keys 'prediction_time' / 'y_pred_total'")
        part = part.rename(columns={"prediction_time": "timestamp", "y_pred_total": "y_pred"})
        part["timestamp"] = pd.to_datetime(part["timestamp"], errors="coerce")
        part = part.loc[:, ["timestamp", "sensor_id", "y_pred"]]

        outs.append(part)
        print(f"[API] processed batch {i} ({len(batch)} rows)")

    out_df = pd.concat(outs, axis=0, ignore_index=True) if outs else pd.DataFrame()
    out_df.sort_values(["timestamp", "sensor_id"], inplace=True, kind="mergesort")
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def local_predict_full(artifact_path: str | Path,
                       raw_test: pd.DataFrame,
                       batch_size: int = 20000) -> pd.DataFrame:
    """
    Local inference (no HTTP). Returns: timestamp, sensor_id, y_pred (total).
    Critical fix: slice timestamp/sensor_id to the prediction length to avoid length mismatches.
    """
    rt = InferenceRuntime(str(artifact_path))
    outs: list[pd.DataFrame] = []

    for i, batch in enumerate(iter_batches(raw_test, batch_size), start=1):
        pred_delta, feats = rt.predict_df(batch)  # delta
        # Build total prediction
        if "value" not in feats.columns:
            raise RuntimeError("Engineered features missing 'value' column.")
        y_pred_total = (pred_delta + feats["value"].to_numpy()).astype(float)

        # Align timestamp/sensor_id to the **prediction length**
        n_pred = len(y_pred_total)
        dt = pd.to_datetime(batch["date"].iloc[:n_pred], errors="coerce")
        timestamp = dt + pd.to_timedelta(rt.horizon, unit="m")
        sensor_id = batch["sensor_id"].iloc[:n_pred].to_numpy()

        df = pd.DataFrame({
            "timestamp": timestamp.to_numpy(),
            "sensor_id": sensor_id,
            "y_pred": y_pred_total,  # already length n_pred
        })
        outs.append(df)
        print(f"[LOCAL] processed batch {i} (in={len(batch)} -> out={n_pred})")

    out_df = pd.concat(outs, axis=0, ignore_index=True) if outs else pd.DataFrame()
    out_df.sort_values(["timestamp", "sensor_id"], inplace=True, kind="mergesort")
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def attach_y_act(pred_df: pd.DataFrame, truth_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join smoothed y_act to predictions via (sensor_id, timestamp)."""
    merged = pred_df.merge(truth_lookup, on=["sensor_id", "timestamp"], how="left")
    return merged.loc[:, ["timestamp", "sensor_id", "y_act", "y_pred"]]


def check_identical(df_api: pd.DataFrame,
                    df_local: pd.DataFrame,
                    tolerance: float = 1e-8) -> Tuple[bool, float]:
    """Compare API vs LOCAL predictions aligned on (sensor_id, timestamp)."""
    key = ["sensor_id", "timestamp"]
    a = df_api.rename(columns={"y_pred": "y_pred_api"}).copy()
    l = df_local.rename(columns={"y_pred": "y_pred_local"}).copy()
    for d in (a, l):
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")

    merged = pd.merge(a, l, on=key, how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping rows to compare. Check time windows.")
    merged["diff"] = merged["y_pred_api"] - merged["y_pred_local"]
    max_abs = float(np.max(np.abs(merged["diff"].to_numpy())))
    return bool(max_abs <= tolerance), max_abs


# --------------------- Main CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Run service, predict via API and locally, attach *smoothed* y_act, and compare.")
    default_artifact = ROOT / "artifacts/traffic_pipeline_h-15.joblib"
    ap.add_argument("--artifact", default=str(default_artifact),
                    help="Path to joblib artifact used by both API and local.")
    default_raw = ROOT.parent / "data/NDW/ndw_three_weeks.parquet"
    ap.add_argument("--raw", default=str(default_raw),
                    help="RAW parquet path.")
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="Base URL of the service.")
    ap.add_argument("--start-server", action="store_true",
                    help="Start the Flask app in-process.")
    ap.add_argument("--batch-size", type=int, default=20000,
                    help="Batch size for API and local predictions.")
    ap.add_argument("--tolerance", type=float, default=1e-8,
                    help="Numeric tolerance for equality.")
    ap.add_argument("--save", action="store_true",
                    help="Save outputs to CSV.")
    args = ap.parse_args()

    server = None
    try:
        # 0) Load artifact states & horizon
        art = load_artifact_states(args.artifact)
        states = art["states"]
        horizon = art["horizon"]

        # 1) Optionally start the API here
        if args.start_server:
            os.environ["ARTIFACT_PATH"] = args.artifact
            host, port = "127.0.0.1", int(args.url.rsplit(":", 1)[-1])
            server = ServerThread(artifact_path=args.artifact, host=host, port=port)
            server.start()
            # Wait for /healthz
            for _ in range(60):
                try:
                    r = requests.get(f"{args.url}/healthz", timeout=1.0)
                    if r.ok:
                        break
                except Exception:
                    time.sleep(0.25)
            else:
                raise RuntimeError("Service did not become healthy on /healthz")

        # 2) RAW test rows (aligned boundary) and drop the two weather cols now
        raw_test, first_ts = get_test_raw_with_states(args.raw, states)

        # 3) Build *smoothed* ground truth y_act
        truth_lookup = build_smoothed_y_act(args.raw, states, horizon)

        # 4) Predictions via API and locally
        print("\n== Running API predictions ==")
        df_api_pred = api_predict_full(args.url, raw_test, batch_size=args.batch_size)
        print("\n== Running LOCAL predictions ==")
        df_local_pred = local_predict_full(args.artifact, raw_test, batch_size=args.batch_size)

        # 5) Attach y_act to each
        df_api = attach_y_act(df_api_pred, truth_lookup)
        df_local = attach_y_act(df_local_pred, truth_lookup)

        # 6) Compare API vs LOCAL
        ok, max_abs = check_identical(df_api, df_local, tolerance=args.tolerance)
        print(f"\nIdentical within tolerance ({args.tolerance}): {ok}  |  max |Δ| = {max_abs:.3g}")

        # 7) Show & optionally save
        print("\nAPI df (head):")
        print(df_api.head())
        print("\nLOCAL df (head):")
        print(df_local.head())
        print(f"\nShapes: API={df_api.shape}, LOCAL={df_local.shape}")

        if args.save:
            Path("outputs").mkdir(exist_ok=True)
            df_api.to_csv("outputs/api_predictions_with_y_act.csv", index=False)
            df_local.to_csv("outputs/local_predictions_with_y_act.csv", index=False)
            print("\nSaved to outputs/*.csv")

    finally:
        if server:
            print("\nShutting down in-process server...")
            server.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")