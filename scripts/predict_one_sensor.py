# scripts/predict_one_sensor.py
from __future__ import annotations
import os, json, time, argparse, threading
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib
import sys

# repo bootstrap
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_flow.service.app import create_app
from traffic_flow.service.runtime import InferenceRuntime
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
from traffic_flow.preprocessing.cleaning import clean_and_cast

DROP_RAW_COLS = [
    "Snow_depth_surface",
    "Water_equivalent_of_accumulated_snow_depth_surface",
]

class ServerThread(threading.Thread):
    def __init__(self, artifact_path: str, host="127.0.0.1", port=8080):
        super().__init__(daemon=True)
        self.app = create_app(artifact_path=artifact_path)
        from werkzeug.serving import make_server
        self.srv = make_server(host, port, self.app)
        self.ctx = self.app.app_context(); self.ctx.push()
    def run(self): self.srv.serve_forever()
    def shutdown(self): self.srv.shutdown(); self.ctx.pop()

def to_json_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    if "date" in out.columns:
        ts = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        out.loc[ts.isna(), "date"] = None
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.astype(object).where(pd.notna(out), None)
    # numpy → python scalars for JSON
    def _py(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.bool_,)):    return bool(v)
        return v
    return [{k: _py(v) for k, v in row.items()} for row in out.to_dict("records")]

def load_artifact_states(artifact: str) -> dict:
    b = joblib.load(artifact)
    return {"states": b["states"], "horizon": int(b.get("horizon", 15))}

def make_tdp_for_split(raw_path: str | Path, states: dict) -> TrafficDataPipelineOrchestrator:
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

def get_sensor_raw_test(raw_path: str | Path, states: dict, sensor_id: str) -> pd.DataFrame:
    tdp = make_tdp_for_split(raw_path, states)
    raw = pd.read_parquet(raw_path)
    raw = clean_and_cast(raw, value_col="value")
    # drop 2 weather cols early
    drop_now = [c for c in DROP_RAW_COLS if c in raw.columns]
    if drop_now: raw = raw.drop(columns=drop_now)
    test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
    test = test.loc[test["sensor_id"] == sensor_id].copy()
    if test.empty:
        raise ValueError(f"Sensor '{sensor_id}' not present in test period.")
    test.sort_values(["date", "sensor_id"], kind="mergesort", inplace=True)
    return test

def build_smoothed_y_act(raw_path: str | Path, states: dict, horizon: int, sensor_id: str) -> pd.DataFrame:
    tdp = make_tdp_for_split(raw_path, states)
    tdp.finalise_for_horizon(
        horizon=horizon, drop_weather=True,
        add_previous_weekday_feature=False, drop_datetime=False
    )
    df = tdp.df.loc[tdp.df["test_set"], ["sensor_id","date_of_prediction","target_total_speed"]].copy()
    df.rename(columns={"date_of_prediction":"timestamp","target_total_speed":"y_act"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.loc[df["sensor_id"] == sensor_id].copy()
    df.sort_values(["timestamp","sensor_id"], inplace=True, kind="mergesort")
    return df

def api_predict_sensor(base_url: str, raw_df: pd.DataFrame, batch_size: int = 5000) -> pd.DataFrame:
    out = []
    for i in range(0, len(raw_df), batch_size):
        batch = raw_df.iloc[i:i+batch_size]
        payload = {"records": to_json_records(batch)}
        r = requests.post(f"{base_url}/predict", json=payload, timeout=300)
        r.raise_for_status()
        part = pd.DataFrame(r.json()["predictions"])
        part = part.rename(columns={"prediction_time":"timestamp","y_pred_total":"y_pred"})
        part["timestamp"] = pd.to_datetime(part["timestamp"], errors="coerce")
        out.append(part[["timestamp","sensor_id","y_pred"]])
    df = pd.concat(out, ignore_index=True)
    df.sort_values(["timestamp","sensor_id"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def local_predict_sensor(artifact: str, raw_df: pd.DataFrame, batch_size: int = 5000) -> pd.DataFrame:
    rt = InferenceRuntime(artifact)
    out = []
    for i in range(0, len(raw_df), batch_size):
        batch = raw_df.iloc[i:i+batch_size].copy()
        pred_delta, feats = rt.predict_df(batch)
        y_pred_total = (pred_delta + feats["value"].to_numpy()).astype(float)
        # Align to prediction length
        n_pred = len(y_pred_total)
        dt = pd.to_datetime(batch["date"].iloc[:n_pred], errors="coerce")
        ts = dt + pd.to_timedelta(rt.horizon, unit="m")
        sid = batch["sensor_id"].iloc[:n_pred].to_numpy()
        part = pd.DataFrame({"timestamp": ts.to_numpy(), "sensor_id": sid, "y_pred": y_pred_total})
        out.append(part)
    df = pd.concat(out, ignore_index=True)
    df.sort_values(["timestamp","sensor_id"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def compare(df_api: pd.DataFrame, df_local: pd.DataFrame, tol=1e-8) -> tuple[bool,float]:
    a = df_api.rename(columns={"y_pred":"y_pred_api"}).copy()
    l = df_local.rename(columns={"y_pred":"y_pred_local"}).copy()
    for d in (a,l): d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    m = a.merge(l, on=["sensor_id","timestamp"], how="inner")
    if m.empty:
        raise RuntimeError("No overlap between API and local outputs.")
    m["diff"] = m["y_pred_api"] - m["y_pred_local"]
    max_abs = float(np.max(np.abs(m["diff"].to_numpy())))
    return (max_abs <= tol), max_abs

def main():
    ap = argparse.ArgumentParser(description="Predict one sensor via API and locally, compare with smoothed y_act.")
    default_artifact = ROOT / "artifacts/traffic_pipeline_h-15.joblib"
    default_raw = ROOT.parent / "data/NDW/ndw_three_weeks.parquet"
    ap.add_argument("--artifact", default=str(default_artifact))
    ap.add_argument("--raw", default=str(default_raw))
    ap.add_argument("--sensor-id", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    server = None
    try:
        art = load_artifact_states(args.artifact)
        states, horizon = art["states"], art["horizon"]

        if args.start_server:
            os.environ["ARTIFACT_PATH"] = args.artifact
            host, port = "127.0.0.1", int(args.url.rsplit(":",1)[-1])
            server = ServerThread(args.artifact, host=host, port=port)
            server.start()
            # wait for health
            for _ in range(60):
                try:
                    if requests.get(f"{args.url}/healthz", timeout=1).ok: break
                except Exception: time.sleep(0.25)

        # RAW test for this sensor
        raw_test = get_sensor_raw_test(args.raw, states, args.sensor_id)

        # Smoothed ground truth
        truth = build_smoothed_y_act(args.raw, states, horizon, args.sensor_id)

        # Predictions via API and locally
        print("\n== API predictions ==")
        df_api = api_predict_sensor(args.url, raw_test, batch_size=args.batch_size)
        print("== LOCAL predictions ==")
        df_local = local_predict_sensor(args.artifact, raw_test, batch_size=args.batch_size)

        # Attach y_act
        df_api = df_api.merge(truth, on=["sensor_id","timestamp"], how="left")
        df_local = df_local.merge(truth, on=["sensor_id","timestamp"], how="left")

        # Compare API vs local
        ok, max_abs = compare(df_api, df_local)
        print(f"\nParity check (API vs LOCAL): ok={ok} | max|Δ|={max_abs:.3g}")

        print("\nAPI (head):");   print(df_api.head())
        print("\nLOCAL (head):"); print(df_local.head())

        if args.save:
            Path("outputs").mkdir(exist_ok=True)
            df_api.to_csv(f"outputs/{args.sensor_id}_api.csv", index=False)
            df_local.to_csv(f"outputs/{args.sensor_id}_local.csv", index=False)
            print(f"\nSaved outputs/* for sensor {args.sensor_id}")

    finally:
        if server:
            print("\nShutting down in-process server...")
            server.shutdown()

if __name__ == "__main__":
    main()