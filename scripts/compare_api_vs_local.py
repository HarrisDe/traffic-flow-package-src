# scripts/compare_api_vs_local.py
from __future__ import annotations
import os, json, time, argparse, threading
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---- Our package imports (editable install assumed) ----
from traffic_flow.service.runtime import InferenceRuntime
from traffic_flow.service.app import create_app
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator


# ----------------------- Utilities -----------------------

class ServerThread(threading.Thread):
    """
    Start the Flask app in-process so this script is standalone.
    If --start-server is not used, we assume the API is already running.
    """
    def __init__(self, artifact_path: str, host="127.0.0.1", port=8080):
        super().__init__(daemon=True)
        self.app = create_app(artifact_path=artifact_path)
        from werkzeug.serving import make_server
        self.srv = make_server(host, port, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()
        self.ctx.pop()


def json_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/±inf with None so JSON serialization is strict."""
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.where(pd.notna(df), None)


def ensure_payload(args) -> dict:
    """
    Build a RAW-rows payload if --payload was not provided.
    Uses the orchestrator ONLY to read the test split timestamp.
    """
    if args.payload and Path(args.payload).exists():
        return json.loads(Path(args.payload).read_text())

    # Discover test boundary
    tdp = TrafficDataPipelineOrchestrator(
        file_path=args.raw, sensor_encoding_type="mean"
    )
    tdp.prepare_base_features(window_size=5)  # sets first_test_timestamp

    raw = pd.read_parquet(args.raw)
    raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
    raw_test = raw_test.sort_values(["date", "sensor_id"], kind="mergesort")

    sample = raw_test.head(args.nrows).copy()
    # Make sure datetime is a string for JSON
    sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    sample = json_friendly(sample)
    payload = {"records": sample.to_dict(orient="records")}
    return payload


def local_predictions(artifact_path: str, payload: dict) -> pd.DataFrame:
    """
    Run local (offline) predictions from the same artifact that the API uses.
    Returns a DataFrame with:
      sensor_id, date (input), prediction_time, y_pred_delta, y_pred_total
    """
    # Load RAW rows back to DataFrame
    raw_df = pd.DataFrame(payload["records"]).copy()
    # Convert date -> datetime for math; keep original string for echo
    dt_col = "date"
    dt = pd.to_datetime(raw_df[dt_col], errors="coerce")

    # Use the same runtime used by the service
    rt = InferenceRuntime(artifact_path)
    pred_delta, feats = rt.predict_df(raw_df)      # DELTA only
    if "value" not in feats.columns:
        raise RuntimeError("Engineered features missing 'value' column.")
    pred_total = (pred_delta + feats["value"].to_numpy()).astype(float)

    pred_time = dt + pd.to_timedelta(rt.horizon, unit="m")

    out = pd.DataFrame({
        "sensor_id": raw_df.get("sensor_id", None),
        "date": dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_time": pred_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "y_pred_delta": pred_delta.astype(float),
        "y_pred_total": pred_total,
    })
    return out


def api_predictions(base_url: str, payload: dict) -> pd.DataFrame:
    """POST payload to /predict and return the response as a DataFrame."""
    r = requests.post(f"{base_url}/predict", json=payload, timeout=180)
    r.raise_for_status()
    obj = r.json()
    df = pd.DataFrame(obj["predictions"])
    return df


def compare_frames(df_local: pd.DataFrame,
                   df_api: pd.DataFrame,
                   tolerance: float = 1e-6) -> dict:
    """
    Align by (sensor_id, date, prediction_time) and compute error metrics.
    Returns a metrics dict and the merged frame with diffs.
    """
    for col in ("date", "prediction_time"):
        for d in (df_local, df_api):
            d[col] = pd.to_datetime(d[col], errors="coerce")

    key = ["sensor_id", "date", "prediction_time"]
    merged = pd.merge(
        df_local, df_api, on=key, how="inner", suffixes=("_local", "_api")
    )

    if merged.empty:
        raise RuntimeError("No common rows after alignment; check inputs/time windows.")

    # Differences
    merged["diff_delta"] = merged["y_pred_delta_local"] - merged["y_pred_delta_api"]
    merged["diff_total"] = merged["y_pred_total_local"] - merged["y_pred_total_api"]

    def metrics_for(col: str) -> dict:
        arr = merged[col].to_numpy()
        mae = float(np.mean(np.abs(arr)))
        rmse = float(np.sqrt(np.mean(arr ** 2)))
        maxabs = float(np.max(np.abs(arr)))
        within = float(np.mean(np.abs(arr) <= tolerance))
        return {"mae": mae, "rmse": rmse, "max_abs": maxabs, "pct_within_tol": within}

    m_delta = metrics_for("diff_delta")
    m_total = metrics_for("diff_total")

    metrics = {
        "n_aligned": int(len(merged)),
        "tolerance": float(tolerance),
        "delta": m_delta,
        "total": m_total,
    }
    return metrics, merged


# ----------------------- Main CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Compare API vs local predictions.")
    ap.add_argument("--artifact", default="artifacts/traffic_pipeline_h-15.joblib",
                    help="Path to joblib artifact used by both API and local.")
    ap.add_argument("--raw", default="data/NDW/ndw_three_weeks.parquet",
                    help="RAW parquet to build a payload from (if --payload not given).")
    ap.add_argument("--payload", default=None,
                    help="Optional path to an existing payload.json.")
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="Base URL of the running service.")
    ap.add_argument("--start-server", action="store_true",
                    help="Start the Flask app in-process for this run.")
    ap.add_argument("--nrows", type=int, default=200,
                    help="How many RAW rows to include when auto-building payload.")
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="Numeric tolerance for equality.")
    ap.add_argument("--out", default="compare_api_vs_local.csv",
                    help="Where to save the merged comparison CSV.")
    args = ap.parse_args()

    # Optionally spin up the server here so we need only one command
    server = None
    if args.start_server:
        os.environ["ARTIFACT_PATH"] = args.artifact
        server = ServerThread(artifact_path=args.artifact, port=int(args.url.rsplit(":", 1)[-1]))
        server.start()
        # wait for health
        for _ in range(40):
            try:
                r = requests.get(f"{args.url}/healthz", timeout=1.0)
                if r.ok:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            if server:
                server.shutdown()
            raise RuntimeError("Service did not become healthy on /healthz")

    try:
        # Build or load the same RAW input for both paths
        payload = ensure_payload(args)

        # 1) Local predictions (artifact only)
        df_local = local_predictions(args.artifact, payload)

        # 2) API predictions (HTTP)
        df_api = api_predictions(args.url, payload)

        # 3) Compare
        metrics, merged = compare_frames(df_local, df_api, tolerance=args.tolerance)
        merged.to_csv(args.out, index=False)

        print("\n=== Comparison Summary ===")
        print(f"Aligned rows: {metrics['n_aligned']}")
        print(f"Tolerance: {metrics['tolerance']}")
        print("Delta  ->  MAE: {mae:.8f}  RMSE: {rmse:.8f}  Max|err|: {max_abs:.8f}  %within_tol: {pct_within_tol:.3%}"
              .format(**metrics["delta"]))
        print("Total  ->  MAE: {mae:.8f}  RMSE: {rmse:.8f}  Max|err|: {max_abs:.8f}  %within_tol: {pct_within_tol:.3%}"
              .format(**metrics["total"]))
        print(f"\nSaved merged comparison to: {args.out}")

        # Optional hard assertion:
        if (metrics["total"]["max_abs"] > args.tolerance) or (metrics["delta"]["max_abs"] > args.tolerance):
            print("\nWARNING: Differences exceed tolerance. Inspect the CSV for details.")
        else:
            print("\nOK: API and local predictions match within tolerance.")

    finally:
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()