#!/usr/bin/env python3
# scripts/compare_api_vs_local.py

import argparse, json, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Your package imports
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
from traffic_flow.service.runtime import InferenceRuntime

# Optional inline server for convenience (same as in your notebook)
import threading
from flask import Flask, request, jsonify
from werkzeug.serving import make_server
from traceback import format_exc


class InlineServer(threading.Thread):
    """Run the service in-process for quick tests."""
    def __init__(self, artifact_path: str, host: str = "127.0.0.1", port: int = 0):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.rt  = InferenceRuntime(artifact_path, keep_datetime=True)

        @self.app.get("/healthz")
        def healthz():
            return jsonify({"status": "ok"}), 200

        @self.app.post("/predict")
        def predict():
            try:
                payload = request.get_json(force=True)
                if not payload or "records" not in payload:
                    return jsonify(error="Payload must include 'records' list"), 400
                df_raw = pd.DataFrame(payload["records"])

                # Minimal validation
                for c in (self.rt.sensor_col, self.rt.datetime_col, self.rt.value_col):
                    if c not in df_raw.columns:
                        return jsonify(error=f"Missing column: {c}"), 400
                df_raw[self.rt.datetime_col] = pd.to_datetime(df_raw[self.rt.datetime_col], errors="coerce")

                pred_df, _ = self.rt.predict_df(
                    df_raw, trim_warmup=True, add_total=True, add_y_act=False
                )
                out = pred_df.copy()
                for c in ("input_time", "prediction_time"):
                    out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d %H:%M:%S")

                return jsonify(
                    horizon=self.rt.horizon,
                    n=int(len(out)),
                    predictions=out.to_dict(orient="records")
                ), 200
            except Exception as e:
                return jsonify(error=str(e), traceback="".join(format_exc())), 500

        self.srv = make_server(self.host, self.port, self.app)
        self.port = self.srv.server_port
        self.ctx = self.app.app_context(); self.ctx.push()

    def run(self): self.srv.serve_forever()
    def shutdown(self):
        self.srv.shutdown()
        self.ctx.pop()


def state_get(d: dict, block: str, key: str, default=None):
    """Read states[block][key] with fallback to states[block]['params'][key]."""
    b = d.get(block, {}) or {}
    if key in b:
        return b[key]
    return (b.get("params") or {}).get(key, default)


def build_test_slice(artifact_path: Path, raw_path: Path, max_rows: int) -> Tuple[pd.DataFrame, dict]:
    """Recreate the test window (starting at the training split) and take first `max_rows` raw rows."""
    import joblib
    bundle = joblib.load(artifact_path)
    states = bundle["states"]

    schema      = states.get("schema_state", {})
    sensor_col  = schema.get("sensor_col", "sensor_id")
    dt_col      = schema.get("datetime_col", states["datetime_state"]["datetime_col"])
    value_col   = schema.get("value_col", "value")
    clean       = states["clean_state"]

    # pull adjacency config from the saved state
    adj_are_relative      = state_get(states, "adjacency_state", "adj_are_relative", True)
    normalize_by_distance = state_get(states, "adjacency_state", "normalize_by_distance", True)
    spatial_adj           = state_get(states, "adjacency_state", "spatial_adj", 1)

    # Use orchestrator to recover the split boundary
    tdp = TrafficDataPipelineOrchestrator(
        file_path=str(raw_path),
        sensor_encoding_type="mean",
        sensor_col=sensor_col,
        datetime_col=dt_col,
        value_col=value_col,
    )
    tdp.prepare_base_features(
        window_size=clean["smoothing_window"],
        filter_extreme_changes=clean.get("filter_extreme_changes", True),
        filter_on_train_only=clean.get("filter_on_train_only", False),
        smooth_speeds=clean.get("smooth_speeds", True),
        relative_threshold=clean["relative_threshold"],
        use_median_instead_of_mean_smoothing=clean["use_median"],
        spatial_adj=spatial_adj,
        normalize_by_distance=normalize_by_distance,
        adj_are_relative=adj_are_relative,
    )

    raw = pd.read_parquet(raw_path)
    raw_test = (
        raw.loc[raw[dt_col] >= tdp.first_test_timestamp]
           .sort_values([dt_col, sensor_col], kind="mergesort")
           .reset_index(drop=True)
    )
    return raw_test.iloc[:max_rows].copy(), {
        "sensor_col": sensor_col,
        "datetime_col": dt_col,
        "value_col": value_col,
        "horizon": int(bundle.get("horizon", 15)),
        "states": states,
    }


def call_api_in_batches(host: str, port: int, df_raw: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """POST /predict in batches and concatenate results."""
    out_frames: List[pd.DataFrame] = []
    n = len(df_raw)
    for i in range(0, n, batch_size):
        chunk = df_raw.iloc[i:i+batch_size].copy()
        payload = {"records": json.loads(chunk.to_json(orient="records", date_format="iso"))}
        r = requests.post(f"http://{host}:{port}/predict", json=payload, timeout=180)
        if r.status_code != 200:
            try:
                print("API error payload:", r.json())
            except Exception:
                print("API error text:", r.text)
            raise SystemExit("Stopping due to API error")

        df = pd.DataFrame(r.json()["predictions"])
        df["prediction_time"] = pd.to_datetime(df["prediction_time"])
        df["input_time"]      = pd.to_datetime(df["input_time"])
        out_frames.append(df)

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    out = out.sort_values(["prediction_time", "sensor_id"], kind="mergesort").reset_index(drop=True)
    return out


def run_local_predictions(artifact_path: Path, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Use the same runtime locally (no server) to predict on the same raw slice."""
    rt = InferenceRuntime(str(artifact_path), keep_datetime=True)
    pred_df, _ = rt.predict_df(df_raw=df_raw, trim_warmup=True, add_total=True, add_y_act=False)
    pred_df = pred_df.sort_values(["prediction_time", "sensor_id"], kind="mergesort").reset_index(drop=True)
    return pred_df


def align_and_report(api_pred: pd.DataFrame, local_pred: pd.DataFrame) -> pd.DataFrame:
    """Join on keys and print MAE stats; return the aligned frame."""
    key = ["prediction_time", "sensor_id"]
    merged = api_pred.merge(local_pred, on=key, suffixes=("_api", "_loc"), how="inner")
    mae_delta = float(np.mean(np.abs(merged["y_pred_delta_api"] - merged["y_pred_delta_loc"])))
    mae_total = float(np.mean(np.abs(merged["y_pred_total_api"] - merged["y_pred_total_loc"])))
    mx_total  = float(np.max(np.abs(merged["y_pred_total_api"] - merged["y_pred_total_loc"])))
    print(f"Aligned rows: {len(merged)}")
    print(f"Δ MAE (y_pred_delta): {mae_delta:.6g}")
    print(f"Total MAE (y_pred_total): {mae_total:.6g}   Max|Δ| (total): {mx_total:.6g}")
    return merged


def plot_parity(merged: pd.DataFrame, out_path: Path):
    """Parity plot y_pred_total: API vs Local."""
    x = merged["y_pred_total_loc"]
    y = merged["y_pred_total_api"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, alpha=0.5, label="Predictions")
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="x = y")

    ax.set_xlabel("Offline Prediction (y_pred_total)", fontsize=14)
    ax.set_ylabel("API Prediction (y_pred_total)", fontsize=14)
    ax.set_title("Offline vs. API Prediction Comparison", fontsize=18, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved parity plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare API vs local predictions on a raw test slice.")
    ap.add_argument("--artifact", required=True, help="Path to joblib artifact")
    ap.add_argument("--raw", required=True, help="Path to raw parquet")
    ap.add_argument("--host", default="127.0.0.1", help="API host")
    ap.add_argument("--port", type=int, default=8080, help="API port")
    ap.add_argument("--max-rows", type=int, default=30000, help="Number of raw rows to test")
    ap.add_argument("--batch-size", type=int, default=5000, help="Rows per API request")
    ap.add_argument("--out-plot", default="parity_api_vs_local.png", help="Output parity plot")
    ap.add_argument("--out-csv-local", default="", help="Optional: path to save local predictions CSV")
    ap.add_argument("--out-csv-api", default="", help="Optional: path to save API predictions CSV")
    ap.add_argument("--start-inline", action="store_true", help="Start an inline Flask server for this run")
    args = ap.parse_args()

    artifact_path = Path(args.artifact)
    raw_path = Path(args.raw)

    # Build the same raw slice used by both paths
    slice_raw, meta = build_test_slice(artifact_path, raw_path, args.max_rows)

    # Optionally start an inline server (if you don't already have one running)
    srv = None
    if args.start_inline:
        srv = InlineServer(str(artifact_path), host=args.host, port=0 if args.port == 0 else args.port)
        srv.start()
        # wait until healthy
        for _ in range(100):
            try:
                if requests.get(f"http://{srv.host}:{srv.port}/healthz", timeout=0.5).ok:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        print(f"Inline server on http://{srv.host}:{srv.port}")
        host, port = srv.host, srv.port
    else:
        host, port = args.host, args.port

    # === API path (batched) ===
    api_pred = call_api_in_batches(host, port, slice_raw, args.batch_size)

    # === Local path ===
    local_pred = run_local_predictions(artifact_path, slice_raw)

    # Save CSVs if requested
    if args.out_csv_api:
        pd.DataFrame(api_pred).to_csv(args.out_csv_api, index=False)
        print(f"Saved API predictions to {args.out_csv_api}")
    if args.out_csv_local:
        pd.DataFrame(local_pred).to_csv(args.out_csv_local, index=False)
        print(f"Saved local predictions to {args.out_csv_local}")

    # Align & report
    merged = align_and_report(api_pred, local_pred)

    # Plot
    plot_parity(merged, Path(args.out_plot))

    # Clean up inline server
    if srv is not None:
        srv.shutdown()
        srv.join(timeout=2)


if __name__ == "__main__":
    main()