# scripts/make_payload.py
from __future__ import annotations
import sys, json, joblib, argparse
from pathlib import Path
import pandas as pd

# Make the package importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator

ARTIFACT = Path("artifacts/traffic_pipeline_h-15.joblib")
RAW_PATH = Path("../data/NDW/ndw_three_weeks.parquet")

# Columns we drop to avoid JSON NaN issues
DROP_RAW_COLS = [
    "Snow_depth_surface",
    "Water_equivalent_of_accumulated_snow_depth_surface",
]

def main(sensor_id: str | None, total_rows: int | None):
    # --- 0) Enforce mutual exclusivity as requested
    if sensor_id is not None and total_rows is not None:
        raise ValueError("Use either --sensor-id OR --total-rows (not both).")

    # --- 1) Load artifact (mainly to prove things are wired)
    bundle = joblib.load(ARTIFACT)
    states = bundle["states"]
    print(f"[make_payload] model horizon={bundle.get('horizon', 15)} | "
          f"#features={len(states['feature_cols'])}")

    # --- 2) Use orchestrator only for the train/test split boundary
    tdp = TrafficDataPipelineOrchestrator(file_path=str(RAW_PATH), sensor_encoding_type="mean")
    tdp.prepare_base_features(window_size=5)  # sets first_test_timestamp

    raw = pd.read_parquet(RAW_PATH)
    # Drop the two problematic weather columns right away (leaner payload)
    drop_now = [c for c in DROP_RAW_COLS if c in raw.columns]
    if drop_now:
        raw = raw.drop(columns=drop_now)

    raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
    raw_test.sort_values(["date", "sensor_id"], kind="mergesort", inplace=True)

    # --- 3) Choose rows: either one full sensor OR first N rows overall
    if sensor_id is not None:
        if sensor_id not in raw_test["sensor_id"].unique():
            raise ValueError(f"--sensor-id '{sensor_id}' not found in test set.")
        sample = raw_test.loc[raw_test["sensor_id"] == sensor_id].copy()
        print(f"[make_payload] Selected sensor_id={sensor_id} | rows={len(sample)}")
    else:
        # total_rows may be None → take all rows
        sample = raw_test.head(total_rows) if total_rows is not None else raw_test.copy()
        print(f"[make_payload] Selected first rows: {len(sample)}")

    # --- 4) Make JSON-safe
    sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    # Use pandas' JSON converter to automatically turn NaN → null
    records = json.loads(sample.to_json(orient="records"))
    payload = {"records": records}

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "payload.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"[make_payload] Wrote {out_file} with {len(records)} rows.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create a JSON payload for the traffic service.")
    p.add_argument("--sensor-id", type=str, default=None,
                   help="If set, include all test rows for this sensor (mutually exclusive with --total-rows).")
    p.add_argument("--total-rows", type=int, default=None,
                   help="If set, include first N rows across all sensors (mutually exclusive with --sensor-id).")
    args = p.parse_args()
    main(sensor_id=args.sensor_id, total_rows=args.total_rows)