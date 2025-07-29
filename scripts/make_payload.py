
from __future__ import annotations
import sys, json, joblib
from pathlib import Path
import argparse
# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator



    

ARTIFACT = Path("artifacts/traffic_pipeline_h-15.joblib")
RAW_PATH = Path("../data/NDW/ndw_three_weeks.parquet")

def main(total_rows = None):
    # 1) Load artifact to discover expected feature columns
    bundle = joblib.load(ARTIFACT)
    states = bundle["states"]
    expected_cols = states["feature_cols"]
    print(f"Expected features from artifact: {expected_cols}")
    
    # 2) Use orchestrator only for split boundary
    tdp = TrafficDataPipelineOrchestrator(file_path=str(RAW_PATH), sensor_encoding_type="mean")
    tdp.prepare_base_features(window_size=5)
    raw = pd.read_parquet(RAW_PATH)
    raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
    print(f"Raw test data has {raw_test.columns} columns.")
    #3) Ensure the sample is sorted by date and sensor_id
    
    if total_rows is None:
        sample = raw_test.sort_values(["date", "sensor_id"], kind="mergesort").copy()
    else:
        sample = raw_test.sort_values(["date", "sensor_id"], kind="mergesort").head(total_rows).copy()
    sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5) Replace NaN/inf with None for strict JSON
    sample = sample.where(pd.notna(sample), None)

    # <-- this is the key: to_json → nulls
    records = json.loads(sample.to_json(orient="records"))
    payload = {"records": records}
    #payload = {"records": sample.to_dict(orient="records")}
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "payload.json"
    output_path.write_text(json.dumps(payload, indent = 2))
    #Path("payload.json").write_text(json.dumps(payload, indent=2))
    print(f"Payload written with {len(records)} rows and columns: {sample.columns.tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a JSON payload for traffic flow service.")
    parser.add_argument("--total-rows", type=int, default=None,
                        help="Total number of rows to include in the payload. If None, use all available rows.")
    args = parser.parse_args()
    main(total_rows=args.total_rows)
    
