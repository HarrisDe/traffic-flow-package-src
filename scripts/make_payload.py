
# # add the repo root (parent of scripts/) to sys.path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator

# # 1) Use the orchestrator to get the first_test_timestamp (split boundary)
# tdp = TrafficDataPipelineOrchestrator(
#     file_path="../data/NDW/ndw_three_weeks.parquet",
#     sensor_encoding_type="mean"   # or 'ordinal', this only finds the split
# )
# tdp.prepare_base_features(window_size=5)  # enough to set first_test_timestamp

# # 2) Load RAW data and filter to test period
# raw = pd.read_parquet("../data/NDW/ndw_three_weeks.parquet")
# raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()

# # 3) (Optional) pick one sensor to keep it tiny/clear
# one_sensor = raw_test.loc[raw_test["sensor_id"] == raw_test["sensor_id"].iloc[0]]
# sample = one_sensor.head(500)[["sensor_id", "date", "value"]]   # 5 rows is plenty

# # 4) Save as JSON payload the service expects
# payload = {"records": sample.to_dict(orient="records")}
# Path("payload.json").write_text(pd.io.json.dumps(payload, indent=2))
# print("Wrote payload.json with", len(payload["records"]), "records.")


from __future__ import annotations
import sys, json, joblib
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator


# def main():
#     # 1) Use the orchestrator only to get the train/test split boundary
#     tdp = TrafficDataPipelineOrchestrator(
#         file_path="../data/NDW/ndw_three_weeks.parquet",
#         sensor_encoding_type="mean",   # just to get split; doesn't matter here
#     )
#     tdp.prepare_base_features(window_size=5)  # sets first_test_timestamp

#     # 2) Load RAW data and filter to test period
#     raw = pd.read_parquet("../data/NDW/ndw_three_weeks.parquet")
#     raw_test = raw.loc[raw["date"] >= tdp.first_test_timestamp].copy()
#     raw_test.sort_values(by=[ "date",'sensor_id'], inplace=True)

#     # 3) (Optional) pick one sensor to keep it tiny/clear
#     one_sensor_id = raw_test["sensor_id"].iloc[0]
#     one_sensor = raw_test.loc[raw_test["sensor_id"] == one_sensor_id].copy()

#     # IMPORTANT: keep only the columns your service expects as raw inputs.
#     # The service builds the rest of features.
#     #sample = one_sensor.head(5)[["sensor_id", "date", "value"]].copy()
#     # ensure datetime is a string
#     #sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
#     sample = raw_test.head(200)
#     sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
#     # 4) Save as JSON payload the service expects
#     payload = {"records": sample.to_dict(orient="records")}
    
#     Path("payload.json").write_text(json.dumps(payload, indent=2))
#     print(f"Wrote payload.json with {len(payload['records'])} records for sensor {one_sensor_id}.")


# if __name__ == "__main__":
#     main()
    
    
    

ARTIFACT = Path("artifacts/traffic_pipeline_h-15.joblib")
RAW_PATH = Path("../data/NDW/ndw_three_weeks.parquet")

def main():
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
    sample = raw_test.sort_values(["date", "sensor_id"], kind="mergesort").head(200).copy()
    sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5) Replace NaN/inf with None for strict JSON
    sample = sample.where(pd.notna(sample), None)

    payload = {"records": sample.to_dict(orient="records")}
    Path("payload.json").write_text(json.dumps(payload, indent=2))
    print(f"Payload written with {len(sample)} rows and columns: {sample.columns.tolist()}")

if __name__ == "__main__":
    main()
    
    
 # 3) Build the minimal raw schema required by the service:
    #    mandatory + any passthrough columns that were used during training
    # must_have = {"sensor_id", "date", "value"}
    # print(f"Raw test columns: {set(raw_test.columns)}")
    # passthrough_needed = expected.intersection(raw_test.columns)
    # print(f"Passthrough columns needed: {passthrough_needed}")
    # input_cols = sorted(must_have.union(passthrough_needed))
    # print(f"Input columns for the service: {input_cols}")
    # # 4) Small sample (you can increase .head(200) if you want)
    # sample = raw_test.sort_values(["date", "sensor_id"], kind="mergesort").head(200)[input_cols].copy()
    # sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")