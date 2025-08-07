#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Create payload.json for /predict")
    p.add_argument("--raw",  help="Path to raw data (.parquet or .csv)",default="../data/NDW/ndw_three_weeks.parquet")
    p.add_argument("--out", default="payload.json", help="Output JSON file")
    p.add_argument("--n-rows", type=int, default=30000, help="Number of rows to include")
    p.add_argument("--sensor-col", default="sensor_id")
    p.add_argument("--datetime-col", default="date")
    p.add_argument("--value-col", default="value")
    args = p.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        sys.exit(f"Raw file not found: {raw_path}")

    # Load
    if raw_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_csv(raw_path)

    # (Optional) sort to be deterministic and align with server assumptions
    for c in (args.datetime_col, args.sensor_col, args.value_col):
        if c not in df.columns:
            sys.exit(f"Missing required column: {c}")
    df = df.sort_values([args.datetime_col, args.sensor_col]).reset_index(drop=True)

    # Take first N rows (and keep ALL raw columns)
    slice_df = df.iloc[:args.n_rows].copy()

    # Build JSON payload
    payload = {
        "records": json.loads(slice_df.to_json(orient="records", date_format="iso"))
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path} with {len(slice_df)} records and {slice_df.shape[1]} columns.")

if __name__ == "__main__":
    main()