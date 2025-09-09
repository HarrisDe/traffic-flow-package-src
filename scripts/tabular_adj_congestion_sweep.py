#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sweep adjacency + congestion knobs for the tabular XGB pipeline.

Swept (in prepare_base_features):
- spatial_adj ∈ {1,2,3}
- quantile_threshold ∈ {0.90, 0.80, 0.70}
- quantile_percentage ∈ {0.55, 0.65, 0.75}
- add_adjacency_congestion_features ∈ {True, False}
- normalize_by_distance_congested ∈ {True, False}

Skips only COMPLETED runs (i.e., matching run_id with model_MAE and model_RMSE present).
"""

import argparse
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np

from traffic_flow.tabular.experiment import DataCfg, TabularExperiment

def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    # treat as completed if metrics exist and are finite
    ok = (~df.get("model_MAE", pd.Series([np.nan]*len(df))).isna()) & \
         (~df.get("model_RMSE", pd.Series([np.nan]*len(df))).isna())
    return set(df.loc[ok, "run_id"].astype(str))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", default="../../data/NDW/ndw_three_weeks.parquet")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--artifacts-dir", default="./results_tabular")
    ap.add_argument("--results-csv", default="tabular_xgb_results_adj_cong_horizon_15.csv")
    ap.add_argument("--use-gpu", action="store_true")
    args = ap.parse_args()

    results_csv = Path(args.artifacts_dir) / args.results_csv
    completed = load_completed(results_csv)

    # Grids
    spatial_grid   = [3, 2, 1]
    qthr_grid      = [0.90, 0.80, 0.70]
    qpct_grid      = [0.55, 0.65, 0.75]
    adj_cong_grid  = [True, False]
    norm_cong_grid = [True, False]

    combos = list(product(spatial_grid, qthr_grid, qpct_grid, adj_cong_grid, norm_cong_grid))
    print(f"Total combos: {len(combos)}")

    for k, qt, qp, add_adj_cong, norm_adj_cong in combos:
        cfg = DataCfg(
            artifacts_dir=args.artifacts_dir,
            results_csv_name=args.results_csv,
            horizon=args.horizon,
            orchestrator_kwargs={
                "file_path": args.file_path,
                # you can set sensor/datetime/value col names here if different from defaults
            },
            prepare_kwargs={
                # everything else remains default; override only what we sweep or must set
                "spatial_adj": k,
                "quantile_threshold": qt,
                "quantile_percentage": qp,
                "add_adjacency_congestion_features": add_adj_cong,
                "normalize_by_distance_congested": norm_adj_cong,
                # keep adjacency speed features on (and in the same style) if you want:
                # "adj_are_relative": True,
                # "normalize_by_distance": True,
            },
            xgb_use_gpu=args.use_gpu,
        )

        exp = TabularExperiment(cfg)
        run_id = exp._make_run_id()  # uses the params we just set

        if run_id in completed:
            print(f"[skip] {run_id} (already completed)")
            continue

        print(f"[run ] {run_id}")
        try:
            exp.run()
        except Exception as e:
            # don't crash the sweep; just log and continue
            print(f"[fail] {run_id}: {e!r}")

if __name__ == "__main__":
    main()