#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run four experiments toggling:
    add_prediction_time_cyclical_features ∈ {True, False}
    include_current_time_cyclical       ∈ {True, False}

Usage:
  python scripts/run_tabular_time_cyc_sweep.py --file-path /path/to/raw.parquet --horizon 15

Notes:
- We forward `file_path` (and any other pipeline args) via `orchestrator_kwargs`.
- Adjust datetime/target/sensor keys below if your orchestrator expects different names.
"""

import argparse
from itertools import product

from traffic_flow.tabular.experiment import DataCfg, TabularExperiment


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, required=True, help="Raw data path for TrafficDataPipelineOrchestrator")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--artifacts-dir", type=str, default="./results_tabular")
    ap.add_argument("--results-csv", type=str, default="tabular_xgb_results.csv")
    ap.add_argument("--test-size", type=float, default=1.0/3.0)
    ap.add_argument("--use-gpu", action="store_true", help="If set, use GPU for XGB (gpu_hist)")
    return ap.parse_args()


def main():
    args = parse_args()

    orchestrator_kwargs = {
        "file_path": args.file_path,
    }

    prepare_kwargs = {"test_size": args.test_size}

    base_cfg = dict(
        artifacts_dir=args.artifacts_dir,
        results_csv_name=args.results_csv,
        horizon=args.horizon,
        orchestrator_kwargs=orchestrator_kwargs,
        prepare_kwargs=prepare_kwargs,
        xgb_model_name="xgb_timecyc",
        xgb_use_gpu=bool(args.use_gpu),
    )

    for add_cyc, include_curr in product([True, False], [True, False]):
        cfg = DataCfg(
            **base_cfg,
            add_prediction_time_cyclical_features=add_cyc,
            include_current_time_cyclical=include_curr,
        )
        exp = TabularExperiment(cfg)
        row = exp.run()
        print(
            f"[done] {row['run_id']} | "
            f"MAE={row.get('model_MAE', float('nan'))} "
            f"RMSE={row.get('model_RMSE', float('nan'))} "
            f"MAPE%={row.get('model_MAPE', float('nan'))} "
            f"SMAPE%={row.get('model_SMAPE', float('nan'))}"
        )


if __name__ == "__main__":
    main()