#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run several experiments varying MomentumFeatureEngineer settings.

Defaults in this sweep:
- include_current_time_cyclical = False
- add_prediction_time_cyclical_features = True

Usage:
  python scripts/run_tabular_momentum_sweep.py --file-path /path/to/raw.parquet --horizon 15

Notes:
- We forward file_path (and any other pipeline args) via orchestrator_kwargs.
- Momentum features are injected via prepare_base_features(add_momentum_features=True, momentum_params=...).
"""

import argparse
from traffic_flow.tabular.experiment import DataCfg, TabularExperiment


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, required=True, help="Raw data path for TrafficDataPipelineOrchestrator")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--artifacts-dir", type=str, default="./results_tabular")
    ap.add_argument("--results-csv", type=str, default="tabular_xgb_results_momentum.csv")
    ap.add_argument("--test-size", type=float, default=1.0/3.0)
    ap.add_argument("--use-gpu", action="store_true", help="If set, use GPU for XGB (gpu_hist)")
    return ap.parse_args()


def main():
    args = parse_args()

    orchestrator_kwargs = {
        "file_path": args.file_path,
        # Add explicit column names here if your orchestrator requires them
        # "datetime_col": "date",
        # "target_col": "value",
        # "sensor_id_col": "sensor_id",
    }

    # Fixed time-encoding defaults for this sweep
    add_prediction_time_cyclical_features = True
    include_current_time_cyclical = False

    # Base prepare kwargs (we'll inject momentum toggles/config per run)
    base_prepare_kwargs = {
        "test_size": args.test_size,
        # We'll add: "add_momentum_features": True, "momentum_params": {...} per run
    }

    # A small set of momentum configurations to probe
    momentum_grid = [
        # ID,   params
        (
            "m1_fast",
            {
                "slope_windows": (5, 10),
                "ewm_halflives": (5.0,),
                "vol_windows": (10,),
                "minmax_windows": (15,),
                "thresholds_kph": (80.0,),
                "drop_fast_flag": True,
                "fast_flag_window": 5,
                "fast_flag_thresh": -0.9,
            },
        ),
        (
            "m2_balanced",
            {
                "slope_windows": (5, 10, 15),
                "ewm_halflives": (5.0, 10.0),
                "vol_windows": (10, 30),
                "minmax_windows": (15, 30),
                "thresholds_kph": (70.0, 80.0, 90.0),
                "drop_fast_flag": True,
                "fast_flag_window": 5,
                "fast_flag_thresh": -1.0,
            },
        ),
        (
            "m3_steady",
            {
                "slope_windows": (10, 15, 30),
                "ewm_halflives": (10.0,),
                "vol_windows": (30,),
                "minmax_windows": (30,),
                "thresholds_kph": (80.0, 90.0),
                "drop_fast_flag": True,
                "fast_flag_window": 10,
                "fast_flag_thresh": -0.6,  # fires less often due to longer window
            },
        ),
        (
            "m4_very_fast",
            {
                "slope_windows": (3, 5),
                "ewm_halflives": (3.0, 5.0),
                "vol_windows": (10,),
                "minmax_windows": (10, 15),
                "thresholds_kph": (80.0,),
                "drop_fast_flag": True,
                "fast_flag_window": 3,
                "fast_flag_thresh": -0.7,
            },
        ),
    ]

    for mom_id, mom_params in momentum_grid:
        prepare_kwargs = dict(base_prepare_kwargs)
        prepare_kwargs.update({
            "add_momentum_features": True,
            "momentum_params": mom_params,
        })

        cfg = DataCfg(
            artifacts_dir=args.artifacts_dir,
            results_csv_name=args.results_csv,
            horizon=args.horizon,
            orchestrator_kwargs=orchestrator_kwargs,
            prepare_kwargs=prepare_kwargs,
            xgb_model_name=f"xgb_momentum_{mom_id}",
            xgb_use_gpu=bool(args.use_gpu),

            # Time-encoding defaults as requested:
            add_prediction_time_cyclical_features=add_prediction_time_cyclical_features,
            include_current_time_cyclical=include_current_time_cyclical,
        )

        exp = TabularExperiment(cfg)
        row = exp.run()

        # quick console feedback
        print(
            f"[done] {row.get('run_id','<no_id>')} | tag={mom_id} | "
            f"MAE={row.get('model_MAE', float('nan'))} "
            f"RMSE={row.get('model_RMSE', float('nan'))} "
            f"MAPE%={row.get('model_MAPE', float('nan'))} "
            f"SMAPE%={row.get('model_SMAPE', float('nan'))}"
        )

if __name__ == "__main__":
    main()