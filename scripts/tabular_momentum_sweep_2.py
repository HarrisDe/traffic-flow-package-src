#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sweep MomentumFeatureEngineer configs targeted at bottlenecks (30/50/70 kph).

Fixed in this sweep:
- add_prediction_time_cyclical_features = True
- include_current_time_cyclical = False

Usage:
  python scripts/run_tabular_momentum_bottleneck_sweep.py --file-path /path/to/raw.parquet --horizon 15
"""

import argparse
from traffic_flow.tabular.experiment import DataCfg, TabularExperiment


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, required=True, help="Raw data path for TrafficDataPipelineOrchestrator")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--artifacts-dir", type=str, default="./results_tabular")
    ap.add_argument("--results-csv", type=str, default="tabular_xgb_results_momentum_bottlenecks.csv")
    ap.add_argument("--test-size", type=float, default=1.0/3.0)
    ap.add_argument("--use-gpu", action="store_true", help="If set, use GPU for XGB (gpu_hist)")
    return ap.parse_args()


def main():
    args = parse_args()

    orchestrator_kwargs = {
        "file_path": args.file_path,
    }

    # Fixed time-encoding defaults for this sweep
    add_prediction_time_cyclical_features = True
    include_current_time_cyclical = False

    # Base prepare kwargs (we inject momentum params per run)
    base_prepare_kwargs = {"test_size": args.test_size}

    # --- Bottleneck-focused momentum configs ---
    momentum_grid = [
        # Very reactive; short windows; low thresholds; sensitive flag
        ("bn_veryfast_305070", {
            "slope_windows": (3, 5, 7),
            "ewm_halflives": (3.0, 5.0),
            "vol_windows": (5, 10),
            "minmax_windows": (10, 15),
            "thresholds_kph": (30.0, 50.0, 70.0),
            "drop_fast_flag": True,
            "fast_flag_window": 3,
            "fast_flag_thresh": -0.8,
        }),
        # Fast but stricter flag (fires only on strong drops)
        ("bn_fast_strictflag", {
            "slope_windows": (3, 5),
            "ewm_halflives": (3.0, 5.0),
            "vol_windows": (10,),
            "minmax_windows": (10, 15),
            "thresholds_kph": (30.0, 50.0, 70.0),
            "drop_fast_flag": True,
            "fast_flag_window": 5,
            "fast_flag_thresh": -1.3,
        }),
        # Balanced set (good starting point)
        ("bn_balanced_305070", {
            "slope_windows": (5, 10, 15),
            "ewm_halflives": (5.0, 10.0),
            "vol_windows": (10, 30),
            "minmax_windows": (15, 30),
            "thresholds_kph": (30.0, 50.0, 70.0),
            "drop_fast_flag": True,
            "fast_flag_window": 5,
            "fast_flag_thresh": -1.0,
        }),
        # Steadier (less jumpy), still watches low thresholds
        ("bn_steady_305070", {
            "slope_windows": (10, 15, 30),
            "ewm_halflives": (10.0,),
            "vol_windows": (30,),
            "minmax_windows": (30,),
            "thresholds_kph": (30.0, 50.0, 70.0),
            "drop_fast_flag": True,
            "fast_flag_window": 10,
            "fast_flag_thresh": -0.6,
        }),
        # Threshold signals only (ablate slopes/volatility to see their impact)
        ("bn_threshold_only", {
            "slope_windows": (),
            "ewm_halflives": (),
            "vol_windows": (),
            "minmax_windows": (15,),        # keep distances to help recovery timing
            "thresholds_kph": (30.0, 50.0, 70.0),
            "drop_fast_flag": False,
        }),
        # Light footprint (for large datasets)
        ("bn_light", {
            "slope_windows": (5,),
            "ewm_halflives": (5.0,),
            "vol_windows": (10,),
            "minmax_windows": (15,),
            "thresholds_kph": (50.0, 70.0),
            "drop_fast_flag": True,
            "fast_flag_window": 5,
            "fast_flag_thresh": -0.9,
        }),
    ]

    for tag, mom_params in momentum_grid:
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
            xgb_model_name=f"xgb_momentum_{tag}",
            xgb_use_gpu=bool(args.use_gpu),

            # time-encoding defaults per your request
            add_prediction_time_cyclical_features=add_prediction_time_cyclical_features,
            include_current_time_cyclical=include_current_time_cyclical,
        )

        exp = TabularExperiment(cfg)
        row = exp.run()
        print(
            f"[done] {row.get('run_id','<no_id>')} | tag={tag} | "
            f"MAE={row.get('model_MAE', float('nan'))} "
            f"RMSE={row.get('model_RMSE', float('nan'))} "
            f"MAPE%={row.get('model_MAPE', float('nan'))} "
            f"SMAPE%={row.get('model_SMAPE', float('nan'))}"
        )


if __name__ == "__main__":
    main()