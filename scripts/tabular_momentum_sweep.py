#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sweep MomentumFeatureEngineer configs targeted at bottlenecks (30/50/70 kph)
across multiple horizons, with skip-if-completed logic.

Fixed in this sweep:
- add_prediction_time_cyclical_features = True
- include_current_time_cyclical        = False

Usage:
  python scripts/run_tabular_momentum_bottleneck_sweep.py \
      --file-path /path/to/raw.parquet --horizon 15   # <-- --horizon is ignored here; we sweep fixed list
"""

import argparse
from pathlib import Path
from typing import Set, Tuple

import pandas as pd

from traffic_flow.tabular.experiment import DataCfg, TabularExperiment


# ---------------------------- config ---------------------------- #

HORIZONS = [5, 10, 15, 30, 45, 60]

MOMENTUM_GRID = [
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


# ----------------------------- CLI ------------------------------ #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, required=True,
                    help="Raw data path for TrafficDataPipelineOrchestrator")
    ap.add_argument("--artifacts-dir", type=str, default="./results_tabular")
    ap.add_argument("--results-csv", type=str,
                    default="tabular_xgb_results_momentum_bottlenecks.csv")
    ap.add_argument("--test-size", type=float, default=1.0/3.0)
    ap.add_argument("--use-gpu", action="store_true",
                    help="If set, use GPU for XGB (gpu_hist)")
    return ap.parse_args()


# ------------------------- helpers ------------------------------ #

def load_done_pairs(csv_path: Path) -> Set[Tuple[str, str]]:
    """
    Return set of (run_id, xgb_model_name) pairs already present in the CSV.
    Safeguards against duplicates across different momentum configs.
    """
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()

    need_cols = {"run_id", "xgb_model_name"}
    if not need_cols.issubset(df.columns):
        return set()

    return set((str(rid), str(name)) for rid, name in zip(df["run_id"], df["xgb_model_name"]))


def make_run_id(h: int, cyc_pred: bool, cyc_curr: bool) -> str:
    """Mirror TabularExperiment._make_run_id."""
    return f"h{h}_cyc{int(cyc_pred)}_curr{int(cyc_curr)}"


# ----------------------------- main ----------------------------- #

def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_csv = artifacts_dir / args.results_csv

    # Fixed time encodings for this sweep
    add_prediction_time_cyclical_features = True
    include_current_time_cyclical = False

    # Prepare base orchestrator kwargs and base prepare kwargs
    orchestrator_kwargs = {"file_path": args.file_path}
    base_prepare_kwargs = {"test_size": args.test_size}

    # Load (run_id, model_name) pairs already done
    done = load_done_pairs(results_csv)

    for h in HORIZONS:
        # ---------- Baseline: momentum OFF ----------
        run_id = make_run_id(h, add_prediction_time_cyclical_features, include_current_time_cyclical)
        model_name = f"xgb_nomomentum_h{h}"

        if (run_id, model_name) in done:
            print(f"[skip] {run_id} | {model_name} already in {results_csv.name}")
        else:
            cfg = DataCfg(
                artifacts_dir=str(artifacts_dir),
                results_csv_name=results_csv.name,
                horizon=h,
                orchestrator_kwargs=orchestrator_kwargs,
                prepare_kwargs={**base_prepare_kwargs,
                                "add_momentum_features": False},  # disable
                xgb_model_name=model_name,
                xgb_use_gpu=bool(args.use_gpu),
                add_prediction_time_cyclical_features=add_prediction_time_cyclical_features,
                include_current_time_cyclical=include_current_time_cyclical,
            )
            row = TabularExperiment(cfg).run()
            print(f"[done] {row['run_id']} | {model_name} | "
                  f"MAE={row.get('model_MAE', float('nan'))}  RMSE={row.get('model_RMSE', float('nan'))}")

        # ---------- Momentum configs ----------
        for tag, mom_params in MOMENTUM_GRID:
            model_name = f"xgb_momentum_{tag}_h{h}"

            if (run_id, model_name) in done:
                print(f"[skip] {run_id} | {model_name} already in {results_csv.name}")
                continue

            cfg = DataCfg(
                artifacts_dir=str(artifacts_dir),
                results_csv_name=results_csv.name,
                horizon=h,
                orchestrator_kwargs=orchestrator_kwargs,
                prepare_kwargs={**base_prepare_kwargs,
                                "add_momentum_features": True,
                                "momentum_params": mom_params},
                xgb_model_name=model_name,
                xgb_use_gpu=bool(args.use_gpu),
                add_prediction_time_cyclical_features=add_prediction_time_cyclical_features,
                include_current_time_cyclical=include_current_time_cyclical,
            )
            row = TabularExperiment(cfg).run()
            print(f"[done] {row['run_id']} | {model_name} | "
                  f"MAE={row.get('model_MAE', float('nan'))}  RMSE={row.get('model_RMSE', float('nan'))}")


if __name__ == "__main__":
    main()

