#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sweep the UpstreamTravelTimeShiftedFeatures toggle for the tabular XGB pipeline.

Only swept knob (in prepare_base_features):
- add_upstream_shifted_features ∈ {False, True}

Everything else in `prepare_base_features` and `finalise_for_horizon` stays at defaults.

Skips runs already present in the results CSV (completed = both model_MAE and model_RMSE present).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from traffic_flow.tabular.experiment import DataCfg, TabularExperiment


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    ok = (~df.get("model_MAE", pd.Series([np.nan]*len(df))).isna()) & \
         (~df.get("model_RMSE", pd.Series([np.nan]*len(df))).isna())
    return set(df.loc[ok, "run_id"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, default="../../data/NDW/ndw_three_weeks.parquet")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--artifacts-dir", type=str, default="./results_tabular")
    ap.add_argument("--results-csv", type=str, default="tabular_xgb_results_upstream_shift.csv")
    ap.add_argument("--use-gpu", action="store_true", help="If set, use GPU for XGB (gpu_hist)")
    args = ap.parse_args()

    results_csv = Path(args.artifacts_dir) / args.results_csv
    completed = load_completed(results_csv)

    # Only toggle this one boolean; everything else remains default
    toggles = [False, True]

    for use_upshift in toggles:
        cfg = DataCfg(
            artifacts_dir=args.artifacts_dir,
            results_csv_name=args.results_csv,
            horizon=args.horizon,
            orchestrator_kwargs={
                "file_path": args.file_path,
            },
            prepare_kwargs={
                # REPLACE semantics in your orchestrator: we only set the toggle,
                # leaving all other params at their defaults.
                "add_upstream_shifted_features": use_upshift,
            },
            xgb_use_gpu=bool(args.use_gpu),
        )

        exp = TabularExperiment(cfg)
        run_id = exp._make_run_id()  # includes horizon & prepare kwargs in your implementation

        if run_id in completed:
            print(f"[skip] {run_id} (already completed)")
            continue

        print(f"[run ] {run_id} (add_upstream_shifted_features={use_upshift})")
        try:
            row = exp.run()
            print(
                f"[done] {row.get('run_id','<no_id>')} | "
                f"MAE={row.get('model_MAE', float('nan'))} "
                f"RMSE={row.get('model_RMSE', float('nan'))} "
                f"MAPE%={row.get('model_MAPE', float('nan'))} "
                f"SMAPE%={row.get('model_SMAPE', float('nan'))}"
            )
        except Exception as e:
            print(f"[fail] {run_id}: {e!r}")


if __name__ == "__main__":
    main()