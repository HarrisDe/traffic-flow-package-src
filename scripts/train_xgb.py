# scripts/train_xgb.py
import argparse, pathlib, time, joblib, json
from traffic_flow import (
    TrafficDataPipelineOrchestrator,
    ModelTunerXGB,
)

def main(args):
    t0 = time.perf_counter()

    # 1. run the feature-engineering pipeline
    tdp = TrafficDataPipelineOrchestrator(
        file_path=str(args.file),
        sensor_col=args.sensor_col,
        datetime_col=args.datetime_col,
        value_col=args.value_col,
        new_sensor_id_col=args.new_sensor_id_col,
        weather_cols=args.weather_cols.split(',') if args.weather_cols else None,
        disable_logs=args.disable_logs
    )

    X_train, X_test, y_train, y_test = tdp.run_pipeline(
        test_size=args.test_size,
        filter_extreme_changes=args.filter_extreme_changes,
        smooth_speeds=args.smooth_speeds,
        relative_threshold=args.relative_threshold,
        diagnose_extreme_changes=args.diagnose_extreme_changes,
        add_gman_predictions=args.add_gman_predictions,
        window_size=args.window_size,
        spatial_adj=args.spatial_adj,
        normalize_by_distance=args.normalize_by_distance,
        lag_steps=args.lag_steps,
        relative_lags=args.relative_lags,
        horizon=args.horizon,
        filter_on_train_only=args.filter_on_train_only,
        use_gman_target=args.use_gman_target,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        quantile_threshold=args.quantile_threshold,
        quantile_percentage=args.quantile_percentage,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
    )

    # 2. hyper-parameter tuning
    tuner = ModelTunerXGB(
        X_train, X_test, y_train, y_test,
        random_state=args.random_state,
        use_ts_split=args.use_ts_split,
        n_splits=args.n_splits,
        best_model_name_string_start=args.best_model_name_string_start,
        model_path=args.model_path,
        XGBoost_model_name=args.xgboost_model_name,
        predict_in_batches=args.predict_in_batches,
        gpu_memory_gb=args.gpu_memory_gb
    )

    model_path, best_params, train_time, total_time = tuner.tune_xgboost(
        model_name=args.xgboost_model_name,
        params=None,  # extend if you want param grid from CLI
        use_gpu=args.use_gpu,
        objective=args.objective,
        suppress_output=args.suppress_output,
        n_jobs=args.n_jobs
    )
    best_model = tuner.best_model

    # 3. bundle preprocessor + model into ONE artifact
    bundle = {
        "preprocessor": tdp,
        "model": best_model,
        "feature_cols": list(X_train.columns),
        "best_params": best_params,
        "training_time": train_time,
        "total_time": total_time,
    }

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "traffic_pipeline.joblib"
    joblib.dump(bundle, out_file)

    # 4. save a lightweight JSON with just metrics / params (optional)
    meta_file = out_dir / "training_metadata.json"
    meta_file.write_text(json.dumps(
        {k: v for k, v in bundle.items() if k not in ("preprocessor", "model")},
        indent=2,
        default=str,
    ))

    elapsed = time.perf_counter() - t0
    print(f"Saved {out_file.relative_to(pathlib.Path.cwd())}  (total {elapsed:,.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train & bundle the XGB traffic model")

    # === Required ===
    p.add_argument("--file", required=True, help="Parquet with raw sensor data")

    # === General outputs ===
    p.add_argument("--out-dir", default="artifacts", help="Output folder for artefact")

    # === TrafficDataPipelineOrchestrator __init__ ===
    p.add_argument("--sensor-col", default="sensor_id")
    p.add_argument("--datetime-col", default="date")
    p.add_argument("--value-col", default="value")
    p.add_argument("--new-sensor-id-col", default="sensor_uid")
    p.add_argument("--weather-cols", default=None, help="Comma-separated list or leave None")
    p.add_argument("--disable-logs", action="store_true")

    # === run_pipeline ===
    p.add_argument("--test-size", type=float, default=1/3)
    p.add_argument("--filter-extreme-changes", action="store_true")
    p.add_argument("--no-filter-extreme-changes", dest="filter_extreme_changes", action="store_false")
    p.set_defaults(filter_extreme_changes=True)

    p.add_argument("--smooth-speeds", action="store_true")
    p.add_argument("--no-smooth-speeds", dest="smooth_speeds", action="store_false")
    p.set_defaults(smooth_speeds=True)

    p.add_argument("--relative-threshold", type=float, default=0.7)
    p.add_argument("--diagnose-extreme-changes", action="store_true")
    p.add_argument("--add-gman-predictions", action="store_true")
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--spatial-adj", type=int, default=1)
    p.add_argument("--normalize-by-distance", action="store_true")
    p.add_argument("--no-normalize-by-distance", dest="normalize_by_distance", action="store_false")
    p.set_defaults(normalize_by_distance=True)

    p.add_argument("--lag-steps", type=int, default=25)
    p.add_argument("--relative-lags", action="store_true")
    p.add_argument("--no-relative-lags", dest="relative_lags", action="store_false")
    p.set_defaults(relative_lags=True)

    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--filter-on-train-only", action="store_true")
    p.add_argument("--use-gman-target", action="store_true")
    p.add_argument("--hour-start", type=int, default=6)
    p.add_argument("--hour-end", type=int, default=19)
    p.add_argument("--quantile-threshold", type=float, default=0.9)
    p.add_argument("--quantile-percentage", type=float, default=0.65)
    p.add_argument("--lower-bound", type=float, default=0.01)
    p.add_argument("--upper-bound", type=float, default=0.99)

    # === ModelTunerXGB ===
    p.add_argument("--random-state", type=int, default=69)
    p.add_argument("--use-ts-split", action="store_true")
    p.add_argument("--no-use-ts-split", dest="use_ts_split", action="store_false")
    p.set_defaults(use_ts_split=True)

    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--best-model-name-string-start", default="best_model_")
    p.add_argument("--model-path", default="./models")
    p.add_argument("--xgboost-model-name", default="XGBoost")
    p.add_argument("--predict-in-batches", action="store_true")
    p.add_argument("--gpu-memory-gb", type=float, default=40.0)

    # === tune_xgboost ===
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--no-use-gpu", dest="use_gpu", action="store_false")
    p.set_defaults(use_gpu=True)

    p.add_argument("--objective", default="reg:pseudohubererror")
    p.add_argument("--suppress-output", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)

    args = p.parse_args()
    main(args)