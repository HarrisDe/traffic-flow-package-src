# -*- coding: utf-8 -*-
"""
Tabular experiment runner that respects the pipeline's two-phase API:
1) prepare_base_features(...)
2) finalise_for_horizon(...)

- DataCfg: dataclass for all inputs to TrafficDataPipelineOrchestrator and this experiment.
- TabularExperiment: builds pipeline, prepares features, trains XGB via ModelTunerXGB,
  evaluates with ModelEvaluator, and appends a CSV row per run.

CSV columns include:
- run_id (based on the two time-feature toggles + horizon)
- all DataCfg settings
- model metrics + stds
- naive metrics + stds
- best_model_path, best_params (JSON), training_time, total_time
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import pandas as pd
from filelock import FileLock

# ---- Project imports (adjust package root if needed) ----
from .pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
from .modeling.model_tuning import ModelTunerXGB
from .evaluation.model_comparison import ModelEvaluator


# ============================== DataCfg ============================== #
@dataclass
class DataCfg:
    """
    All inputs for TrafficDataPipelineOrchestrator + experiment toggles.
    Put any constructor args for the orchestrator in `orchestrator_kwargs`.

    Typical keys for `orchestrator_kwargs`:
        - file_path (or data_path)
        - datetime_col, target_col, sensor_id_col
        - any other project-specific knobs

    `prepare_kwargs` go to prepare_base_features(...)
        e.g. {"test_size": 1/3, "shuffle": False, ...}

    Time-feature toggles are forwarded to finalise_for_horizon(...).
    """
    # Where to store outputs
    artifacts_dir: str = "./results_tabular"
    results_csv_name: str = "tabular_xgb_results.csv"

    # Horizon and time-feature toggles we sweep
    horizon: int = 15
    add_prediction_time_cyclical_features: bool = True
    include_current_time_cyclical: bool = True

    # Orchestrator configuration
    orchestrator_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Arguments for prepare_base_features(...)
    prepare_kwargs: Dict[str, Any] = field(default_factory=lambda: {"test_size": 1/3})

    # Optional: name the XGB model in artifacts
    xgb_model_name: str = "xgb_tabular"

    # Optional: choose CPU by default (works everywhere)
    xgb_use_gpu: bool = False



# ============================ Experiment ============================= #
class TabularExperiment:
    def __init__(self, cfg: DataCfg) -> None:
        self.cfg = cfg
        self.artifacts_dir = Path(cfg.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _make_run_id(self) -> str:
        # Run id based on the features we sweep (+ horizon for clarity)
        return "h{h}_cyc{cyc}_curr{curr}".format(
            h=self.cfg.horizon,
            cyc=int(self.cfg.add_prediction_time_cyclical_features),
            curr=int(self.cfg.include_current_time_cyclical),
        )

    def _build_pipeline(self) -> TrafficDataPipelineOrchestrator:
        """Create orchestrator from cfg.orchestrator_kwargs (no assumptions here)."""
        ork = TrafficDataPipelineOrchestrator(**(self.cfg.orchestrator_kwargs or {}))
        return ork

    def _prepare_features(self, ork: TrafficDataPipelineOrchestrator) -> None:
        """Phase 1: base features & split flags."""
        ork.prepare_base_features(**(self.cfg.prepare_kwargs or {}))

    def _finalise_for_horizon(self, ork: TrafficDataPipelineOrchestrator):
        """Phase 2: horizon-specific features and return train/test splits."""
        X_train, X_test, y_train, y_test = ork.finalise_for_horizon(
            horizon=self.cfg.horizon,
            add_prediction_time_cyclical_features=self.cfg.add_prediction_time_cyclical_features,
            include_current_time_cyclical=self.cfg.include_current_time_cyclical,
        )

        return X_train, X_test, y_train, y_test

    def _flatten_metrics(self, eval_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Flatten ModelEvaluator output into a single-level dict with prefixes:
          model_* , model_*_std , naive_* , naive_*_std
        """
        out: Dict[str, float] = {}
        for prefix, d in [
            ("model", eval_dict.get("metrics", {})),
            ("model", eval_dict.get("metrics_std", {})),      # already has *_std keys
            ("naive", eval_dict.get("naive_metrics", {})),
            ("naive", eval_dict.get("naive_metrics_std", {})),# already has *_std keys
        ]:
            for k, v in d.items():
                key = f"{prefix}_{k}"
                out[key] = v
        return out

    def run(self) -> Dict[str, Any]:
        # 1) Build pipeline
        ork = self._build_pipeline()

        # 2) Base features
        self._prepare_features(ork)

        # 3) Horizon-specific features & splits
        X_train, X_test, y_train, y_test = self._finalise_for_horizon(ork)

        # 4) Train/tune XGB with your ModelTunerXGB defaults
        mt = ModelTunerXGB(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            XGBoost_model_name=self.cfg.xgb_model_name,
        )
        best_model_path, best_params, training_time, total_time = mt.tune_xgboost(
            use_gpu=self.cfg.xgb_use_gpu,
            suppress_output=False,  # keep logs tidy; set False to see GridSearchCV progress
        )

        # 5) Evaluate with ModelEvaluator (from_path, using your ML frame)
        me = ModelEvaluator(
            X_test=X_test,
            df_for_ML=ork.df,   # expects df with 'test_set' flag and columns used internally
            y_train=y_train,
            y_test=y_test,
        )
        eval_out = me.evaluate_model_from_path(best_model_path, print_results=False)

        # Flatten metrics
        flat_metrics = self._flatten_metrics(eval_out)

        # 6) Assemble row
        row: Dict[str, Any] = {
            "run_id": self._make_run_id(),
            **asdict(self.cfg),
            "best_model_path": str(best_model_path),
            "best_params": json.dumps(best_params),
            "training_time_s": float(training_time),
            "tuning_time_s": float(total_time),
            **flat_metrics,
        }

        # 7) Append to CSV (lock-safe)
        out_csv = self.artifacts_dir / self.cfg.results_csv_name
        lock = FileLock(str(out_csv) + ".lock")
        with lock:
            if out_csv.exists():
                df_out = pd.read_csv(out_csv)
            else:
                df_out = pd.DataFrame()
            df_out = pd.concat([df_out, pd.DataFrame([row])], ignore_index=True)
            df_out.to_csv(out_csv, index=False)

        return row