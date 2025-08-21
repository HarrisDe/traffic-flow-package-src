from __future__ import annotations

"""traffic_flow_package_src/experiments/xgb_horizon_experiment.py

End‑to‑end wrapper for running an XGBoost study across *multiple* forecasting
horizons.  Compared with the notebook‑style prototype, the module:

* Follows **SOLID** design – configuration, orchestration and collaborators are
  cleanly separated.
* Can **resume** after an interrupted run (per‑horizon checkpoints).
* Produces a *single* ground‑truth column **``y_test``** taken from the
  *smallest* horizon – no more duplicates like ``y_test_h_15``.
* Accepts horizons in **any order**; they are processed in ascending order so
  that the smallest horizon (and thus the longest y‑test vector) is handled
  first.

Example
-------
>>> from traffic_flow_package_src.experiments.xgb_horizon_experiment import XGBHorizonExperiment, ExperimentConfig
>>> cfg = ExperimentConfig(orig_file_path="../data/NDW/ndw_one_year_orig.parquet", horizons=[20, 14, 12])
>>> XGBHorizonExperiment(config=cfg).run()
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Type

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 3rd‑party dependencies from your package
# ---------------------------------------------------------------------------
from traffic_flow_package_src import (
    TrafficDataPipelineOrchestrator,
    ModelTunerXGB,
    ModelEvaluator,
)

__all__ = [
    "ExperimentConfig",
    "HorizonResult",
    "XGBHorizonExperiment",
]

# =============================================================================
#  Immutable configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Experiment‑wide knobs (filesystem + training meta‑params)."""

    orig_file_path: Path | str
    horizons: Sequence[int] = field(default_factory=lambda: list(range(12, 21, 2)))
    date_str: str = field(default_factory=lambda: datetime.now().strftime("%y%m%d"))
    results_dir: Path | str = "results"
    test_size: float = 0.5

    # reproducibility / training
    random_state: int = 69
    n_splits: int = 3
    objective: str = "reg:pseudohubererror"
    use_gpu: bool = True

    # resolved paths (filled post‑init)
    csv_path: Path = field(init=False)
    pred_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    def __post_init__(self):  # noqa: D401
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.results_dir / f"{self.date_str}_xgb_results.csv"
        pred_dir = self.results_dir / f"{self.date_str}_xgb_predictions"
        model_dir = Path("models") / f"{self.date_str}_xgb"
        pred_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "csv_path", csv_path)
        object.__setattr__(self, "pred_dir", pred_dir)
        object.__setattr__(self, "model_dir", model_dir)

# =============================================================================
#  Typed results row
# =============================================================================

@dataclass(slots=True)
class HorizonResult:
    """Flattened metrics ready to write into the CSV."""

    model_name: str
    horizon: int
    training_time_without_validation: float
    hp_tuning_time_with_validation: float
    metrics: Dict[str, float]
    metrics_std: Dict[str, float]
    naive_metrics: Dict[str, float]
    naive_metrics_std: Dict[str, float]

    def to_series(self) -> pd.Series:
        flat: Dict[str, float | int | str] = {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "training_time_without_validation": self.training_time_without_validation,
            "hp_tuning_time_with_validation": self.hp_tuning_time_with_validation,
        }
        flat.update(self.metrics)
        flat.update({f"{k}_std": v for k, v in self.metrics_std.items()})
        flat.update({f"Naive_{k}": v for k, v in self.naive_metrics.items()})
        flat.update({f"Naive_{k}_std": v for k, v in self.naive_metrics_std.items()})
        return pd.Series(flat)

# =============================================================================
#  Main orchestration class
# =============================================================================

class XGBHorizonExperiment:  # pylint: disable=too-many-instance-attributes
    """Runs the complete multi‑horizon experiment.

    The class is *Open for extension* – pass alternative orchestrator / tuner /
    evaluator classes – while keeping the default behaviour untouched.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        orchestrator_cls: Type[TrafficDataPipelineOrchestrator] = TrafficDataPipelineOrchestrator,
        tuner_cls: Type[ModelTunerXGB] = ModelTunerXGB,
        evaluator_cls: Type[ModelEvaluator] = ModelEvaluator,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = config
        self._orchestrator_cls = orchestrator_cls
        self._tuner_cls = tuner_cls
        self._evaluator_cls = evaluator_cls

        self.log = logger or logging.getLogger(self.__class__.__name__)
        if not self.log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)

        # Sort & deduplicate horizons so that the *smallest* is processed first.
        self._horizons_sorted: List[int] = sorted(set(self.cfg.horizons))
        self._min_horizon: int = self._horizons_sorted[0]

        # runtime state
        self._results_frame: pd.DataFrame | None = None
        self._merged_predictions: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.log.info("Starting experiment – horizons: %s", self._horizons_sorted)

        completed: List[int] = []
        if self.cfg.csv_path.exists():
            completed = pd.read_csv(self.cfg.csv_path)["horizon"].unique().tolist()
            self.log.info("Resuming – completed horizons: %s", completed)

        for horizon in self._horizons_sorted:
            if horizon in completed:
                self.log.info("Skipping horizon %s (already processed)", horizon)
                continue
            try:
                result = self._run_single_horizon(horizon)
                self._append_result_to_csv(result)
            except Exception as exc:  # pylint: disable=broad-except
                # Keep going even if one horizon blows up.
                self.log.error("Horizon %s failed: %s", horizon, exc)
                self.log.debug("%s", traceback.format_exc())

        # Return the consolidated DataFrame for immediate analysis.
        if self._results_frame is None:
            self._results_frame = pd.read_csv(self.cfg.csv_path)
        self.log.info("Finished – metrics at %s", self.cfg.csv_path)
        return self._results_frame

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_single_horizon(self, horizon: int) -> HorizonResult:
        self.log.info("▶ Horizon %s min", horizon)

        # 1. Data preparation ------------------------------------------------
        tdp = self._orchestrator_cls(self.cfg.orig_file_path, sensor_encoding_type="mean")
        tdp.prepare_base_features(
            test_size=self.cfg.test_size,
            filter_extreme_changes=True,
            smooth_speeds=True,
            relative_threshold=0.7,
            diagnose_extreme_changes=False,
            window_size=5,
            spatial_adj=1,
            normalize_by_distance=True,
            lag_steps=25,
            relative_lags=True,
            filter_on_train_only=False,
            hour_start=6,
            hour_end=19,
            quantile_threshold=0.9,
            quantile_percentage=0.65,
            lower_bound=0.01,
            upper_bound=0.99,
            use_median_instead_of_mean_smoothing=False,
        )
        X_train, X_test, y_train, y_test = tdp.finalise_for_horizon(
            horizon=horizon,
            df_gman=None,
            convert_gman_prediction_to_delta_speed=None,
            add_previous_weekday_feature=True,
            previous_weekday_window_min=0,
            drop_weather=True,
            use_gman_target=False,
            drop_missing_gman_rows=False,
        )

        # Persist raw frames for debuggability
        X_train.to_parquet(self.cfg.pred_dir / f"X_train_h{horizon}.parquet")
        X_test.to_parquet(self.cfg.pred_dir / f"X_test_h{horizon}.parquet")
        tdp.df.to_parquet(self.cfg.pred_dir / f"tdp_full_df_h{horizon}.parquet")

        # 2. Training / hyper‑parameter tuning ------------------------------
        model_name = f"{self.cfg.date_str}_XGB_h{horizon}_ts_split"
        tuner = self._tuner_cls(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            random_state=self.cfg.random_state,
            use_ts_split=True,
            n_splits=self.cfg.n_splits,
            best_model_name_string_start="best_model_",
            XGBoost_model_name=model_name,
            predict_in_batches=False,
            model_path=str(self.cfg.model_dir),
        )
        model_path, best_params, tr_time, total_time = tuner.tune_xgboost(
            objective=self.cfg.objective,
            use_gpu=self.cfg.use_gpu,
        )
        self.log.debug("Best params for h=%s: %s", horizon, best_params)

        # 3. Evaluation ------------------------------------------------------
        evaluator = self._evaluator_cls(
            X_test=X_test,
            y_test=y_test,
            y_train=y_train,
            target_is_gman_error_prediction=False,
            df_for_ML=tdp.df,
            y_is_normalized=False,
            rounding=2,
        )
        res_dict = evaluator.evaluate_model_from_path(model_path)

        # 4. Predictions parquet -------------------------------------------
        time_axis = tdp.df.loc[tdp.df["test_set"], ["date_of_prediction", "sensor_id"]]

        if horizon == self._min_horizon:
            # Keep ground truth once, name it simply "y_test".
            df_pred = pd.concat([time_axis, evaluator.y_test.rename("y_test"), evaluator.y_pred], axis=1)
            df_pred.columns = ["date", "sensor_id", "y_test", f"y_pred_h_{horizon}"]
        else:
            # Later horizons: *only* predictions.
            df_pred = pd.concat([time_axis, evaluator.y_pred], axis=1)
            df_pred.columns = ["date", "sensor_id", f"y_pred_h_{horizon}"]

        df_pred.to_parquet(self.cfg.pred_dir / f"{self.cfg.date_str}_predictions_h{horizon}.parquet")

        # Merge across horizons -------------------------------------------
        if self._merged_predictions is None:
            self._merged_predictions = df_pred
        else:
            self._merged_predictions = self._merged_predictions.merge(
                df_pred, on=["date", "sensor_id"], how="outer"
            )
        self._merged_predictions.to_parquet(
            self.cfg.pred_dir / f"{self.cfg.date_str}_all_predictions_all_horizons.parquet"
        )

        # 5. Package metrics into typed dataclass ---------------------------
        result = HorizonResult(
            model_name=model_name,
            horizon=horizon,
            training_time_without_validation=tr_time,
            hp_tuning_time_with_validation=total_time,
            metrics=res_dict["metrics"],
            metrics_std=res_dict["metrics_std"],
            naive_metrics=res_dict["naive_metrics"],
            naive_metrics_std=res_dict["naive_metrics_std"],
        )
        self.log.info("Finished horizon %s", horizon)
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _append_result_to_csv(self, result: HorizonResult) -> None:
        row_df = result.to_series().to_frame().T
        header = not self.cfg.csv_path.exists()
        row_df.to_csv(self.cfg.csv_path, mode="a", header=header, index=False)

        if self._results_frame is None:
            self._results_frame = row_df.copy()
        else:
            self._results_frame = pd.concat([self._results_frame, row_df], ignore_index=True)
