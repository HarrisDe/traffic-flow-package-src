# gman_hybrid_runner.py
# -*- coding: utf-8 -*-
"""
GMANHybridExperiment: A SOLID, testable runner for GMAN hybrid experiments.

Key ideas
---------
- Single Responsibility: this class only orchestrates data loading, config
  resolution, GPU/seed setup, OOM backoff, and saving results.
- Open-Closed: behaviour is configurable via dataclasses & injected callables
  (train_fn, loader_cls), without modifying the class.
- Liskov Substitution: injected `train_fn` only needs to honour the expected
  signature/return; any compatible implementation can be substituted.
- Interface Segregation: small configs (smoothing/splits/train) avoid bloated
  initializers.
- Dependency Inversion: GMAN training is injected (default uses your existing
  `run_gman_light_test_set_as_input_column`), and the data loader class is
  injected as well.

Drop this file in:  src/traffic_flow/hybrid/gman_hybrid_runner.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

# ---- Logging -----------------------------------------------------------------
logger = logging.getLogger("GMANHybridExperiment")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---- Configs -----------------------------------------------------------------
@dataclass(frozen=True)
class SmoothingCfg:
    window_size: int = 5
    filter_on_train_only: bool = False
    filter_extreme_changes: bool = True
    smooth_speeds: bool = True
    use_median_instead_of_mean: bool = False
    relative_threshold: float = 0.7


@dataclass(frozen=True)
class SplitCfg:
    # If test_start_time is provided, it takes precedence over ratios in the data loader.
    train_ratio: float = 2 / 9
    test_ratio: float = 6 / 9
    test_start_time: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class TrainCfg:
    max_epoch: int = 10
    patience: int = 3
    initial_batch_size: int = 32
    oom_backoff: bool = True
    min_batch_size: int = 1
    seed: int = 69
    use_gpu: bool = True


@dataclass(frozen=True)
class GMANParams:
    P: int  # history length
    Q: int  # horizon
    L: int  # temporal attention blocks
    K: int  # spatial attention heads


# ---- Runner ------------------------------------------------------------------
class GMANHybridExperiment:
    """
    Orchestrates a single GMAN run for a given horizon (Q), handling data prep,
    hyperparam selection (L,K), OOM backoff, and persistence.

    Parameters
    ----------
    orig_file_path : str | Path
        Path to the parquet (or supported) file consumed by InitialTrafficDataLoader.
    se_file : str | Path
        Path to SE_new.txt used by GMAN.
    results_dir : str | Path
        Directory to write parquet results + CSV logs (from train_fn).
    previous_results_csv : Optional[str | Path]
        CSV with past GMAN sweeps; used to auto-pick (L,K) if not explicitly provided.
        Must include columns ['Q','L','K','MAE'].
    results_filename : str
        File name (CSV) that the underlying training function will append to.
    train_fn : Optional[Callable]
        Callable used to train/evaluate GMAN. If None, tries to import and use
        `run_gman_light_test_set_as_input_column` from your code base.
    loader_cls : Optional[type]
        Data loader class. If None, tries to import InitialTrafficDataLoader.
    """

    def __init__(
        self,
        orig_file_path: Union[str, Path],
        se_file: Union[str, Path],
        results_dir: Union[str, Path],
        previous_results_csv: Optional[Union[str, Path]] = None,
        results_filename: str = "model_results_gman_one_year_reduced_33_67_split_standalone.csv",
        train_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        loader_cls: Optional[type] = None,
    ) -> None:
        self.orig_file_path = Path(orig_file_path)
        self.se_file = Path(se_file)
        self.results_dir = Path(results_dir)
        self.previous_results_csv = Path(previous_results_csv) if previous_results_csv else None
        self.results_filename = results_filename

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports to keep this file standalone & testable
        if train_fn is None:
            try:
                from traffic_flow.hybrid.experimental_studies_functions import (
                    run_gman_light_test_set_as_input_column as _default_train_fn,
                )
                self.train_fn = _default_train_fn
            except Exception as e:
                raise ImportError(
                    "Could not import run_gman_light_test_set_as_input_column. "
                    "Pass a `train_fn` explicitly to GMANHybridExperiment."
                ) from e
        else:
            self.train_fn = train_fn

        if loader_cls is None:
            try:
                from traffic_flow.data_loading.data_loader_orchestrator import InitialTrafficDataLoader as _Loader
                self.loader_cls = _Loader
            except Exception as e:
                raise ImportError(
                    "Could not import InitialTrafficDataLoader. Pass `loader_cls` explicitly."
                ) from e
        else:
            self.loader_cls = loader_cls

    # --------------------------- utilities -----------------------------------
    @staticmethod
    def _setup_seed_and_gpu(seed: int, use_gpu: bool) -> None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Limit GPU memory growth to reduce OOMs
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus and use_gpu:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Enabled GPU memory growth for %d GPU(s).", len(gpus))
            else:
                logger.info("GPU not used or not available; running on CPU.")
        except Exception as e:
            logger.warning("Failed to set GPU memory growth: %s", e)

    @staticmethod
    def _experiment_name(q: int, smoothing: SmoothingCfg) -> str:
        # Keep the familiar naming — reproducible & human-readable
        smooth_str = (
            f"smooth-{smoothing.smooth_speeds}_"
            f"filter_on_train_only-{smoothing.filter_on_train_only}_"
            f"w-{smoothing.window_size}-mean_"
            f"smoothing_train_ratio-0.2"
        )
        return f"GMAN__h-{q}_{smooth_str}"

    def _resolve_lk(self, q: int, provided_L: Optional[int], provided_K: Optional[int]) -> tuple[int, int]:
        if provided_L is not None and provided_K is not None:
            return int(provided_L), int(provided_K)

        if self.previous_results_csv is None or not self.previous_results_csv.exists():
            raise ValueError(
                "L and K not provided, and previous_results_csv was not supplied or does not exist."
            )

        df_prev = pd.read_csv(self.previous_results_csv)
        if q not in df_prev["Q"].unique():
            logger.info("Horizon Q=%s not found in previous results. Falling back to best for Q=60.", q)
            df_filtered = df_prev[df_prev["Q"] == 60]
        else:
            df_filtered = df_prev[df_prev["Q"] == q]

        if df_filtered.empty:
            raise ValueError("No rows available in previous_results_csv to infer (L,K).")

        row = df_filtered.sort_values(by="MAE").iloc[0]
        L, K = int(row["L"]), int(row["K"])
        logger.info("Resolved (L,K) = (%d,%d) from previous results.", L, K)
        return L, K

    def _build_df_gman(self, smoothing: SmoothingCfg, split: SplitCfg) -> pd.DataFrame:
        loader = self.loader_cls(self.orig_file_path)
        df_gman = loader.convert_to_ts_dataframe(
            window_size=smoothing.window_size,
            filter_on_train_only=smoothing.filter_on_train_only,
            filter_extreme_changes=smoothing.filter_extreme_changes,
            smooth_speeds=smoothing.smooth_speeds,
            use_median_instead_of_mean=smoothing.use_median_instead_of_mean,
            relative_threshold=smoothing.relative_threshold,
            test_size=None if split.test_start_time is not None else split.test_ratio,
            test_start_time=split.test_start_time,
            diagnose_extreme_changes=False,
        )
        return df_gman

    # ------------------------------ public API --------------------------------
    def run(
        self,
        gman: GMANParams,
        smoothing: Optional[SmoothingCfg] = None,
        split: Optional[SplitCfg] = None,
        train: Optional[TrainCfg] = None,
        skip_if_exists: bool = True,
        extra_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one GMAN training/eval run and persist parquet results.

        Returns
        -------
        Dict[str, Any]
            Whatever `train_fn` returns. Expecting a dict with 'results' (pd.DataFrame).
        """
        smoothing = smoothing or SmoothingCfg()
        split = split or SplitCfg()
        train = train or TrainCfg()
        extra_train_kwargs = dict(extra_train_kwargs or {})

        self._setup_seed_and_gpu(seed=train.seed, use_gpu=train.use_gpu)

        # Resolve L/K if not provided explicitly
        L, K = self._resolve_lk(q=gman.Q, provided_L=gman.L, provided_K=gman.K)
        gman = GMANParams(P=gman.P, Q=gman.Q, L=L, K=K)

        # Prepare data frame
        logger.info("Preparing df_gman with cfg: %s | split: %s", asdict(smoothing), asdict(split))
        df_gman = self._build_df_gman(smoothing, split)

        # Naming & output paths
        experiment_name = self._experiment_name(q=gman.Q, smoothing=smoothing)
        file_path = self.results_dir / f"{experiment_name}.parquet"
        logger.info("Experiment: %s", experiment_name)
        logger.info("Output path: %s", file_path)

        if skip_if_exists and file_path.exists():
            logger.info("Skipping: already exists at %s", file_path)
            # Return a small manifest to be explicit
            return {
                "skipped": True,
                "experiment_name": experiment_name,
                "path": str(file_path),
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Train with OOM backoff if needed
        batch_size = train.initial_batch_size
        last_err: Optional[BaseException] = None

        while True:
            try:
                logger.info("Running GMAN: (L=%d, K=%d), batch_size=%d", gman.L, gman.K, batch_size)

                results = self.train_fn(
                    Q=gman.Q,
                    P=gman.P,
                    L=gman.L,
                    K=gman.K,
                    traffic_file=None,
                    df_gman=df_gman,
                    max_epoch=train.max_epoch,
                    patience=train.patience,
                    batch_size=batch_size,
                    train_ratio=split.train_ratio,
                    test_ratio=split.test_ratio,
                    results_filename=self.results_filename,
                    results_dir=str(self.results_dir),
                    save_results_to_csv=True,
                    experiment_filename=experiment_name,
                    smooth_speeds=smoothing.smooth_speeds,
                    filter_on_train_only=smoothing.filter_on_train_only,
                    window_size=smoothing.window_size,
                    use_median_instead_of_mean_smoothing=smoothing.use_median_instead_of_mean,
                    SE_file=str(self.se_file),
                    **extra_train_kwargs,
                )

                df_out = results.get("results")
                if isinstance(df_out, pd.DataFrame):
                    df_out.to_parquet(file_path)
                    logger.info("Saved parquet results to %s", file_path)
                else:
                    logger.warning("Train fn returned no 'results' DataFrame; skipping parquet save.")

                # Attach manifest for traceability
                manifest = {
                    "experiment_name": experiment_name,
                    "params": asdict(gman),
                    "smoothing": asdict(smoothing),
                    "split": {**asdict(split), "test_start_time": str(split.test_start_time)},
                    "train": asdict(train),
                    "output_path": str(file_path),
                }
                results["manifest"] = manifest
                # Also drop a JSON manifest next to parquet
                with open(self.results_dir / f"{experiment_name}.json", "w") as f:
                    json.dump(manifest, f, indent=2)
                return results

            except tf.errors.ResourceExhaustedError as e:
                last_err = e
                if not train.oom_backoff or batch_size <= train.min_batch_size:
                    logger.error("OOM and cannot back off further (batch=%d).", batch_size)
                    raise
                new_batch = max(train.min_batch_size, batch_size // 2)
                logger.warning("ResourceExhaustedError; reducing batch size %d → %d", batch_size, new_batch)
                batch_size = new_batch
                continue
            except Exception as e:
                last_err = e
                logger.exception("Training failed with an unexpected error.")
                raise
            finally:
                if last_err:
                    logger.debug("Last error: %s", repr(last_err))

