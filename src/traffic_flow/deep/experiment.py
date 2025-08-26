
# energy_forecasting/deep_tf/experiment.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence
import json
import os
import uuid
import numpy as np
import pandas as pd
import tensorflow as tf

from ..common.utils import LoggingMixin
from .data_interface import DeepTFDataInterface
from .windowing import TFWindowedDatasetBuilder
from .label_scaling import LabelScaler
from .modeling import LSTMBuilder
from .training import TFTrainer
from ..common.results import assemble_results_dataframe
from ..evaluation import ModelEvaluator
from .custom_features import make_demand_context_feature_fn

# ---------------------- configuration containers ---------------------- #

@dataclass(frozen=True)
class DataCfg:
    """
    Configuration of data-related choices for a deep TF run.

    Fields
    ------
    seq_len : lookback length (window length) used to build each training example
    horizons : forecast horizons to predict (e.g. ("1h",) or ("1d",) or ("1w",))
    feature_mode : "value_only" or "value_plus_time" for cyclic time encodings
    batch_size : batch size for tf.data pipelines
    val_fraction_of_train : fraction of the pre-test portion reserved as validation
    target_mode : "absolute" for direct MW, "delta" for future change vs window end
    """
    seq_len: int = 168
    horizons: Sequence[str] = ("1h",)
    feature_mode: str = "value_plus_time"
    batch_size: int = 256
    val_fraction_of_train: float = 0.1
    target_mode: str = "delta"  # "absolute" | "delta"
    
    # demand-context feature toggles/params (JSON-friendly)
    add_demand_context: bool = True
    dc_rolling_hours: int = 168            # ~1 week
    dc_dev_last_hours: int = 24            # previous day
    dc_dev_week_hours: int = 168           # previous week same hour
    dc_add_zscore: bool = True
    dc_add_dev_last: bool = True
    dc_add_dev_week: bool = True
    dc_add_flags: bool = True



@dataclass(frozen=True)
class ModelCfg:
    """
    LSTM architecture and training hyperparameters.

    Fields
    ------
    units : hidden size for each LSTM layer
    n_layers : number of stacked LSTM layers
    dropout : dropout rate after each LSTM
    use_norm : whether to apply a Normalization layer on inputs
    add_dense : whether to add a Dense head after the last LSTM
    dense_units : width of the Dense head (only if add_dense=True)
    dense_activation : activation for Dense head
    epochs, patience, learning_rate, loss, optimizer : standard Keras training knobs
    """
    units: int = 32
    n_layers: int = 2
    dropout: float = 0.2
    use_norm: bool = True

    add_dense: bool = False
    dense_units: int = 16
    dense_activation: Optional[str] = "relu"

    epochs: int = 200
    patience: int = 20
    learning_rate: float = 3e-4
    loss: str = "huber"        # "huber" | "mae" | "mse"
    optimizer: str = "adam"


class DeepTFExperiment(LoggingMixin):
    """
    High-level runner for a single deep-learning experiment (one configured horizon set).

    Pipeline
    --------
    1) Load & clean using DeepTFDataInterface
    2) Window into train/val/test using TFWindowedDatasetBuilder (absolute or delta targets)
    3) Scale labels per-output using LabelScaler
    4) Build model via LSTMBuilder.from_dataset (optionally with Dense head)
    5) Train via TFTrainer (EarlyStopping + optional LR schedule)
    6) Predict on test, unscale to MW, assemble a wide results DataFrame,
       then evaluate **for the configured horizon(s)** using ModelEvaluator

    Artifacts written under: artifacts_dir / run_id
    - history.json, metrics.json, predictions.parquet, config.json
    Optionally appends a summary row to a CSV (for sweeps).
    """

    def __init__(
        self,
        *,
        data_path: str,
        artifacts_dir: str = "./results",
        datetime_col: str = "Datetime",
        value_col: str = "PJME_MW",
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.datetime_col = datetime_col
        self.value_col = value_col
        os.makedirs(self.artifacts_dir, exist_ok=True)

    # ------------------------------- main API ------------------------------- #
    def run(
        self,
        *,
        data_cfg: DataCfg,
        model_cfg: ModelCfg,
        run_name: Optional[str] = None,
        results_csv: Optional[str] = None,
        save_predictions: bool = True,
    ) -> Dict[str, object]:
        """
        Execute a full training→prediction→evaluation cycle.

        Returns
        -------
        dict with keys:
          - 'preds_df' : wide DataFrame with per-horizon y_true*/y_pred* columns
          - 'summary'  : flat dict of metrics + configs + run_id (suitable for CSV)
          - 'run_id'   : artifact folder name under artifacts_dir
          - 'paths'    : dict of artifact file paths
        """
        run_id = run_name or uuid.uuid4().hex[:8]
        paths = self._paths(run_id)

        # Persist configs used (for reproducibility)
        cfg_payload = {
            "data_cfg": asdict(data_cfg),
            "model_cfg": asdict(model_cfg),
            "run_id": run_id,
        }
        self._dump_json(paths["cfg"], cfg_payload)

        # 1) Load & clean
        iface = DeepTFDataInterface(
            file_path=self.data_path,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            disable_logs=self.disable_logs,
        )
        df_clean = iface.prepare_clean_frame(
            window_size=3,
            filter_on_train_only=False,
            filter_extreme_changes=True,
            smooth_series=True,                 # keep or turn off based on your preference
            use_median_instead_of_mean=False,
            relative_threshold=0.7,
            test_size=1 / 3,
            test_start_time=None,
            diagnose_extreme_changes=False,
        )
        splits = iface.build_splits(
            use_loader_test_split=True,
            val_fraction_of_train=data_cfg.val_fraction_of_train,
        )
        custom_fn = None
        if data_cfg.add_demand_context:
            
            dc_windows = sorted({
                    int(data_cfg.dc_rolling_hours),
                    int(data_cfg.dc_dev_last_hours),
                    int(data_cfg.dc_dev_week_hours),
                })

            # custom_fn = make_demand_context_feature_fn(
            #     datetime_col=self.datetime_col,
            #     value_col=self.value_col,
            #     rolling_hours=data_cfg.dc_rolling_hours,
            #     dev_last_hours=data_cfg.dc_dev_last_hours,
            #     dev_week_hours=data_cfg.dc_dev_week_hours,
            #     add_zscore=data_cfg.dc_add_zscore,
            #     add_dev_last=data_cfg.dc_add_dev_last,
            #     add_dev_week=data_cfg.dc_add_dev_week,
            #     add_flags=data_cfg.dc_add_flags,
            # )
            
            custom_fn = make_demand_context_feature_fn(
                datetime_col=self.datetime_col,
                value_col=self.value_col,
                windows=tuple(dc_windows),
                add_deviation=True,                        # always useful; keep on
                add_zscore=bool(data_cfg.dc_add_zscore),
                add_ratio=True,                           # optional; keep off for now
                min_periods=None,                          # default: each window size
                include_business_hour=bool(data_cfg.dc_add_flags),
                include_weekend_flag=bool(data_cfg.dc_add_flags),
                holiday_dates=None,                        # plug a list if you have holidays
            )
            # log the config attached by the factory
            cfg = getattr(custom_fn, "_dc_config", {})
            self._log(f"[experiment] demand_context_features enabled with {cfg}")
            
        # 2) Window datasets (handles absolute|delta)
        win = TFWindowedDatasetBuilder(
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            seq_len=data_cfg.seq_len,
            horizons=list(data_cfg.horizons),
            stride=1,
            feature_mode=data_cfg.feature_mode,
            custom_feature_fn=custom_fn,
            target_mode=data_cfg.target_mode,
        )
        train_ds, meta_train = win.build_split(
            df_clean, split=splits["train"],
            batch_size=data_cfg.batch_size, shuffle=True, cache=True
        )
        val_ds, meta_val = win.build_split(
            df_clean, split=splits["val"],
            batch_size=data_cfg.batch_size, shuffle=False, cache=True
        )
        test_ds, meta_test = win.build_split(
            df_clean, split=splits["test"],
            batch_size=data_cfg.batch_size, shuffle=False, cache=True
        )

        total_feats = len(meta_train["feature_cols"])
        custom_feats = meta_train.get("custom_feature_cols", []) or []
        preview = ", ".join(custom_feats[:8])
        more = "" if len(custom_feats) <= 8 else f" (+{len(custom_feats)-8} more)"
        self._log(f"[experiment] feature_cols: total={total_feats}, custom={len(custom_feats)} → {preview}{more}")
        
        # 3) Scale labels
        scaler = LabelScaler.from_dataset(train_ds)
        train_ds_s = scaler.scale_ds(train_ds)
        val_ds_s   = scaler.scale_ds(val_ds)
        test_ds_s  = scaler.scale_ds(test_ds)

        # 4) Build model
        model, builder = LSTMBuilder.from_dataset(
            train_ds_s,
            units=model_cfg.units,
            n_layers=model_cfg.n_layers,
            dropout=model_cfg.dropout,
            use_norm=model_cfg.use_norm,
            add_dense=model_cfg.add_dense,
            dense_units=model_cfg.dense_units,
            dense_activation=model_cfg.dense_activation,
            adapt_batches=None,
        )

        # 5) Compile + train
        trainer = TFTrainer(model)
        trainer.compile(
            loss=model_cfg.loss,
            optimizer=model_cfg.optimizer,
            metrics=["mae"],  # overall MAE on standardized outputs
            learning_rate=model_cfg.learning_rate,
        )
        history = trainer.fit(
            train_ds_s, val_ds_s,
            epochs=model_cfg.epochs,
            patience=model_cfg.patience,
            reduce_lr=True,
            reduce_lr_patience=5,
            reduce_lr_factor=0.5,
            verbose=1,
        )
        self._dump_json(paths["history"], history.history)

        # 6) Predict (unscale), assemble results, evaluate
        y_pred_s = trainer.predict(test_ds_s)
        y_pred   = scaler.unscale(y_pred_s)

        res_df = assemble_results_dataframe(
            meta=meta_test,
            y_pred=y_pred,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
        )

        if save_predictions:
            res_df.to_parquet(paths["preds"], index=False)

        # Evaluate for the configured horizon(s).
        h_label = meta_test["h_labels"][0]
        me = ModelEvaluator(
            df_res=res_df,
            horizon=h_label,
            datetime_col=self.datetime_col,
            value_col=self.value_col,
            disable_logs=True,
        )
        metrics, naive = me.evaluate(rounding=3)

        all_metrics = {**metrics, **{f"naive_{k}": v for k, v in naive.items()}}
        self._dump_json(paths["metrics"], all_metrics)
        self._log(f"[deep_tf] Run {run_id}: metrics → {paths['metrics']}")

        # Optional: append summary row to CSV
        summary_row = {
            "run_id": run_id,
            **{f"data.{k}": v for k, v in asdict(data_cfg).items()},
            **{f"model.{k}": v for k, v in asdict(model_cfg).items()},
            **all_metrics,
            "data.feature_count": total_feats,
            "data.custom_feature_count": len(custom_feats),
            "data.custom_feature_sample": ";".join(custom_feats[:12]),
        }
        if results_csv:
            self._append_row(results_csv, summary_row)

        return {
            "preds_df": res_df,
            "summary": summary_row,
            "run_id": run_id,
            "paths": paths,
        }

    # ------------------------------- helpers ------------------------------- #
    def _paths(self, run_id: str) -> Dict[str, str]:
        base = os.path.join(self.artifacts_dir, run_id)
        os.makedirs(base, exist_ok=True)
        return {
            "root": base,
            "history": os.path.join(base, "history.json"),
            "preds": os.path.join(base, "predictions.parquet"),
            "metrics": os.path.join(base, "metrics.json"),
            "cfg": os.path.join(base, "config.json"),
        }

    @staticmethod
    def _dump_json(path: str, obj: Dict) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_fallback)

    @staticmethod
    def _append_row(csv_path: str, row: Dict[str, object]) -> None:
        """
        Append a single row to a CSV, creating it if missing. New columns are added
        without failing; existing rows keep NaN for new keys.
        """
        df_row = pd.DataFrame([row])
        if not os.path.exists(csv_path):
            df_row.to_csv(csv_path, index=False)
            return

        old = pd.read_csv(csv_path)
        # align schemas (union of columns)
        for c in df_row.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in df_row.columns:
                df_row[c] = np.nan

        out = pd.concat(
            [old[sorted(old.columns)], df_row[sorted(df_row.columns)]],
            ignore_index=True
        )
        out.to_csv(csv_path, index=False)


def _json_fallback(o):
    """Make numpy types JSON-serializable (minor robustness for history dicts)."""
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")