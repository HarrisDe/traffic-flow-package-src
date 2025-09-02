
# traffic_flow/deep/experiment.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Iterable, Tuple, List
import json, os, uuid
import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils.helper_utils import LoggingMixin
from ..data_loading.data_loader_orchestrator import InitialTrafficDataLoader
from .data_interface import TrafficDeepSplitInterface   # your splitter that builds target-index splits
from .windowing import TFMultiSeriesSeq2OneBuilder
from .label_scaling import LabelScaler
from .modeling import LSTMBuilder
from .training import TFTrainer
from .evaluation import ModelEvaluator, assemble_results_dataframe
from .custom_features import make_demand_context_feature_fn, adapt_single_series_context_fn
from .custom_features_extra import make_short_term_dynamics_fn, compose_feature_fns  # NEW



@dataclass(frozen=True)
class DataCfg:
    seq_len: int = 30
    horizon_minutes: int = 15
    feature_mode: str = "value_plus_time"        # value_only | value_plus_time
    batch_size: int = 256
    val_fraction_of_train: float = 0.3
    target_mode: str = "delta"                   # absolute | delta

    # global demand-context (minutes)
    add_demand_context: bool = True
    dc_windows_minutes: Tuple[int, int] = (60, 1440)
    dc_add_deviation: bool = True
    dc_add_zscore: bool = True
    dc_add_ratio: bool = False
    dc_add_flags: bool = True
    holiday_dates: Optional[Iterable[str]] = None
    
    
    # Short-term dynamics (unit-consistent)
    add_short_term_dynamics: bool = False
    std_base_unit: str = "kph"
    std_short_windows: Tuple[int, ...] = (5, 15, 30)   # minutes
    std_diff_windows:  Tuple[int, ...] = (1, 3, 5, 10) # minutes
    std_ema_fast: int = 5
    std_ema_slow: int = 15
    std_z_threshold: float = 1.5

    # loader knobs
    smooth_series: bool = True
    filter_extreme_changes: bool = True
    filter_on_train_only: bool = False
    use_median_instead_of_mean: bool = False
    relative_threshold: float = 0.7
    test_size: float = 1/3
    test_start_time: Optional[str] = None


@dataclass(frozen=True)
class ModelCfg:
    """
    All LSTMBuilder knobs + training hyperparameters.
    """

    # ---- LSTMBuilder core ----
    units: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    use_norm: bool = True                 # tf.keras.layers.Normalization on inputs
    add_dense: bool = False
    dense_units: int = 128
    dense_activation: Optional[str] = "relu"
    adapt_batches: Optional[int] = None   # if not None, run Normalization.adapt() on N batches

    # ---- LSTMBuilder extras ----
    bidirectional: bool = True
    recurrent_dropout: float = 0.1
    conv_frontend: bool = True
    conv_filters: int = 64
    conv_kernel: int = 5                  # typical odd kernel (3/5/7)
    layer_norm_in_lstm: bool = True
    attention_pooling: bool = True
    residual_head: bool = True
    conv_padding: str = "causal"

    # ---- training ----
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 3e-4
    loss: str = "huber"                   # "huber" | "mae" | "mse"
    optimizer: str = "adam"



from dataclasses import asdict
from typing import Optional, Dict, List
import os, json, uuid
import numpy as np
import pandas as pd

class TrafficDeepExperiment(LoggingMixin):
    """
    Run a deep-learning experiment end-to-end.

    - If artifacts_dir is None OR save_artifacts=False, nothing is written to disk.
    - Model weights are not saved here (no model.save).
    """

    def __init__(
        self,
        *,
        data_path: str,                         # parquet for InitialTrafficDataLoader
        artifacts_dir: Optional[str] = "./results",
        datetime_col: str = "date",
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.datetime_col = datetime_col
        if self.artifacts_dir:  # only create if provided
            os.makedirs(self.artifacts_dir, exist_ok=True)

    def _paths(self, run_id: str) -> Dict[str, Optional[str]]:
        """Return file paths for this run, or all-None if no artifacts dir."""
        if not self.artifacts_dir:
            return {"root": None, "history": None, "preds": None, "metrics": None, "cfg": None}
        base = os.path.join(self.artifacts_dir, run_id)
        os.makedirs(base, exist_ok=True)
        return {
            "root": base,
            "history": os.path.join(base, "history.json"),
            "preds": os.path.join(base, "predictions.parquet"),
            "metrics": os.path.join(base, "metrics.json"),
            "cfg": os.path.join(base, "config.json"),
        }

    def _log_dataset_shapes(
        self,
        *,
        ds_train,
        ds_val,
        ds_test,
        meta: dict,
        data_cfg: "DataCfg",
    ) -> None:
        """Pretty-print dataset/window shapes without changing behavior."""
        def _card(ds):
            if ds is None:
                return 0
            try:
                return int(tf.data.experimental.cardinality(ds).numpy())
            except Exception:
                return None

        def _peek_shapes(ds):
            if ds is None:
                return (None, None)
            try:
                for xb, yb in ds.take(1):
                    def _shape(t):
                        try:
                            return tuple(t.shape.as_list())
                        except Exception:
                            return tuple(t.shape)
                    return (_shape(xb), _shape(yb))
            except Exception:
                return (None, None)

        features_per_step = len(meta.get("feature_cols", []))
        n_sensors         = len(meta.get("sensor_cols", []))
        h_label           = meta.get("h_labels", [f"{data_cfg.horizon_minutes}m"])[0]

        train_n, val_n, test_n = _card(ds_train), _card(ds_val), _card(ds_test)
        xb_shape, yb_shape     = _peek_shapes(ds_train)

        lines = [
            "[dataset] -------------------------------",
            f" horizon: {h_label} | seq_len: {data_cfg.seq_len}",
            f" features_per_step: {features_per_step} | n_sensors: {n_sensors}",
            f" windows -> train: {train_n} | val: {val_n} | test: {test_n}",
            f" batch shapes -> X: {xb_shape} | y: {yb_shape}",
            "-----------------------------------------",
        ]
        msg = "\n".join(lines)

        if hasattr(self, "log") and callable(getattr(self, "log")):
            self.log(msg)  # type: ignore[attr-defined]
        else:
            print(msg, flush=True)

    def run(
        self,
        *,
        data_cfg: "DataCfg",
        model_cfg: "ModelCfg",
        run_name: Optional[str] = None,
        results_csv: Optional[str] = None,
        save_predictions: bool = True,
        save_artifacts: bool = True,          # <-- NEW
        log_dataset_shapes: bool = False,
    ) -> Dict[str, object]:

        run_id = run_name or uuid.uuid4().hex[:8]
        paths = self._paths(run_id) if save_artifacts else {"root": None, "history": None, "preds": None, "metrics": None, "cfg": None}

        # Config snapshot
        self._dump_json(paths["cfg"], {
            "data_cfg": asdict(data_cfg),
            "model_cfg": asdict(model_cfg),
            "run_id": run_id,
        })

        # 1) Wide dataframe (date + sensors + test_set)
        loader = InitialTrafficDataLoader(self.data_path, disable_logs=self.disable_logs)
        df_wide = loader.convert_to_ts_dataframe(
            window_size=5,
            filter_on_train_only=data_cfg.filter_on_train_only,
            filter_extreme_changes=data_cfg.filter_extreme_changes,
            smooth_speeds=data_cfg.smooth_series,
            use_median_instead_of_mean=data_cfg.use_median_instead_of_mean,
            relative_threshold=data_cfg.relative_threshold,
            test_size=data_cfg.test_size,
            test_start_time=data_cfg.test_start_time,
            diagnose_extreme_changes=False,
        )

        # 2) Target-time splits
        splitter = TrafficDeepSplitInterface(datetime_col=self.datetime_col, test_col="test_set", disable_logs=self.disable_logs)
        splitter.attach_frame(df_wide)
        splits = splitter.build_target_splits(val_fraction_of_train=data_cfg.val_fraction_of_train)

        # 3) Optional extra features (global context + short-term dynamics)
        custom_fn = None
        parts: List = []
        if data_cfg.add_demand_context:
            dc_fn_single = make_demand_context_feature_fn(
                datetime_col=self.datetime_col,
                value_col="value_ref",
                windows=tuple(int(x) for x in data_cfg.dc_windows_minutes),
                add_deviation=bool(data_cfg.dc_add_deviation),
                add_zscore=bool(data_cfg.dc_add_zscore),
                add_ratio=bool(data_cfg.dc_add_ratio),
                min_periods=None,
                include_business_hour=bool(data_cfg.dc_add_flags),
                include_weekend_flag=bool(data_cfg.dc_add_flags),
                holiday_dates=data_cfg.holiday_dates,
            )
            global_ref_fn = adapt_single_series_context_fn(dc_fn_single, datetime_col=self.datetime_col, ref_col="value_ref")
            parts.append(global_ref_fn)

        if data_cfg.add_short_term_dynamics:
            dyn_fn = make_short_term_dynamics_fn(
                datetime_col=self.datetime_col,
                base_unit=data_cfg.std_base_unit,
                short_windows=tuple(int(x) for x in data_cfg.std_short_windows),
                diff_windows=tuple(int(x) for x in data_cfg.std_diff_windows),
                ema_fast=int(data_cfg.std_ema_fast),
                ema_slow=int(data_cfg.std_ema_slow),
                z_k=float(data_cfg.std_z_threshold),
            )
            parts.append(dyn_fn)

        custom_fn = compose_feature_fns(*parts) if parts else None

        # 4) Windowing
        win = TFMultiSeriesSeq2OneBuilder(
            datetime_col=self.datetime_col,
            seq_len=int(data_cfg.seq_len),
            horizon_minutes=int(data_cfg.horizon_minutes),
            stride=1,
            feature_mode=data_cfg.feature_mode,
            custom_feature_fn=custom_fn,
            target_mode=data_cfg.target_mode,
            batch_size=int(data_cfg.batch_size),
            dropna=True,
            disable_logs=self.disable_logs,
        )
        ds_train, ds_val, ds_test, meta_all = win.make_datasets(df_wide, splits)

        if log_dataset_shapes:
            self._log_dataset_shapes(
                ds_train=ds_train,
                ds_val=ds_val,
                ds_test=ds_test,
                meta=meta_all,
                data_cfg=data_cfg,
            )

        # Dense arrays for export/eval assembly
        X_all, y_all, base_times, base_vals, target_times, sensor_cols, raw_meta = win.make_arrays(df_wide)
        u = raw_meta["target_idx"]
        h_label: str = meta_all["h_labels"][0]

        # 5) Label scaling (per-sensor)
        scaler = LabelScaler.from_dataset(ds_train)
        train_ds_s = scaler.scale_ds(ds_train)
        val_ds_s   = scaler.scale_ds(ds_val)  if ds_val  is not None else None
        test_ds_s  = scaler.scale_ds(ds_test) if ds_test is not None else None

        # 6) Build & train
        model, _ = LSTMBuilder.from_dataset(
            train_ds_s,
            # --- core ---
            units=model_cfg.units,
            n_layers=model_cfg.n_layers,
            dropout=model_cfg.dropout,
            use_norm=model_cfg.use_norm,
            add_dense=model_cfg.add_dense,
            dense_units=model_cfg.dense_units,
            dense_activation=model_cfg.dense_activation,
            adapt_batches=model_cfg.adapt_batches,
            # --- extras ---
            bidirectional=model_cfg.bidirectional,
            recurrent_dropout=model_cfg.recurrent_dropout,
            conv_frontend=model_cfg.conv_frontend,
            conv_filters=model_cfg.conv_filters,
            conv_kernel=model_cfg.conv_kernel,
            layer_norm_in_lstm=model_cfg.layer_norm_in_lstm,
            attention_pooling=model_cfg.attention_pooling,
            residual_head=model_cfg.residual_head,
            padding=model_cfg.conv_padding,
        )

        trainer = TFTrainer(model)
        trainer.compile(
            loss=model_cfg.loss,
            optimizer=model_cfg.optimizer,
            metrics=["mae"],
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

        # 7) Predict on TEST and evaluate
        def _mask(r):
            i0, i1 = r
            if i0 < 0 or i1 < 0 or i1 < i0:
                return np.zeros_like(u, dtype=bool)
            return (u >= i0) & (u <= i1)

        m_te = _mask(splits["test"])
        if not m_te.any():
            self._dump_json(paths["metrics"], {"note": "no test windows", "run_id": run_id})
            return {"preds_df": None, "summary": {"run_id": run_id}, "run_id": run_id, "paths": paths}

        X_te = X_all[m_te]
        y_te = y_all[m_te]
        b_te = base_vals[m_te]
        t_te = pd.to_datetime(target_times[m_te])
        issued_times = pd.to_datetime(base_times[m_te])

        y_pred_s = model.predict(X_te, verbose=0)
        y_pred   = scaler.unscale(y_pred_s)

        # Absolute space (if delta, reconstruct)
        if data_cfg.target_mode == "delta":
            y_true_abs = b_te + y_te
            y_pred_abs = b_te + y_pred
        else:
            y_true_abs = y_te
            y_pred_abs = y_pred

        wide_parts: List[pd.DataFrame] = []
        metrics_list: List[Dict[str, float]] = []

        for k, sensor in enumerate(sensor_cols):
            meta_sensor = {
                "y": y_te[:, [k]],
                "base_times": base_times[m_te],
                "base_values": b_te[:, k],
                "pred_times": {h_label: t_te},
                "h_labels": [h_label],
                "target_mode": data_cfg.target_mode,
            }
            y_pred_sensor = y_pred[:, [k]]

            df_res = assemble_results_dataframe(
                meta=meta_sensor,
                y_pred=y_pred_sensor,                  # (N_te, 1) or (N_te,) is fine
                value_col=sensor)


            out_k = df_res[["date", "prediction_time", sensor, f"{sensor}_pred", f"{sensor}_at_issued_time"]]
            wide_parts.append(out_k)

            ev = ModelEvaluator(
                df_res=df_res.assign(test_set=True),
                horizon=h_label,
                datetime_col="prediction_time",
                value_col=sensor,
                disable_logs=True,
            )
            m_mod, m_nv = ev.evaluate(rounding=3)
            naive_row = {f"naive_{kk[:-6]}": v for kk, v in m_nv.items() if kk.endswith("_naive")}
            mrow = {**m_mod, **naive_row, "sensor": sensor}
            metrics_list.append(mrow)

        preds_wide = wide_parts[0]
        for part in wide_parts[1:]:
            preds_wide = preds_wide.merge(part, on=["date","prediction_time"], how="outer")
        
        lead = ["date", "prediction_time"]
        sensors = [c for c in preds_wide.columns if c not in set(lead) and not c.endswith("_pred") and not c.endswith("_at_issued_time")]
        ordered = lead + [x for s in sensors for x in (s, f"{s}_pred", f"{s}_at_issued_time") if x in preds_wide.columns]
        preds_wide = preds_wide[ordered]

        if save_predictions and save_artifacts and paths["preds"]:
            preds_wide.to_parquet(paths["preds"], index=False)

        met_df = pd.DataFrame(metrics_list).set_index("sensor")
        agg = met_df.mean(numeric_only=True).to_dict()

        summary = {
            "run_id": run_id,
            "horizon_minutes": int(data_cfg.horizon_minutes),
            "target_mode": data_cfg.target_mode,
            "seq_len": int(data_cfg.seq_len),
            "features_per_step": int(len(meta_all["feature_cols"])),
            "n_sensors": int(len(sensor_cols)),
            "n_train_windows": int(((u >= splits["train"][0]) & (u <= splits["train"][1])).sum()) if splits["train"][0] >= 0 else 0,
            "n_val_windows":   int(((u >= splits["val"][0])   & (u <= splits["val"][1])).sum())   if splits["val"][0]   >= 0 else 0,
            "n_test_windows":  int(m_te.sum()),
            "MAE":   float(agg.get("MAE", np.nan)),
            "MedianAE": float(agg.get("MedianAE", np.nan)),
            "RMSE":  float(agg.get("RMSE", np.nan)),
            "MAPE":  float(agg.get("MAPE", np.nan)),
            "SMAPE": float(agg.get("SMAPE", np.nan)),
            "naive_MAE":   float(agg.get("naive_MAE", np.nan)),
            "naive_MedianAE": float(agg.get("naive_MedianAE", np.nan)),
            "naive_RMSE":  float(agg.get("naive_RMSE", np.nan)),
            "naive_MAPE":  float(agg.get("naive_MAPE", np.nan)),
            "naive_SMAPE": float(agg.get("naive_SMAPE", np.nan)),
        }
        self._dump_json(paths["metrics"], summary)

        if results_csv and save_artifacts:
            row = {
                "run_id": run_id,
                **{f"data.{k}": v for k, v in asdict(data_cfg).items()},
                **{f"model.{k}": v for k, v in asdict(model_cfg).items()},
                **summary,
            }
            self._append_row(results_csv, row)

        return {"preds_df": preds_wide, "summary": summary, "run_id": run_id, "paths": paths}

    # --------------- helpers ---------------

    @staticmethod
    def _dump_json(path: Optional[str], obj: Dict) -> None:
        """Write JSON if a path is provided; otherwise no-op."""
        if not path:
            return
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def _append_row(csv_path: Optional[str], row: Dict[str, object]) -> None:
        """Append a row to CSV if a path is provided; otherwise no-op."""
        if not csv_path:
            return
        df_row = pd.DataFrame([row])
        if not os.path.exists(csv_path):
            df_row.to_csv(csv_path, index=False); return
        old = pd.read_csv(csv_path)
        for c in df_row.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in df_row.columns: df_row[c] = np.nan
        out = pd.concat([old[sorted(old.columns)], df_row[sorted(df_row.columns)]], ignore_index=True)
        out.to_csv(csv_path, index=False)