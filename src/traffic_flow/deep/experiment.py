
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
from .custom_features import make_demand_context_feature_fn


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
    units: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    use_norm: bool = True
    add_dense: bool = False
    dense_units: int = 16
    dense_activation: Optional[str] = "relu"
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 3e-4
    loss: str = "huber"
    optimizer: str = "adam"


def adapt_single_series_context_fn(single_series_fn, *, datetime_col: str = "date", ref_col: str = "value_ref"):
    def _wide_fn(df_wide: pd.DataFrame) -> pd.DataFrame:
        sensors = [c for c in df_wide.columns if c not in (datetime_col, "test_set")]
        tmp = pd.DataFrame({
            datetime_col: pd.to_datetime(df_wide[datetime_col]),
            ref_col: df_wide[sensors].mean(axis=1).astype(np.float32)
        }, index=df_wide.index)
        return single_series_fn(tmp)
    _wide_fn.__name__ = f"wide_{getattr(single_series_fn, '__name__', 'custom')}"
    return _wide_fn


class TrafficDeepExperiment(LoggingMixin):
    def __init__(
        self,
        *,
        data_path: str,                         # parquet for InitialTrafficDataLoader
        artifacts_dir: str = "./results",
        datetime_col: str = "date",
        disable_logs: bool = False,
    ) -> None:
        super().__init__(disable_logs=disable_logs)
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.datetime_col = datetime_col
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def run(
        self,
        *,
        data_cfg: DataCfg,
        model_cfg: ModelCfg,
        run_name: Optional[str] = None,
        results_csv: Optional[str] = None,
        save_predictions: bool = True,
    ) -> Dict[str, object]:

        run_id = run_name or uuid.uuid4().hex[:8]
        paths = self._paths(run_id)
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

        # 3) Global demand-context → adapter over reference series
        custom_fn = None
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
            custom_fn = adapt_single_series_context_fn(dc_fn_single, datetime_col=self.datetime_col, ref_col="value_ref")

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

        # Also keep dense arrays (needed for joined export)
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
            units=model_cfg.units,
            n_layers=model_cfg.n_layers,
            dropout=model_cfg.dropout,
            use_norm=model_cfg.use_norm,
            add_dense=model_cfg.add_dense,
            dense_units=model_cfg.dense_units,
            dense_activation=model_cfg.dense_activation,
            adapt_batches=None,
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

        # 7) Predict on TEST, make per-sensor results with assemble_results_dataframe
        #    and evaluate with ModelEvaluator (drop MASE keys).
        def _mask(r):
            i0, i1 = r
            if i0 < 0 or i1 < 0 or i1 < i0:
                return np.zeros_like(u, dtype=bool)
            return (u >= i0) & (u <= i1)

        m_te = _mask(splits["test"])
        if not m_te.any():
            self._dump_json(paths["metrics"], {"note": "no test windows"})
            return {"preds_df": None, "summary": {"run_id": run_id}, "run_id": run_id, "paths": paths}

        X_te = X_all[m_te]
        y_te = y_all[m_te]             # (N_te, F)  in model target space
        b_te = base_vals[m_te]         # (N_te, F)
        t_te = pd.to_datetime(target_times[m_te])

        y_pred_s = model.predict(X_te, verbose=0)       # scaled space
        y_pred   = scaler.unscale(y_pred_s)             # back to model target space

        # If model target is delta ⇒ reconstruct absolute; otherwise use abs directly
        if data_cfg.target_mode == "delta":
            y_true_abs = b_te + y_te
            y_pred_abs = b_te + y_pred
        else:
            y_true_abs = y_te
            y_pred_abs = y_pred

        # Build one wide predictions frame by merging per-sensor assembled frames
        wide_parts: List[pd.DataFrame] = []
        metrics_list: List[Dict[str, float]] = []

        for k, sensor in enumerate(sensor_cols):
            # Per-sensor meta in the shape expected by assemble_results_dataframe
            meta_sensor = {
                "y": y_te[:, [k]],                        # (N_te, 1)
                "base_times": base_times[m_te],           # (N_te,)
                "base_values": b_te[:, k],                # (N_te,)
                "pred_times": {h_label: t_te},            # dict[label] -> (N_te,)
                "h_labels": [h_label],
                "target_mode": data_cfg.target_mode,
            }
            # Per-sensor predictions in model target space
            y_pred_sensor = y_pred[:, [k]]               # (N_te, 1)

            df_res = assemble_results_dataframe(
                meta=meta_sensor,
                y_pred=y_pred_sensor,
                datetime_col=self.datetime_col,
                target_mode=data_cfg.target_mode,
                value_col=sensor,                         # base/current value col name
            )

            # Keep absolute columns for this horizon and rename to <sensor> / <sensor>_pred
            pt_col = f"prediction_time_{h_label}"
            if pt_col not in df_res.columns:              # fallback name from your helper
                pt_col = "prediction_time"
            out_k = pd.DataFrame({
                "prediction_time": pd.to_datetime(df_res[pt_col]),
                sensor:           df_res.get(f"y_true_abs_{h_label}", df_res[f"y_true_delta_{h_label}"] + df_res[sensor]),
                f"{sensor}_pred": df_res.get(f"y_pred_abs_{h_label}", df_res[f"y_pred_delta_{h_label}"] + df_res[sensor]),
            })

            wide_parts.append(out_k)

            # Evaluate (absolute space); drop MASE keys afterwards
            ev = ModelEvaluator(
                df_res=df_res.assign(test_set=True),    # mark rows as test
                horizon=h_label,
                datetime_col=self.datetime_col,
                value_col=sensor,
                disable_logs=True,
            )
            m_mod, m_nv = ev.evaluate(rounding=3)
            # attach sensor id for aggregation
            naive_row = {f"naive_{k[:-6]}": v for k, v in m_nv.items() if k.endswith("_naive")}
            mrow = {**m_mod, **naive_row, "sensor": sensor}
            mrow["sensor"] = sensor
            metrics_list.append(mrow)

        # Merge per-sensor frames on prediction_time
        preds_wide = wide_parts[0]
        for part in wide_parts[1:]:
            preds_wide = preds_wide.merge(part, on="prediction_time", how="outer")

        if save_predictions:
            preds_wide.to_parquet(paths["preds"], index=False)

        # Aggregate metrics across sensors (mean)
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
            # model metrics (averaged over sensors)
            "MAE":   float(agg.get("MAE", np.nan)),
            "MedianAE": float(agg.get("MedianAE", np.nan)),
            "RMSE":  float(agg.get("RMSE", np.nan)),
            "MAPE":  float(agg.get("MAPE", np.nan)),
            "SMAPE": float(agg.get("SMAPE", np.nan)),
            # naive (averaged)
            "naive_MAE":   float(agg.get("naive_MAE", np.nan)),
            "naive_MedianAE": float(agg.get("naive_MedianAE", np.nan)),
            "naive_RMSE":  float(agg.get("naive_RMSE", np.nan)),
            "naive_MAPE":  float(agg.get("naive_MAPE", np.nan)),
            "naive_SMAPE": float(agg.get("naive_SMAPE", np.nan)),
        }
        self._dump_json(paths["metrics"], summary)

        # Optional: append to CSV
        if results_csv:
            row = {
                "run_id": run_id,
                **{f"data.{k}": v for k, v in asdict(data_cfg).items()},
                **{f"model.{k}": v for k, v in asdict(model_cfg).items()},
                **summary,
            }
            self._append_row(results_csv, row)

        return {"preds_df": preds_wide, "summary": summary, "run_id": run_id, "paths": paths}

    # --------------- helpers ---------------
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
            json.dump(obj, f, indent=2)

    @staticmethod
    def _append_row(csv_path: str, row: Dict[str, object]) -> None:
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