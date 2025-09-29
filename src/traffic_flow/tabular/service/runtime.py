from __future__ import annotations
import joblib
from typing import Dict, Any, Tuple, Optional
from ..inference.inference_pipeline import TrafficInferencePipeline
from ..inference.prediction_protocol import make_prediction_frame
from ...utils.helper_utils import LoggingMixin
import numpy as np
import pandas as pd

from ..inference.inference_pipeline import TrafficInferencePipeline
from ..inference.prediction_protocol import make_prediction_frame
from ...utils.helper_utils import LoggingMixin  


class InferenceRuntime(LoggingMixin):
    """
    Load a saved artifact (model + states), rebuild the inference pipeline,
    and provide:
      - preprocess(): raw -> sorted/trimmed + engineered features
      - predict():    features -> predictions frame
      - predict_df(): convenience wrapper (backward compatible)

    Canonical prediction frame columns:
      - sensor_id (or whatever schema_state names it)
      - input_time
      - prediction_time
      - y_pred_delta
      - horizon
      - y_pred_total  (when add_total=True)
    """

    def __init__(self, artifact_path: str, *, keep_datetime: bool = False) -> None:
        super().__init__(disable_logs=False)

        bundle: Dict[str, Any] = joblib.load(artifact_path)
        self.states: Dict[str, Any] = bundle["states"]
        self.model = bundle["model"]
        self.horizon: int = int(bundle.get("horizon", 15))
        self.feature_cols: list[str] = bundle["feature_cols"]

        schema = self.states.get("schema_state", {})
        self.sensor_col: str = schema.get("sensor_col", "sensor_id")
        self.datetime_col: str = schema.get(
            "datetime_col",
            self.states.get("datetime_state", {}).get("datetime_col", "date"),
        )
        self.value_col: str = schema.get("value_col", "value")

        # Rebuild inference pipeline from states
        self.pipeline = TrafficInferencePipeline(self.states, keep_datetime=keep_datetime)
        self.feats: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # 1) Preprocess: raw -> sorted/trimmed + engineered features
    # -------------------------------------------------------------------------
    def preprocess(
        self,
        df_raw: pd.DataFrame,
        *,
        df_for_ML: Optional[pd.DataFrame] = None,
        trim_warmup: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare inputs for prediction:
          - deterministic sort of df_raw
          - optional warm-up trim per sensor
          - feature engineering (feats)
          - (optionally) deterministically sort df_for_ML if provided

        Returns
        -------
        dict with:
            df_sorted, feats, df_for_ML_sorted
        """
        # 1) sort & dtype-fix
        if (self.datetime_col in df_raw.columns) and (self.sensor_col in df_raw.columns):
            df_sorted = df_raw.sort_values(
                [self.datetime_col, self.sensor_col], kind="mergesort"
            ).reset_index(drop=True)
        else:
            raise ValueError(
                f"df_raw must contain columns {self.datetime_col!r} and {self.sensor_col!r}"
            )

        if not np.issubdtype(df_sorted[self.datetime_col].dtype, np.datetime64):
            df_sorted[self.datetime_col] = pd.to_datetime(
                df_sorted[self.datetime_col], errors="coerce"
            )

        # 2) optional warm-up trim (based on lag depth in states)
        if trim_warmup:
            df_sorted = self._trim_warmup_rows(df_sorted)

        # 3) features
        feats = self.pipeline.transform(df_sorted).reindex(self.feature_cols, axis=1, copy=False)

        # 4) optional df_for_ML sorting (for alignment/debugging)
        df_for_ML_sorted = None
        if df_for_ML is not None:
            if (self.datetime_col not in df_for_ML.columns) or (self.sensor_col not in df_for_ML.columns):
                raise ValueError(
                    f"`df_for_ML` must include {self.datetime_col!r} and {self.sensor_col!r}"
                )
            df_for_ML_sorted = df_for_ML.sort_values(
                [self.datetime_col, self.sensor_col], kind="mergesort"
            ).reset_index(drop=True)

        return {
            "df_sorted": df_sorted,
            "feats": feats,
            "df_for_ML_sorted": df_for_ML_sorted,
        }

    # -------------------------------------------------------------------------
    # 2) Predict: features -> prediction frame
    # -------------------------------------------------------------------------
    def predict(
        self,
        preprocessed: Dict[str, pd.DataFrame],
        *,
        add_total: bool = True,
        add_y_act: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Consume the dict returned by preprocess() and produce:
          - pred_df: canonical predictions DataFrame
          - feats:   the feature matrix used
        """
        df_sorted = preprocessed["df_sorted"]
        feats = preprocessed["feats"]
        df_for_ML_sorted = preprocessed.get("df_for_ML_sorted", None)

        pred_delta = self.model.predict(feats)

        pred_df = make_prediction_frame(
            df_raw=df_sorted,
            df_for_ML=df_for_ML_sorted,
            feats=feats,
            pred_delta=pred_delta,
            states=self.states,
            horizon_min=self.horizon,
            add_total=add_total,
            add_y_act=add_y_act,
        )
        return pred_df, feats

    # -------------------------------------------------------------------------
    # 3) Legacy one-shot API (backward-compatible)
    # -------------------------------------------------------------------------
    def predict_df(
        self,
        df_raw: pd.DataFrame,
        *,
        df_for_ML: Optional[pd.DataFrame] = None,
        trim_warmup: bool = True,
        add_total: bool = True,
        add_y_act: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Equivalent to:
            prep = self.preprocess(df_raw, df_for_ML=df_for_ML, trim_warmup=trim_warmup)
            return self.predict(prep, add_total=add_total, add_y_act=add_y_act)
        """
        prep = self.preprocess(df_raw, df_for_ML=df_for_ML, trim_warmup=trim_warmup)
        return self.predict(prep, add_total=add_total, add_y_act=add_y_act)

    # ------------------------------ internals --------------------------------
    def _trim_warmup_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_steps = int(self.states.get("lag_state", {}).get("lags", 0))
        if lag_steps <= 0:
            return df.copy()
        g = df.groupby(self.sensor_col, sort=False)
        rank = g.cumcount()
        kept = rank >= lag_steps
        return df.loc[kept].reset_index(drop=True)