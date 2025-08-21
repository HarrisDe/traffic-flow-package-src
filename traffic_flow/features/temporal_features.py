import pandas as pd
from typing import List, Dict, Any
from .base import BaseFeatureTransformer 



class TemporalLagFeatureAdder(BaseFeatureTransformer):
    """
    Adds lag-based features to capture past values or relative differences for each sensor.

    Parameters
    ----------
    lags
        How many lag steps to add (1 … `lags`)
    relative
        If *True* create relative differences,
        else create absolute (shifted - current) deltas
    fill_nans_value
        What to write in the first `lags` rows after shifting
    sensor_col, value_col, datetime_col
        Column names
    epsilon
        Added to denominators in relative mode to avoid /0
    smoothing_id
        Optional fingerprint of the smoothing configuration; if the same lag
        columns already exist **and** the smoothing_id matches, transform()
        will skip recomputation - useful for hyper-parameter sweeps.
    """

    def __init__(
        self,
        *,
        lags: int = 25,
        relative: bool = False,
        fill_nans_value: float = -1.0,
        sensor_col: str = "sensor_id",
        value_col: str = "value",
        datetime_col: str = "datetime",
        epsilon: float = 1e-5,
        smoothing_id: str | None = None,
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)
        self.lags              = lags
        self.relative          = relative
        self.fill_nans_value   = fill_nans_value
        self.sensor_col        = sensor_col
        self.value_col         = value_col
        self.datetime_col      = datetime_col
        self.epsilon           = epsilon
        self.smoothing_id      = smoothing_id        # reference fingerprint
        self.feature_names_out_: List[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # sklearn API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to learn, but we can validate input columns
        missing = [c for c in (self.sensor_col, self.value_col) if c not in X.columns]
        if missing:
            raise ValueError(f"TemporalLagFeatureAdder: missing columns {missing}")
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() (or from_state()) before transform().")

        current_id = X.attrs.get("smoothing_id", None)
        prefix     = "relative_diff_lag" if self.relative else "lag"
        expected   = [f"{prefix}{i}" for i in range(1, self.lags + 1)]

        df = X.copy()
        new_cols: List[str] = []

        # Drop lag columns that should not exist for the chosen `lags`
        extra = [c for c in df.columns if c.startswith(prefix) and c not in expected]
        if extra:
            df.drop(columns=extra, inplace=True)
            self._log(f"Dropped obsolete lag cols {extra}")

        # Generate / regenerate each lag column
        for i in range(1, self.lags + 1):
            col = f"{prefix}{i}"
            already = col in df.columns
            same_smoothing = (current_id == self.smoothing_id)

            if already and same_smoothing:
                self._log(f"Skip {col} (exists & same smoothing_id).")
                new_cols.append(col)
                continue

            shifted = df.groupby(self.sensor_col)[self.value_col].shift(i)

            if self.relative:
                df[col] = (df[self.value_col] - shifted) / (shifted + self.epsilon)
            else:
                df[col] = shifted - df[self.value_col]

            df[col] = df[col].fillna(self.fill_nans_value)
            new_cols.append(col)

        # Record names for .export_state()
        self.feature_names_out_ = new_cols
        return df

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def export_state(self) -> Dict[str, Any]:
        """Serialize constructor arguments - no learned state to save."""
        return {
            "type": "temporal_lag",
            "lags": self.lags,
            "relative": self.relative,
            "fill_nans_value": self.fill_nans_value,
            "sensor_col": self.sensor_col,
            "value_col": self.value_col,
            "datetime_col": self.datetime_col,
            "epsilon": self.epsilon,
            "smoothing_id": self.smoothing_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TemporalLagFeatureAdder":
        inst = cls(
            lags             = state["lags"],
            relative         = state["relative"],
            fill_nans_value  = state["fill_nans_value"],
            sensor_col       = state["sensor_col"],
            value_col        = state["value_col"],
            datetime_col     = state["datetime_col"],
            epsilon          = state["epsilon"],
            smoothing_id     = state["smoothing_id"],
        )
        inst.fitted_ = True   # nothing to learn
        return inst