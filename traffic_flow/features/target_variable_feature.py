from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from .base import BaseFeatureTransformer


class TargetVariableCreator(BaseFeatureTransformer):
    """
    Adds target variables for supervised learning.

    Supports both delta-speed regression and correction-based targets (e.g., GMAN).

    Attributes:
        horizon (int): Forecasting horizon in time steps.
        sensor_col (str): Name of sensor ID column.
        datetime_col (str): Name of datetime column.
        value_col (str): Column with observed traffic speed.
        gman_col (str): Column with GMAN predictions, if correction is used.
        use_gman (bool): Whether to compute correction target using GMAN.
    """

    def __init__(
        self,
        horizon: int = 15,
        sensor_col: str = 'sensor_id',
        datetime_col: str = 'date',
        value_col: str = 'value',
        gman_col: str = 'gman_prediction_orig',
        use_gman: bool = False,
        disable_logs: bool = False
    ):
        super().__init__(disable_logs)
        self.horizon = horizon
        self.sensor_col = sensor_col
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.gman_col = gman_col
        self.use_gman = use_gman
        self.fitted_ = False
        
        self.feature_names_out_ = [
            "target_total_speed",
            "target_speed_delta",
            "target",
        ]
    
    # -----------------------------------------------------------------
    def fit(self, X, y=None):
        # Just validate needed columns
        missing = [c for c in (self.sensor_col, self.datetime_col, self.value_col) if c not in X.columns]
        if missing:
            raise ValueError(f"TargetVariableCreator missing {missing}")
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        
        """
        Adds prediction targets to the DataFrame.

        Args:
            df (pd.DataFrame): Input data sorted by sensor and datetime.

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with new target columns and list of added columns.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        self._log("Creating target variables.")
        df = df.sort_values(by=[self.sensor_col, self.datetime_col]).copy()

        # Basic target: raw speed + delta
        df['target_total_speed'] = df.groupby(self.sensor_col)[self.value_col].shift(-self.horizon)
        df['target_speed_delta'] = df['target_total_speed'] - df[self.value_col]
        self._log("Computed 'target_total_speed' and 'target_speed_delta'.")

        used_cols = ['target_total_speed', 'target_speed_delta']

        # Optional correction target (e.g., GMAN residual)
        if self.use_gman:
            df['target'] = df['target_total_speed'] - df[self.gman_col]
            self._log("GMAN correction target created.")
        else:
            df['target'] = df['target_speed_delta']
            self._log("Using delta speed as final target.")

        used_cols.append('target')

        # Drop incomplete rows (NaNs at horizon edges)
        df = df.dropna(subset=['target'])
        self._log(f"Final target column ready. {df.shape[0]} rows retained after dropping NaNs.")
        self.feature_names_out_ = used_cols
        return df
    
        # ------------------------ persistence ----------------------------
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "target_creator",
            "params": dict(
                horizon=self.horizon,
                sensor_col=self.sensor_col,
                datetime_col=self.datetime_col,
                value_col=self.value_col,
                gman_col=self.gman_col,
                use_gman=self.use_gman,
            ),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TargetVariableCreator":
        params = state["params"]
        inst = cls(
            horizon=params["horizon"],
            sensor_col=params["sensor_col"],
            datetime_col=params["datetime_col"],
            value_col=params["value_col"],
            gman_col=params["gman_col"],
            use_gman=params["use_gman"],
            disable_logs=False,
        )
        inst.fitted_ = True
        return inst
    