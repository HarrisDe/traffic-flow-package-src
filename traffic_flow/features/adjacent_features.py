import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseFeatureTransformer


class AdjacentSensorFeatureAdder(BaseFeatureTransformer):
    """
    Adds features representing the value of upstream and downstream adjacent sensors.

    The number of adjacent sensors is configurable with `spatial_adj`. Optionally,
    distances can be used to normalize the values, and the difference can be expressed
    as a relative delta compared to the current sensor value.

    Args:
        sensor_dict_path (str): Path to the folder containing upstream/downstream sensor JSONs.
        spatial_adj (int): Number of adjacent sensors to extract in each direction.
        normalize_by_distance (bool): Whether to normalize speed by inter-sensor distance.
        datetime_col (str): Name of the timestamp column.
        value_col (str): Name of the traffic value column (e.g., speed).
        sensor_col (str): Name of the sensor ID column.
        adj_are_relative (bool): Whether to represent adjacents as relative differences.
        fill_nans_value (float): Value used to fill missing entries.
        epsilon (float): Small constant to prevent division by zero.
        disable_logs (bool): If True, suppress logging.
    """

    def __init__(
        self,
        *,
        downstream_dict: Dict[str, Any] | None = None,
        upstream_dict:   Dict[str, Any] | None = None,
        sensor_dict_path: str | None = "../data",
        spatial_adj: int = 1,
        normalize_by_distance: bool = True,
        adj_are_relative: bool = False,
        datetime_col: str = "datetime",
        value_col: str = "value",
        sensor_col: str = "sensor_id",
        fill_nans_value: float = -1,
        epsilon: float = 1e-5,
        smoothing_id: str | None = None,
        disable_logs: bool = False
    ):
        super().__init__(disable_logs)
        self.sensor_dict_path = sensor_dict_path
        self.spatial_adj = spatial_adj
        self.normalize_by_distance = normalize_by_distance
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.sensor_col = sensor_col
        self.adj_are_relative = adj_are_relative
        self.fill_nans_value = fill_nans_value
        self.epsilon = epsilon
        self.smoothing_id = smoothing_id

        # # Load precomputed sensor neighbor dictionaries
        # self.downstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, "downstream_dict.json")))
        # self.upstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, "upstream_dict.json")))
        # Load or accept ready-made dictionaries
        if downstream_dict is None or upstream_dict is None:
            if sensor_dict_path is None:
                raise ValueError(
                    "Provide either sensor_dict_path or both downstream_dict/upstream_dict."
                )
            downstream_path = os.path.join(sensor_dict_path, "downstream_dict.json")
            upstream_path   = os.path.join(sensor_dict_path, "upstream_dict.json")
            self.downstream_dict_ = json.load(open(downstream_path))
            self.upstream_dict_   = json.load(open(upstream_path))
        else:
            self.downstream_dict_ = downstream_dict
            self.upstream_dict_   = upstream_dict
         
        self.fitted_ = False
        self.feature_names_out_ = []

        
    def fit(self, X: pd.DataFrame, y=None):
        """Validate that every sensor_id is present in the neighbour dicts."""
        missing = [
            sid
            for sid in X[self.sensor_col].unique()
            if sid not in self.downstream_dict_ or sid not in self.upstream_dict_
        ]
        if missing:
            self._log(
                f"{len(missing)} sensor IDs missing from neighbour dictionaries; "
                f"they will receive NaNs during transform."
            )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame,) -> pd.DataFrame:
        """
        Transforms the dataframe by adding adjacent sensor features (upstream/downstream).

        Args:
            df (pd.DataFrame): Input dataframe containing timestamped sensor readings.
            current_smoothing (str, optional): Label for current smoothing configuration.
            prev_smoothing (str, optional): Label for previous smoothing configuration.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Modified dataframe and list of added column names.
        """
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() (or from_state()) before transform().")
        
        current_id  = X.attrs.get("smoothing_id", None)   

        df = X.copy()
        new_cols: List[str] = [] 
        self._log("Adding adjacent sensor features.")

        if not self.spatial_adj or self.spatial_adj < 1:
            self._log("No adjacent sensors to add. Skipping.")
            return df

        # Create pivoted lookup table: (datetime, sensor_id) → value
        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        pivot_stacked = pivot.stack().to_frame(self.value_col).rename_axis(
            [self.datetime_col, self.sensor_col]
        ).sort_index()


        for i in range(self.spatial_adj):
            down_col, up_col = f"downstream_sensor_{i+1}", f"upstream_sensor_{i+1}"
            
            # --- Skip logic ------------------------------------------------
            already_have = down_col in df.columns and up_col in df.columns
            same_smoothing   = (current_id == self.smoothing_id)# <─ read from attrs
            if already_have and same_smoothing:
                self._log(f"Skip {down_col}/{up_col} (already computed, same smoothing_id).")
                new_cols += [down_col, up_col]
                continue

            # Get adjacent sensor mappings
            down_map = {
                sid: self.downstream_dict_.get(sid, {}).get("downstream_sensor", [None] * self.spatial_adj)[i]
                for sid in df[self.sensor_col].unique()
            }
            up_map = {
                sid: self.upstream_dict_.get(sid, {}).get("upstream_sensor", [None] * self.spatial_adj)[i]
                for sid in df[self.sensor_col].unique()
            }

            df[f"{down_col}_id"] = df[self.sensor_col].map(down_map)  # type: ignore[arg-type]
            df[f"{up_col}_id"] = df[self.sensor_col].map(up_map)      # type: ignore[arg-type]

            # Perform lookup in pivot table
            down_values = pivot_stacked[self.value_col].reindex(
                list(zip(df[self.datetime_col], df[f"{down_col}_id"]))
            ).values
            up_values = pivot_stacked[self.value_col].reindex(
                list(zip(df[self.datetime_col], df[f"{up_col}_id"]))
            ).values

            df[down_col] = down_values
            df[up_col] = up_values

            # Optionally compute relative difference
            if self.adj_are_relative:
                df[down_col] = (df[down_col] - df[self.value_col]) / (df[self.value_col] + self.epsilon)
                df[up_col] = (df[self.value_col] - df[up_col]) / (df[up_col] + self.epsilon)

            # Optionally normalize by distance
            if self.normalize_by_distance:
                down_dist_map = {
                    sid: self.downstream_dict_.get(sid, {}).get("downstream_distance", [np.nan] * self.spatial_adj)[i]
                    for sid in df[self.sensor_col].unique()
                }
                up_dist_map = {
                    sid: self.upstream_dict_.get(sid, {}).get("upstream_distance", [np.nan] * self.spatial_adj)[i]
                    for sid in df[self.sensor_col].unique()
                }
                df[down_col] = df[down_col] / 3.6 / df[self.sensor_col].map(down_dist_map)  # type: ignore[arg-type]
                df[up_col] = df[up_col] / 3.6 / df[self.sensor_col].map(up_dist_map)  # type: ignore[arg-type]

            # Finalize and clean
            df.drop(columns=[f"{down_col}_id", f"{up_col}_id"], inplace=True)
            df[down_col].fillna(self.fill_nans_value, inplace=True)
            df[up_col].fillna(self.fill_nans_value, inplace=True)
            new_cols += [down_col, up_col]
            self._log(f"Added features: {down_col}, {up_col}")
            self.feature_names_out_ = new_cols

        return df
    
    # Persistence
    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "adjacent",
            "downstream_dict": self.downstream_dict_,
            "upstream_dict": self.upstream_dict_,
            "spatial_adj": self.spatial_adj,
            "normalize_by_distance": self.normalize_by_distance,
            "adj_are_relative": self.adj_are_relative,
            "sensor_col": self.sensor_col,
            "datetime_col": self.datetime_col,
            "value_col": self.value_col,
            "fill_nans_value": self.fill_nans_value,
            "epsilon": self.epsilon,
            "smoothing_id": self.smoothing_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AdjacentSensorFeatureAdder":
        return cls(
            downstream_dict=state["downstream_dict"],
            upstream_dict=state["upstream_dict"],
            spatial_adj=state["spatial_adj"],
            normalize_by_distance=state["normalize_by_distance"],
            adj_are_relative=state["adj_are_relative"],
            sensor_col=state["sensor_col"],
            datetime_col=state["datetime_col"],
            value_col=state["value_col"],
            fill_nans_value=state["fill_nans_value"],
            epsilon=state["epsilon"],
            smoothing_id=state["smoothing_id"],
        )