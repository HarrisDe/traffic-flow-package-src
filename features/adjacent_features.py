import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from .base import FeatureTransformer


class AdjacentSensorFeatureAdder(FeatureTransformer):
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
        sensor_dict_path: str = "../data",
        spatial_adj: int = 5,
        normalize_by_distance: bool = True,
        datetime_col: str = "datetime",
        value_col: str = "value",
        sensor_col: str = "sensor_id",
        adj_are_relative: bool = False,
        fill_nans_value: float = -1,
        epsilon: float = 1e-5,
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
        self.new_columns: List[str] = []

        # Load precomputed sensor neighbor dictionaries
        self.downstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, "downstream_dict.json")))
        self.upstream_sensor_dict = json.load(open(os.path.join(sensor_dict_path, "upstream_dict.json")))

    def transform(
        self,
        df: pd.DataFrame,
        current_smoothing: Optional[str] = None,
        prev_smoothing: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transforms the dataframe by adding adjacent sensor features (upstream/downstream).

        Args:
            df (pd.DataFrame): Input dataframe containing timestamped sensor readings.
            current_smoothing (str, optional): Label for current smoothing configuration.
            prev_smoothing (str, optional): Label for previous smoothing configuration.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Modified dataframe and list of added column names.
        """
        self._log("Adding adjacent sensor features.")

        if not self.spatial_adj or self.spatial_adj < 1:
            self._log("No adjacent sensors to add. Skipping.")
            return df, []

        # Create pivoted lookup table: (datetime, sensor_id) → value
        pivot = df.pivot(index=self.datetime_col, columns=self.sensor_col, values=self.value_col)
        pivot_stacked = pivot.stack().to_frame(self.value_col).rename_axis(
            [self.datetime_col, self.sensor_col]
        ).sort_index()

        # Drop excess adjacent columns if leftover from earlier runs
        for direction in ["upstream", "downstream"]:
            existing_cols = [col for col in df.columns if col.startswith(f"{direction}_sensor_") and not col.endswith("_id")]
            expected_cols = [f"{direction}_sensor_{i+1}" for i in range(self.spatial_adj)]
            to_drop = list(set(existing_cols) - set(expected_cols))
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
                self._log(f"Dropped excess {direction} columns: {to_drop}")

        for i in range(self.spatial_adj):
            down_col, up_col = f"downstream_sensor_{i+1}", f"upstream_sensor_{i+1}"

            # Skip if already computed and smoothing config hasn't changed
            if down_col in df.columns and up_col in df.columns and current_smoothing == prev_smoothing:
                self._log(f"Skipping {down_col} and {up_col}, already exist.")
                self.new_columns += [down_col, up_col]
                continue

            # Get adjacent sensor mappings
            down_map = {
                sid: self.downstream_sensor_dict.get(sid, {}).get("downstream_sensor", [None] * self.spatial_adj)[i]
                for sid in df[self.sensor_col].unique()
            }
            up_map = {
                sid: self.upstream_sensor_dict.get(sid, {}).get("upstream_sensor", [None] * self.spatial_adj)[i]
                for sid in df[self.sensor_col].unique()
            }

            df[f"{down_col}_id"] = df[self.sensor_col].map(down_map)
            df[f"{up_col}_id"] = df[self.sensor_col].map(up_map)

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
                    sid: self.downstream_sensor_dict.get(sid, {}).get("downstream_distance", [np.nan] * self.spatial_adj)[i]
                    for sid in df[self.sensor_col].unique()
                }
                up_dist_map = {
                    sid: self.upstream_sensor_dict.get(sid, {}).get("upstream_distance", [np.nan] * self.spatial_adj)[i]
                    for sid in df[self.sensor_col].unique()
                }
                df[down_col] = df[down_col] / 3.6 / df[self.sensor_col].map(down_dist_map)
                df[up_col] = df[up_col] / 3.6 / df[self.sensor_col].map(up_dist_map)

            # Finalize and clean
            df.drop(columns=[f"{down_col}_id", f"{up_col}_id"], inplace=True)
            df[down_col].fillna(self.fill_nans_value, inplace=True)
            df[up_col].fillna(self.fill_nans_value, inplace=True)
            self.new_columns += [down_col, up_col]
            self._log(f"Added features: {down_col}, {up_col}")

        return df, self.new_columns