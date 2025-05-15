import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Callable, List, Union, Dict, Tuple, str
import warnings
from traffic_flow_package_src.helper_utils import load_adjacency_dicts
from traffic_flow_package_src.post_processing import PredictionCorrectionPerSensor

# Type alias for upstream/downstream dictionary structure
UpDownDict = Dict[str, Dict[str, List[Union[str, float, None]]]]

class SensorPredictionPlotter:
    """
    A class to visualize model predictions for traffic sensors, including their
    upstream and downstream neighbors.

    Automatically applies correction methods to the model predictions, either through
    an injected factory or defaulting to naive-based correction.
    Also caches correction objects per sensor for efficiency.
    """

    def __init__(self,
                 X_test: pd.DataFrame,
                 df_for_ML: pd.DataFrame,
                 y_test: pd.Series,
                 y_pred: pd.Series,
                 upstream_dict: Optional[UpDownDict] = None,
                 downstream_dict: Optional[UpDownDict] = None,
                 correction_factory: Optional[Callable[[int], PredictionCorrectionPerSensor]] = None,
                 time_axis_col: str = 'date'):
        """
        Initialize the SensorPredictionPlotter.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set with 'sensor_uid' column.
        df_for_ML : pd.DataFrame
            Full ML dataframe including 'sensor_id', 'sensor_uid', and 'date'.
        y_test : pd.Series
            Ground truth target values.
        y_pred : pd.Series
            Model predictions.
        upstream_dict : Optional[UpDownDict]
            Pre-loaded upstream adjacency dictionary. If None, will be loaded from JSON.
        downstream_dict : Optional[UpDownDict]
            Pre-loaded downstream adjacency dictionary. If None, will be loaded from JSON.
        correction_factory : Optional[Callable[[int], PredictionCorrectionPerSensor]]
            Factory function to create correction objects per sensor_uid. If None, uses default.
        """
        self.X_test = X_test
        self.df_for_ML = df_for_ML
        self.y_test = y_test
        self.y_pred = y_pred
        self.time_axis_col = time_axis_col

        if upstream_dict is None or downstream_dict is None:
            self.upstream_dict, self.downstream_dict = load_adjacency_dicts()
        else:
            self.upstream_dict = upstream_dict
            self.downstream_dict = downstream_dict

        self._correction_factory = correction_factory or self._default_correction_factory
        self._corr_cache: Dict[int, PredictionCorrectionPerSensor] = {}

    def clear_cache(self) -> None:
        """
        Clear the internal cache of correction objects.
        Use this if you want to force rebuilding correction logic.
        """
        self._corr_cache.clear()

    def plot(self,
             sensor_uid: int,
             correction_pipeline: Optional[Union[Callable, List[Callable]]] = None,
             show_pred: bool = True,
             show_corrected_pred: bool = True,
             title: str = "Predictions vs Time") -> None:
        """
        Plot actual vs predicted values for a single sensor, optionally applying correction.

        Parameters
        ----------
        sensor_uid : int
            Unique ID of the sensor to plot.
        correction_pipeline : Optional[Union[Callable, List[Callable]]]
            Function(s) that take in y_pred and return corrected predictions.
            If None, naive_based_correction is applied by default.
        show_pred : bool
            Whether to show the raw prediction line.
        show_corrected_pred : bool
            Whether to show the corrected prediction line.
        title : str
            Plot title prefix.
        """
        try:
            sensor_id = self._get_sensor_id(sensor_uid)
            idx = self._get_indices_for_uid(sensor_uid)
        except ValueError as e:
            warnings.warn(str(e))
            return

        time_axis = self.df_for_ML.loc[idx, self.time_axis_col]
        y_pred_s = self.y_pred.loc[idx]
        y_test_s = self.y_test.loc[idx]

        if correction_pipeline is None:
            # Default to naive-based correction
            corr = self._get_correction(sensor_uid)
            correction_pipeline = corr.naive_based_correction

        y_corr_s = self._apply_corrections(y_pred_s, correction_pipeline)

        full_title = f"<b>{title} [{sensor_id}]</b>"
        self._plot_single(time_axis, y_test_s, y_pred_s, y_corr_s,
                          title=full_title,
                          show_pred=show_pred,
                          show_corrected_pred=show_corrected_pred)

    def plot_with_adjacent(self,
                           sensor_uid: int,
                           correction_pipeline: Optional[Union[Callable, List[Callable]]] = None,
                           show_pred: bool = True,
                           show_corrected_pred: bool = True,
                           title: str = "Predictions vs Time") -> None:
        """
        Plot predictions for a sensor and its upstream/downstream neighbors.

        Parameters
        ----------
        sensor_uid : int
            UID of the target sensor.
        correction_pipeline : Optional[Union[Callable, List[Callable]]]
            Correction method(s). Defaults to naive_based_correction.
        show_pred : bool
            Show raw model predictions.
        show_corrected_pred : bool
            Show corrected predictions.
        title : str
            Plot title prefix.
        """
        try:
            sensor_id = self._get_sensor_id(sensor_uid)
        except ValueError as e:
            warnings.warn(str(e))
            return

        all_ids = [sensor_id]
        all_ids += self.upstream_dict.get(str(sensor_id), {}).get("upstream_sensor", [])
        all_ids += self.downstream_dict.get(str(sensor_id), {}).get("downstream_sensor", [])

        for sid in all_ids:
            uid = self._get_sensor_uid_from_id(sid)
            if uid is not None:
                self.plot(sensor_uid=uid,
                          correction_pipeline=correction_pipeline,
                          show_pred=show_pred,
                          show_corrected_pred=show_corrected_pred,
                          title=title)

    def _get_correction(self, sensor_uid: int) -> PredictionCorrectionPerSensor:
        """Return cached correction object for the sensor or create a new one."""
        if sensor_uid not in self._corr_cache:
            self._corr_cache[sensor_uid] = self._correction_factory(sensor_uid)
        return self._corr_cache[sensor_uid]

    def _default_correction_factory(self, sensor_uid: int) -> PredictionCorrectionPerSensor:
        """Default correction factory that uses PredictionCorrectionPerSensor."""
        return PredictionCorrectionPerSensor(
            X_test=self.X_test,
            y_test=self.y_test,
            df_for_ML=self.df_for_ML,
            sensor_uid=sensor_uid
        )

    def _get_sensor_id(self, sensor_uid: int) -> str:
        """Map a sensor UID to its external sensor_id."""
        match = self.df_for_ML[self.X_test['sensor_uid'] == sensor_uid]
        if match.empty:
            raise ValueError(f"Sensor UID {sensor_uid} not found.")
        return match['sensor_id'].iloc[0]

    def _get_sensor_uid_from_id(self, sensor_id: Union[str, int]) -> Optional[int]:
        """Map a sensor_id to its internal sensor_uid."""
        match = self.df_for_ML[self.df_for_ML['sensor_id'] == sensor_id]
        return match['sensor_uid'].iloc[0] if not match.empty else None

    def _get_indices_for_uid(self, sensor_uid: int) -> pd.Index:
        """Return index mask for selected sensor UID."""
        idx = self.X_test.index[self.X_test['sensor_uid'] == sensor_uid]
        if idx.empty:
            raise ValueError(f"No data found for sensor_uid={sensor_uid}")
        return idx

    def _apply_corrections(self,
                           y_pred_s: pd.Series,
                           pipeline: Optional[Union[Callable, List[Callable]]]) -> Optional[pd.Series]:
        """Apply one or more correction functions to the predictions."""
        if pipeline is None:
            return None
        y_corr = y_pred_s.copy()
        if callable(pipeline):
            return pipeline(y_corr)
        for fn in pipeline:
            if not callable(fn):
                raise TypeError("Each item in correction pipeline must be callable.")
            y_corr = fn(y_corr)
        return y_corr

    def _plot_single(self,
                     time_axis: pd.Series,
                     y_test_s: pd.Series,
                     y_pred_s: pd.Series,
                     y_corr_s: Optional[pd.Series],
                     title: str,
                     show_pred: bool,
                     show_corrected_pred: bool) -> None:
        """Create and show the interactive Plotly figure for one sensor."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name='Actual', line=dict(color='green')))
        if show_pred:
            fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name='Prediction',
                                     line=dict(color='blue', dash='dash')))
        if show_corrected_pred and y_corr_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name='Corrected',
                                     line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=title,
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=400
        )
        fig.show()




class SensorPredictionPlotter:
    """
    A class to visualize model predictions for traffic sensors, including their
    upstream and downstream neighbors.

    Automatically applies correction methods to the model predictions, either through
    an injected factory or defaulting to naive-based correction.
    Also caches correction objects per sensor for efficiency.
    """

    def __init__(self,
                 X_test: pd.DataFrame,
                 df_for_ML: pd.DataFrame,
                 y_test: pd.Series,
                 y_pred: pd.Series,
                 upstream_dict: Optional[UpDownDict] = None,
                 downstream_dict: Optional[UpDownDict] = None,
                 correction_factory: Optional[Callable[[int], PredictionCorrectionPerSensor]] = None):
        self.X_test = X_test
        self.df_for_ML = df_for_ML
        self.y_test = y_test
        self.y_pred = y_pred

        if upstream_dict is None or downstream_dict is None:
            self.upstream_dict, self.downstream_dict = load_adjacency_dicts()
        else:
            self.upstream_dict = upstream_dict
            self.downstream_dict = downstream_dict

        self._correction_factory = correction_factory or self._default_correction_factory
        self._corr_cache: Dict[int, PredictionCorrectionPerSensor] = {}

    def clear_cache(self) -> None:
        self._corr_cache.clear()

    def plot(self,
             sensor_uid: int,
             correction_pipeline: Optional[Union[str, List[str], Callable, List[Callable]]] = None,
             show_pred: bool = True,
             show_corrected_pred: bool = True,
             title: str = "Predictions vs Time") -> None:
        try:
            sensor_id = self._get_sensor_id(sensor_uid)
            idx = self._get_indices_for_uid(sensor_uid)
        except ValueError as e:
            warnings.warn(str(e))
            return

        time_axis = self.df_for_ML.loc[idx, 'date']
        y_pred_s = self.y_pred.loc[idx]
        y_test_s = self.y_test.loc[idx]

        corr_obj = self._get_correction(sensor_uid)
        if correction_pipeline is None:
            correction_pipeline = corr_obj.naive_based_correction
        elif isinstance(correction_pipeline, str):
            correction_pipeline = getattr(corr_obj, self._resolve_correction_name(correction_pipeline))
        elif isinstance(correction_pipeline, list) and all(isinstance(c, str) for c in correction_pipeline):
            funcs = [getattr(corr_obj, self._resolve_correction_name(name)) for name in correction_pipeline]
            correction_pipeline = funcs

        y_corr_s = self._apply_corrections(y_pred_s, correction_pipeline)

        full_title = f"<b>{title} [{sensor_id}]</b>"
        self._plot_single(time_axis, y_test_s, y_pred_s, y_corr_s,
                          title=full_title,
                          show_pred=show_pred,
                          show_corrected_pred=show_corrected_pred)

    def plot_with_adjacent(self,
                           sensor_uid: int,
                           correction_pipeline: Optional[Union[str, List[str], Callable, List[Callable]]] = None,
                           show_pred: bool = True,
                           show_corrected_pred: bool = True,
                           title: str = "Predictions vs Time") -> None:
        try:
            sensor_id = self._get_sensor_id(sensor_uid)
        except ValueError as e:
            warnings.warn(str(e))
            return

        all_ids = [sensor_id]
        all_ids += self.upstream_dict.get(str(sensor_id), {}).get("upstream_sensor", [])
        all_ids += self.downstream_dict.get(str(sensor_id), {}).get("downstream_sensor", [])

        for sid in all_ids:
            uid = self._get_sensor_uid_from_id(sid)
            if uid is not None:
                self.plot(sensor_uid=uid,
                          correction_pipeline=correction_pipeline,
                          show_pred=show_pred,
                          show_corrected_pred=show_corrected_pred,
                          title=title)

    def _get_correction(self, sensor_uid: int) -> PredictionCorrectionPerSensor:
        if sensor_uid not in self._corr_cache:
            self._corr_cache[sensor_uid] = self._correction_factory(sensor_uid)
        return self._corr_cache[sensor_uid]

    def _default_correction_factory(self, sensor_uid: int) -> PredictionCorrectionPerSensor:
        return PredictionCorrectionPerSensor(
            X_test=self.X_test,
            y_test=self.y_test,
            df_for_ML=self.df_for_ML,
            sensor_uid=sensor_uid
        )

    def _resolve_correction_name(self, name: str) -> str:
        mapping = {
            "naive": "naive_based_correction",
            "rolling_median": "rolling_median_correction",
            "ewma": "ewma_smoothing",
            "constrained": "constrain_predictions",
            "kalman": "kalman_smoothing",
            "all": "apply_all_corrections"
            # Add more mappings here if you extend methods in PredictionCorrectionPerSensor
        }
        if name not in mapping:
            raise ValueError(f"Unsupported correction method name: '{name}'. Supported: {list(mapping.keys())}")
        return mapping[name]

    def _get_sensor_id(self, sensor_uid: int) -> str:
        match = self.df_for_ML[self.X_test['sensor_uid'] == sensor_uid]
        if match.empty:
            raise ValueError(f"Sensor UID {sensor_uid} not found.")
        return match['sensor_id'].iloc[0]

    def _get_sensor_uid_from_id(self, sensor_id: Union[str, int]) -> Optional[int]:
        match = self.df_for_ML[self.df_for_ML['sensor_id'] == sensor_id]
        return match['sensor_uid'].iloc[0] if not match.empty else None

    def _get_indices_for_uid(self, sensor_uid: int) -> pd.Index:
        idx = self.X_test.index[self.X_test['sensor_uid'] == sensor_uid]
        if idx.empty:
            raise ValueError(f"No data found for sensor_uid={sensor_uid}")
        return idx

    def _apply_corrections(self,
                           y_pred_s: pd.Series,
                           pipeline: Optional[Union[Callable, List[Callable]]]) -> Optional[pd.Series]:
        if pipeline is None:
            return None
        y_corr = y_pred_s.copy()
        if callable(pipeline):
            return pipeline(y_corr)
        for fn in pipeline:
            if not callable(fn):
                raise TypeError("Each item in correction pipeline must be callable.")
            y_corr = fn(y_corr)
        return y_corr

    def _plot_single(self,
                     time_axis: pd.Series,
                     y_test_s: pd.Series,
                     y_pred_s: pd.Series,
                     y_corr_s: Optional[pd.Series],
                     title: str,
                     show_pred: bool,
                     show_corrected_pred: bool) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name='Actual', line=dict(color='green')))
        if show_pred:
            fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name='Prediction',
                                     line=dict(color='blue', dash='dash')))
        if show_corrected_pred and y_corr_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name='Corrected',
                                     line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=title,
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=400
        )
        fig.show()
