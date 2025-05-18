import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Callable, List, Union, Dict, Tuple
from traffic_flow_package_src.helper_utils import load_adjacency_dicts
from traffic_flow_package_src.post_processing import PredictionCorrectionPerSensor
from traffic_flow_package_src.post_processing import UpDownDict
import warnings
import logging
from plotly.subplots import make_subplots
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                correction_pipeline: Optional[Union[str, List[Union[str, Tuple[str, Dict]]], Callable, List[Callable]]] = None,
                show_pred: bool = True,
                show_corrected_pred: bool = True,
                title: str = "Predictions vs Time",
                verbose: bool = False,
                export_path: Optional[str] = None,
                zoom_range: Optional[Tuple[str, str]] = None,
                y_range: Optional[Tuple[float, float]] = None,
                save: bool = True,
                export_html: bool = False) -> None:
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
            applied_names = []

            # Resolve correction pipeline into list of (function, kwargs) pairs
            if correction_pipeline is None:
                correction_pipeline = [(corr_obj.naive_based_correction, {})]
                applied_names.append("naive")
            elif isinstance(correction_pipeline, str):
                method = getattr(corr_obj, self._resolve_correction_name(correction_pipeline))
                correction_pipeline = [(method, {})]
                applied_names.append(correction_pipeline.__name__)
            elif isinstance(correction_pipeline, list):
                funcs = []
                for entry in correction_pipeline:
                    if isinstance(entry, str):
                        method = getattr(corr_obj, self._resolve_correction_name(entry))
                        funcs.append((method, {}))
                        applied_names.append(entry)
                    elif isinstance(entry, tuple) and isinstance(entry[0], str) and isinstance(entry[1], dict):
                        method = getattr(corr_obj, self._resolve_correction_name(entry[0]))
                        funcs.append((method, entry[1]))
                        applied_names.append(f"{entry[0]}({entry[1]})")
                    else:
                        raise ValueError(f"Invalid correction pipeline entry: {entry}")
                correction_pipeline = funcs

            y_corr_s = self._apply_corrections(y_pred_s, correction_pipeline)

            if show_corrected_pred and verbose:
                logging.info(f"[Sensor {sensor_uid}] Applied corrections: {applied_names}")

            full_title = f"<b>{title} [{sensor_id}]</b>"
            fig = self._plot_single(time_axis, y_test_s, y_pred_s, y_corr_s,
                                    title=full_title,
                                    show_pred=show_pred,
                                    show_corrected_pred=show_corrected_pred,
                                    x_range=zoom_range,
                                    y_range=y_range)

            if save and export_path:
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                if export_path.endswith(".html") or export_html:
                    fig.write_html(export_path)
                else:
                    fig.write_image(export_path)

    def _apply_corrections(self,
                           y_pred_s: pd.Series,
                           pipeline: Optional[List[Tuple[Callable, Dict]]]) -> Optional[pd.Series]:
        if pipeline is None:
            return None
        y_corr = y_pred_s.copy()
        for fn_entry in pipeline:
            if isinstance(fn_entry, tuple):
                fn, kwargs = fn_entry
                y_corr = pd.Series(fn(y_corr, **kwargs), index=y_corr.index)
            else:
                y_corr = pd.Series(fn_entry(y_corr), index=y_corr.index)
        return y_corr

    def plot_with_adjacent(self,
                       sensor_uid: int,
                       correction_pipeline: Optional[Union[str, List[Union[str, Tuple[str, Dict]]], Callable, List[Callable]]] = None,
                       show_pred: bool = True,
                       show_corrected_pred: bool = True,
                       title: str = "Predictions vs Time",
                       verbose: bool = False,
                       export_dir: Optional[str] = None,
                       zoom_range: Optional[Tuple[str, str]] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       save: bool = True,
                       export_html: bool = False) -> None:
        """
        Plot predictions for a main sensor and its first upstream/downstream neighbors.
        Stacked vertically with 16:9 aspect ratio and export support.
        """
        try:
            sensor_id = self._get_sensor_id(sensor_uid)
        except ValueError as e:
            warnings.warn(str(e))
            return

        upstream = self.upstream_dict.get(str(sensor_id), {}).get("upstream_sensor", [])
        downstream = self.downstream_dict.get(str(sensor_id), {}).get("downstream_sensor", [])
        upstream_id = upstream[0] if upstream else None
        downstream_id = downstream[0] if downstream else None

        layout_titles = ["Downstream Sensor", "Main Sensor", "Upstream Sensor"]
        sensors_to_plot = [downstream_id, sensor_id, upstream_id]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=layout_titles)

        for i, sid in enumerate(sensors_to_plot):
            if sid is None:
                continue
            uid = self._get_sensor_uid_from_id(sid)
            if uid is None:
                continue
            try:
                idx = self._get_indices_for_uid(uid)
                time_axis = self.df_for_ML.loc[idx, 'date']
                y_pred_s = self.y_pred.loc[idx]
                y_test_s = self.y_test.loc[idx]
                corr_obj = self._get_correction(uid)

                # Resolve pipeline
                applied_names = []
                if correction_pipeline is None:
                    resolved_pipeline = [(corr_obj.naive_based_correction, {})]
                    applied_names.append("naive")
                elif isinstance(correction_pipeline, str):
                    method = getattr(corr_obj, self._resolve_correction_name(correction_pipeline))
                    resolved_pipeline = [(method, {})]
                    applied_names.append(correction_pipeline)
                elif isinstance(correction_pipeline, list):
                    resolved_pipeline = []
                    for entry in correction_pipeline:
                        if isinstance(entry, str):
                            method = getattr(corr_obj, self._resolve_correction_name(entry))
                            resolved_pipeline.append((method, {}))
                            applied_names.append(entry)
                        elif isinstance(entry, tuple) and isinstance(entry[0], str) and isinstance(entry[1], dict):
                            method = getattr(corr_obj, self._resolve_correction_name(entry[0]))
                            resolved_pipeline.append((method, entry[1]))
                            applied_names.append(f"{entry[0]}({entry[1]})")
                        else:
                            raise ValueError(f"Invalid correction pipeline entry: {entry}")
                else:
                    raise ValueError("Invalid correction pipeline format.")

                y_corr_s = self._apply_corrections(y_pred_s, resolved_pipeline)

                if show_corrected_pred and verbose:
                    logging.info(f"[Sensor {sid}] Applied corrections: {applied_names}")

                row = i + 1
                fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name=f'{sid} Actual'), row=row, col=1)
                if show_pred:
                    fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name=f'{sid} Pred'), row=row, col=1)
                if show_corrected_pred and y_corr_s is not None:
                    fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name=f'{sid} Corrected'), row=row, col=1)
            except Exception as e:
                logging.warning(f"Could not plot sensor {sid}: {e}")

        fig.update_layout(
            title=f"<b>{title} [{sensor_id} and Adjacent]</b>",
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=720,
            width=1280,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            )
        )

        if zoom_range:
            fig.update_xaxes(range=list(zoom_range))
        if y_range:
            fig.update_yaxes(range=list(y_range))

        if save and export_dir:
            os.makedirs(export_dir, exist_ok=True)
            export_file = os.path.join(export_dir, f"{sensor_id}_w_adj.html" if export_html else f"{sensor_id}_w_adj.png")
            if export_html:
                fig.write_html(export_file)
            else:
                fig.write_image(export_file, width=1280, height=720)

        fig.show()
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
        }
        if name not in mapping:
            raise ValueError(f"Unsupported correction method name: '{name}'. Supported: {list(mapping.keys())}")
        return mapping[name]

    def _get_sensor_id(self, sensor_uid: int) -> str:
        sensor_idx = self.X_test.loc[self.X_test['sensor_uid'] == sensor_uid].index
        match = self.df_for_ML.loc[sensor_idx]
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

    def _plot_single(self,
                     time_axis: pd.Series,
                     y_test_s: pd.Series,
                     y_pred_s: pd.Series,
                     y_corr_s: Optional[pd.Series],
                     title: str,
                     show_pred: bool,
                     show_corrected_pred: bool,
                     x_range: Optional[Tuple[str, str]] = None,
                     y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name='Actual', line=dict(color='green')))
        if show_pred:
            fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name='Prediction',
                                     line=dict(color='blue', dash='dash')))
        if show_corrected_pred and y_corr_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name='Corrected',
                                     line=dict(color='red', dash='dash')))

        layout_config = dict(
            title=title,
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=400
        )
        if x_range is not None:
            layout_config['xaxis'] = dict(range=list(x_range))
        if y_range is not None:
            layout_config['yaxis'] = dict(range=list(y_range))

        fig.update_layout(**layout_config)
        fig.show()
        return fig
    
    
    
    
    



