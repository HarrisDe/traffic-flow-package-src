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

class SensorPredictionPlotterResidualCorrection:
    def __init__(
        self,
        X_test: pd.DataFrame,
        df_for_ML: pd.DataFrame,
        y_test: pd.Series,
        y_pred: pd.Series,
        y_pred_residual: Optional[pd.Series] = None,
        upstream_dict: Optional[UpDownDict] = None,
        downstream_dict: Optional[UpDownDict] = None,
        correction_factory: Optional[Callable[[int], PredictionCorrectionPerSensor]] = None
    ):
        self.X_test = X_test
        self.df_for_ML = df_for_ML
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_residual = y_pred_residual

        if upstream_dict is None or downstream_dict is None:
            self.upstream_dict, self.downstream_dict = load_adjacency_dicts()
        else:
            self.upstream_dict = upstream_dict
            self.downstream_dict = downstream_dict

        self._correction_factory = correction_factory or self._default_correction_factory
        self._corr_cache: Dict[int, PredictionCorrectionPerSensor] = {}

    def plot(self,
             sensor_uid: int,
             correction_pipeline=None,
             show_pred=True,
             show_corrected_pred=True,
             show_residual_pred=False,
             show_corrected_residual_pred=False,
             title="Predictions vs Time",
             verbose=False,
             export_path=None,
             zoom_range=None,
             y_range=None,
             save=True,
             export_html=False):

        sensor_id = self._get_sensor_id(sensor_uid)
        idx = self._get_indices_for_uid(sensor_uid)

        time_axis = self.df_for_ML.loc[idx, 'date']
        y_pred_s = self.y_pred.loc[idx]
        y_test_s = self.y_test.loc[idx]
        y_pred_res_s = self.y_pred_residual.loc[idx] if self.y_pred_residual is not None else None

        corr_obj = self._get_correction(sensor_uid)

        # Handle correction pipeline
        correction_pipeline = self._resolve_pipeline(correction_pipeline, corr_obj, verbose, sensor_uid)
        y_corr_s = self._apply_corrections(y_pred_s, correction_pipeline)
        y_corr_res_s = self._apply_corrections(y_pred_res_s, correction_pipeline) if y_pred_res_s is not None else None

        title_full = f"<b>{title} [{sensor_id}]</b>"
        fig = self._plot_single(
            time_axis, y_test_s, y_pred_s, y_corr_s,
            y_pred_res_s if show_residual_pred else None,
            y_corr_res_s if show_corrected_residual_pred else None,
            title=title_full, show_pred=show_pred,
            show_corrected_pred=show_corrected_pred,
            x_range=zoom_range, y_range=y_range
        )

        if save and export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            if export_path.endswith(".html") or export_html:
                fig.write_html(export_path)
            else:
                fig.write_image(export_path)

    def plot_with_adjacent(self, sensor_uid, correction_pipeline=None,
                           show_pred=True, show_corrected_pred=True,
                           show_residual_pred=False, show_corrected_residual_pred=False,
                           title="Predictions vs Time", verbose=False,
                           export_dir=None, zoom_range=None, y_range=None,
                           save=True, export_html=False):

        sensor_id = self._get_sensor_id(sensor_uid)

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
            idx = self._get_indices_for_uid(uid)
            time_axis = self.df_for_ML.loc[idx, 'date']
            y_pred_s = self.y_pred.loc[idx]
            y_test_s = self.y_test.loc[idx]
            y_pred_res_s = self.y_pred_residual.loc[idx] if self.y_pred_residual is not None else None

            corr_obj = self._get_correction(uid)
            pipeline = self._resolve_pipeline(correction_pipeline, corr_obj, verbose, uid)
            y_corr_s = self._apply_corrections(y_pred_s, pipeline)
            y_corr_res_s = self._apply_corrections(y_pred_res_s, pipeline) if y_pred_res_s is not None else None

            row = i + 1
            fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name=f'{sid} Actual'), row=row, col=1)
            if show_pred:
                fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name=f'{sid} Pred'), row=row, col=1)
            if show_corrected_pred:
                fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name=f'{sid} Corrected'), row=row, col=1)
            if show_residual_pred and y_pred_res_s is not None:
                fig.add_trace(go.Scatter(x=time_axis, y=y_pred_res_s, mode='lines', name=f'{sid} Residual Pred'), row=row, col=1)
            if show_corrected_residual_pred and y_corr_res_s is not None:
                fig.add_trace(go.Scatter(x=time_axis, y=y_corr_res_s, mode='lines', name=f'{sid} Corrected Residual'), row=row, col=1)

        fig.update_layout(
            title=f"<b>{title} [{sensor_id} and Adjacent]</b>",
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=720,
            width=1280,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10))
        )

        if zoom_range:
            fig.update_xaxes(range=list(zoom_range))
        if y_range:
            fig.update_yaxes(range=list(y_range))

        if save and export_dir:
            os.makedirs(export_dir, exist_ok=True)
            file = os.path.join(export_dir, f"{sensor_id}_w_adj.html" if export_html else f"{sensor_id}_w_adj.png")
            if export_html:
                fig.write_html(file)
            else:
                fig.write_image(file, width=1280, height=720)

        fig.show()

    def _apply_corrections(self, y_pred_s, pipeline):
        if y_pred_s is None or pipeline is None:
            return None
        y_corr = y_pred_s.copy()
        for fn, kwargs in pipeline:
            y_corr = pd.Series(fn(y_corr, **kwargs), index=y_corr.index)
        return y_corr

    def _resolve_pipeline(self, correction_pipeline, corr_obj, verbose, sensor_uid):
        applied_names = []
        if correction_pipeline is None:
            return [(corr_obj.naive_based_correction, {})]
        if isinstance(correction_pipeline, str):
            return [(getattr(corr_obj, self._resolve_correction_name(correction_pipeline)), {})]
        if isinstance(correction_pipeline, list):
            resolved = []
            for entry in correction_pipeline:
                if isinstance(entry, str):
                    resolved.append((getattr(corr_obj, self._resolve_correction_name(entry)), {}))
                elif isinstance(entry, tuple) and isinstance(entry[0], str):
                    resolved.append((getattr(corr_obj, self._resolve_correction_name(entry[0])), entry[1]))
            return resolved
        raise ValueError("Unsupported correction pipeline format.")

    def _get_correction(self, sensor_uid: int):
        if sensor_uid not in self._corr_cache:
            self._corr_cache[sensor_uid] = self._correction_factory(sensor_uid)
        return self._corr_cache[sensor_uid]

    def _default_correction_factory(self, sensor_uid: int):
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
            raise ValueError(f"Unsupported correction method name: '{name}'.")
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

    def _plot_single(self, time_axis, y_test_s, y_pred_s, y_corr_s,
                     y_pred_res_s=None, y_corr_res_s=None,
                     title=None, show_pred=True, show_corrected_pred=True,
                     x_range=None, y_range=None):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y_test_s, mode='lines', name='Actual', line=dict(color='green')))
        if show_pred:
            fig.add_trace(go.Scatter(x=time_axis, y=y_pred_s, mode='lines', name='Prediction', line=dict(color='blue', dash='dash')))
        if show_corrected_pred and y_corr_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_corr_s, mode='lines', name='Corrected', line=dict(color='red', dash='dash')))
        if y_pred_res_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_pred_res_s, mode='lines', name='Residual Pred', line=dict(color='orange', dash='dot')))
        if y_corr_res_s is not None:
            fig.add_trace(go.Scatter(x=time_axis, y=y_corr_res_s, mode='lines', name='Corrected Residual', line=dict(color='black', dash='dot')))

        layout_config = dict(
            title=title,
            xaxis_title="<b>Datetime</b>",
            yaxis_title="<b>Speed [kph]</b>",
            template='plotly_white',
            height=400
        )
        if x_range:
            layout_config['xaxis'] = dict(range=list(x_range))
        if y_range:
            layout_config['yaxis'] = dict(range=list(y_range))

        fig.update_layout(**layout_config)
        fig.show()
        return fig