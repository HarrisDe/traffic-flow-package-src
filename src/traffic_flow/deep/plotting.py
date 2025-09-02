from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

class TimeSeriesPlotter:
    """Overlay true vs predicted as time series."""
    def plot(
        self,
        df: pd.DataFrame,
        *,
        time_col: str,
        y_true_col: str,
        y_pred_col: str,
        split_time=None,
        title: str = "Time Series: true vs predicted",
    ):
        d = df.sort_values(time_col).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[time_col], y=d[y_true_col], mode="lines", name="True"))
        fig.add_trace(go.Scatter(x=d[time_col], y=d[y_pred_col], mode="lines", name="Predicted"))

        if split_time is not None:
            fig.add_shape(type="line", x0=split_time, x1=split_time, y0=0, y1=1, yref="paper",
                          line=dict(dash="dash"))
            fig.add_annotation(x=split_time, y=1, yref="paper", text=f"Split @ {split_time}",
                               showarrow=False, xanchor="left")

        fig.update_layout(
            title=title, xaxis_title="Time", yaxis_title="Value",
            hovermode="x unified", xaxis_rangeslider_visible=True
        )
        fig.show()


class ScatterPlotterSeaborn:
    """
    Simple density-revealing scatter (no cmap).
    - Bivariate KDE background with a single color & varying alpha.
    - Light scatter overlay so individual points are visible.
    - x=y line, square axes, PNG save support (creates folders).
    """

    def __init__(self):
        pass

    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        alpha: float = 0.6,
        title: str = "Predicted vs True",
        save_path: Optional[str] = None,
        levels: int = 50,        # KDE contour levels
        thresh: float = 0.02,    # lower density threshold (0..1)
        s: int = 8,              # scatter marker size
        color: Optional[str] = None,  # single color; None -> seaborn default
        bw_adjust: float = 1.0,  # KDE bandwidth (lower = wigglier, higher = smoother)
        show: bool = True,
    ):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # keep only finite points
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]; y_pred = y_pred[mask]
        if y_true.size == 0:
            raise ValueError("No finite data points to plot.")

        lo = float(np.min([y_true.min(), y_pred.min()]))
        hi = float(np.max([y_true.max(), y_pred.max()]))
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        lo -= pad; hi += pad

        # style
        sns.set_theme(style="darkgrid", context="talk")
        if color is None:
            color = sns.color_palette()[0]  # first color in current palette

        fig, ax = plt.subplots(figsize=(6.4, 6.4))

        # --- density background (single-color, no cmap) ---
        # sns.kdeplot(
        #     x=y_true, y=y_pred,
        #     fill=True, color=color, levels=levels, thresh=thresh,
        #     bw_adjust=bw_adjust,
        #     alpha=0.95, ax=ax
        # )

        # --- light scatter overlay ---
        sns.scatterplot(
            x=y_true, y=y_pred,
            ax=ax, s=s, alpha=alpha,
            color=color, edgecolor=None, linewidth=0
        )

        # x = y reference line
        ax.plot([lo, hi], [lo, hi], "--", color="red", lw=2, label="x = y")

        # axes & labels
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("True", fontweight="bold")
        ax.set_ylabel("Predicted", fontweight="bold")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()

        # save handling (PNG only)
        if save_path:
            p = Path(save_path)
            if p.suffix.lower() != ".png":
                raise ValueError(f"save_path must end with .png (got '{p.suffix}')")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)