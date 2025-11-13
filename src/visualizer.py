"""
visualizer.py — Plotting & Reporting
======================================
Turns DSP results from dsp_engine.py into charts.

Chart types
-----------
1. Noise Heatmap (PSD overlay)
   All three axes plotted on one frequency-domain chart. Colored bands
   highlight the D-term risk zone (200–500 Hz) and typical propeller
   harmonics.

2. Spectrogram
   2-D plot of frequency vs. time, colored by power (dBFS). Reveals
   whether noise spikes are throttle-dependent.

3. Summary Dashboard
   Multi-panel Matplotlib figure suitable for PDF export / sharing.

TODO:
  - Add interactive Plotly versions of each chart
  - Add filter effectiveness overlay (pre vs post)
  - Add PDF report generator
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.dsp_engine import AnalysisReport, DTERM_NOISE_BAND_HZ

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Visual style constants
# ──────────────────────────────────────────────
AXIS_COLORS = {"Roll": "#00BFFF", "Pitch": "#FF6B6B", "Yaw": "#FFD700"}
STYLE = "dark_background"
DTERM_BAND_COLOR = "rgba(255,100,0,0.15)"   # Plotly RGBA string


def plot_noise_heatmap(
    report: AnalysisReport,
    max_freq_hz: float = 1000.0,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Power Spectral Density for all gyro axes on a single chart.

    The chart highlights:
      - The D-term risk zone (shaded orange band, 200–500 Hz)
      - Each axis's dominant peak (annotated with frequency label)
      - Warning markers on axes that triggered the D-term heuristic

    Parameters
    ----------
    report : AnalysisReport
        Output from dsp_engine.run_fft_analysis().
    max_freq_hz : float
        Upper frequency limit for the x-axis.
    output_path : str or Path, optional
        Save figure to this path (PNG/PDF). If None, not saved.
    show : bool
        Display the interactive Matplotlib window. Set False for batch mode.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0D0D0D")
        ax.set_facecolor("#0D0D0D")

        # ── D-term risk-zone shading ──────────────────────────
        lo, hi = DTERM_NOISE_BAND_HZ
        ax.axvspan(lo, hi, color="#FF6500", alpha=0.08, label="_D-term zone")
        ax.axvline(lo, color="#FF6500", linewidth=0.6, alpha=0.5, linestyle="--")
        ax.axvline(hi, color="#FF6500", linewidth=0.6, alpha=0.5, linestyle="--")
        ax.text(
            (lo + hi) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] < -100 else -100,
            "D-term\nrisk zone",
            ha="center", va="bottom", color="#FF6500",
            fontsize=7, alpha=0.7,
        )

        # ── Per-axis PSD lines ────────────────────────────────
        for axis_result in report.axes:
            mask = axis_result.frequencies <= max_freq_hz
            freqs = axis_result.frequencies[mask]
            psd = axis_result.psd_db[mask]
            color = AXIS_COLORS.get(axis_result.axis, "white")

            label = axis_result.axis
            if axis_result.dterm_warning:
                label += "  ⚠"

            ax.plot(freqs, psd, color=color, linewidth=1.2, alpha=0.9, label=label)

            # Annotate the dominant peak
            peak_idx = int(np.argmax(psd))
            ax.annotate(
                f"{freqs[peak_idx]:.0f} Hz",
                xy=(freqs[peak_idx], psd[peak_idx]),
                xytext=(8, 4),
                textcoords="offset points",
                color=color,
                fontsize=7,
                alpha=0.85,
            )

        # ── Labels & legend ───────────────────────────────────
        ax.set_xlabel("Frequency (Hz)", color="0.8")
        ax.set_ylabel("Power Spectral Density (dB)", color="0.8")
        ax.set_title(
            "Gyro Noise Heatmap — Frequency Domain Analysis\n"
            f"Sample rate: {report.sample_rate_hz:.0f} Hz  |  "
            f"Nyquist: {report.sample_rate_hz / 2:.0f} Hz",
            color="white", fontsize=11, pad=12,
        )
        ax.set_xlim(0, max_freq_hz)
        ax.tick_params(colors="0.6")
        ax.spines[:].set_color("0.2")
        legend = ax.legend(
            loc="upper right", framealpha=0.2, labelcolor="white",
            facecolor="#1A1A1A", edgecolor="0.3",
        )

        if report.any_warning:
            fig.text(
                0.5, 0.01,
                "⚠  D-term oscillation detected — see axis labels above",
                ha="center", color="#FF6500", fontsize=9,
            )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved heatmap → %s", output_path)

        if show:
            plt.show()

    return fig


def plot_spectrogram(
    t: np.ndarray,
    f: np.ndarray,
    Sxx_db: np.ndarray,
    axis_name: str,
    max_freq_hz: float = 1000.0,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a 2-D spectrogram (frequency vs. time) for one gyro axis.

    Parameters
    ----------
    t : np.ndarray
        Time axis in seconds (from dsp_engine.compute_spectrogram).
    f : np.ndarray
        Frequency axis in Hz.
    Sxx_db : np.ndarray
        Power in dB, shape (len(f), len(t)).
    axis_name : str
        Label for the chart title ("Roll" / "Pitch" / "Yaw").
    max_freq_hz : float
        Upper frequency limit for the y-axis.
    output_path : str or Path, optional
        Save figure to disk.
    show : bool
        Display the Matplotlib window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    freq_mask = f <= max_freq_hz
    f_plot = f[freq_mask]
    Sxx_plot = Sxx_db[freq_mask, :]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0D0D0D")

        im = ax.pcolormesh(
            t, f_plot, Sxx_plot,
            shading="gouraud",
            cmap="inferno",
            norm=mcolors.Normalize(vmin=Sxx_plot.min(), vmax=Sxx_plot.max()),
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Power (dB)", color="0.8")
        cbar.ax.yaxis.set_tick_params(color="0.6")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="0.6")

        # D-term band marker
        lo, hi = DTERM_NOISE_BAND_HZ
        ax.axhline(lo, color="#FF6500", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(hi, color="#FF6500", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.fill_between(
            [t.min(), t.max()], lo, hi,
            color="#FF6500", alpha=0.06, label="D-term risk zone",
        )

        ax.set_xlabel("Time (s)", color="0.8")
        ax.set_ylabel("Frequency (Hz)", color="0.8")
        ax.set_title(
            f"Gyro Spectrogram — {axis_name} Axis\n"
            "(STFT: frequency content over time)",
            color="white", fontsize=11, pad=10,
        )
        ax.tick_params(colors="0.6")
        ax.spines[:].set_color("0.2")
        legend = ax.legend(loc="upper right", framealpha=0.2, labelcolor="white",
                           facecolor="#1A1A1A", edgecolor="0.3")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved spectrogram → %s", output_path)

        if show:
            plt.show()

    return fig
