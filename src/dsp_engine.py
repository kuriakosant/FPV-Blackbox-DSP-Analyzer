"""
dsp_engine.py — DSP & FFT Analysis Core
=========================================
The heart of the analyzer. Converts time-domain gyro data into the frequency
domain using Fast Fourier Transforms (FFT) and computes diagnostics.

Key concepts
------------
FFT (Fast Fourier Transform)
    Decomposes the gyro signal x[n] into its constituent frequencies:

        X[k] = Σ x[n] · e^(−i·2π·k·n / N)   for k = 0 … N−1

    This turns a wiggly gyro trace into a spectrum showing *which frequencies*
    carry the most energy — propeller harmonics, motor noise, frame resonances.

Power Spectral Density (PSD)
    |X[k]|² / N — the squared magnitude of each frequency bin, representing
    how much power exists at that frequency. We plot this on a dB scale.

Nyquist Limit
    The highest frequency resolvable is fs/2 (half the sample rate). At 8 kHz
    logging, that's 4 kHz. Any energy above that folds back (aliasing).

D-Term Oscillation
    Betaflight's D-term amplifies high-frequency gyro noise. If excessive noise
    lives in the 200–500 Hz band it drives motors hard at high duty cycle →
    heat, vibration, potential ESC desyncs. We flag this automatically.

TODO:
  - Add spectrogram (short-time FFT) computation
  - Add Welch's PSD estimator for smoother spectra
  - Add notch filter effectiveness comparison (pre vs post filter)
  - Add batch analysis across multiple logs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.parser import GYRO_COLS, TIME_COL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Thresholds (community-derived FPV baselines)
# ──────────────────────────────────────────────
DTERM_NOISE_BAND_HZ = (200.0, 500.0)   # Typical D-term resonance window
DTERM_WARN_THRESHOLD_DB = -20.0        # dB relative to peak — tune to taste
HIGH_FREQ_WARN_HZ = 300.0              # Absolute Hz above which we start caring


@dataclass
class AxisFFTResult:
    """Stores FFT results for a single gyro axis."""
    axis: str                           # e.g. "Roll", "Pitch", "Yaw"
    frequencies: np.ndarray             # Hz array
    psd_db: np.ndarray                  # Power spectral density in dBFS
    peak_freq_hz: float                 # Dominant noise frequency
    peak_power_db: float                # Power at dominant frequency
    dterm_warning: bool = False         # True if D-term oscillation flagged
    dterm_warning_message: str = ""


@dataclass
class AnalysisReport:
    """Aggregates results across all three gyro axes."""
    sample_rate_hz: float
    axes: list[AxisFFTResult] = field(default_factory=list)

    @property
    def any_warning(self) -> bool:
        return any(ax.dterm_warning for ax in self.axes)


def run_fft_analysis(df: pd.DataFrame, sample_rate_hz: float) -> AnalysisReport:
    """
    Run FFT-based noise analysis on all three gyro axes.

    Uses Welch's method (averaged periodogram) for a smoother, more reliable
    spectrum than a single FFT on the full signal. Each segment is windowed
    with a Hann window to reduce spectral leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Blackbox DataFrame from parser.load_csv().
    sample_rate_hz : float
        Gyro sample rate in Hz (from parser.estimate_sample_rate()).

    Returns
    -------
    AnalysisReport
        Object containing per-axis FFT results and any diagnostic warnings.
    """
    report = AnalysisReport(sample_rate_hz=sample_rate_hz)

    for col in GYRO_COLS:
        if col not in df.columns:
            logger.warning("Column %s not found, skipping.", col)
            continue

        signal = df[col].dropna().values.astype(np.float64)
        axis_label = col  # e.g. "gyroADC[0]"

        from src.parser import AXIS_LABELS
        axis_name = AXIS_LABELS.get(col, col)

        result = _analyse_axis(signal, sample_rate_hz, axis_name)
        report.axes.append(result)

        if result.dterm_warning:
            logger.warning("⚠  %s", result.dterm_warning_message)
        else:
            logger.info("✓  %s axis — peak @ %.1f Hz (%.1f dB)",
                        axis_name, result.peak_freq_hz, result.peak_power_db)

    return report


def _analyse_axis(
    signal: np.ndarray,
    sample_rate_hz: float,
    axis_name: str,
) -> AxisFFTResult:
    """
    Compute Welch PSD for one axis and run D-term oscillation heuristics.

    Parameters
    ----------
    signal : np.ndarray
        Raw gyro ADC samples for this axis.
    sample_rate_hz : float
        Sampling frequency in Hz.
    axis_name : str
        Human-readable axis label ("Roll" / "Pitch" / "Yaw").

    Returns
    -------
    AxisFFTResult
    """
    # Welch's method: nperseg controls frequency resolution vs. variance trade-off
    nperseg = min(4096, len(signal) // 4)
    freqs, psd = welch(signal, fs=sample_rate_hz, nperseg=nperseg, window="hann")

    # Convert to dB (add small epsilon to avoid log(0))
    psd_db = 10.0 * np.log10(psd + 1e-12)

    # Find dominant noise peak
    peak_idx = int(np.argmax(psd_db))
    peak_freq = float(freqs[peak_idx])
    peak_power = float(psd_db[peak_idx])

    # ── D-term oscillation heuristic ──────────────────────────────────────
    lo, hi = DTERM_NOISE_BAND_HZ
    band_mask = (freqs >= lo) & (freqs <= hi)
    dterm_warning = False
    dterm_msg = ""

    if band_mask.any():
        band_peak_power = float(psd_db[band_mask].max())
        band_peak_freq = float(freqs[band_mask][np.argmax(psd_db[band_mask])])
        relative_power = band_peak_power - peak_power  # 0 dB = as loud as dominant peak

        if relative_power >= DTERM_WARN_THRESHOLD_DB and band_peak_freq >= HIGH_FREQ_WARN_HZ:
            dterm_warning = True
            dterm_msg = (
                f"[{axis_name}] ⚠  WARNING: D-term gain too high — "
                f"high-frequency oscillation detected at {band_peak_freq:.0f} Hz "
                f"({band_peak_power:.1f} dB). Potential motor overheat risk. "
                f"Consider reducing D-term or enabling a notch filter at {band_peak_freq:.0f} Hz."
            )

    return AxisFFTResult(
        axis=axis_name,
        frequencies=freqs,
        psd_db=psd_db,
        peak_freq_hz=peak_freq,
        peak_power_db=peak_power,
        dterm_warning=dterm_warning,
        dterm_warning_message=dterm_msg,
    )


def compute_spectrogram(
    df: pd.DataFrame,
    col: str,
    sample_rate_hz: float,
    nperseg: int = 512,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Short-Time Fourier Transform (STFT) spectrogram for one axis.

    A spectrogram shows how the noise spectrum *changes over time* — invaluable
    for spotting resonances that only appear at certain throttle levels.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Blackbox DataFrame.
    col : str
        Column name to analyse (e.g. "gyroADC[0]").
    sample_rate_hz : float
        Gyro sample rate in Hz.
    nperseg : int
        FFT window size (samples). Larger = better freq resolution, worse time res.
    noverlap : int or None
        Samples overlapping between windows. Default: nperseg // 2.

    Returns
    -------
    (t, f, Sxx_db)
        t : time bins (seconds from log start)
        f : frequency bins (Hz)
        Sxx_db : power in dB, shape (len(f), len(t))
    """
    from scipy.signal import spectrogram as sp_spectrogram

    signal = df[col].dropna().values.astype(np.float64)
    time_us = df[TIME_COL].values[: len(signal)]

    if noverlap is None:
        noverlap = nperseg // 2

    f, t_rel, Sxx = sp_spectrogram(
        signal,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    # Map relative time bins to absolute seconds from log start
    t_abs = (time_us[0] / 1e6) + t_rel

    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    return t_abs, f, Sxx_db
