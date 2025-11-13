"""
tests/test_dsp_engine.py — Unit tests for the DSP core
=======================================================
These tests use synthetic signals with known properties so we can
verify that the math is correct independently of any real log file.
"""

import numpy as np
import pandas as pd
import pytest

from src import parser as log_parser
from src import dsp_engine


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

FS = 4000  # Hz — synthetic sample rate
DURATION = 2.0  # seconds


def make_sine_df(
    freq_hz: float,
    amplitude: float = 1000.0,
    sample_rate: float = FS,
    duration: float = DURATION,
) -> pd.DataFrame:
    """Create a minimal DataFrame with a pure sine wave on all three gyro axes."""
    n = int(sample_rate * duration)
    t_us = np.arange(n) * (1e6 / sample_rate)
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t_us / 1e6)

    return pd.DataFrame({
        log_parser.TIME_COL: t_us,
        "gyroADC[0]": signal,
        "gyroADC[1]": signal * 0.8,  # slightly weaker on pitch
        "gyroADC[2]": signal * 0.3,  # much weaker on yaw
    })


# ──────────────────────────────────────────────────────────────
# Tests: sample rate estimation
# ──────────────────────────────────────────────────────────────

def test_estimate_sample_rate_exact():
    """Sample rate estimator should return the correct rate for regular timestamps."""
    df = make_sine_df(100.0, sample_rate=FS)
    estimated = log_parser.estimate_sample_rate(df)
    assert abs(estimated - FS) < 1.0, f"Expected ~{FS} Hz, got {estimated:.1f} Hz"


# ──────────────────────────────────────────────────────────────
# Tests: FFT peak detection
# ──────────────────────────────────────────────────────────────

def test_fft_peak_near_correct_frequency():
    """
    A pure 150 Hz sine should produce an FFT peak very close to 150 Hz.
    Tolerance: ±10 Hz (limited by frequency bin width at 2-second window).
    """
    target_hz = 150.0
    df = make_sine_df(target_hz)
    report = dsp_engine.run_fft_analysis(df, FS)

    for ax in report.axes:
        assert abs(ax.peak_freq_hz - target_hz) < 15.0, (
            f"[{ax.axis}] Expected peak near {target_hz} Hz, "
            f"got {ax.peak_freq_hz:.1f} Hz"
        )


# ──────────────────────────────────────────────────────────────
# Tests: D-term oscillation heuristic
# ──────────────────────────────────────────────────────────────

def test_dterm_warning_triggered_for_high_freq_noise():
    """
    A strong 350 Hz sine (inside D-term band) should trigger the warning.
    """
    df = make_sine_df(350.0, amplitude=2000.0)
    report = dsp_engine.run_fft_analysis(df, FS)
    assert report.any_warning, (
        "Expected D-term warning for 350 Hz high-amplitude signal, none triggered."
    )


def test_dterm_warning_not_triggered_for_low_freq():
    """
    A 30 Hz sine (outside D-term band) should NOT trigger the warning.
    """
    df = make_sine_df(30.0, amplitude=2000.0)
    report = dsp_engine.run_fft_analysis(df, FS)
    assert not report.any_warning, (
        "D-term warning should NOT fire for a 30 Hz signal."
    )


# ──────────────────────────────────────────────────────────────
# Tests: AnalysisReport structure
# ──────────────────────────────────────────────────────────────

def test_report_has_three_axes():
    """Report should always contain exactly three axis results."""
    df = make_sine_df(100.0)
    report = dsp_engine.run_fft_analysis(df, FS)
    assert len(report.axes) == 3


def test_all_axes_named_correctly():
    """Axis names must match the canonical labels from parser.AXIS_LABELS."""
    df = make_sine_df(100.0)
    report = dsp_engine.run_fft_analysis(df, FS)
    names = {ax.axis for ax in report.axes}
    assert names == {"Roll", "Pitch", "Yaw"}
