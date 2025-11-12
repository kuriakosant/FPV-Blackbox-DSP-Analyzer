"""
parser.py — Blackbox Log Parser
================================
Responsible for ingesting raw Betaflight Blackbox exports (CSV or decoded BBL)
and returning a clean, normalized DataFrame ready for DSP processing.

Betaflight logs exported via Blackbox Explorer or `blackbox_decode` contain:
  - Timestamp in microseconds
  - Gyro axes: gyroADC[0] (roll), gyroADC[1] (pitch), gyroADC[2] (yaw)
  - PID terms: axisP[], axisI[], axisD[]
  - Motor outputs: motor[0..3]
  - RC commands, setpoint, etc.

TODO:
  - Add support for raw .bbl binary parsing via blackbox_decode subprocess
  - Add multi-log batch loader
  - Add validation for missing columns / corrupted logs
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names the rest of the pipeline expects
GYRO_COLS = ["gyroADC[0]", "gyroADC[1]", "gyroADC[2]"]
AXIS_LABELS = {
    "gyroADC[0]": "Roll",
    "gyroADC[1]": "Pitch",
    "gyroADC[2]": "Yaw",
}
TIME_COL = "time (us)"


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a Betaflight Blackbox CSV export and return a cleaned DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .csv file exported from Blackbox Explorer.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with a float64 time column (microseconds),
        gyro columns in raw ADC units, and all other original columns preserved.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    logger.info("Loading Blackbox CSV: %s", path.name)

    # Betaflight CSVs can have leading spaces in header names — strip them
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    _validate_columns(df, path.name)

    # Ensure time column is numeric
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df.dropna(subset=[TIME_COL], inplace=True)
    df.sort_values(TIME_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Coerce gyro columns to float
    for col in GYRO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Loaded %d samples | Duration: %.2f s",
        len(df),
        (df[TIME_COL].iloc[-1] - df[TIME_COL].iloc[0]) / 1e6,
    )
    return df


def estimate_sample_rate(df: pd.DataFrame) -> float:
    """
    Estimate the gyro sample rate from the time column.

    Betaflight logs are typically sampled at 1kHz, 2kHz, 4kHz, or 8kHz.
    The actual rate may differ from the configured rate due to USB logging
    overhead.

    Returns
    -------
    float
        Estimated sample rate in Hz.
    """
    time_us = df[TIME_COL].values
    dt_us = np.diff(time_us)
    # Median is robust against dropped frames / outliers
    median_dt_us = float(np.median(dt_us))
    sample_rate = 1e6 / median_dt_us
    logger.info("Estimated sample rate: %.1f Hz", sample_rate)
    return sample_rate


def _validate_columns(df: pd.DataFrame, filename: str) -> None:
    """Raise ValueError if mandatory columns are absent."""
    missing = [c for c in [TIME_COL] + GYRO_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{filename}] Missing required columns: {missing}\n"
            "Make sure you exported the CSV from Blackbox Explorer with "
            "all fields enabled."
        )
