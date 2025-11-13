#!/usr/bin/env python3
"""
main.py — FPV Blackbox DSP Analyzer CLI Entry Point
=====================================================
Usage examples:

    # Basic FFT heatmap
    python main.py analyze data/my_flight.csv

    # Specify a custom sample rate (override auto-detect)
    python main.py analyze data/my_flight.csv --sample-rate 4000

    # Generate and save a spectrogram for the roll axis
    python main.py spectrogram data/my_flight.csv --axis roll --output exports/roll_spec.png

    # Run analysis without displaying windows (batch/CI mode)
    python main.py analyze data/my_flight.csv --no-show --output exports/heatmap.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

# ── Local imports
from src import parser as log_parser
from src import dsp_engine
from src import visualizer

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("fpv_analyzer")


# ──────────────────────────────────────────────────────────────
# CLI setup
# ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0-alpha", prog_name="FPV Blackbox DSP Analyzer")
def cli():
    """
    🛸 FPV Blackbox DSP Analyzer

    Analyze Betaflight Blackbox logs using FFT and DSP to diagnose
    noise, D-term oscillations, and filter effectiveness.
    """
    _print_banner()


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--sample-rate", "-s", type=float, default=None,
              help="Override auto-detected sample rate (Hz).")
@click.option("--max-freq", "-f", type=float, default=1000.0,
              help="Upper frequency limit for the heatmap chart (Hz). Default: 1000.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save heatmap PNG to this path.")
@click.option("--no-show", is_flag=True, default=False,
              help="Suppress the interactive chart window (useful for batch/CI).")
def analyze(log_file, sample_rate, max_freq, output, no_show):
    """
    Run FFT noise analysis on a Blackbox log and display a Noise Heatmap.

    LOG_FILE — path to a Betaflight Blackbox CSV export.
    """
    log_path = Path(log_file)
    console.rule(f"[bold cyan]Analyzing:[/] {log_path.name}")

    # 1. Parse
    df = log_parser.load_csv(log_path)

    # 2. Sample rate
    fs = sample_rate or log_parser.estimate_sample_rate(df)
    console.print(f"  Sample rate : [bold]{fs:.0f} Hz[/]")
    console.print(f"  Nyquist     : [bold]{fs / 2:.0f} Hz[/]")
    console.print(f"  Samples     : [bold]{len(df):,}[/]")

    # 3. FFT analysis
    with console.status("[bold green]Running FFT analysis…"):
        report = dsp_engine.run_fft_analysis(df, fs)

    # 4. Print summary
    _print_report_summary(report)

    # 5. Plot
    out_path = Path(output) if output else None
    visualizer.plot_noise_heatmap(
        report,
        max_freq_hz=max_freq,
        output_path=out_path,
        show=not no_show,
    )


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--axis", "-a",
              type=click.Choice(["roll", "pitch", "yaw"], case_sensitive=False),
              default="roll", show_default=True,
              help="Gyro axis to plot.")
@click.option("--sample-rate", "-s", type=float, default=None,
              help="Override auto-detected sample rate (Hz).")
@click.option("--max-freq", "-f", type=float, default=1000.0,
              help="Upper frequency limit for the spectrogram y-axis (Hz).")
@click.option("--nperseg", type=int, default=512,
              help="STFT window size in samples. Larger = better freq resolution.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save spectrogram PNG to this path.")
@click.option("--no-show", is_flag=True, default=False,
              help="Suppress the interactive chart window.")
def spectrogram(log_file, axis, sample_rate, max_freq, nperseg, output, no_show):
    """
    Generate a spectrogram (frequency vs. time) for one gyro axis.

    Shows whether noise spikes are constant or only appear at certain
    throttle levels / flight phases.
    """
    axis_map = {"roll": "gyroADC[0]", "pitch": "gyroADC[1]", "yaw": "gyroADC[2]"}
    axis_label_map = {"roll": "Roll", "pitch": "Pitch", "yaw": "Yaw"}
    col = axis_map[axis.lower()]
    axis_name = axis_label_map[axis.lower()]

    log_path = Path(log_file)
    console.rule(f"[bold cyan]Spectrogram:[/] {axis_name} — {log_path.name}")

    df = log_parser.load_csv(log_path)
    fs = sample_rate or log_parser.estimate_sample_rate(df)

    with console.status("[bold green]Computing STFT spectrogram…"):
        t, f, Sxx_db = dsp_engine.compute_spectrogram(df, col, fs, nperseg=nperseg)

    out_path = Path(output) if output else None
    visualizer.plot_spectrogram(
        t, f, Sxx_db,
        axis_name=axis_name,
        max_freq_hz=max_freq,
        output_path=out_path,
        show=not no_show,
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _print_banner():
    banner = Text()
    banner.append("  🛸 FPV Blackbox DSP Analyzer  ", style="bold cyan")
    banner.append("v0.1.0-alpha", style="dim")
    console.print(Panel(banner, border_style="cyan", expand=False))


def _print_report_summary(report: dsp_engine.AnalysisReport):
    from rich.table import Table

    table = Table(title="FFT Analysis Summary", border_style="cyan", show_lines=True)
    table.add_column("Axis", style="bold")
    table.add_column("Peak Freq (Hz)")
    table.add_column("Peak Power (dB)")
    table.add_column("D-term Warning")

    for ax in report.axes:
        warn_cell = "[bold red]⚠ YES[/]" if ax.dterm_warning else "[green]✓ OK[/]"
        table.add_row(ax.axis, f"{ax.peak_freq_hz:.1f}", f"{ax.peak_power_db:.1f}", warn_cell)

    console.print(table)

    if report.any_warning:
        for ax in report.axes:
            if ax.dterm_warning:
                console.print(Panel(ax.dterm_warning_message, style="bold red", title="D-term Warning"))


if __name__ == "__main__":
    cli()
