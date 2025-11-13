<div align="center">

# 🛸 FPV-Blackbox-DSP-Analyzer

**A Python-based diagnostic suite for Betaflight Blackbox logs**

*Leveraging Digital Signal Processing and Fast Fourier Transforms to diagnose gyro noise, D-term oscillations, and filter effectiveness.*

[![Status](https://img.shields.io/badge/status-Work%20In%20Progress-orange?style=flat-square)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/)

</div>

---

> **🚧 Work In Progress** — Core FFT engine and visualizer are implemented. Batch processing, Streamlit dashboard, and notch filter comparison are on the roadmap.

---

## 📖 Overview

Standard log viewers like [Blackbox Explorer](https://github.com/betaflight/blackbox-log-viewer) show you *what* your drone did — this tool explains *why* it happened.

When your quad vibrates, motors heat up, or your tune "feels off", the root cause lives in the **frequency domain**. By running your raw gyro data through a **Fast Fourier Transform**, this analyzer converts a messy, time-domain gyro trace into a clean frequency spectrum — immediately showing you *which frequencies* are carrying dangerous energy levels, and whether they fall in the D-term oscillation band.

**Who this is for:**
- FPV pilots doing PID tuning sessions who want to go deeper than visual inspection
- Drone engineers verifying filter configurations (RPM filters, notch filters, lowpass)
- Developers interested in real-world DSP applications on high-frequency embedded sensor data

---

## ✨ Features

| Feature | Status | Description |
|---|---|---|
| **FFT Noise Heatmap** | ✅ Ready | Power Spectral Density across all 3 gyro axes, frequency-domain chart |
| **D-term Oscillation Detector** | ✅ Ready | Automated heuristic flags 200–500 Hz high-energy oscillations |
| **Gyro Spectrogram** | ✅ Ready | STFT-based frequency-vs-time chart showing throttle-dependent noise |
| **CLI Interface** | ✅ Ready | `analyze` and `spectrogram` subcommands with rich terminal output |
| **Unit Tests** | ✅ Ready | Synthetic signal fixtures — no real log needed to run tests |
| **Batch Processing** | 🔲 Planned | Compare multiple flight logs in one run |
| **Filter Effectiveness** | 🔲 Planned | Pre-filter vs post-filter PSD overlay |
| **Streamlit Dashboard** | 🔲 Planned | Interactive web UI for non-CLI users |
| **PDF Report Export** | 🔲 Planned | Auto-generated diagnostic summary PDF |

---

## 🔬 The Engineering — How It Works

### 1. The FFT: Moving to the Frequency Domain

A Betaflight gyro captures data in the **time domain** — thousands of samples per second of how fast the craft is rotating. To find noise, you need to see *which frequencies* carry energy. The **Fast Fourier Transform** does this:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i \cdot 2\pi k n / N}$$

This decomposes the signal `x[n]` into `N` frequency bins. Each bin `X[k]` tells you the amplitude and phase at that specific frequency. We then compute the **Power Spectral Density (PSD)**:

$$\text{PSD}[k] = \frac{|X[k]|^2}{N}$$

The PSD is plotted on a dB scale. A spike at 350 Hz means *a lot* of energy at 350 Hz — and if you find it in the D-term band, that's your problem.

### 2. Welch's Method — Better Than a Raw FFT

Instead of one giant FFT, the analyzer uses [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method): it splits the signal into overlapping windows, computes an FFT on each, then **averages** the power. The result is a smoother, more statistically reliable spectrum with far less variance — critical for distinguishing real resonances from noise floor noise.

### 3. D-term Oscillation Detection

The **D-term** in Betaflight's PID controller is a derivative term — it amplifies high-frequency changes in the error signal. This makes it very effective at stopping overshoot, but it also amplifies high-frequency gyro noise. When D-term gain is too high:

```
D-noise → motor output commands → motors heat up → potential ESC desync
```

The analyzer monitors the **200–500 Hz** band (where D-term resonance typically manifests in 5-inch racing quads). If the power in that band is within **20 dB** of the dominant spectral peak *and* exceeds 300 Hz, it triggers:

```
⚠  WARNING: D-term gain too high — high-frequency oscillation detected at 350 Hz
    (-18.3 dB). Potential motor overheat risk.
    Consider reducing D-term or enabling a notch filter at 350 Hz.
```

### 4. The Spectrogram — Throttle-Dependent Noise

A PSD chart averages over the entire flight. The **spectrogram** uses the **Short-Time Fourier Transform (STFT)** to show how the spectrum evolves over time. You'll see noise clusters appear at specific throttle levels — classic signs of propeller resonance or ESC switching noise that only manifests under load.

### 5. The Nyquist Limit — Knowing Your Boundaries

If your gyro logs at **4 kHz**, the highest frequency you can ever resolve is **2 kHz** (the Nyquist frequency = fs/2). Any energy *above* that folds back into the spectrum — this is **aliasing**. The analyzer's charts are capped accordingly and label the Nyquist limit so you never misread folded artifacts as real noise.

---

## 📂 Project Structure

```
FPV-Blackbox-DSP-Analyzer/
├── data/                       # Raw .bbl or .csv log files (gitignored)
│   └── README.md               # Instructions for adding log files
├── exports/                    # Generated heatmaps, spectrograms, PDFs
├── src/
│   ├── __init__.py
│   ├── parser.py               # CSV/BBL ingestion, column validation, sample rate estimation
│   ├── dsp_engine.py           # FFT (Welch PSD), STFT spectrogram, D-term heuristic
│   └── visualizer.py           # Matplotlib: noise heatmap & spectrogram charts
├── tests/
│   ├── __init__.py
│   └── test_dsp_engine.py      # Unit tests using synthetic sine-wave fixtures
├── main.py                     # CLI entry point (Click + Rich)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- A Betaflight Blackbox log exported as CSV from [Blackbox Explorer](https://github.com/betaflight/blackbox-log-viewer)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/FPV-Blackbox-DSP-Analyzer.git
cd FPV-Blackbox-DSP-Analyzer

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Running Your First Analysis

```bash
# Drop your CSV into the data/ folder, then:
python main.py analyze data/my_flight.csv
```

You'll get a rich terminal output with a summary table and any D-term warnings, then a dark-themed frequency heatmap.

```bash
# Generate a spectrogram for the roll axis
python main.py spectrogram data/my_flight.csv --axis roll

# Save outputs without displaying windows (batch mode)
python main.py analyze data/my_flight.csv --no-show --output exports/heatmap.png
python main.py spectrogram data/my_flight.csv --axis pitch --no-show --output exports/pitch_spec.png

# See all options
python main.py --help
python main.py analyze --help
python main.py spectrogram --help
```

### Exporting Your Log from Blackbox Explorer

1. Open [Blackbox Explorer](https://github.com/betaflight/blackbox-log-viewer/releases)
2. Load your `.bbl` file
3. **File → Export CSV** (make sure "Gyro" fields are checked)
4. Save the `.csv` into the `data/` directory

---

## 📊 Reading the Charts

### Noise Heatmap

```
Y-axis : Power (dB) — higher = more energy at that frequency
X-axis : Frequency (Hz)
Orange band : D-term risk zone (200–500 Hz)
Colored lines : Roll (blue), Pitch (red), Yaw (gold)
⚠ labels : Axes where D-term warning was triggered
```

**What to look for:**
- **Flat noise floor** → clean tune, filters working well
- **Spike at 50/100 Hz** → prop imbalance or frame resonance
- **Energy in 200–500 Hz band** → D-term too high, consider reducing or adding a dynamic notch filter
- **Wall at high frequency (>1 kHz)** → check your gyro lowpass filter cutoff

### Spectrogram

```
Y-axis : Frequency (Hz)
X-axis : Time (seconds into flight)
Color  : Power (dark = quiet, bright yellow = loud)
```

**What to look for:**
- **Horizontal bright bands** → constant-frequency noise (usually frame/motor resonance)
- **Diagonal or curved bands** → throttle-correlated noise (prop harmonics — use RPM filter)
- **Sporadic bright patches** → transient events (hard maneuvers, prop strikes)

---

## 🛠 Tech Stack

| Library | Role |
|---|---|
| **NumPy** | Array math, FFT math |
| **Pandas** | Log ingestion and time-series manipulation |
| **SciPy** | `welch()` PSD estimator, `spectrogram()` STFT |
| **Matplotlib** | Static charts (heatmap, spectrogram) |
| **Plotly** | *(Planned)* Interactive charts |
| **Click** | CLI framework |
| **Rich** | Terminal formatting, progress, tables |

---

## 🧪 Running Tests

The test suite uses **synthetic sine-wave signals** with mathematically known properties — no real log file required to verify the DSP math.

```bash
pytest tests/ -v
```

Tests verify:
- Sample rate estimation accuracy
- FFT peak detection within ±15 Hz of truth frequency
- D-term warning triggers correctly for 350 Hz high-amplitude signals
- D-term warning does NOT fire for low-frequency (30 Hz) signals
- Report structure (3 axes, correct labels)

---

## 📈 Roadmap

- [x] Core FFT engine (Welch PSD)
- [x] D-term oscillation heuristic
- [x] Gyro noise heatmap visualization
- [x] STFT spectrogram
- [x] CLI with rich terminal output
- [x] Unit test suite with synthetic signals
- [ ] `blackbox_decode` integration for raw `.bbl` binary files
- [ ] Batch processing — compare multiple flights in one report
- [ ] Pre-filter vs post-filter PSD overlay (filter effectiveness map)
- [ ] RPM harmonic overlay (ESC telemetry integration)
- [ ] Streamlit interactive web dashboard
- [ ] PDF diagnostic report generator

---

## ⚠️ Disclaimer

This project is a **Work in Progress**. D-term oscillation thresholds are based on general FPV community standards and calibrated for typical 5-inch racing quads. They are intended as **diagnostic guidance only** and should not be treated as absolute flight-safety thresholds.

Always make final tuning decisions with a full understanding of your specific hardware.

---

## 🤝 Contributing

Contributions are welcome! If you have Blackbox log files you'd like to use as test cases, or expertise in FPV tuning heuristics, open an issue or pull request.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
