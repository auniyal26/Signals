# CC01 — PSD Rhythm Metric (Welch)

## What this evaluates
Detects whether a signal has a clear rhythmic component (ECG-ish demo: ~1–2 Hz) using Welch PSD.

## Inputs
- Signal `x` (1D or channels×samples)
- Sampling rate `fs`

## Method
1) Compute PSD with Welch:
- `f, Pxx = welch_psd(x, fs, nperseg=1024)`
2) Dominant rhythm frequency:
- `peak_hz = argmax(Pxx) within rhythm_band (default 0.8–2.0 Hz)`
- Convert: `BPM = peak_hz * 60`
3) Rhythm clarity ratio:
- `ratio = bandpower(Pxx, 0.8–2.0 Hz) / bandpower(Pxx, 0.5–40 Hz)`

## Interpretation
- `peak_hz` (or BPM): estimated dominant rhythm.
- `ratio` (0–1-ish):
  - High (~0.6–0.9): rhythm dominates (clean synthetic / clean ECG)
  - Mid (~0.2–0.6): rhythm present but noisy / mixed components
  - Low (<0.2): rhythm weak or absent; peak may be unreliable

## Parameter sensitivity (important)
Frequency resolution ≈ `fs / nperseg`.
Example: fs=250, nperseg=1024 ⇒ bin width ≈ 0.244 Hz.
So a true 1.2 Hz rhythm may land at 1.2207 Hz.

## Failure modes / gotchas
- Very short signals: Welch becomes unstable (nperseg too large).
- Strong baseline wander: can inflate low-frequency power (fix via bandpass).
- Multi-rhythm / arrhythmia: may show multiple peaks → peak_hz alone is not enough.
- If channels differ: compute per-channel peak/ratio.

## Demo reference
Saved plot: `Artifacts/signals/2025-12-29_psd_peak_ratio.png`
Example output (synthetic):
- peak = 1.2207 Hz (73.24 BPM)
- ratio = 0.894
