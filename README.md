Here. Copy-paste this as **Signals/README.md** (it’s complete and clean).

````md
# Signals (ECG/EEG)

A small, reproducible sandbox for ECG/EEG-style signal processing (NOT EMG).

This repo contains:
- a minimal signal toolkit (`signal_toolkit.py`)
- an end-to-end demo script (`run_ecg_demo.py`) that prints metrics and saves plots

---

## Quickstart (one command)

### Windows (PowerShell / CMD)
```bash
cd "D:\Study Material\LifeReset\Signals"
python run_ecg_demo.py
````

If your environment uses `python3`, run:

```bash
python3 run_ecg_demo.py
```

---

## What you get

### Console output

For each segment (default: 0s / 60s / 120s, 20s each) and each case (clean/noisy/t_wave_boost), the script prints:

* **BEFORE (simple)**: peak count, BPM, RR mean ± std
* **AFTER (robust)**: peak count, BPM, RR mean ± std

### Saved plots

Plots are written to:

* `RESULTS/robustness/`

Each plot overlays the ECG (0.5–40 Hz) with:

* `x` markers = simple peaks
* `o` markers = robust peaks

---

## Robust peak detection (what changed)

`ecg_peaks_hr()` is the robust detector and uses **one mechanism**:

**QRS-envelope + prominence + RR constraint**

* bandpass the signal in a QRS-focused band (default 5–15 Hz)
* rectify and smooth (moving RMS envelope)
* run `find_peaks()` with:

  * minimum peak distance derived from `max_hr`
  * adaptive prominence threshold (based on envelope std)
  * RR outlier clamp around median RR (to reduce missed/extra beat damage)

The old behavior is kept as:

* `ecg_peaks_hr_simple()` (for before/after comparison)

---

## Repo structure

* `signal_toolkit.py`
  Core utilities:

  * `bandpass_filter()`
  * `moving_rms()`
  * `ecg_peaks_hr_simple()` (baseline)
  * `ecg_peaks_hr()` (robust)
  * `welch_psd()`
  * `bandpower_from_psd()`
  * `rhythm_psd_metric()`

* `run_ecg_demo.py`
  End-to-end script to reproduce metrics + plots.

* `RESULTS/`
  Small, shareable outputs committed to GitHub (example plots).

* `Artifacts/`
  Local-only scratch outputs (ignored by `.gitignore`).

---

## Notes

* The demo tries to load a real ECG dataset via SciPy. If not available, it falls back to a realistic synthetic ECG-like waveform.
* If you want to change segment positions or length, edit `starts` and `seg_s` in `run_ecg_demo.py`.

---