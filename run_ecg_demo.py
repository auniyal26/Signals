import os
import numpy as np
import matplotlib.pyplot as plt

from signal_toolkit import (
    bandpass_filter,
    ecg_peaks_hr_simple,
    ecg_peaks_hr,
    welch_psd,
)

DATE_TAG = "2026-01-09"
SAVE_PSD = False  # flip to True if you want PSD plots too
RNG_SEED = 42


def load_ecg():
    """Load one ECG array + fs. Falls back to synthetic if dataset unavailable."""
    # Try new SciPy location
    try:
        from scipy.datasets import electrocardiogram
        x = electrocardiogram().astype(float)
        fs = 360.0
        return x, fs, "scipy.datasets.electrocardiogram"
    except Exception:
        pass

    # Try older SciPy location (deprecated but might exist)
    try:
        from scipy.misc import electrocardiogram
        x = electrocardiogram().astype(float)
        fs = 360.0
        return x, fs, "scipy.misc.electrocardiogram (deprecated)"
    except Exception:
        pass

    return None, None, None


def make_synth_ecg(fs=250.0, dur_s=180.0, bpm=72.0, noise=0.03, wander_hz=0.2):
    """Realistic-ish ECG: Gaussian P-QRS-T + baseline wander + noise."""
    t = np.arange(0, dur_s, 1 / fs)
    rr = 60.0 / bpm
    beat_times = np.arange(0.5, dur_s - 0.5, rr)

    def gauss(tt, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((tt - mu) / sigma) ** 2)

    x = np.zeros_like(t)
    for bt in beat_times:
        x += gauss(t, bt - 0.18, 0.03, 0.12)     # P
        x += gauss(t, bt - 0.02, 0.010, -0.20)   # Q
        x += gauss(t, bt,       0.012, 1.00)     # R
        x += gauss(t, bt + 0.02, 0.012, -0.30)   # S
        x += gauss(t, bt + 0.25, 0.060, 0.30)    # T

    x += 0.15 * np.sin(2 * np.pi * wander_hz * t) + noise * np.random.randn(len(t))
    return x.astype(float), float(fs)


def rr_stats(rr):
    if len(rr) == 0:
        return np.nan, np.nan
    return float(rr.mean()), float(rr.std())


def main():
    np.random.seed(RNG_SEED)

    x, fs, src = load_ecg()
    if x is None:
        print("WARNING: ECG dataset not available. Falling back to realistic synthetic QRS.")
        x, fs = make_synth_ecg(fs=250.0, dur_s=180.0)  # 3 minutes → segments at 0/60/120
        src = "synthetic_qrs"

    # Project root assumed: Signals/ is inside LifeReset/
    repo_dir = os.path.abspath(os.path.dirname(__file__))  # Signals/
    out_dir = os.path.join(repo_dir, "RESULTS", "robustness")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving plots to: {out_dir}")

    seg_s = 20
    starts = [0, 60, 120]

    print(f"ECG source: {src} | fs={fs} Hz")
    print(f"Saving plots to: {out_dir}")

    for start_sec in starts:
        i0 = int(start_sec * fs)
        i1 = i0 + int(seg_s * fs)
        x_seg = x[i0:i1]
        tseg = np.arange(len(x_seg)) / fs

        cases = {
            "clean": x_seg,
            "noisy": x_seg + 0.25 * np.sin(2 * np.pi * 0.2 * tseg) + 0.08 * np.random.randn(len(x_seg)),
            "t_wave_boost": x_seg + 0.35 * np.sin(2 * np.pi * (72 / 60) * tseg + 0.8),
        }

        for cname, x_case in cases.items():
            # Display bandpass for plotting (not detection)
            x_disp = bandpass_filter(x_case, fs=fs, low_hz=0.5, high_hz=40.0, order=4)

            # BEFORE: simple peaks on displayed signal
            p0, rr0, bpm0 = ecg_peaks_hr_simple(x_disp, fs=fs, max_hr=180, prominence=0.6)

            # AFTER: robust peaks (internal QRS envelope)
            p1, rr1, bpm1 = ecg_peaks_hr(x_case, fs=fs, max_hr=200)

            rr0m, rr0s = rr_stats(rr0)
            rr1m, rr1s = rr_stats(rr1)

            print(f"\nSegment start={start_sec}s ({seg_s}s) | case={cname}")
            print(f"  BEFORE(simple): peaks={len(p0):2d}  BPM={bpm0:6.1f}  RR={rr0m:.3f}±{rr0s:.3f}")
            print(f"  AFTER(robust):  peaks={len(p1):2d}  BPM={bpm1:6.1f}  RR={rr1m:.3f}±{rr1s:.3f}")

            # Overlay plot (single plot per case)
            t = np.arange(len(x_disp)) / fs
            plt.figure()
            plt.plot(t, x_disp, label="ECG (0.5–40 Hz)")
            if len(p0):
                plt.plot(t[p0], x_disp[p0], "x", label="simple peaks")
            if len(p1):
                plt.plot(t[p1], x_disp[p1], "o", label="robust peaks")

            plt.title(f"Before/after | start={start_sec}s | {cname} | simple BPM={bpm0:.1f} | robust BPM={bpm1:.1f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()

            out_path = os.path.join(out_dir, f"{DATE_TAG}_before_after_{cname}_start{start_sec:03d}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

            if SAVE_PSD:
                f, Pxx = welch_psd(x_disp, fs=fs, nperseg=2048)
                plt.figure()
                plt.semilogy(f, Pxx)
                plt.title(f"Welch PSD (display) | start={start_sec}s | {cname}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("PSD")
                out_path_psd = os.path.join(out_dir, f"{DATE_TAG}_psd_{cname}_start{start_sec:03d}.png")
                plt.savefig(out_path_psd, dpi=200, bbox_inches="tight")
                plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
