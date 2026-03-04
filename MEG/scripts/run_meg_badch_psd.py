import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib.pyplot as plt
import mne

from MEG.src.dataset import load_sample_raw
from MEG.src.preprocess import notch_and_bandpass
from MEG.src.report import ensure_run_dir, write_summary

DATE_TAG = "2026-03-04"

def mean_psd_db(raw, fmin=0.5, fmax=100, n_fft=2048):
    spec = raw.compute_psd(picks="meg", fmin=fmin, fmax=fmax, n_fft=n_fft)
    freqs = spec.freqs
    psd = spec.get_data()
    psd_mean = 10 * np.log10(np.mean(psd, axis=0) + 1e-30)
    return freqs, psd_mean

def detect_bad_channels_by_variance(raw, z_thresh=6.0):
    """Mark channels with unusually high variance as bad (rough heuristic)."""
    data = raw.get_data(picks="meg")  # (n_ch, n_times)
    v = np.var(data, axis=1)
    z = (v - v.mean()) / (v.std() + 1e-12)
    bad_idx = np.where(z > z_thresh)[0]
    bad_names = [raw.copy().pick("meg").ch_names[i] for i in bad_idx]
    return bad_names, z, v

def main():
    meg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_root = os.path.join(meg_dir, "RESULTS")
    run_dir = ensure_run_dir(results_root, DATE_TAG)

    raw, raw_path = load_sample_raw(crop_sec=180, preload=True)

    # basic filtering first (otherwise variance dominated by drift)
    raw_f, notches = notch_and_bandpass(raw, line_freq=50.0, l_freq=1.0, h_freq=40.0)

    # detect bad channels on filtered data
    bads, z, v = detect_bad_channels_by_variance(raw_f, z_thresh=3.5)
    raw_f.info["bads"] = bads

    # copy with bads dropped
    raw_drop = raw_f.copy().drop_channels(bads) if len(bads) else raw_f.copy()

    # PSD before/after dropping channels
    f1, p1 = mean_psd_db(raw_f)
    f2, p2 = mean_psd_db(raw_drop)

    out_path = os.path.join(run_dir, "07_badch_psd_before_after.png")

    plt.figure(figsize=(10, 4))
    plt.plot(f1, p1, label=f"Before drop (n_ch={len(raw_f.copy().pick('meg').ch_names)})")
    plt.plot(f2, p2, label=f"After drop (n_ch={len(raw_drop.copy().pick('meg').ch_names)})")
    plt.title("MEG PSD: before vs after dropping high-variance channels")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, MEG avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    summary = [
        f"MEG bad-channel pass: {DATE_TAG}",
        f"Dataset: {raw_path}",
        f"Preproc: notch={notches if notches else 'none'} | bandpass=1–40 Hz",
        f"Bad channels (variance z>6): {bads}",
        f"Saved: {out_path}",
    ]
    write_summary(os.path.join(run_dir, "07_badch_summary.txt"), summary)

    print("\n".join(summary))

if __name__ == "__main__":
    main()