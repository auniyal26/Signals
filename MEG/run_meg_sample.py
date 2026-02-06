import os
import numpy as np
import matplotlib.pyplot as plt
import mne


DATE_TAG = "2026-02-02"  # change if you care
SEGMENT_SEC = 10
CROP_SEC = 60

# Filtering defaults (good “first contact” settings)
BP_LOW = 1.0
BP_HIGH = 40.0


def _pick_one_meg_channel(raw):
    """Pick a single MEG channel name (prefer magnetometer if available)."""
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
    if len(meg_picks) == 0:
        raise RuntimeError("No MEG channels found in this recording.")
    # Prefer first magnetometer channel if present
    mag_picks = mne.pick_types(raw.info, meg="mag", eeg=False, eog=False, ecg=False, stim=False)
    pick = mag_picks[0] if len(mag_picks) else meg_picks[0]
    return raw.ch_names[pick]


def _line_freq(raw, default=50.0):
    """Try to infer mains line frequency; fall back to 50 Hz (Germany)."""
    lf = raw.info.get("line_freq", None)
    if lf is None or not np.isfinite(lf):
        return float(default)
    return float(lf)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "meg_raw_vs_filtered.png")

    # --- Load sample MEG dataset (downloads on first run) ---
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)
    raw.crop(tmin=0, tmax=CROP_SEC)

    fs = float(raw.info["sfreq"])
    ch_name = _pick_one_meg_channel(raw)

    # --- Filtered copy (notch + bandpass) ---
    lf = _line_freq(raw, default=50.0)
    raw_f = raw.copy()

    # Notch at line freq + harmonics (up to 4th, within Nyquist)
    nyq = fs / 2.0
    notches = [lf * k for k in range(1, 5) if (lf * k) < nyq]
    if len(notches) > 0:
        raw_f.notch_filter(freqs=notches, picks="meg", verbose=False)

    raw_f.filter(l_freq=BP_LOW, h_freq=BP_HIGH, picks="meg", verbose=False)

    # --- Time series (same channel, same segment) ---
    n = int(SEGMENT_SEC * fs)
    x_raw = raw.get_data(picks=[ch_name])[0][:n]
    x_flt = raw_f.get_data(picks=[ch_name])[0][:n]
    t = np.arange(n) / fs

    # Convert Tesla -> fT for readability
    x_raw_ft = x_raw * 1e15
    x_flt_ft = x_flt * 1e15

    # --- PSD raw vs filtered (average across MEG channels) ---
    # Use MNE Spectrum objects and pull numeric arrays for plotting.
    spec_raw = raw.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
    spec_flt = raw_f.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)

    freqs = spec_raw.freqs
    psd_raw = spec_raw.get_data()  # shape: (n_channels, n_freqs)
    psd_flt = spec_flt.get_data()

    # Average across channels; add tiny epsilon to avoid log(0)
    eps = 1e-30
    psd_raw_mean = 10 * np.log10(np.mean(psd_raw, axis=0) + eps)
    psd_flt_mean = 10 * np.log10(np.mean(psd_flt, axis=0) + eps)

    # --- Plot: 2x2 ---
    plt.figure(figsize=(12, 7))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, x_raw_ft)
    ax1.set_title(f"Raw MEG (channel: {ch_name}) — {SEGMENT_SEC}s")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (fT)")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, x_flt_ft)
    ax2.set_title(f"Filtered MEG (notch {notches if notches else 'none'}, BP {BP_LOW}-{BP_HIGH} Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (fT)")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(freqs, psd_raw_mean)
    ax3.set_title("PSD (raw) — MEG average")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (dB)")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(freqs, psd_flt_mean)
    ax4.set_title("PSD (filtered) — MEG average")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Power (dB)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)
    print(f"fs={fs} Hz | line_freq={lf} Hz | notch={notches if notches else 'none'} | bandpass={BP_LOW}-{BP_HIGH} Hz")
    print("What to look for:")
    print("- Time series: filtered should look less 'drifty' + less high-frequency fuzz.")
    print("- PSD: filtered should suppress <1 Hz drift and >40 Hz content; notch should reduce line peaks.")


if __name__ == "__main__":
    main()
