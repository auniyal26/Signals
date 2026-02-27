import os
import numpy as np
import matplotlib.pyplot as plt
import mne

DATE_TAG = "2026-02-28"  # change to today if needed
CROP_SEC = 180           # 3 min so ICA has enough data
FMIN, FMAX = 0.5, 100.0

def mean_psd_db(raw, fmin=0.5, fmax=100, n_fft=2048):
    spec = raw.compute_psd(picks="meg", fmin=fmin, fmax=fmax, n_fft=n_fft)
    freqs = spec.freqs
    psd = spec.get_data()  # (n_ch, n_freq)
    psd_mean = 10 * np.log10(np.mean(psd, axis=0) + 1e-30)
    return freqs, psd_mean

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{DATE_TAG}_meg_ica_psd_before_after.png")

    # Load sample dataset
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)

    # Crop to keep runtime reasonable
    raw.crop(tmin=0, tmax=CROP_SEC)

    # Copy for ICA pipeline
    raw_ica = raw.copy()

    # Basic filtering for ICA stability (common practice)
    raw_ica.filter(l_freq=1.0, h_freq=40.0, picks="meg", verbose=False)

    # Fit ICA on MEG channels
    ica = mne.preprocessing.ICA(
        n_components=20, method="fastica", random_state=97, max_iter="auto"
    )
    ica.fit(raw_ica, picks="meg")

    # Try to identify EOG/ECG-related components if channels exist
    exclude = []

    try:
        eog_inds, _ = ica.find_bads_eog(raw_ica)
        exclude += eog_inds
    except Exception:
        pass

    try:
        ecg_inds, _ = ica.find_bads_ecg(raw_ica)
        exclude += ecg_inds
    except Exception:
        pass

    exclude = sorted(set(exclude))
    ica.exclude = exclude

    # Apply ICA to a filtered copy for comparison
    raw_clean = raw.copy()
    raw_clean.filter(l_freq=1.0, h_freq=40.0, picks="meg", verbose=False)
    ica.apply(raw_clean)

    # PSD before/after (both filtered 1–40 so comparison is fair)
    f1, p_before = mean_psd_db(raw_ica, fmin=FMIN, fmax=FMAX)
    f2, p_after  = mean_psd_db(raw_clean, fmin=FMIN, fmax=FMAX)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(f1, p_before, label="Before ICA (filtered 1–40)")
    plt.plot(f2, p_after,  label=f"After ICA (exclude={len(exclude)})")
    plt.title("MEG PSD before vs after ICA")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, MEG avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)
    print("ICA excluded components:", exclude)

if __name__ == "__main__":
    main()