import os
import numpy as np
import matplotlib.pyplot as plt
import mne

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)

    # Download / locate MNE sample dataset (MEG/EEG example)
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")

    # Load raw MEG
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=True)

    # Crop to 60 seconds to keep it light
    raw.crop(tmin=0, tmax=60)

    # Pick a couple of MEG channels for a time-series screenshot
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
    picks = picks[:5]  # first 5 MEG channels

    data, times = raw.get_data(picks=picks, return_times=True)

    # Plot a short segment (10s) of first channel (convert to fT for readability)
    ch0 = data[0]
    fs = raw.info["sfreq"]
    n = int(10 * fs)

    plt.figure()
    plt.plot(times[:n], ch0[:n] * 1e15)
    plt.title("MEG raw (first MEG channel) — 10s segment")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (fT)")
    plt.savefig(os.path.join(out_dir, "2026-02-04_meg_raw_segment.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # PSD plot (MEG only)
    # Newer MNE returns a Spectrum object from compute_psd
    spectrum = raw.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
    fig = spectrum.plot(average=True, show=False)
    fig.suptitle("MEG PSD (0.5–100 Hz) — averaged", y=0.98)
    fig.savefig(os.path.join(out_dir, "2026-02-04_meg_psd.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
