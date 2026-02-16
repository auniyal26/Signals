import os
import numpy as np
import matplotlib.pyplot as plt
import mne

DATE_TAG = "2026-02-16"

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)

    # --- Load MNE sample dataset ---
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)

    # Keep it light
    raw.crop(tmin=0, tmax=120)

    # --- Locate annotations ---
    ann = raw.annotations
    n_ann = len(ann)

    # --- Locate events (stim channel) ---
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=True)
    unique_ids, counts = np.unique(events[:, 2], return_counts=True)

    # --- Plot 1: raw segment (one MEG channel, 10s) ---
    fs = float(raw.info["sfreq"])
    n = int(10 * fs)

    # pick one MEG channel (prefer magnetometer)
    mag_picks = mne.pick_types(raw.info, meg="mag", eeg=False, stim=False)
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False)
    pick = mag_picks[0] if len(mag_picks) else meg_picks[0]
    ch_name = raw.ch_names[pick]

    x = raw.get_data(picks=[ch_name])[0][:n] * 1e15  # T -> fT
    t = np.arange(n) / fs

    raw_fig = os.path.join(out_dir, f"{DATE_TAG}_meg_raw_segment.png")
    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.title(f"MEG raw segment (10s) | channel={ch_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (fT)")
    plt.tight_layout()
    plt.savefig(raw_fig, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 2: PSD (MEG average, 0.5–100 Hz) ---
    psd_fig = os.path.join(out_dir, f"{DATE_TAG}_meg_psd.png")
    spectrum = raw.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
    fig = spectrum.plot(average=True, show=False)
    fig.suptitle("MEG PSD (0.5–100 Hz) — averaged", y=0.98)
    fig.savefig(psd_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Console summary (for your note) ---
    print("DATASET:", "MNE sample | sample_audvis_raw.fif")
    print("RAW:", raw)
    print("sfreq:", fs, "Hz | duration:", raw.times[-1], "s")
    print("channels:", len(raw.ch_names), "| MEG:", len(meg_picks), "| MAG:", len(mag_picks))
    print("annotations:", n_ann)
    if n_ann:
        # show first few descriptions
        descs = list(dict.fromkeys(ann.description))  # unique, keep order
        print("annotation types (first):", descs[:6])
    print("events:", len(events), "| event_ids:", {int(k): int(v) for k, v in zip(unique_ids, counts)})
    print("Saved:", raw_fig)
    print("Saved:", psd_fig)

if __name__ == "__main__":
    main()
