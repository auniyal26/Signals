import os
import numpy as np
import matplotlib.pyplot as plt
import mne

DATE_TAG = "2026-02-23"

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{DATE_TAG}_meg_tfr.png")

    # Load sample MEG
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)
    raw.crop(tmin=0, tmax=120)

    # Events
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=True)
    ids, counts = np.unique(events[:, 2], return_counts=True)
    event_id = int(ids[np.argmax(counts)])
    event_dict = {f"event_{event_id}": event_id}

    # Picks + epochs
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
    epochs = mne.Epochs(
        raw, events, event_id=event_dict, tmin=-0.2, tmax=0.8,
        picks=picks, baseline=(None, 0.0), preload=True, reject_by_annotation=True, verbose=True
    )

    # Compute TFR on a small subset of channels to keep it fast
    epochs.pick(picks[:20])

    freqs = np.arange(4, 41, 2)          # 4..40 Hz
    n_cycles = freqs / 2.0               # classic choice
    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True, decim=5, n_jobs=1, verbose=True
    )

    # Plot average power (across channels) as an image
    P = power.data.mean(axis=0)  # (freqs, times)
    times = power.times

    plt.figure(figsize=(10, 4))
    plt.imshow(
        P, aspect="auto", origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    plt.axvline(0, linestyle="--")
    plt.colorbar(label="Power")
    plt.title(f"MEG TFR (Morlet) | event={event_id} | avg over 20 MEG ch")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)
    print("Event used:", event_id, "| epochs:", len(epochs))

if __name__ == "__main__":
    main()