import os
import numpy as np
import matplotlib.pyplot as plt
import mne

DATE_TAG = "2026-02-13"

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2026-02-20_meg_evoked_before_after.png")

    # Load MNE sample dataset
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)

    # Keep it light (2 minutes)
    raw.crop(tmin=0, tmax=120)

    # Find events (stim channel)
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=True)

    # If no events found, fail loudly (so you know what's missing)
    if events is None or len(events) == 0:
        raise RuntimeError("No events found in STI 014. Cannot epoch.")

    # Pick one common event code to demonstrate (most frequent)
    event_ids, counts = np.unique(events[:, 2], return_counts=True)
    event_id = int(event_ids[np.argmax(counts)])
    event_dict = {f"event_{event_id}": event_id}

    # Epoch window
    tmin, tmax = -0.2, 0.5
    baseline = (None, 0.0)

    # -------------------------
    # BEFORE: epochs on raw
    # -------------------------
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
    epochs_raw = mne.Epochs(
        raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
        picks=picks, baseline=baseline, preload=True, reject_by_annotation=True, verbose=True
    )
    evoked_raw = epochs_raw.average()

    # -------------------------
    # AFTER: filtered copy + epochs
    # -------------------------
    raw_f = raw.copy()

    # Notch + bandpass (simple “first pipeline”)
    line_freq = raw.info.get("line_freq", None)
    if line_freq is None or not np.isfinite(line_freq):
        line_freq = 50.0  # Germany default
    nyq = raw.info["sfreq"] / 2.0
    notches = [line_freq * k for k in range(1, 5) if (line_freq * k) < nyq]
    if notches:
        raw_f.notch_filter(freqs=notches, picks="meg", verbose=False)

    raw_f.filter(l_freq=1.0, h_freq=40.0, picks="meg", verbose=False)

    epochs_f = mne.Epochs(
        raw_f, events, event_id=event_dict, tmin=tmin, tmax=tmax,
        picks=picks, baseline=baseline, preload=True, reject_by_annotation=True, verbose=True
    )
    evoked_f = epochs_f.average()

    # -------------------------
    # Plot: evoked global field power (GFP) style
    # We'll plot RMS across MEG channels vs time for raw vs filtered.
    # -------------------------
    t = evoked_raw.times

    def gfp(evoked):
        data = evoked.data  # shape: (n_channels, n_times)
        return np.sqrt(np.mean(data ** 2, axis=0))

    g_raw = gfp(evoked_raw)
    g_flt = gfp(evoked_f)

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(t, g_raw)
    ax1.axvline(0, linestyle="--")
    ax1.set_title(f"Evoked (RAW) | event={event_id} | n={len(epochs_raw)}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS (T)")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(t, g_flt)
    ax2.axvline(0, linestyle="--")
    ax2.set_title(f"Evoked (FILTERED 1–40Hz + notch) | event={event_id} | n={len(epochs_f)}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RMS (T)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)
    print(f"Events found: {len(events)} | Using event_id={event_id} (most frequent)")
    print(f"Epochs: raw={len(epochs_raw)} | filtered={len(epochs_f)}")
    print(f"Filters: notch={notches if notches else 'none'} | bandpass=1–40 Hz")


if __name__ == "__main__":
    main()
