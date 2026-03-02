import numpy as np
import mne

def notch_and_bandpass(raw, line_freq=50.0, l_freq=1.0, h_freq=40.0):
    raw_f = raw.copy()

    fs = float(raw.info["sfreq"])
    nyq = fs / 2.0
    notches = [line_freq * k for k in range(1, 5) if (line_freq * k) < nyq]
    if notches:
        raw_f.notch_filter(freqs=notches, picks="meg", verbose=False)

    raw_f.filter(l_freq=l_freq, h_freq=h_freq, picks="meg", verbose=False)
    return raw_f, notches

def pick_one_meg_channel(raw):
    mag_picks = mne.pick_types(raw.info, meg="mag", eeg=False, stim=False)
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False)
    if len(meg_picks) == 0:
        raise RuntimeError("No MEG channels found.")
    pick = mag_picks[0] if len(mag_picks) else meg_picks[0]
    return raw.ch_names[pick]

def compute_events(raw):
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=True)
    return events

def most_frequent_event_id(events):
    ids, counts = np.unique(events[:, 2], return_counts=True)
    return int(ids[counts.argmax()]), {int(k): int(v) for k, v in zip(ids, counts)}