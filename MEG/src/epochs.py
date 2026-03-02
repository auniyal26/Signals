import numpy as np
import mne

def make_epochs(raw, events, event_id, tmin, tmax, baseline):
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False, stim=False)
    epochs = mne.Epochs(
        raw, events, event_id={f"event_{event_id}": event_id},
        tmin=tmin, tmax=tmax, baseline=baseline,
        picks=picks, preload=True, reject_by_annotation=True, verbose=True
    )
    return epochs

def evoked_rms(evoked):
    data = evoked.data
    return np.sqrt(np.mean(data ** 2, axis=0))