import os
import mne

def load_sample_raw(crop_sec: int, preload: bool = True):
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=preload, verbose=True)
    raw.crop(tmin=0, tmax=crop_sec)
    return raw, raw_path