import mne

def run_ica(raw_filt, n_components=20, random_state=97):
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica",
        random_state=random_state, max_iter="auto"
    )
    ica.fit(raw_filt, picks="meg")

    exclude = []
    try:
        eog_inds, _ = ica.find_bads_eog(raw_filt)
        exclude += eog_inds
    except Exception:
        eog_inds = []

    try:
        ecg_inds, _ = ica.find_bads_ecg(raw_filt)
        exclude += ecg_inds
    except Exception:
        ecg_inds = []

    exclude = sorted(set(exclude))
    ica.exclude = exclude
    return ica, exclude, eog_inds, ecg_inds