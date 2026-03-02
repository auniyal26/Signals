import os
import numpy as np
import matplotlib.pyplot as plt
import mne

DATE_TAG = "2026-03-02"
CROP_SEC = 180

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{DATE_TAG}_meg_ica_report.png")

    # Load sample
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_path = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True)
    raw.crop(tmin=0, tmax=CROP_SEC)

    # Prep for ICA
    raw_ica = raw.copy()
    raw_ica.filter(l_freq=1.0, h_freq=40.0, picks="meg", verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=20, method="fastica", random_state=97, max_iter="auto"
    )
    ica.fit(raw_ica, picks="meg")

    # Find bads
    exclude = []

    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw_ica)
        exclude += eog_inds
    except Exception:
        eog_inds, eog_scores = [], None

    try:
        ecg_inds, ecg_scores = ica.find_bads_ecg(raw_ica)
        exclude += ecg_inds
    except Exception:
        ecg_inds, ecg_scores = [], None

    exclude = sorted(set(exclude))
    ica.exclude = exclude

    # Make a single figure: plot scores if available, otherwise a components overview
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    if eog_scores is not None:
        ax1.plot(eog_scores)
        ax1.set_title(f"EOG scores | suggested={eog_inds}")
    else:
        ax1.text(0.1, 0.5, "EOG scores not available", fontsize=12)
        ax1.set_title("EOG scores")
    ax1.set_xlabel("ICA component")
    ax1.set_ylabel("score")

    ax2 = plt.subplot(1, 2, 2)
    if ecg_scores is not None:
        ax2.plot(ecg_scores)
        ax2.set_title(f"ECG scores | suggested={ecg_inds}")
    else:
        ax2.text(0.1, 0.5, "ECG scores not available", fontsize=12)
        ax2.set_title("ECG scores")
    ax2.set_xlabel("ICA component")
    ax2.set_ylabel("score")

    plt.suptitle(f"MEG ICA report | excluded={exclude} | crop={CROP_SEC}s", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)
    print("Excluded components:", exclude)
    print("EOG suggested:", eog_inds)
    print("ECG suggested:", ecg_inds)

if __name__ == "__main__":
    main()