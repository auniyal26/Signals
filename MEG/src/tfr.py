import numpy as np
import mne

def compute_tfr(epochs, fmin=8, fmax=40, fstep=2, n_ch=20, decim=5):
    # keep it fast
    picks = epochs.ch_names[:min(n_ch, len(epochs.ch_names))]
    ep = epochs.copy().pick(picks)

    freqs = np.arange(fmin, fmax + 1, fstep)

    # shorter wavelets so they fit inside short epochs
    # (constant 2 cycles is a safe default here)
    n_cycles = np.full_like(freqs, 2.0, dtype=float)

    power = mne.time_frequency.tfr_morlet(
        ep, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True, decim=decim, n_jobs=1, verbose=True
    )
    return power, freqs