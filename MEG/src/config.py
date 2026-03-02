from dataclasses import dataclass
from datetime import date

@dataclass
class MEGConfig:
    date_tag: str = date.today().isoformat()   # e.g. 2026-03-02
    crop_sec: int = 180                        # how much raw to use
    segment_sec: int = 10                      # raw segment plot length

    # filters
    line_freq: float = 50.0                    
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0

    # epochs
    tmin: float = -0.2
    tmax: float = 0.8
    baseline = (None, 0.0)

    # TFR
    tfr_fmin: int = 4
    tfr_fmax: int = 40
    tfr_fstep: int = 2
    tfr_decim: int = 5
    tfr_n_ch: int = 20

    # ICA
    ica_n_components: int = 20
    ica_random_state: int = 97