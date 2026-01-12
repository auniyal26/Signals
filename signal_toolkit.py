import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

def bandpass_filter(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    Bandpass filter a 1D signal using a Butterworth filter + zero-phase filtering.

    Args:
        x: 1D signal (shape: [n_samples])
        fs: sampling frequency (Hz)
        low_hz: low cutoff frequency (Hz)
        high_hz: high cutoff frequency (Hz)
        order: filter order (typical: 2–6)

    Returns:
        Filtered signal (same shape as x)
    """
    x = np.asarray(x).astype(float)
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Bad cutoffs: low={low_hz}Hz high={high_hz}Hz for fs={fs}Hz")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)

def moving_rms(x: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Compute moving RMS (useful for envelopes / activity level).

    Args:
        x: 1D signal
        window_samples: window length in samples (e.g., int(0.2 * fs))

    Returns:
        RMS envelope (same length as x)
    """
    x = np.asarray(x).astype(float)
    if window_samples < 1:
        raise ValueError("window_samples must be >= 1")

    # RMS = sqrt(mean(x^2))
    x2 = x ** 2
    kernel = np.ones(window_samples) / window_samples
    mean_x2 = np.convolve(x2, kernel, mode="same")
    return np.sqrt(mean_x2)

def ecg_peaks_hr_simple(x, fs, min_hr=40, max_hr=200, prominence=None, height=None):
    """
    Simple peak detector on the input signal.
    Returns: peaks_idx, rr_sec, bpm
    """
    x = np.asarray(x, dtype=float)
    if fs <= 0:
        raise ValueError("fs must be > 0")

    min_dist = int(fs * 60 / max_hr)
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence, height=height)

    rr = np.diff(peaks) / fs
    bpm = 60.0 / rr.mean() if len(rr) else np.nan
    return peaks, rr, bpm

def ecg_peaks_hr(
    x,
    fs,
    min_hr=40,
    max_hr=200,
    qrs_band=(5.0, 15.0),
    env_win_s=0.15,
    prom_k=0.5,
    rr_clamp=(0.5, 1.5),
):
    """
    Robust ECG-ish peak detector (R-peak style) using a QRS envelope:
    bandpass (qrs_band) -> rectify -> moving RMS envelope -> prominence + RR constraints.
    Returns: peaks_idx, rr_sec_filtered, bpm_filtered
    """
    x = np.asarray(x, dtype=float)
    if fs <= 0:
        raise ValueError("fs must be > 0")

    # QRS-focused detection signal
    x_qrs = bandpass_filter(x, fs=fs, low_hz=qrs_band[0], high_hz=qrs_band[1], order=4)

    # Envelope (rectify + smooth)
    win = max(3, int(env_win_s * fs))
    env = moving_rms(np.abs(x_qrs), window_samples=win)

    # RR constraint via min distance
    min_dist = int(fs * 60 / max_hr)

    # Adaptive prominence
    prom = prom_k * float(np.std(env))

    peaks, _ = find_peaks(env, distance=min_dist, prominence=prom)

    rr = np.diff(peaks) / fs
    if len(rr) == 0:
        return peaks, rr, np.nan

    # RR outlier clamp around median
    if rr_clamp is not None and len(rr) >= 2:
        med = float(np.median(rr))
        keep = (rr > rr_clamp[0] * med) & (rr < rr_clamp[1] * med)
        rr_f = rr[keep]
        if len(rr_f) > 0:
            bpm_f = 60.0 / rr_f.mean()
            return peaks, rr_f, bpm_f

    bpm = 60.0 / rr.mean()
    return peaks, rr, bpm

def welch_psd(
    x,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | None = "constant",
    axis: int = -1,
):
    """
    Compute Power Spectral Density (PSD) using Welch's method.

    Parameters
    ----------
    x : array-like
        Signal array. Supports 1D (samples,) or 2D (channels, samples).
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each segment for Welch.
    noverlap : int | None
        Number of points to overlap between segments.
    window : str
        Window type passed to scipy.signal.welch.
    detrend : str | None
        Detrend option passed to scipy.signal.welch.
    axis : int
        Axis corresponding to time/samples.

    Returns
    -------
    f : np.ndarray
        Frequency bins in Hz.
    Pxx : np.ndarray
        PSD. Shape matches x with the time axis replaced by frequency.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    x = np.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError("x must be 1D (samples,) or 2D (channels, samples)")

    f, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=axis,
    )
    return f, Pxx

def bandpower_from_psd(f, Pxx, band: tuple[float, float], axis: int = -1):
    """
    Integrate PSD over a frequency band using trapezoidal rule.

    Parameters
    ----------
    f : np.ndarray
        Frequency bins (Hz), 1D.
    Pxx : np.ndarray
        PSD values. Last dimension should correspond to f if axis=-1.
    band : (float, float)
        Frequency band (low, high) in Hz.
    axis : int
        Frequency axis in Pxx corresponding to f.

    Returns
    -------
    bp : np.ndarray
        Band power (same shape as Pxx with freq axis removed).
    """
    f = np.asarray(f)
    Pxx = np.asarray(Pxx)
    lo, hi = band
    if lo < 0 or hi <= lo:
        raise ValueError("band must be (lo, hi) with 0 <= lo < hi")

    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        raise ValueError("No frequency bins inside the requested band")

    f_band = f[mask]

    # Move freq axis to last for easy slicing/integration
    P = np.moveaxis(Pxx, axis, -1)
    P_band = P[..., mask]

    bp = np.trapz(P_band, x=f_band, axis=-1)
    return bp


def rhythm_psd_metric(
    x,
    fs: float,
    rhythm_band: tuple[float, float] = (0.8, 2.0),
    full_band: tuple[float, float] = (0.5, 40.0),
    nperseg: int = 1024,
    noverlap: int | None = None,
):
    """
    PSD-based rhythm clarity metric.

    Returns
    -------
    peak_hz : float | np.ndarray
        Dominant frequency (Hz) within rhythm_band.
    ratio : float | np.ndarray
        bandpower(rhythm_band) / bandpower(full_band)
    """
    f, Pxx = welch_psd(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Peak inside rhythm band
    mask = (f >= rhythm_band[0]) & (f <= rhythm_band[1])
    if not np.any(mask):
        raise ValueError("No PSD bins inside rhythm_band")

    P = np.asarray(Pxx)
    # handle 1D or 2D: we assume freq axis is last (welch default axis=-1)
    if P.ndim == 1:
        peak_hz = f[mask][np.argmax(P[mask])]
    else:
        # channels x freq
        peak_hz = f[mask][np.argmax(P[..., mask], axis=-1)]

    bp_r = bandpower_from_psd(f, Pxx, rhythm_band, axis=-1)
    bp_f = bandpower_from_psd(f, Pxx, full_band, axis=-1)
    ratio = bp_r / (bp_f + 1e-12)

    return peak_hz, ratio
