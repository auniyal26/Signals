import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_run_dir(results_root, date_tag):
    run_dir = os.path.join(results_root, f"report_{date_tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_raw_segment(raw, ch_name, segment_sec, out_path):
    fs = float(raw.info["sfreq"])
    n = int(segment_sec * fs)
    x = raw.get_data(picks=[ch_name])[0][:n] * 1e15
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.title(f"Raw MEG segment | ch={ch_name} | {segment_sec}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (fT)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_psd(raw, out_path, title):
    spec = raw.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
    fig = spec.plot(average=True, show=False)
    fig.suptitle(title, y=0.98)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_raw_vs_filtered(raw, raw_f, ch_name, segment_sec, notches, bp, out_path):
    fs = float(raw.info["sfreq"])
    n = int(segment_sec * fs)

    x1 = raw.get_data(picks=[ch_name])[0][:n] * 1e15
    x2 = raw_f.get_data(picks=[ch_name])[0][:n] * 1e15
    t = np.arange(n) / fs

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, x1)
    plt.title("Raw")
    plt.xlabel("Time (s)"); plt.ylabel("fT")

    plt.subplot(1, 2, 2)
    plt.plot(t, x2)
    plt.title(f"Filtered | notch={notches if notches else 'none'} | BP={bp[0]}–{bp[1]} Hz")
    plt.xlabel("Time (s)"); plt.ylabel("fT")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_evoked_before_after(times, g_raw, g_flt, event_id, n_epochs_raw, n_epochs_flt, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(times, g_raw, label=f"RAW (n={n_epochs_raw})")
    plt.plot(times, g_flt, label=f"FILTERED (n={n_epochs_flt})")
    plt.axvline(0, linestyle="--")
    plt.title(f"Evoked RMS before/after | event={event_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_tfr(power, freqs, out_path, event_id, n_ch):
    P = power.data.mean(axis=0)
    times = power.times

    plt.figure(figsize=(10, 4))
    plt.imshow(P, aspect="auto", origin="lower",
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.axvline(0, linestyle="--")
    plt.colorbar(label="Power")
    plt.title(f"TFR (Morlet) | event={event_id} | avg over {n_ch} ch")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_ica_psd_before_after(raw_before, raw_after, out_path, exclude):
    def mean_psd_db(raw):
        spec = raw.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
        freqs = spec.freqs
        psd = spec.get_data()
        return freqs, 10 * np.log10(np.mean(psd, axis=0) + 1e-30)

    f1, p1 = mean_psd_db(raw_before)
    f2, p2 = mean_psd_db(raw_after)

    plt.figure(figsize=(10, 4))
    plt.plot(f1, p1, label="Before ICA")
    plt.plot(f2, p2, label=f"After ICA (exclude={len(exclude)})")
    plt.title("PSD before vs after ICA")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, MEG avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def write_summary(out_path, lines):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")