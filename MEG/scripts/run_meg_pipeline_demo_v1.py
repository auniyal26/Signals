import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib.pyplot as plt

from MEG.src.config import MEGConfig
from MEG.src.dataset import load_sample_raw
from MEG.src.preprocess import (
    notch_and_bandpass,
    pick_one_meg_channel,
    compute_events,
    most_frequent_event_id,
)
from MEG.src.epochs import make_epochs, evoked_rms
from MEG.src.report import ensure_run_dir, write_summary


def main():
    cfg = MEGConfig()
    cfg.date_tag = "2026-03-09"  # set today

    meg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_dir = ensure_run_dir(os.path.join(meg_dir, "RESULTS"), cfg.date_tag)

    raw, raw_path = load_sample_raw(cfg.crop_sec, preload=True)
    events = compute_events(raw)
    event_id, event_counts = most_frequent_event_id(events)

    raw_f, notches = notch_and_bandpass(
        raw, line_freq=cfg.line_freq, l_freq=cfg.bandpass_low, h_freq=cfg.bandpass_high
    )

    ch = pick_one_meg_channel(raw)
    fs = float(raw.info["sfreq"])
    n = int(cfg.segment_sec * fs)
    tseg = np.arange(n) / fs

    x_raw = raw.get_data(picks=[ch])[0][:n] * 1e15
    x_flt = raw_f.get_data(picks=[ch])[0][:n] * 1e15

    # PSD mean (raw vs filtered)
    def mean_psd_db(r):
        spec = r.compute_psd(picks="meg", fmin=0.5, fmax=100, n_fft=2048)
        freqs = spec.freqs
        psd = spec.get_data()
        return freqs, 10 * np.log10(np.mean(psd, axis=0) + 1e-30)

    f1, p1 = mean_psd_db(raw)
    f2, p2 = mean_psd_db(raw_f)

    # Evoked RMS (raw vs filtered)
    epochs_raw = make_epochs(raw, events, event_id, cfg.tmin, cfg.tmax, cfg.baseline)
    epochs_flt = make_epochs(raw_f, events, event_id, cfg.tmin, cfg.tmax, cfg.baseline)
    ev_raw = epochs_raw.average()
    ev_flt = epochs_flt.average()
    g_raw = evoked_rms(ev_raw)
    g_flt = evoked_rms(ev_flt)

    # --- Plot: 3 panels ---
    out_path = os.path.join(run_dir, "00_meg_pipeline_demo_v1.png")
    plt.figure(figsize=(14, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(tseg, x_raw, label="raw")
    ax1.plot(tseg, x_flt, label="filtered")
    ax1.set_title(f"Raw vs filtered segment | {ch} | {cfg.segment_sec}s")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (fT)")
    ax1.legend()

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(f1, p1, label="raw")
    ax2.plot(f2, p2, label="filtered")
    ax2.set_title("PSD (MEG avg) | 0.5–100 Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(ev_raw.times, g_raw, label=f"raw (n={len(epochs_raw)})")
    ax3.plot(ev_flt.times, g_flt, label=f"filtered (n={len(epochs_flt)})")
    ax3.axvline(0, linestyle="--")
    ax3.set_title(f"Evoked RMS | event={event_id}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RMS (T)")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    caption = (
        "MEG pipeline demo v1: raw→notch+bandpass (1–40 Hz) → event-locked epochs → evoked RMS.\n"
        f"Dataset: {os.path.basename(raw_path)} | events from STI 014 | chosen event_id={event_id}."
    )
    write_summary(os.path.join(run_dir, "00_caption.txt"), [caption])

    log_lines = [
        f"MEG pipeline demo v1 ({cfg.date_tag})",
        f"Dataset: {raw_path}",
        f"Preproc: notch={notches if notches else 'none'} | bandpass={cfg.bandpass_low}-{cfg.bandpass_high} Hz",
        f"Events: {event_counts} | chosen event_id={event_id} | epochs raw={len(epochs_raw)} filtered={len(epochs_flt)}",
        f"Saved: {out_path} (+ 00_caption.txt)",
    ]
    write_summary(os.path.join(run_dir, "00_log.txt"), log_lines)

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()