import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MEG.src.config import MEGConfig
from MEG.src.dataset import load_sample_raw
from MEG.src.preprocess import notch_and_bandpass, pick_one_meg_channel, compute_events, most_frequent_event_id
from MEG.src.epochs import make_epochs, evoked_rms
from MEG.src.tfr import compute_tfr
from MEG.src.ica import run_ica
from MEG.src import report as R

def main():
    cfg = MEGConfig()

    meg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_root = os.path.join(meg_dir, "RESULTS")
    run_dir = R.ensure_run_dir(results_root, cfg.date_tag)

    raw, raw_path = load_sample_raw(cfg.crop_sec, preload=True)

    # dataset inventory
    events = compute_events(raw)
    event_id, event_counts = most_frequent_event_id(events)
    n_ann = len(raw.annotations)

    # preprocessing
    raw_f, notches = notch_and_bandpass(raw, line_freq=cfg.line_freq, l_freq=cfg.bandpass_low, h_freq=cfg.bandpass_high)

    # pick channel for raw plots
    ch = pick_one_meg_channel(raw)

    # 1) raw segment
    R.save_raw_segment(raw, ch, cfg.segment_sec, os.path.join(run_dir, "01_raw_segment.png"))

    # 2) PSD raw
    R.save_psd(raw, os.path.join(run_dir, "02_psd_raw.png"), "PSD (raw) — MEG average")

    # 3) raw vs filtered segment
    R.save_raw_vs_filtered(raw, raw_f, ch, cfg.segment_sec, notches, (cfg.bandpass_low, cfg.bandpass_high),
                           os.path.join(run_dir, "03_raw_vs_filtered.png"))

    # 4) evoked before/after
    epochs_raw = make_epochs(raw, events, event_id, cfg.tmin, cfg.tmax, cfg.baseline)
    epochs_flt = make_epochs(raw_f, events, event_id, cfg.tmin, cfg.tmax, cfg.baseline)
    ev_raw = epochs_raw.average()
    ev_flt = epochs_flt.average()
    g_raw = evoked_rms(ev_raw)
    g_flt = evoked_rms(ev_flt)
    R.save_evoked_before_after(ev_raw.times, g_raw, g_flt, event_id, len(epochs_raw), len(epochs_flt),
                               os.path.join(run_dir, "04_evoked_before_after.png"))

    # 5) TFR (on filtered epochs to reduce junk)
    power, freqs = compute_tfr(epochs_flt, fmin=cfg.tfr_fmin, fmax=cfg.tfr_fmax, fstep=cfg.tfr_fstep,
                               n_ch=cfg.tfr_n_ch, decim=cfg.tfr_decim)
    R.save_tfr(power, freqs, os.path.join(run_dir, "05_tfr.png"), event_id, cfg.tfr_n_ch)

    # 6) ICA before/after PSD (run ICA on filtered data for stability)
    ica, exclude, eog_inds, ecg_inds = run_ica(raw_f.copy(), n_components=cfg.ica_n_components, random_state=cfg.ica_random_state)
    raw_ica_before = raw_f.copy()
    raw_ica_after = raw_f.copy()
    ica.apply(raw_ica_after)
    R.save_ica_psd_before_after(raw_ica_before, raw_ica_after, os.path.join(run_dir, "06_ica_psd_before_after.png"), exclude)

    # summary
    summary_lines = [
        f"MEG report run: {cfg.date_tag}",
        f"Dataset: {raw_path}",
        f"Crop: {cfg.crop_sec}s | sfreq={raw.info['sfreq']}",
        f"Channels: total={len(raw.ch_names)} | annotations={n_ann}",
        f"Events: {event_counts} | chosen_event_id={event_id}",
        f"Filter: notch={notches if notches else 'none'} | bandpass={cfg.bandpass_low}-{cfg.bandpass_high} Hz",
        f"Epochs: raw={len(epochs_raw)} | filtered={len(epochs_flt)} | window={cfg.tmin}..{cfg.tmax}s baseline={cfg.baseline}",
        f"TFR: {cfg.tfr_fmin}-{cfg.tfr_fmax} Hz step={cfg.tfr_fstep} | channels={cfg.tfr_n_ch} | decim={cfg.tfr_decim}",
        f"ICA: n_components={cfg.ica_n_components} | excluded={exclude} | eog={eog_inds} | ecg={ecg_inds}",
        f"Outputs saved in: {run_dir}",
    ]
    R.write_summary(os.path.join(run_dir, "summary.txt"), summary_lines)

    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()