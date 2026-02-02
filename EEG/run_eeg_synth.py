import os
import numpy as np
import matplotlib.pyplot as plt

def synth_eeg(fs=250, dur_s=10, alpha_hz=10.0, alpha_amp=1.0, noise_amp=0.6, wander_hz=0.3):
    t = np.arange(0, dur_s, 1/fs)

    # Alpha rhythm + slow drift + broadband noise
    alpha = alpha_amp * np.sin(2*np.pi*alpha_hz*t)
    wander = 0.5 * np.sin(2*np.pi*wander_hz*t)
    noise = noise_amp * np.random.randn(len(t))

    x = alpha + wander + noise
    return t, x

def main():
    fs = 250
    dur_s = 10

    t, x = synth_eeg(fs=fs, dur_s=dur_s)

    out_dir = os.path.join(os.path.dirname(__file__), "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2026-02-02_synth_eeg_segment.png")

    # Plot first 5 seconds
    n = int(5 * fs)
    plt.figure()
    plt.plot(t[:n], x[:n])
    plt.title("Synthetic EEG segment (5s) | alpha~10 Hz + drift + noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
