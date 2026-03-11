import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATE_TAG = "2026-03-11"
SRC_REPORT = "report_2026-03-09"
SRC_FIG = "00_meg_pipeline_demo_v1.png"

BULLETS = [
    "What: One-step MEG pipeline summary (raw→filter→PSD→event-locked evoked).",
    "Shows: Filtering removes drift/line-noise and stabilizes the spectrum without erasing signal.",
    "Shows: Event-locked averaging reveals a consistent response near ~100 ms that raw time-series hides.",
    "Why it matters: This is the minimal reproducible MEG workflow used before any fancy modeling/source mapping.",
    "Output: Single-run artifact + run folder makes results comparable across future preprocessing changes.",
]

def main():
    meg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(meg_dir, "RESULTS", SRC_REPORT, SRC_FIG)

    out_dir = os.path.join(meg_dir, "RESULTS", f"report_{DATE_TAG}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "00_meg_results_card_v1.png")

    img = mpimg.imread(src_path)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title("MEG Results Card v1 — Pipeline Demo", fontsize=18, pad=10)

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    y = 0.95
    for b in BULLETS:
        ax1.text(0.02, y, f"• {b}", fontsize=12, va="top")
        y -= 0.18

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    main()