"""
Three-experiment comparison: distance / monocular-goggles / disparity-goggles.

Notebook-free by design: a plain top-to-bottom script rather than a .ipynb, so there
is exactly one artifact (this file) and one way to run it (`python
goggles-distance-comparison.py`) -- no separate kernel state that can drift from
what's on disk. Every result is written under `results/comparison/` rather than
displayed inline; open those files afterward.

Only the reliability summary + overlap/breakdown charts are generated -- no per-cell
figures (see the note near the bottom for how to add those back). RHO, the split-half
rho floor used throughout, is the one knob worth editing directly rather than
re-running interactively.

Screen calibration (the deg<->pixel px vs distance-RF scatters) is a separate concern
and lives in distance-disparity-calibration.ipynb, not here.
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless: nothing is shown inline, only saved to disk

base_dir = r"C:\Users\acostamendoza\projects\distance-project"
for p in (base_dir, base_dir + r"\radial-analysis", base_dir + r"\disparity-analysis"):
    if p not in sys.path:
        sys.path.append(p)

import matplotlib.pyplot as plt

from src.data.RadialDevice import RadialDevice
from src.data.MonocularDevice import MonocularDevice
from src.data.DisparityDevice import DisparityDevice
from src.comparison.CrossExperimentAnalysis import CrossExperimentAnalysis

mouse, date = "SAM045", "260715"
R = RadialDevice(mouse_name=mouse, record_date=date, experiment_name="distance")
M = MonocularDevice(mouse_name=mouse, record_date=date, experiment_name="goggles-monocular")
D = DisparityDevice(mouse_name=mouse, record_date=date, experiment_name="goggles-disparity")

RHO = 0.8   # split-half rho floor for the per-experiment breakdown below

cea = CrossExperimentAnalysis.from_devices(R, M, D)

out_dir = os.path.join(D.data_dir, "results", "comparison")
os.makedirs(out_dir, exist_ok=True)

all_three = cea.reliable_all()   # cells reliable in all three experiments (boolean flags)

# --------------------------------------------------------------- 1. text summary
# Captured to a buffer so the same text goes to the console AND to a durable file --
# unlike a notebook's inline output, this script's console output disappears once it
# exits, so the report has to be written down to be worth anything afterward.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cea.print_summary()
    print()
    print(f"Per-experiment, independently thresholded (rho >= {RHO}):")
    for name, ids in cea.reliable_each_thresh(RHO, monocular="all").items():
        print(f"  {name:10s} {len(ids)}")
    print()
    print(f"Reliable in all three (boolean, no threshold): {len(all_three)}")

summary_text = buf.getvalue()
print(summary_text)
with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
    fh.write(summary_text)

# ------------------------------------------------------------- 2. overlap figures
fig = cea.plot_overlap()
fig.savefig(os.path.join(out_dir, "overlap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig = cea.plot_reliable_each(RHO, monocular="all")
fig.savefig(os.path.join(out_dir, "reliable_each.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"saved overlap.png, reliable_each.png -> {out_dir}")

# Per-cell combined figures (CrossExperimentVisualization.make_combined_plot) are not
# generated here -- that's a separate, much slower step (~267 PDFs for this session).
# To add it back: build `viz = CrossExperimentVisualization(R, M, D)` and loop
# `all_three`, saving each `viz.make_combined_plot(int(cell))` under out_dir/cells/.
