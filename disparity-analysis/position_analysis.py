#!/usr/bin/env python3
"""position_analysis.py – Per-position dF/F response extraction.

Stim parameters are loaded from any *.mat in IMAGE_DIR if present;
otherwise falls back to positions.csv + position_order.csv from CSV_DIR.

Output: {IMAGE_DIR}/output/preprocessing.npz
        {IMAGE_DIR}/output/position_responses.png
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from natsort import natsorted
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_DIR    = Path(r"D:\data-distance-coding-project\raw-data\260219_SAM041_goggles_left")
CSV_DIR      = Path(r"C:\Users\acostamendoza\projects\distance-project\hardware-control\disparity-device")
FS           = 10.0  # imaging frame rate (Hz)
BASELINE_DUR = 2.0   # seconds of off period to use for F0
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = IMAGE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
npz_path = OUTPUT_DIR / "preprocessing.npz"

if npz_path.exists():
    print(f"Loading cached results from {npz_path}")
    _data              = np.load(npz_path)
    f0                 = _data["f0"]
    position_responses = _data["position_responses"]
    positions          = _data["positions"]
    on_time            = float(_data["on_time"])

else:
    # ── Load stimulus data ────────────────────────────────────────────────────
    mat_files = sorted(IMAGE_DIR.glob("*.mat"))
    if mat_files:
        print(f"Loading stim from {mat_files[0].name}")
        raw = loadmat(str(mat_files[0]), squeeze_me=True, struct_as_record=False)
        key = next(k for k in raw if not k.startswith("__"))
        sd             = raw[key]
        stim_on        = np.atleast_2d(sd.stim_on)               # (n_repeats, n_positions), seconds
        position_order = np.atleast_2d(sd.position_order).astype(int)  # 1-based
        positions      = np.atleast_1d(sd.positions)
        on_time        = float(sd.on_time)
        off_time       = float(sd.off_time)
        n_repeats      = int(sd.repeats)
    else:
        print("No .mat in IMAGE_DIR – reconstructing timing from CSVs")
        positions      = np.loadtxt(str(CSV_DIR / "positions.csv"), delimiter=",")
        position_order = np.loadtxt(str(CSV_DIR / "position_order.csv"), delimiter=",", dtype=int)
        on_time        = 2.0
        off_time       = 4.0
        n_repeats      = position_order.shape[0]
        n_pos          = len(positions)
        stim_on = np.array([
            (rep * n_pos + np.arange(n_pos)) * (on_time + off_time) + off_time
            for rep in range(n_repeats)
        ])

    n_positions     = len(positions)
    on_frames       = int(round(on_time * FS))
    baseline_frames = int(round(BASELINE_DUR * FS))

    print(f"  {n_positions} positions, on={on_time}s, off={off_time}s, repeats={n_repeats}")
    print(f"  on_frames={on_frames}, baseline_frames={baseline_frames}")

    # ── Discover frames ───────────────────────────────────────────────────────
    frame_paths = natsorted(IMAGE_DIR.glob("*.tif"))
    n_frames    = len(frame_paths)
    print(f"Found {n_frames} frames")

    sample = tifffile.imread(str(frame_paths[0]))
    H, W   = sample.shape
    print(f"Frame size: {H}x{W}, dtype: {sample.dtype}")

    # ── Block reader ──────────────────────────────────────────────────────────
    _executor = ThreadPoolExecutor(max_workers=16)

    def _read_block(start: int, end: int) -> np.ndarray:
        """Read frames [start, end) and return (H, W, count) float32."""
        paths  = frame_paths[start:end]
        frames = list(_executor.map(lambda p: tifffile.imread(str(p)).astype(np.float32), paths))
        return np.stack(frames, axis=-1)

    # ── Compute global F0 (running mean across all blank windows) ─────────────
    # Mean keeps memory at O(H×W); median would require ~23 GB at float32.
    print("Computing global F0 ...")
    f0_sum   = np.zeros((H, W), dtype=np.float64)
    f0_count = 0

    for rep in range(n_repeats):
        for trial in range(stim_on.shape[1]):
            stim_start = int(round(stim_on[rep, trial] * FS))
            bl_start   = max(stim_start - baseline_frames, 0)
            chunk      = _read_block(bl_start, stim_start)  # (H, W, baseline_frames)
            f0_sum    += chunk.sum(axis=-1)
            f0_count  += chunk.shape[-1]

    f0 = (f0_sum / f0_count).astype(np.float32)
    print(f"F0 shape: {f0.shape}  ({f0_count} blank frames averaged)")

    # ── Extract per-position responses ────────────────────────────────────────
    print("Extracting per-position responses ...")
    f0_3d          = f0[:, :, np.newaxis]   # (H, W, 1) for broadcasting
    response_accum = np.zeros((n_positions, H, W, on_frames), dtype=np.float64)
    response_count = np.zeros(n_positions, dtype=int)

    for rep in tqdm(range(n_repeats), desc="Repeats"):
        for trial in range(stim_on.shape[1]):
            stim_start = int(round(stim_on[rep, trial] * FS))
            stim_end   = stim_start + on_frames
            if stim_end > n_frames:
                continue

            stim_block = _read_block(stim_start, stim_end)       # (H, W, on_frames)
            dff        = (stim_block - f0_3d) / f0_3d * 100

            pos_idx = int(position_order[rep, trial]) - 1        # 0-based
            response_accum[pos_idx] += dff
            response_count[pos_idx] += 1

    position_responses = response_accum / response_count[:, None, None, None]
    print(f"position_responses: {position_responses.shape}")
    print(f"Trials per position: {response_count}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        npz_path,
        f0=f0,
        position_responses=position_responses,
        positions=positions,
        position_order=position_order,
        on_time=np.float64(on_time),
        off_time=np.float64(off_time),
        repeats=np.int64(n_repeats),
    )
    print(f"Saved → {npz_path}")

# ── Plot mean response per position ──────────────────────────────────────────
n_positions = len(positions)
n_rows      = 2
n_cols      = math.ceil(n_positions / n_rows)
pos_deg     = -25 + (positions / 240) * (120 - (-25))

mean_maps = position_responses.mean(axis=-1)   # (n_positions, H, W)
vmin = np.percentile(mean_maps, 1)
vmax = np.percentile(mean_maps, 99)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = np.array(axes).ravel()

for i in range(n_positions):
    im = axes[i].imshow(mean_maps[i], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[i].set_title(f"{positions[i]:.0f} px ({pos_deg[i]:.1f}°)")
    axes[i].axis("off")

for ax in axes[n_positions:]:
    ax.set_visible(False)

fig.suptitle(f"Mean dF/F per position (time-averaged) — {IMAGE_DIR.name}", fontsize=14)
plt.tight_layout()
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="dF/F (%)")

fig_path = OUTPUT_DIR / "position_responses.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {fig_path}")
