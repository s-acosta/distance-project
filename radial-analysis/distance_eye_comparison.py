"""
Right / left / both-eye comparison on the distance device.

The same stimulus is shown with the right eye only, the left eye only and both eyes, as three
sessions ``<date>_<mouse>_distance-right / -left / -both``. All three come from one concatenated
suite2p run, so ROI i is the same cell in every session. Left hemisphere imaged: right eye = contra.
The stimulus code doesn't know which eye is covered, so every session is flagged binocular.

Works for single-distance sessions (e.g. 260928, distance 1 only) and for sessions with every
distance (e.g. 260930, distances 0-3 + control). The three sessions must show the same distances.

Everything is written to ``<processed-data>/<date>_<mouse>_distance-both/results/eye_comparison/``:
    report.txt           everything printed: trial structure, angle axis, scale checks, summary
    eye_table.pkl/.csv   one row per cell: reliable / reliable_corr / RF per eye, resp_*, category
    onset_aligned.png    population mean around stimulus onset (to place the baseline window)
    onset_aligned.npz    the traces behind that figure
    rf_histograms.png    where the RFs of the reliable cells land, per condition + overlaid
    rf_shift.png         RF per eye for the top cells reliable to left and right, one row per cell
    rf_shift_hist.png    histogram of RF right - left for the cells reliable to left and right
    rf_shift_hist_by_distance.png   the same at each distance, one row per distance (several distances only)
    location_map.png     cells on the suite2p field of view, colored by eye (right / left sessions)
    cells/<category>/    one figure per cell. Single distance: left / right / both heatmaps +
                         overlaid means. Several distances: the nasal-vs-temporal figure
                         (CrossSessionVisualizer) with one row per eye: left / right / both.

Each session is first processed on its own with the standard pipeline (``analyze_session``),
cached to the usual ``<session>_dataframe.pkl`` in each session folder (shared with the notebooks).
Only afterwards are the three sessions put on a common angle axis and pooled normalization for
the figures.

Run with the distance-project env. This file sits in radial-analysis/ so that ``src`` resolves to
radial-analysis (disparity-analysis defines its own ``src.analysis.CellAnalysis``):

    python radial-analysis/distance_eye_comparison.py --mouse SAM047 --date 260930

``plot_cell`` can also be imported into a notebook to look at single cells; the script only
switches to the Agg backend when run directly.
"""
import os
import sys
import shutil
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.append(p)

from src.workflows import analyze_session
from src.viz.CellVisualizer import CellVisualizer
from src.viz.CrossSessionVisualizer import CrossSessionVisualizer

EYES = ("right", "left", "both")

# figures: panel / row order and colors (left red, right blue, both purple)
PANEL_ORDER = ("left", "right", "both")
EYE_LABEL = {"left": "Left Eye", "both": "Both Eyes", "right": "Right Eye"}
EYE_COLOR = {"left": "red", "both": "purple", "right": "blue"}
EYE_CMAP = {"left": "Reds", "both": "Purples", "right": "Blues"}
CATEGORY_DIR = {"R-only": "right_only", "L-only": "left_only", "R&L": "right_and_left",
                "B-only": "both_only", "none": "none"}
DIST_COLOR = ["#FFA500", "#35b779", "#31688e", "#440154"]     # distances 0-3, as in CellVisualizer
DIST_CM = {0: 10, 1: 16, 2: 22, 3: 28}                         # ball distance from the axis (right eye)


class Report:
    """prints each line and appends it to a text file as it goes, so a crash keeps what ran"""

    def __init__(self, path):
        self.path = path
        open(path, "w", encoding="utf-8").close()

    def __call__(self, msg=""):
        print(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")


# ---------------------------------------------------------------- Step 0: process each session

def process_sessions(mouse, date, data_dir, n_iterations, recompute, log):
    """
    Standard single-session pipeline per eye (analyze_session), cached to the usual
    <session>_dataframe.pkl. Runs on the untouched devices, i.e. before the common angle axis
    and pooled normalization below, so the cache is the same one the notebooks would make.
    """
    D, stats = {}, {}
    for eye in EYES:
        stats[eye], D[eye] = analyze_session(mouse_name=mouse, record_date=date,
                                             experiment_name=f"distance-{eye}", base_data_dir=data_dir,
                                             show_plots=False, n_iterations=n_iterations,
                                             force_recompute=recompute)
        print()
        log(f"{D[eye].session_name}: stats in {D[eye].data_dir}")
        assert np.array_equal(stats[eye]["cell_id"].values, D[eye].suite2p_id), f"{eye}: cached cells differ"
    # same concatenated suite2p run -> same ROIs in the same order
    for eye in EYES[1:]:
        assert np.array_equal(D[eye].suite2p_id, D["right"].suite2p_id), f"{eye}: ROIs differ from right"
    return D, stats


def shown_distances(D):
    """stimulus distances (control excluded), which must be the same in the three sessions"""
    shown = {eye: tuple(np.unique(d.distance_sequence[d.get_condition_trials()]).tolist())
             for eye, d in D.items()}
    assert len(set(shown.values())) == 1, f"sessions show different distances: {shown}"
    return list(shown["both"])


def describe_sessions(D, log):
    log(f"cells: {len(D['right'].suite2p_id)}")
    for eye, d in D.items():
        dist, n = np.unique(d.distance_sequence, return_counts=True)
        k = d.get_condition_trials()                  # stimulus trials (control excluded)
        log(f"\n{d.session_name}")
        log(f"  frames={d.spikes.shape[1]}  on_length={d.on_length}  "
            f"angles={np.ravel(d.start_angle)[0]:.1f} -> {np.ravel(d.end_angle)[0]:.1f}")
        log(f"  trials per distance={dict(zip(dist.tolist(), n.tolist()))}  "
            f"monocular_sequence={np.unique(d.monocular_sequence).tolist()}")
        log(f"  on duration={np.mean(d.on_end[k] - d.on_start[k]):.2f}s  "
            f"off_start-on_start={np.mean(d.off_start[k] - d.on_start[k]):.2f}s  "
            f"raise={np.ravel(d.raise_time)[0]}  lower={np.ravel(d.lower_time)[0]}  "
            f"off={np.ravel(d.off_time)[0]}")


def align_angle_axis(D):
    """common angle axis: the shortest on-period across the three sessions"""
    sweeps = {(np.ravel(d.start_angle)[0], np.ravel(d.end_angle)[0]) for d in D.values()}
    assert len(sweeps) == 1, f"sweep angles differ: {sweeps}"
    on_length = min(d.on_length for d in D.values())
    for d in D.values():
        d.on_length = on_length
        d.angle_range = np.linspace(d.start_angle, d.end_angle, int(on_length))
    return np.ravel(D["both"].angle_range)


def pooled_normalize(D, log):
    """
    One mu/sd per cell over all three sessions, so amplitudes are comparable across R/L/B
    (RadialDevice z-scores each session on its own). Same smoothing as RadialDevice.
    Reliability and RF position don't change: correlation and argmax are scale-free.
    """
    n_frames = sum(D[eye].spikes.shape[1] for eye in EYES)
    mu = sum(D[eye].spikes.sum(axis=1, keepdims=True) for eye in EYES) / n_frames
    sd = np.sqrt(sum(((D[eye].spikes - mu) ** 2).sum(axis=1, keepdims=True) for eye in EYES) / n_frames)
    for eye in EYES:
        D[eye].response = gaussian_filter1d((D[eye].spikes - mu) / sd, 1.5, axis=1)

    # one suite2p run -> raw spikes should be on the same scale in all three sessions.
    # A session far from 1 would mean it was deconvolved on its own.
    log("\nraw-spike scale per session (median over cells):")
    for eye in EYES:
        s = D[eye].spikes.std(axis=1)
        log(f"  {eye:<5} std={np.median(s):.4g}  std/pooled={np.median(s / sd.ravel()):.3f}")


# ---------------------------------------------------------------- Step 1: trial tensors

def trial_tensor(d):
    """cells x trials x angle for the stimulus trials (control excluded); same slicing as get_on_trace"""
    X = []
    for k in d.get_condition_trials():
        on_start, on_end = d.get_on_times(k)
        mask = (d.frames_time > on_start) & (d.frames_time < on_end)
        X.append(d.response[:, mask][:, :d.on_length])
    return np.stack(X, axis=1)


# ---------------------------------------------------------------- Step 2: reliability

def condition_stats(df):
    # each session holds a single eye condition, so exactly one of the two dicts is filled
    filled = [c for c in ("monocular", "binocular") if df[c].notna().all()]
    assert len(filled) == 1, f"expected one condition per session, got {filled}"
    return pd.DataFrame(df[filled[0]].tolist(), index=df["cell_id"].values)[["reliable", "reliable_corr", "RF"]]


def floor_stats(df, corr_floor):
    """
    Session stats with `reliable` also requiring reliable_corr >= corr_floor, so the per-cell
    figures ("Reliable Condition" / "Reliable General") make the same call as the table.
    """
    if corr_floor is None:
        return df
    df = df.copy()
    for c in ("monocular", "binocular"):
        df[c] = [{**s, "reliable": bool(s["reliable"]) and s["reliable_corr"] >= corr_floor}
                 if isinstance(s, dict) else s for s in df[c]]
    return df


def build_eye_table(stats, corr_floor=None):
    eye_df = pd.concat({eye: condition_stats(stats[eye]) for eye in EYES}, axis=1)
    eye_df.columns = [f"{metric}_{eye}" for eye, metric in eye_df.columns]
    eye_df.index.name = "cell_id"

    for eye in EYES:
        resp = eye_df[f"reliable_{eye}"].astype(bool)
        if corr_floor is not None:
            resp &= eye_df[f"reliable_corr_{eye}"] >= corr_floor
        eye_df[f"resp_{eye}"] = resp

    R, L, B = (eye_df[f"resp_{eye}"] for eye in EYES)
    eye_df["category"] = np.select([R & L, R & ~L, ~R & L, ~R & ~L & B],
                                   ["R&L", "R-only", "L-only", "B-only"], default="none")
    return eye_df


def write_summary(eye_df, log, title, corr_floor=None):
    R, L, B = (eye_df[f"resp_{eye}"] for eye in EYES)
    n_cells, n_any = len(eye_df), (R | L | B).sum()

    log("\n" + "=" * 50)
    log(f" {title}")
    log(f" Total cells: {n_cells}" + (f" | mean_corr floor: {corr_floor}" if corr_floor else ""))
    log("-" * 50)
    for eye, m in zip(EYES, (R, L, B)):
        log(f" {'Reliable ' + eye:<24}: {m.sum():5d}  ({100 * m.mean():5.1f}% of all cells)")
    log(f" {'Reliable in any':<24}: {n_any:5d}  ({100 * n_any / n_cells:5.1f}% of all cells)")
    log("-" * 50)
    log(f" Among those {n_any} cells:")
    for cat, label in [("R-only", "Right only (contra)"), ("L-only", "Left only (ipsi)"),
                       ("R&L", "Both monocular (R & L)"), ("B-only", "Only with both open")]:
        k = (eye_df["category"] == cat).sum()
        log(f" {label:<24}: {k:5d}  ({100 * k / max(n_any, 1):5.1f}%)")
    log("-" * 50)
    log(f" R-reliable also reliable under B: {(R & B).sum()} / {R.sum()}")
    log(f" L-reliable also reliable under B: {(L & B).sum()} / {L.sum()}")
    log("=" * 50)

    # all 8 combinations (rows: right x left, columns: both)
    log("\n" + str(pd.crosstab([R.rename("right"), L.rename("left")], B.rename("both"))))


# ---------------------------------------------------------------- baseline window check

def onset_aligned_mean(d, pre_s):
    """per-cell trial-averaged response aligned to on_start (cells x time), stimulus trials only"""
    t = d.frames_time                                 # ms
    dt = np.median(np.diff(t))
    npre = int(round(pre_s * 1000 / dt))
    acc, n = 0.0, 0
    for k in d.get_condition_trials():
        i0 = np.searchsorted(t, 1000 * d.on_start[k])
        if i0 - npre >= 0 and i0 + d.on_length <= len(t):
            acc = acc + d.response[:, i0 - npre:i0 + d.on_length]
            n += 1
    return acc / n, np.arange(-npre, d.on_length) * dt / 1000


def plot_onset_aligned(D, eye_df, out_dir, pre_s):
    """population mean around stimulus onset, to place the Step 3 baseline window"""
    resp_any = eye_df[[f"resp_{eye}" for eye in EYES]].any(axis=1).to_numpy()
    traces = {}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for eye in PANEL_ORDER:
        m, tt = onset_aligned_mean(D[eye], pre_s)
        traces[eye] = m[resp_any].mean(axis=0)
        ax.plot(tt, traces[eye], label=eye)
    d = D["both"]
    k = d.get_condition_trials()
    ax.axvline(0, color="k", lw=0.8, label="on_start")
    ax.axvline(np.mean(d.off_start[k] - d.on_start[k]), color="gray", ls="--", label="off_start")
    ax.set(xlabel="time from on_start (s)", ylabel="pooled z (reliable in any)",
           title="pick a baseline where nothing is evoked")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "onset_aligned.png"), dpi=150)
    plt.close(fig)
    np.savez(os.path.join(out_dir, "onset_aligned.npz"), t=tt, **traces)


def plot_rf_histograms(eye_df, angles, out_dir, log, title, frames_per_bin=4):
    """
    Where the RFs of the reliable cells land, one panel per condition (RF = sweep angle at the
    peak of the trial-averaged response), then the three overlaid as fractions of each
    condition's reliable cells. Bins are whole groups of frames, so every bin holds the same
    number of possible RF values (5° bins on the ~1.5° angle grid would alias).
    """
    da = angles[1] - angles[0]
    edges = angles[0] - da / 2 + da * np.arange(0, len(angles) + 1, frames_per_bin)
    if edges[-1] < angles[-1] + da / 2:
        edges = np.append(edges, angles[-1] + da / 2)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    log("\nRF of reliable cells (deg):")
    for ax, eye in zip(axes[:3], PANEL_ORDER):
        rf = eye_df.loc[eye_df[f"resp_{eye}"], f"RF_{eye}"].to_numpy(dtype=float)
        ax.hist(rf, bins=edges, color=EYE_COLOR[eye], alpha=0.8, edgecolor="white")
        ax.set(xlim=(edges[0], edges[-1]), xlabel="RF (deg)", ylabel="Cells",
               title=f"{EYE_LABEL[eye]} (n={len(rf)})")
        if len(rf):
            axes[3].hist(rf, bins=edges, weights=np.full(len(rf), 1 / len(rf)), histtype="step",
                         linewidth=2, color=EYE_COLOR[eye], label=f"{EYE_LABEL[eye]} (n={len(rf)})")
            # first / last frame of the sweep: the true peak may lie beyond the swept range
            at_edge = (rf <= angles[0] + da / 2) | (rf >= angles[-1] - da / 2)
            log(f"  {eye:<5} n={len(rf):4d}  median={np.median(rf):6.1f}  at sweep edge: {at_edge.mean():4.0%}")
        else:
            log(f"  {eye:<5} n=   0")

    axes[3].set(xlim=(edges[0], edges[-1]), xlabel="RF (deg)", ylabel="Fraction of reliable cells",
                title="All conditions")
    axes[3].legend(frameon=False, fontsize=12)
    fig.suptitle(f"{title}: RF of reliable cells")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_histograms.png"), dpi=150)
    plt.close(fig)


def top_lr_cells(eye_df, top_n):
    """cells reliable to both left and right, and the top_n of them ranked by the weaker eye's corr"""
    lr = eye_df[eye_df["resp_left"] & eye_df["resp_right"]]
    weaker = lr[["reliable_corr_left", "reliable_corr_right"]].min(axis=1)
    return lr, lr.loc[weaker.sort_values(ascending=False).index[:top_n]]


def plot_rf_shift(eye_df, X, angles, out_dir, log, title, top_n=50, s_max=70):
    """
    RF position per eye for the top_n cells reliable to both left and right, ranked by the weaker
    eye's split-half corr. One row per cell, sorted by mean left/right RF; a gray line spans the
    cell's RFs. Red / blue = left / right RF, circle area = that eye's share of the response,
    peak / (peak_left + peak_right) of the pooled-z trial average (ODI-like). Purple ring = RF
    with both eyes open, drawn only if the cell is also reliable with both eyes.
    """
    lr, top = top_lr_cells(eye_df, top_n)
    log(f"\nRF shift plot: top {len(top)} of {len(lr)} cells reliable to left and right")
    if top.empty:
        return
    n = len(top)

    # eye_df rows follow suite2p order, same as the trial tensors
    idx = eye_df.index.get_indexer(top.index)
    peak = {eye: np.clip(X[eye][idx].mean(axis=1).max(axis=1), 0, None) for eye in ("left", "right")}
    total = peak["left"] + peak["right"]
    share = {eye: np.divide(peak[eye], total, out=np.full(n, 0.5), where=total > 0)
             for eye in ("left", "right")}
    rf = {eye: top[f"RF_{eye}"].to_numpy(dtype=float) for eye in EYES}
    has_both = top["resp_both"].to_numpy(dtype=bool)

    # row 0 at the top
    y = np.empty(n)
    y[np.argsort((rf["left"] + rf["right"]) / 2)] = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(7, min(1.5 + 0.12 * n, 16)))
    lo, hi = np.minimum(rf["left"], rf["right"]), np.maximum(rf["left"], rf["right"])
    lo = np.where(has_both, np.minimum(lo, rf["both"]), lo)
    hi = np.where(has_both, np.maximum(hi, rf["both"]), hi)
    ax.hlines(y, lo, hi, color="0.75", linewidth=0.8, zorder=1)
    for eye in ("left", "right"):
        ax.scatter(rf[eye], y, s=s_max * share[eye], color=EYE_COLOR[eye], edgecolor="white",
                   linewidth=0.4, alpha=0.9, zorder=2)
    # hollow, so it can sit on top of the eye circles without hiding them
    ax.scatter(rf["both"][has_both], y[has_both], s=30, facecolor="none",
               edgecolor=EYE_COLOR["both"], linewidth=1.2, zorder=3)

    ax.set_xlim(angles[0] - 2, angles[-1] + 2)
    ax.set_ylim(-1, n)
    ax.set_xlabel("RF position (deg)", fontsize=12)
    ax.set_ylabel(f"Top {n} L&R cells (sorted by mean RF)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    if n <= 60:
        ax.set_yticks(y)
        ax.set_yticklabels(top.index, fontsize=7)
    else:
        ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [ax.scatter([], [], s=40, color=EYE_COLOR[eye], label=EYE_LABEL[eye]) for eye in ("left", "right")]
    handles.append(ax.scatter([], [], s=30, facecolor="none", edgecolor=EYE_COLOR["both"], linewidth=1.2,
                              label="Both Eyes (if reliable)"))
    for f in (0.25, 0.5, 0.75, 1.0):
        handles.append(ax.scatter([], [], s=s_max * f, color="0.5", label=f"eye share {f:.0%}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              fontsize=9, labelspacing=1.1)

    ax.set_title(f"{title}\nRF position by eye", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_shift.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rf_shift_hist(eye_df, angles, out_dir, log, title, top_n=50, frames_per_bin=3):
    """
    Histogram of RF_right - RF_left for every cell reliable to left and right (gray), with the
    top_n cells of rf_shift.png outlined. Differences are whole multiples of the ~1.5 deg angle
    step, so bins are an odd number of steps centered on 0 and no edge falls on a possible value.
    """
    assert frames_per_bin % 2 == 1, "frames_per_bin must be odd"
    lr, top = top_lr_cells(eye_df, top_n)
    if lr.empty:
        return
    d_all = (lr["RF_right"] - lr["RF_left"]).to_numpy(dtype=float)
    d_top = (top["RF_right"] - top["RF_left"]).to_numpy(dtype=float)

    w = frames_per_bin * (angles[1] - angles[0])
    k = int(np.ceil(np.abs(d_all).max() / w))
    edges = (np.arange(-k, k + 2) - 0.5) * w           # bin centers at -k*w .. k*w

    log("\nRF right - left (deg):")
    for label, d in ((f"all L&R  n={len(d_all):4d}", d_all), (f"top {len(d_top):<4d}   ", d_top)):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        log(f"  {label} median={med:+6.1f}  IQR=[{q1:+.1f}, {q3:+.1f}]  "
            f"within ±{w / 2:.1f}: {np.mean(np.abs(d) < w / 2):4.0%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_all, bins=edges, color="0.75", edgecolor="white", label=f"All L&R cells (n={len(d_all)})")
    ax.hist(d_top, bins=edges, histtype="step", color="black", linewidth=1.5,
            label=f"Top {len(d_top)} (rf_shift.png)")
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    med = np.median(d_all)
    ax.axvline(med, color="black", linewidth=1, label=f"Median, all: {med:+.1f}°")
    ax.set_xlabel("RF right − RF left (deg)", fontsize=12)
    ax.set_ylabel("Cells", fontsize=12)
    ax.tick_params(labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{title}\nRF shift between eyes", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_shift_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def rf_by_distance(d, X_eye, angles):
    """per-distance RF (deg): peak of each distance's trial average; X_eye rows in suite2p order"""
    dist = np.ravel(d.distance_sequence)[d.get_condition_trials()]   # same trial order as trial_tensor
    ds = np.unique(dist)
    return ds, np.stack([angles[X_eye[:, dist == k].mean(axis=1).argmax(axis=1)] for k in ds], axis=1)


def plot_rf_shift_hist_by_distance(D, X, eye_df, angles, out_dir, log, title, top_n=50, frames_per_bin=3):
    """
    RF_right - RF_left at each distance (one row per distance), for the cells reliable to left and
    right (filled, distance colors) with the top_n cells of rf_shift.png outlined. Per-distance RF =
    peak of that distance's trial average, in device angle: 0 means the ball at that distance hits
    both eyes' RFs at the same point of the sweep. Same bins as rf_shift_hist.png.
    """
    assert frames_per_bin % 2 == 1, "frames_per_bin must be odd"
    lr, top = top_lr_cells(eye_df, top_n)
    if lr.empty:
        return
    ds, rf_left = rf_by_distance(D["left"], X["left"], angles)
    ds_right, rf_right = rf_by_distance(D["right"], X["right"], angles)
    assert np.array_equal(ds, ds_right), "left and right show different distances"
    diff = rf_right - rf_left                                          # all cells x distances
    rows_all, rows_top = eye_df.index.get_indexer(lr.index), eye_df.index.get_indexer(top.index)

    w = frames_per_bin * (angles[1] - angles[0])
    k = int(np.ceil(np.abs(diff[rows_all]).max() / w))
    edges = (np.arange(-k, k + 2) - 0.5) * w                           # bin centers at -k*w .. k*w

    fig, axes = plt.subplots(len(ds), 1, figsize=(7, 2.2 * len(ds) + 1), sharex=True, sharey=True,
                             squeeze=False)
    log("\nRF right - left per distance (deg, device angle):")
    for ax, i, dk in zip(axes[:, 0], range(len(ds)), ds):
        d_all, d_top = diff[rows_all, i], diff[rows_top, i]
        q1, med, q3 = np.percentile(d_all, [25, 50, 75])
        log(f"  distance {dk}: all n={len(d_all)} median={med:+6.1f} IQR=[{q1:+.1f}, {q3:+.1f}]   "
            f"top {len(d_top)} median={np.median(d_top):+6.1f}")
        ax.hist(d_all, bins=edges, color=DIST_COLOR[int(dk)], edgecolor="white",
                label=f"All L&R cells (n={len(d_all)})")
        ax.hist(d_top, bins=edges, histtype="step", color="black", linewidth=1.5, label=f"Top {len(d_top)}")
        ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
        ax.axvline(med, color="black", linewidth=1, label="Median")
        ax.text(0.99, 0.92, f"median {med:+.1f}°", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        cm = f" ({DIST_CM[int(dk)]} cm)" if int(dk) in DIST_CM else ""
        ax.set_title(f"Distance {dk}{cm}", fontsize=12, loc="left")
        ax.set_ylabel("Cells", fontsize=11)
        ax.tick_params(labelsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[-1, 0].set_xlabel("RF right − RF left (deg, device angle)", fontsize=12)
    fig.suptitle(f"{title}\nRF shift between eyes, per distance", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_shift_hist_by_distance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_location_map(D, eye_df, out_dir, cell_size=60):
    """
    Cells on the suite2p field of view, colored by which eye reliably drives them, from the
    right and left sessions only (same layout as the monocular map): left red, right blue,
    both eyes purple. Cells reliable to neither eye (both-only cells included) are small and gray.
    """
    plane = os.path.join(D["both"].suite2p_dir, "plane0")
    stat = np.load(os.path.join(plane, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(plane, "ops.npy"), allow_pickle=True).item()
    bg = ops["refImg"] if "refImg" in ops else ops["meanImg"]

    def cell_xy(cell):
        # centroid of the non-overlapping pixels (all of them if the cell is buried under overlaps)
        s = stat[cell]
        keep = ~s["overlap"]
        if not keep.any():
            keep = np.ones_like(s["overlap"], dtype=bool)
        return s["xpix"][keep].mean(), s["ypix"][keep].mean()

    R, L = eye_df["resp_right"], eye_df["resp_left"]
    groups = [("none", ~R & ~L, "lightgray"), ("left", L & ~R, "red"),
              ("right", R & ~L, "blue"), ("left + right", R & L, "purple")]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(bg, cmap="gray", origin="upper", vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99),
              extent=[0, bg.shape[1], bg.shape[0], 0])
    for label, mask, color in groups:
        cells = eye_df.index[mask]
        xy = np.array([cell_xy(c) for c in cells]).reshape(-1, 2)
        faint = label == "none"
        ax.scatter(xy[:, 0], xy[:, 1], c=color, edgecolor="black", linewidth=0.4,
                   s=cell_size * 0.35 if faint else cell_size, alpha=0.5 if faint else 0.95,
                   label=f"{label} (n={len(cells)})")
    ax.invert_xaxis()
    ax.invert_yaxis()                                 # same FOV orientation as the monocular map
    ax.axis("off")
    ax.set_title(f"{D['both'].record_date}_{D['both'].mouse_name} distance — cells in field (right / left)")
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "location_map.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- per-cell figures

def plot_cell(cell, D, X, eye_df, angles):
    """
    Four panels: left-eye, right-eye and both-eyes trial heatmaps (trials x angle), then the
    three mean ± SEM tuning curves overlaid with each condition's RF angle. Traces are pooled z,
    so all three share one color scale and are directly comparable. * = reliable in that condition.
    """
    idx = np.where(D["both"].suite2p_id == cell)[0][0]
    row = eye_df.loc[cell]
    pooled = np.concatenate([D[eye].response[idx] for eye in EYES])
    vmin, vmax = np.percentile(pooled, [1, 99])

    da = angles[1] - angles[0]
    edges = np.append(angles - da / 2, angles[-1] + da / 2)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, eye in zip(axes[:3], PANEL_ORDER):
        traces = X[eye][idx]                          # trials x angle
        ax.pcolormesh(edges, np.arange(traces.shape[0] + 1), traces,
                      vmin=vmin, vmax=vmax, cmap=EYE_CMAP[eye])
        star = " *" if row[f"resp_{eye}"] else ""
        ax.set(xlabel="Angle (deg)", ylabel="Trial",
               title=f"{EYE_LABEL[eye]} (corr: {row[f'reliable_corr_{eye}']:.2f}){star}")

    ax = axes[3]
    lo, hi = vmin, vmax
    for eye in PANEL_ORDER:
        traces = X[eye][idx]
        mean, sem = traces.mean(axis=0), traces.std(axis=0) / np.sqrt(traces.shape[0])
        ax.fill_between(angles, mean - sem, mean + sem, alpha=0.2, color=EYE_COLOR[eye])
        ax.plot(angles, mean, linewidth=2, color=EYE_COLOR[eye],
                label=f"{EYE_LABEL[eye]} (RF: {angles[np.argmax(mean)]:.1f}°)")
        lo, hi = min(lo, (mean - sem).min()), max(hi, (mean + sem).max())
    ax.set(xlim=(angles[0], angles[-1]), ylim=(lo, hi), xlabel="Angle (deg)",
           ylabel="Activity (pooled z)", title="Mean Trace ± SEM")
    ax.legend(frameon=False, fontsize=12)

    fig.suptitle(f"Cell {cell}  ({row['category']};  * = reliable)")
    fig.tight_layout()
    return fig


def distance_viz(D, stats, corr_floor=None):
    """
    CrossSessionVisualizer for several distances: one row per eye (left / right / both), stats
    from the cache with the corr floor applied, so each row's reliable call matches the table.
    """
    return CrossSessionVisualizer([(EYE_LABEL[eye], CellVisualizer(D[eye])) for eye in PANEL_ORDER],
                                  stats={EYE_LABEL[eye]: floor_stats(stats[eye], corr_floor)
                                         for eye in PANEL_ORDER})


def plot_cell_distances(cell, cross_viz, eye_df):
    """
    The nasal-vs-temporal figure with three rows (left eye / right eye / both eyes): all-trace heatmap,
    polar mean, per-distance tuning curves, RF boxplot by distance, stats. Traces are pooled z
    and all rows share one color / y scale.
    """
    fig = cross_viz.plot_cell(cell, show_condition=False)
    labels = " vs ".join(EYE_LABEL[eye] for eye in PANEL_ORDER)
    fig.suptitle(f"Cell {cell} — {labels}  ({eye_df.loc[cell, 'category']})", fontsize=20)
    return fig


def save_cell_plots(plot, eye_df, out_dir, cells=None):
    """
    One figure per cell in cells/<category>/cell_<id>.png, drawn by plot(cell). By default every
    cell reliable in any condition, after clearing cells/ so a rerun with other criteria leaves
    no stale figures.
    """
    cells_dir = os.path.join(out_dir, "cells")
    if cells is None:
        cells = eye_df.index[eye_df["category"] != "none"].tolist()
        shutil.rmtree(cells_dir, ignore_errors=True)

    for i, cell in enumerate(cells, 1):
        print(f"Plotting cell {i} of {len(cells)}", end="\r", flush=True)
        cat_dir = os.path.join(cells_dir, CATEGORY_DIR[eye_df.loc[cell, "category"]])
        os.makedirs(cat_dir, exist_ok=True)
        fig = plot(cell)
        fig.savefig(os.path.join(cat_dir, f"cell_{cell}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print()
    return cells_dir, len(cells)


def main():
    ap = argparse.ArgumentParser(description="Right / left / both-eye comparison on the distance device")
    ap.add_argument("--mouse", default="SAM047")
    ap.add_argument("--date", default="260930")
    ap.add_argument("--data-dir", default=r"D:/data-distance-coding-project/processed-data")
    ap.add_argument("--n-iterations", type=int, default=200)
    ap.add_argument("--corr-floor", type=float, default=0.85,
                    help="mean split-half corr floor on top of `reliable` (default 0.85, as in "
                         "SessionVisualizer.reliable_classes; 0 switches it off)")
    ap.add_argument("--rf-top-n", type=int, default=50,
                    help="cells in rf_shift.png: the top N reliable to left and right")
    ap.add_argument("--pre-s", type=float, default=4.0,
                    help="seconds before onset shown in onset_aligned.png")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the cached <session>_dataframe.pkl files")
    ap.add_argument("--cells", type=int, nargs="+", default=None,
                    help="plot only these cell ids (default: every cell reliable in any condition)")
    ap.add_argument("--skip-cells", action="store_true",
                    help="skip the per-cell figures (the slow part)")
    args = ap.parse_args()
    plt.switch_backend("Agg")

    title = f"{args.date}_{args.mouse} distance: right / left / both"
    out_dir = os.path.join(args.data_dir, f"{args.date}_{args.mouse}_distance-both", "results", "eye_comparison")
    os.makedirs(out_dir, exist_ok=True)
    log = Report(os.path.join(out_dir, "report.txt"))
    log(f"{title}\nargs: {vars(args)}\n")

    # Step 0: process each session on its own (standard pipeline, cached), then checks,
    # common angle axis and pooled normalization for the figures
    D, stats = process_sessions(args.mouse, args.date, args.data_dir, args.n_iterations, args.recompute, log)
    describe_sessions(D, log)
    distances = shown_distances(D)
    log(f"\nstimulus distances (all three sessions): {distances}"
        + ("  -> single-distance figures" if len(distances) == 1 else "  -> distance-tuning figures"))
    angles = align_angle_axis(D)
    log(f"\ncommon on_length: {len(angles)} frames ({angles[0]:.1f} -> {angles[-1]:.1f} deg)")
    pooled_normalize(D, log)

    # Step 1: trial tensors (cells x trials x angle)
    X = {eye: trial_tensor(D[eye]) for eye in EYES}
    log("\ntrial tensors (cells x trials x angle):")
    for eye in EYES:
        log(f"  {eye:<5} {X[eye].shape}")

    # Step 2: reliability per condition and categories
    eye_df = build_eye_table(stats, args.corr_floor)
    write_summary(eye_df, log, title, args.corr_floor)
    eye_df.to_pickle(os.path.join(out_dir, "eye_table.pkl"))
    eye_df.to_csv(os.path.join(out_dir, "eye_table.csv"))
    plot_rf_histograms(eye_df, np.ravel(angles), out_dir, log, title)
    plot_rf_shift(eye_df, X, np.ravel(angles), out_dir, log, title, args.rf_top_n)
    plot_rf_shift_hist(eye_df, np.ravel(angles), out_dir, log, title, args.rf_top_n)
    if len(distances) > 1:
        plot_rf_shift_hist_by_distance(D, X, eye_df, np.ravel(angles), out_dir, log, title, args.rf_top_n)

    plot_onset_aligned(D, eye_df, out_dir, args.pre_s)
    plot_location_map(D, eye_df, out_dir)

    if not args.skip_cells:
        if len(distances) == 1:
            plot = lambda cell: plot_cell(cell, D, X, eye_df, np.ravel(angles))
        else:
            cross_viz = distance_viz(D, stats, args.corr_floor)
            plot = lambda cell: plot_cell_distances(cell, cross_viz, eye_df)
        cells_dir, n = save_cell_plots(plot, eye_df, out_dir, args.cells)
        log(f"\nSaved {n} cell figures to: {cells_dir}")
    log(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
