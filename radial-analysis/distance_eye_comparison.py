"""
Right / left / both-eye comparison on the distance device. Left hemisphere imaged: right eye = contra.

Two recording layouts, picked automatically:

* interleaved (from 261006 on): one session ``<date>_<mouse>_distance-eyes`` recorded with
  radialDeviceControl_v3. The eye condition comes in blocks (eye_sequence 1 left, 2 right,
  3 both) and sweeps alternate direction. The session is split into one trial subset per eye
  (RadialDevice.select_trials). Both directions are pooled: reverse sweeps are flipped onto the
  start -> end angle axis (RadialDevice.get_on_trace), which assumes no response lag
  (checked in direction_check.png).
* separate sessions (260928, 260930): ``<date>_<mouse>_distance-right / -left / -both`` from one
  concatenated suite2p run, so ROI i is the same cell in every session. The stimulus code
  didn't know which eye was covered, so every session is flagged binocular.

Works for single-distance sessions (e.g. 260928, distance 1 only) and for sessions with every
distance (e.g. 260930, distances 0-3 + control). The three eye conditions must show the same distances.

Everything is written to ``<processed-data>/<date>_<mouse>_distance-eyes/results/eye_comparison/``
(``distance-both`` for separate sessions):
    report.txt           everything printed: trial structure, angle axis, scale checks, summary
    eye_table.pkl/.csv   one row per cell: reliable / reliable_corr / RF per eye, resp_*, category
    onset_aligned.png    population mean around stimulus onset (to place the baseline window)
    onset_aligned.npz    the traces behind that figure
    rf_histograms.png    where the RFs of the reliable cells land, per condition + overlaid
    rf_shift.png         RF per eye for the top cells reliable to left and right, one row per cell
    rf_shift_hist.png    histogram of RF right - left for the cells reliable to left and right
    rf_shift_hist_by_distance.png   the same at each distance, one row per distance (several distances only)
    location_map.png     cells on the suite2p field of view, colored by eye (right / left sessions)
    direction_check.png  interleaved only: RF forward - reverse sweep, per eye (pooling check)
    cells/<category>/    one figure per cell. Single distance: left / right / both heatmaps +
                         overlaid means. Several distances: the nasal-vs-temporal figure
                         (CrossSessionVisualizer) with one row per eye: left / right / both.
  interleaved with several distances, distance modulation per sweep direction:
    direction_table.pkl/.csv     per cell, eye and direction (pooled / fwd / rev): reliable, RF,
                                 ANOVA p, modulated; distance x direction ANOVA; preferred distance
    direction_modulation.png     % modulated per direction and 2-way terms per eye; preferred
                                 distance forward vs reverse
    cells_directions/<group>/    both-eye cells behind each bar of direction_modulation.png (forward,
                                 reverse, both_directions, pooled, 2way_distance, 2way_dist_x_dir,
                                 2way_direction), rows pooled / forward / reverse; a cell in several
                                 groups is copied into each

Each eye condition is first processed on its own with the standard pipeline, cached to
``<session>_dataframe.pkl`` in each session folder (interleaved: ``<session>_<eye>_dataframe.pkl``
and per direction ``<session>_<eye>_<fwd|rev>_dataframe.pkl`` in the one session folder), shared
with the notebooks. Only afterwards are the three put on a common angle axis and pooled
normalization for the figures.

Run with the distance-project env. This file sits in radial-analysis/ so that ``src`` resolves to
radial-analysis (disparity-analysis defines its own ``src.analysis.CellAnalysis``):

    python radial-analysis/distance_eye_comparison.py --mouse SAM047 --date 261006
    python radial-analysis/distance_eye_comparison.py --mouse SAM047 --date 260930

``plot_cell`` can also be imported into a notebook to look at single cells; the script only
switches to the Agg backend when run directly.
"""
import io
import os
import sys
import pickle
import shutil
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import f as f_dist

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.append(p)

from src.workflows import analyze_session
from src.data.RadialDevice import RadialDevice
from src.analysis.CellAnalysis import SessionAnalyzer
from src.viz.CellVisualizer import CellVisualizer
from src.viz.CrossSessionVisualizer import CrossSessionVisualizer

EYES = ("right", "left", "both")
INTERLEAVED = "distance-eyes"                                  # radialDeviceControl_v3 session
EYE_CODE = {"left": 1, "right": 2, "both": 3}                  # eye_sequence / block_order
DIRECTIONS = {"fwd": 1, "rev": -1}                             # direction_sequence: start -> end angle / back
DIR_LABEL = {"fwd": "forward", "rev": "reverse"}
MOD_CATEGORIES = ("both-dirs", "fwd-only", "rev-only", "pooled-only")   # direction_table: mod_category_both

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


def is_interleaved(mouse, date, data_dir):
    """one interleaved session (v3 stimulus) rather than three sessions, one per eye"""
    return os.path.isdir(os.path.join(data_dir, f"{date}_{mouse}_{INTERLEAVED}"))


def results_dir(mouse, date, data_dir):
    experiment = INTERLEAVED if is_interleaved(mouse, date, data_dir) else "distance-both"
    return os.path.join(data_dir, f"{date}_{mouse}_{experiment}", "results", "eye_comparison")


def eye_devices(mouse, date, data_dir):
    """
    The three eye conditions as RadialDevices, without stats: the eye views of the interleaved
    session, or the three sessions. Other scripts (disparity_distance.py) load data through this.
    """
    if is_interleaved(mouse, date, data_dir):
        device = RadialDevice(mouse_name=mouse, record_date=date, experiment_name=INTERLEAVED,
                              base_data_dir=data_dir)
        return {eye: interleaved_view(device, eye) for eye in EYES}
    return {eye: RadialDevice(mouse_name=mouse, record_date=date, experiment_name=f"distance-{eye}",
                              base_data_dir=data_dir) for eye in EYES}


def interleaved_view(device, eye, direction=None):
    """trials of one eye condition (and sweep direction, 'fwd' / 'rev') of an interleaved session"""
    keep = device.eye_sequence == EYE_CODE[eye]
    if direction is None:
        return device.select_trials(keep, label=eye)
    return device.select_trials(keep & (device.direction_sequence == DIRECTIONS[direction]),
                                label=f"{eye}_{direction}")


def _save_stats(view, n_iterations, path):
    df = SessionAnalyzer(view).analyze_session_stats(n_iterations=n_iterations)
    with open(path, "wb") as f:
        pickle.dump(df, f)


def _analyze_condition(task):
    """worker process: loads its own RadialDevice, runs one condition, pickles it to path"""
    mouse, date, data_dir, eye, direction, n_iterations, path = task
    with contextlib.redirect_stdout(io.StringIO()):          # per-cell progress lines would interleave
        device = RadialDevice(mouse_name=mouse, record_date=date, experiment_name=INTERLEAVED,
                              base_data_dir=data_dir)
        _save_stats(interleaved_view(device, eye, direction), n_iterations, path)
    return path


def process_interleaved(mouse, date, data_dir, n_iterations, recompute, jobs, log):
    """
    Interleaved session: one trial subset per eye (both directions pooled) and per eye and sweep
    direction, each run through the standard single-session analysis (SessionAnalyzer, as
    analyze_session does) and cached to <session>_<eye>[_fwd|_rev]_dataframe.pkl in the session
    folder. Missing caches are computed `jobs` at a time in separate processes.
    Returns the pooled views / stats per eye and the per-direction ones keyed (eye, 'fwd'|'rev').
    """
    device = RadialDevice(mouse_name=mouse, record_date=date, experiment_name=INTERLEAVED,
                          base_data_dir=data_dir)
    assert hasattr(device, "eye_sequence"), f"{device.session_name}: no eye_sequence (not a v3 stimulus file)"
    conditions = [(eye, None) for eye in EYES] + [(eye, s) for eye in EYES for s in DIRECTIONS]
    views = {c: interleaved_view(device, *c) for c in conditions}
    paths = {c: os.path.join(device.data_dir, f"{v.session_name}_dataframe.pkl") for c, v in views.items()}

    todo = [c for c in conditions if recompute or not os.path.exists(paths[c])]
    if jobs > 1 and len(todo) > 1:
        print(f"[...] Computing analysis for {len(todo)} conditions, {min(jobs, len(todo))} at a time")
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            futures = {ex.submit(_analyze_condition, (mouse, date, data_dir, *c, n_iterations, paths[c])): c
                       for c in todo}
            for k, fut in enumerate(as_completed(futures), 1):
                fut.result()
                print(f"[ok] {views[futures[fut]].session_name} ({k}/{len(todo)})")
    else:
        for c in todo:
            print(f"[...] Computing analysis for {views[c].session_name}")
            _save_stats(views[c], n_iterations, paths[c])
            print()

    stats = {}
    for c in conditions:
        with open(paths[c], "rb") as f:
            stats[c] = pickle.load(f)
        log(f"{views[c].session_name}: stats in {paths[c]}")
        assert np.array_equal(stats[c]["cell_id"].values, device.suite2p_id), f"{c}: cached cells differ"
    D = {eye: views[(eye, None)] for eye in EYES}
    Dd = {c: views[c] for c in conditions if c[1] is not None}
    return D, {eye: stats[(eye, None)] for eye in EYES}, Dd, {c: stats[c] for c in Dd}


def shown_distances(D):
    """stimulus distances (control excluded), which must be the same in the three sessions"""
    shown = {eye: tuple(np.unique(d.distance_sequence[d.get_condition_trials()]).tolist())
             for eye, d in D.items()}
    assert len(set(shown.values())) == 1, f"sessions show different distances: {shown}"
    return list(shown["both"])


def describe_sessions(D, log):
    log(f"cells: {len(D['right'].suite2p_id)}")
    d = D["both"]
    if hasattr(d, "block_order"):
        code_eye = {v: k for k, v in EYE_CODE.items()}
        log(f"blocks: {' '.join(code_eye[int(b)] for b in d.block_order)}  "
            f"(start, min: {', '.join(f'{t / 60:.1f}' for t in d.block_start)}; "
            f"{len(d.switch_start)} blocker switches of {np.mean(d.switch_end - d.switch_start):.0f} s)")
    for eye, d in D.items():
        dist, n = np.unique(d.distance_sequence, return_counts=True)
        k = d.get_condition_trials()                  # stimulus trials (control excluded)
        log(f"\n{d.session_name}")
        log(f"  frames={d.spikes.shape[1]}  on_length={d.on_length}  "
            f"angles={np.ravel(d.start_angle)[0]:.1f} -> {np.ravel(d.end_angle)[0]:.1f}")
        log(f"  trials per distance={dict(zip(dist.tolist(), n.tolist()))}  "
            f"monocular_sequence={np.unique(d.monocular_sequence).tolist()}")
        if np.any(d.direction_sequence < 0):
            per_dir = {int(dk): [int(np.sum((d.distance_sequence == dk) & (d.direction_sequence == s)))
                                 for s in (1, -1)] for dk in dist}
            log(f"  trials per distance [forward, reverse]={per_dir}")
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
    """
    cells x trials x angle for the stimulus trials (control excluded); same frames as
    get_on_trace, so reverse sweeps are flipped onto the angle axis
    """
    return np.stack([d.response[:, d.get_on_frames(k)] for k in d.get_condition_trials()], axis=1)


# ---------------------------------------------------------------- Step 2: reliability

def condition_stats(df, fields=("reliable", "reliable_corr", "RF")):
    # each session holds a single eye condition, so exactly one of the two dicts is filled
    filled = [c for c in ("monocular", "binocular") if df[c].notna().all()]
    assert len(filled) == 1, f"expected one condition per session, got {filled}"
    return pd.DataFrame(df[filled[0]].tolist(), index=df["cell_id"].values)[list(fields)]


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


def plot_direction_check(D, X, eye_df, angles, out_dir, log, title, frames_per_bin=3):
    """
    Interleaved sessions pool both sweep directions (reverse sweeps flipped onto the angle axis),
    which assumes no response lag: a lag t puts the forward-sweep RF speed x t past the true RF
    and the reverse-sweep RF speed x t before it, so RF forward - reverse = 2 x speed x t.
    Per eye, for the cells reliable to it: RF of the forward-only and reverse-only trial averages.
    Same bins as rf_shift_hist.png.
    """
    assert frames_per_bin % 2 == 1, "frames_per_bin must be odd"
    d = D["both"]
    k = d.get_condition_trials()
    speed = abs(np.ravel(d.end_angle)[0] - np.ravel(d.start_angle)[0]) / np.median(d.on_end[k] - d.on_start[k])
    w = frames_per_bin * (angles[1] - angles[0])

    diffs = {}
    for eye in PANEL_ORDER:
        direction = D[eye].direction_sequence[D[eye].get_condition_trials()]   # same trial order as X
        Xr = X[eye][np.flatnonzero(eye_df[f"resp_{eye}"].to_numpy())]
        rf = {s: angles[Xr[:, direction == s].mean(axis=1).argmax(axis=1)] for s in (1, -1)}
        diffs[eye] = rf[1] - rf[-1]
    big = max((np.abs(v).max() for v in diffs.values() if len(v)), default=w)
    kb = int(np.ceil(big / w))
    edges = (np.arange(-kb, kb + 2) - 0.5) * w                          # bin centers at -kb*w .. kb*w

    log(f"\nRF forward - reverse sweep, cells reliable to that eye (sweep {speed:.1f} deg/s; "
        f"lag = shift / (2 x speed)):")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, eye in zip(axes, PANEL_ORDER):
        v = diffs[eye]
        ax.set_title(f"{EYE_LABEL[eye]} (n={len(v)})", fontsize=12)
        if not len(v):
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lag = med / (2 * speed) * 1000
        log(f"  {eye:<5} n={len(v):4d}  median={med:+5.1f}  IQR=[{q1:+.1f}, {q3:+.1f}]  "
            f"within ±{w / 2:.1f}: {np.mean(np.abs(v) < w / 2):4.0%}  implied lag {lag:+.0f} ms")
        ax.hist(v, bins=edges, color=EYE_COLOR[eye], alpha=0.8, edgecolor="white")
        ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
        ax.axvline(med, color="black", linewidth=1)
        ax.text(0.98, 0.95, f"median {med:+.1f}°\nlag {lag:+.0f} ms", transform=ax.transAxes,
                ha="right", va="top", fontsize=10)
        ax.set_xlabel("RF forward − RF reverse (deg)", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Cells", fontsize=11)
    fig.suptitle(f"{title}\nRF shift between sweep directions (0 = no lag, directions can be pooled)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "direction_check.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def block_drift(D, X, eye_df, log, window=5):
    """
    Response per block in recording order, to see slow drift over the session (the mirrored
    order, e.g. L R B B R L, balances a linear drift across eyes). Per cell: mean over the
    block's stimulus trials in a ±window frame window at the cell's RF for that eye; then the
    median over the cells reliable to that eye.
    """
    d = D["both"]
    code_eye = {v: k for k, v in EYE_CODE.items()}
    log("\nresponse at the RF per block, recording order (pooled z, median over cells reliable to that eye):")
    per_eye = {}
    for b, code in enumerate(d.block_order, 1):
        eye = code_eye[int(code)]
        block = D[eye].block_sequence[D[eye].get_condition_trials()]           # same trial order as X
        Xr = X[eye][np.flatnonzero(eye_df[f"resp_{eye}"].to_numpy())]
        if not len(Xr):
            continue
        rf_i = Xr.mean(axis=1).argmax(axis=1)
        a = np.arange(Xr.shape[2])
        mask = (a[None] >= rf_i[:, None] - window) & (a[None] <= rf_i[:, None] + window)
        val = (Xr[:, block == b] * mask[:, None, :]).sum(axis=2) / mask.sum(axis=1)[:, None]
        m = np.median(val.mean(axis=1))
        per_eye.setdefault(eye, []).append(m)
        log(f"  block {b}  {eye:<5} trials={np.sum(block == b):3d}  median={m:.3f}")
    log("  later / earlier block of each eye: " + ", ".join(
        f"{eye} {v[1] / v[0]:.2f}" for eye, v in per_eye.items() if len(v) == 2 and v[0] > 0))


# ---------------------------------------------------------------- distance modulation per direction

def two_way_anova(d, X_eye, window=5):
    """
    Per cell, distance x sweep direction ANOVA on the per-trial response in a ±window frame
    window at the RF of the trial average over both directions (the pipeline's ANOVA window).
    Type II sums of squares, so an unbalanced design (a dropped trial) is fine.
    Returns p-values {distance, direction, interaction} (one per cell), the mean response per
    direction and distance (cells x [fwd, rev] x distances) and the distances.
    """
    trials = d.get_condition_trials()                                   # same trial order as X_eye
    dist, sweep = d.distance_sequence[trials], d.direction_sequence[trials]
    rf_i = X_eye.mean(axis=1).argmax(axis=1)
    a = np.arange(X_eye.shape[2])
    mask = (a[None] >= rf_i[:, None] - window) & (a[None] <= rf_i[:, None] + window)
    Y = ((X_eye * mask[:, None, :]).sum(axis=2) / mask.sum(axis=1)[:, None]).T     # trials x cells

    ds = np.unique(dist)
    one = np.ones((len(trials), 1))
    Dm = (dist[:, None] == ds[None, 1:]).astype(float)                            # distance dummies
    Sm = (sweep < 0).astype(float)[:, None]                                         # direction dummy

    def rss(*blocks):
        M = np.hstack(blocks)
        beta = np.linalg.lstsq(M, Y, rcond=None)[0]
        return ((Y - M @ beta) ** 2).sum(axis=0), M.shape[1]

    rss_full, n_par = rss(one, Dm, Sm, Dm * Sm)
    rss_add = rss(one, Dm, Sm)[0]
    df_res, nd = len(trials) - n_par, len(ds) - 1
    with np.errstate(divide="ignore", invalid="ignore"):                # silent cells: 0 / 0
        ms = rss_full / df_res
        p = {"distance": f_dist.sf((rss(one, Sm)[0] - rss_add) / nd / ms, nd, df_res),
             "direction": f_dist.sf((rss(one, Dm)[0] - rss_add) / ms, 1, df_res),
             "interaction": f_dist.sf((rss_add - rss_full) / nd / ms, nd, df_res)}
    tuning = np.stack([np.stack([Y[(sweep == s) & (dist == k)].mean(axis=0) for k in ds], axis=1)
                       for s in DIRECTIONS.values()], axis=1)
    return p, tuning, ds


def build_direction_table(D, X, stats, stats_d, corr_floor=None):
    """
    One row per cell. Per eye and condition (pooled = both directions, fwd, rev): the pipeline's
    reliable / reliable_corr / RF / one-way ANOVA p, resp_* = reliable with the corr floor and
    mod_* = resp_* & ANOVA p < 0.05 (the pipeline's modulated call). Per eye: the distance x
    direction ANOVA (p2_*), the preferred distance per direction (pref_*, from the mean response
    in the pooled RF window) and resp_<eye>_any = reliable pooled or in either direction.
    mod_category_both: both eyes modulated in both directions / forward only / reverse only /
    pooled only (neither direction alone).
    """
    fields = ("reliable", "reliable_corr", "RF", "anova_pvalue")
    parts = {f"{eye}_pooled": condition_stats(stats[eye], fields) for eye in EYES}
    parts.update({f"{eye}_{s}": condition_stats(df, fields) for (eye, s), df in stats_d.items()})
    t = pd.concat(parts, axis=1)
    t.columns = [f"{metric}_{cond}" for cond, metric in t.columns]
    t.index.name = "cell_id"
    for cond in parts:
        resp = t[f"reliable_{cond}"].astype(bool)
        if corr_floor is not None:
            resp &= t[f"reliable_corr_{cond}"] >= corr_floor
        t[f"resp_{cond}"] = resp
        t[f"mod_{cond}"] = resp & (t[f"anova_pvalue_{cond}"] < 0.05)

    tuning = {}
    for eye in EYES:
        p, tuning[eye], ds = two_way_anova(D[eye], X[eye])               # X rows follow suite2p order
        for term, v in p.items():
            t[f"p2_{eye}_{term}"] = v
        for i, s in enumerate(DIRECTIONS):
            t[f"pref_{eye}_{s}"] = ds[tuning[eye][:, i].argmax(axis=1)]
        t[f"resp_{eye}_any"] = t[[f"resp_{eye}_{c}" for c in ("pooled", *DIRECTIONS)]].any(axis=1)

    f, r = t["mod_both_fwd"], t["mod_both_rev"]
    t["mod_category_both"] = np.select([f & r, f & ~r, ~f & r, t["mod_both_pooled"]], list(MOD_CATEGORIES),
                                       default="none")
    return t, tuning


def direction_summary(t, eye):
    """counts behind the report and figure; cells = reliable to that eye pooled or in either direction"""
    cells = t[f"resp_{eye}_any"]
    n = int(cells.sum())
    m = {c: t[f"mod_{eye}_{c}"] & cells for c in ("fwd", "rev", "pooled")}
    p2 = {term: (t[f"p2_{eye}_{term}"] < 0.05) & cells for term in ("distance", "direction", "interaction")}
    both = m["fwd"] & m["rev"]
    a, b = (t.loc[both, f"pref_{eye}_{s}"].to_numpy(dtype=int) for s in DIRECTIONS)
    chance = np.sum(np.bincount(a, minlength=4) * np.bincount(b, minlength=4)) / len(a) ** 2 if len(a) else np.nan
    return dict(cells=cells, n=n, m=m, p2=p2, both=both, pref=(a, b), chance=chance,
                agree=np.mean(a == b) if len(a) else np.nan,
                expected=m["fwd"].sum() * m["rev"].sum() / n if n else np.nan)


def report_direction_modulation(t, tuning, log):
    log("\n" + "=" * 50)
    log(" Distance modulation by sweep direction")
    log(" forward = -30 -> 60 deg, reverse = 60 -> -30 deg. modulated = reliable in that condition")
    log(" (split-half test + corr floor) & one-way ANOVA p < 0.05. 2-way = distance x direction")
    log(" ANOVA at the pooled RF (type II). Chance: 5% per test.")
    log("=" * 50)
    for eye in PANEL_ORDER:
        s = direction_summary(t, eye)
        n, m, p2 = s["n"], s["m"], s["p2"]
        rf_, rr = t[f"resp_{eye}_fwd"], t[f"resp_{eye}_rev"]
        pc = lambda k: f"{k} ({100 * k / max(n, 1):.0f}%)"
        log(f"\n{EYE_LABEL[eye]}: {n} cells reliable pooled or in either direction")
        log(f"  reliable: pooled {t[f'resp_{eye}_pooled'].sum()}, forward {rf_.sum()}, reverse {rr.sum()}; "
            f"both directions {(rf_ & rr).sum()}, forward only {(rf_ & ~rr).sum()}, reverse only {(~rf_ & rr).sum()}")
        log(f"  modulated: forward {pc(m['fwd'].sum())}, reverse {pc(m['rev'].sum())}, pooled {pc(m['pooled'].sum())}")
        log(f"    both directions {s['both'].sum()} (expected if independent: {s['expected']:.1f}), "
            f"forward only {(m['fwd'] & ~m['rev']).sum()}, reverse only {(~m['fwd'] & m['rev']).sum()}, "
            f"pooled but neither direction {(m['pooled'] & ~m['fwd'] & ~m['rev']).sum()}")
        log(f"  2-way p < 0.05: distance {pc(p2['distance'].sum())}, direction {pc(p2['direction'].sum())}, "
            f"distance x direction {pc(p2['interaction'].sum())}")
        if s["both"].sum():
            rows = t.index.get_indexer(t.index[s["both"]])
            r = np.mean([np.corrcoef(tuning[eye][i, 0], tuning[eye][i, 1])[0, 1] for i in rows])
            log(f"  modulated in both directions: same preferred distance {s['agree']:.0%} "
                f"(chance {s['chance']:.0%}), mean correlation of the distance tuning {r:+.2f}")


def plot_direction_modulation(t, out_dir, title):
    """
    Top: % of the cells reliable to each eye that are distance-modulated forward, reverse, in
    both directions (tick: expected if independent), pooled, and the 2-way ANOVA terms; dashed
    line = 5% chance. Bottom: preferred distance forward vs reverse for cells modulated in both.
    """
    labels = ["forward", "reverse", "both\ndirections", "pooled", "2-way\ndistance", "2-way\ndist×dir",
              "2-way\ndirection"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    for j, eye in enumerate(PANEL_ORDER):
        s = direction_summary(t, eye)
        n = max(s["n"], 1)
        vals = [s["m"]["fwd"].sum(), s["m"]["rev"].sum(), s["both"].sum(), s["m"]["pooled"].sum(),
                s["p2"]["distance"].sum(), s["p2"]["interaction"].sum(), s["p2"]["direction"].sum()]
        ax = axes[0, j]
        x = np.arange(len(vals))
        ax.bar(x, 100 * np.array(vals) / n, color=EYE_COLOR[eye], alpha=0.8)
        for xi, v in zip(x, vals):
            ax.text(xi, 100 * v / n + 0.5, str(v), ha="center", va="bottom", fontsize=9)
        ax.plot([2 - 0.4, 2 + 0.4], [100 * s["expected"] / n] * 2, color="black", linewidth=2)
        ax.axhline(5, color="0.4", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("% of reliable cells", fontsize=11)
        ax.set_title(f"{EYE_LABEL[eye]} (n={s['n']} reliable)", fontsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax = axes[1, j]
        a, b = s["pref"]
        M = np.zeros((4, 4), dtype=int)
        np.add.at(M, (a, b), 1)
        ax.imshow(M, cmap="Greys", vmin=0, vmax=max(M.max(), 1))
        for (i, k), v in np.ndenumerate(M):
            ax.text(k, i, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v > M.max() / 2 else "black")
        ticks = [f"{DIST_CM[k]} cm" for k in range(4)]
        ax.set_xticks(range(4))
        ax.set_xticklabels(ticks)
        ax.set_yticks(range(4))
        ax.set_yticklabels(ticks)
        ax.set_xlabel("preferred distance, reverse", fontsize=11)
        ax.set_ylabel("preferred distance, forward", fontsize=11)
        agree = "" if not len(a) else f": same {s['agree']:.0%} (chance {s['chance']:.0%})"
        ax.set_title(f"modulated in both directions (n={len(a)}){agree}", fontsize=11)
    fig.suptitle(f"{title}\ndistance modulation by sweep direction (forward = −30→60°, reverse = 60→−30°)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "direction_modulation.png"), dpi=150, bbox_inches="tight")
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


def direction_viz(D, Dd, stats, stats_d, corr_floor=None, eye="both"):
    """
    CrossSessionVisualizer for one eye of an interleaved session, one row per condition: both
    directions pooled / forward / reverse, stats from the caches with the corr floor applied.
    """
    rows = [(EYE_LABEL[eye], D[eye], stats[eye])] + [
        (f"{EYE_LABEL[eye]} - {DIR_LABEL[s]}", Dd[(eye, s)], stats_d[(eye, s)]) for s in DIRECTIONS]
    return CrossSessionVisualizer([(label, CellVisualizer(d)) for label, d, _ in rows],
                                  stats={label: floor_stats(df, corr_floor) for label, _, df in rows})


def plot_cell_directions(cell, viz, dir_df, eye="both"):
    p = {term: dir_df.loc[cell, f"p2_{eye}_{term}"] for term in ("distance", "direction", "interaction")}
    fig = viz.plot_cell(cell, show_condition=False)
    fig.suptitle(f"Cell {cell} — pooled / forward (−30→60°) / reverse (60→−30°)     2-way p: "
                 f"distance {p['distance']:.3f}, direction {p['direction']:.3f}, "
                 f"dist×dir {p['interaction']:.3f}", fontsize=18)
    return fig


def direction_groups(dir_df, eye="both"):
    """the cells behind each bar of direction_modulation.png for that eye: {folder: cell ids}"""
    s = direction_summary(dir_df, eye)
    masks = {"forward": s["m"]["fwd"], "reverse": s["m"]["rev"], "both_directions": s["both"],
             "pooled": s["m"]["pooled"], "2way_distance": s["p2"]["distance"],
             "2way_dist_x_dir": s["p2"]["interaction"], "2way_direction": s["p2"]["direction"]}
    return {folder: dir_df.index[m].tolist() for folder, m in masks.items()}


def save_cell_plots(plot, groups, cells_dir, cells=None):
    """
    One figure per cell, drawn once by plot(cell) and saved as cell_<id>.png in every
    <cells_dir>/<group>/ the cell belongs to (groups: {folder: cell ids}; may overlap). By
    default every cell in any group, after clearing cells_dir so a rerun with other criteria
    leaves no stale figures. Cells asked for in `cells` but in no group go to none/.
    """
    member = {}
    for folder, ids in groups.items():
        for cell in ids:
            member.setdefault(cell, []).append(folder)
    if cells is None:
        cells = list(member)
        shutil.rmtree(cells_dir, ignore_errors=True)

    for i, cell in enumerate(cells, 1):
        print(f"Plotting cell {i} of {len(cells)}", end="\r", flush=True)
        paths = [os.path.join(cells_dir, folder, f"cell_{cell}.png") for folder in member.get(cell, ["none"])]
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        fig = plot(cell)
        fig.savefig(paths[0], dpi=150, bbox_inches="tight")
        plt.close(fig)
        for path in paths[1:]:
            shutil.copyfile(paths[0], path)
    print()
    return len(cells)


def main():
    ap = argparse.ArgumentParser(description="Right / left / both-eye comparison on the distance device")
    ap.add_argument("--mouse", default="SAM047")
    ap.add_argument("--date", default="261006")
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
    ap.add_argument("--jobs", type=int, default=6,
                    help="interleaved sessions: conditions (eye, eye x direction) analysed in parallel "
                         "when not cached; ~2 GB of memory each")
    ap.add_argument("--cells", type=int, nargs="+", default=None,
                    help="plot only these cell ids (default: every cell reliable in any condition)")
    ap.add_argument("--skip-cells", action="store_true",
                    help="skip the per-cell figures (the slow part); same as --cell-figures none")
    ap.add_argument("--cell-figures", choices=("all", "eyes", "directions", "none"), default="all",
                    help="which per-cell figures: cells/ (eyes), cells_directions/ (directions), both or none")
    args = ap.parse_args()
    plt.switch_backend("Agg")

    # one interleaved session (v3 stimulus) if there is one, else three sessions (one per eye)
    interleaved = is_interleaved(args.mouse, args.date, args.data_dir)
    title = f"{args.date}_{args.mouse} distance: right / left / both" + (" (interleaved)" if interleaved else "")
    out_dir = results_dir(args.mouse, args.date, args.data_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = Report(os.path.join(out_dir, "report.txt"))
    log(f"{title}\nargs: {vars(args)}\n")

    # Step 0: process each eye condition on its own (standard pipeline, cached), then checks,
    # common angle axis and pooled normalization for the figures
    if interleaved:
        D, stats, Dd, stats_d = process_interleaved(args.mouse, args.date, args.data_dir, args.n_iterations,
                                                    args.recompute, args.jobs, log)
    else:
        D, stats = process_sessions(args.mouse, args.date, args.data_dir, args.n_iterations, args.recompute, log)
    describe_sessions(D, log)
    distances = shown_distances(D)
    log(f"\nstimulus distances (all three sessions): {distances}"
        + ("  -> single-distance figures" if len(distances) == 1 else "  -> distance-tuning figures"))
    angles = align_angle_axis(D)
    log(f"\ncommon on_length: {len(angles)} frames ({angles[0]:.1f} -> {angles[-1]:.1f} deg)")
    if interleaved:
        log("\none recording: RadialDevice's z-score already spans all three eye conditions")
    else:
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
    if any(np.any(D[eye].direction_sequence < 0) for eye in EYES):
        plot_direction_check(D, X, eye_df, np.ravel(angles), out_dir, log, title)
    if interleaved:
        block_drift(D, X, eye_df, log)

    # Step 3 (interleaved, several distances): distance modulation per sweep direction
    by_direction = interleaved and len(distances) > 1
    if by_direction:
        dir_df, tuning = build_direction_table(D, X, stats, stats_d, args.corr_floor)
        report_direction_modulation(dir_df, tuning, log)
        plot_direction_modulation(dir_df, out_dir, title)
        dir_df.to_pickle(os.path.join(out_dir, "direction_table.pkl"))
        dir_df.to_csv(os.path.join(out_dir, "direction_table.csv"))

    plot_onset_aligned(D, eye_df, out_dir, args.pre_s)
    plot_location_map(D, eye_df, out_dir)

    figures = "none" if args.skip_cells else args.cell_figures
    if figures in ("all", "eyes"):
        if len(distances) == 1:
            plot = lambda cell: plot_cell(cell, D, X, eye_df, np.ravel(angles))
        else:
            cross_viz = distance_viz(D, stats, args.corr_floor)
            plot = lambda cell: plot_cell_distances(cell, cross_viz, eye_df)
        groups = {CATEGORY_DIR[c]: eye_df.index[eye_df["category"] == c].tolist() for c in CATEGORY_DIR if c != "none"}
        cells_dir = os.path.join(out_dir, "cells")
        n = save_cell_plots(plot, groups, cells_dir, args.cells)
        log(f"\nSaved {n} cell figures to: {cells_dir}")
    if by_direction and figures in ("all", "directions"):
        dir_viz = direction_viz(D, Dd, stats, stats_d, args.corr_floor)
        groups = direction_groups(dir_df)
        cells_dir = os.path.join(out_dir, "cells_directions")
        n = save_cell_plots(lambda cell: plot_cell_directions(cell, dir_viz, dir_df), groups, cells_dir, args.cells)
        log(f"\nSaved {n} both-eye cell figures (pooled / forward / reverse) to: {cells_dir}, one folder per bar "
            f"of direction_modulation.png: " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()))
    log(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
