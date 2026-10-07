"""
Does a cell's interocular RF offset predict its distance preference? Radial device, right / left /
both-eye conditions: one interleaved session (distance-eyes, v3 stimulus, both sweep directions
pooled) or three sessions (distance-right / -left / -both), loaded as in distance_eye_comparison.py.
Run after distance_eye_comparison.py, whose eye_table.pkl (and direction_table.pkl) it reads:

    python radial-analysis/disparity_distance.py --mouse SAM047 --date 261006 --eye-offset 1.0

Cells: reliable to the left eye, the right eye and both eyes (eye_table resp_*, corr floor included).

Disparity: dRF = RF_right - RF_left (device angle, pooled over distances, as in rf_shift_hist.png),
minus its median over these cells (the "gap", about -6 deg). dRF_rel < 0: the cell's two RFs cross
nearer than a typical cell's; > 0: farther. The gap may be real convergence or partly eye position
differing between sessions; rank statistics don't depend on it either way.

Distance preference, per session: peak of each distance's trial average (each distance's own peak,
so RF shifts with distance don't leak in). Preferred distance = argmax; response-weighted distance =
sum(r_d * w_d) / sum(w_d), w_d = peak_d - min(peak).

Tests: Spearman(dRF_rel, weighted distance) with a cell-shuffle p and bootstrap CI; the same partial
on RF position; the same with the left-eye and right-eye sessions as controls (disparity needs both
eyes; the right eye sees the same stimulus at every distance). Prediction: positive, only for both.

Gap-corrected match: with the gap removed, the predicted RF offset at distance d is
dRF_rel + p(d) - mean(p), p(d) = parallax of the left eye at the cell's right-eye RF and distance d
(left eye --eye-offset cm nasal of the right eye, which is on the rotation axis). Predicted preferred
distance = where that offset is closest to 0, compared with the observed both-eyes preference.

Absolute geometry (disparity_range.png): each cell's preferred disparity, where its two RFs cross,
delta* = mean_d p(d) - dRF (no gap removed), against the disparities the ball has at each distance.

By modulation (disparity_by_modulation.png): the same for distance-modulated vs not distance-modulated
cells (both eyes), with the distance pure geometry predicts (the tested distance whose ball disparity
is closest to the cell's preferred disparity). Distance-modulated (--modulation): 'anova' (default)
= ANOVA distance effect p < 0.05, i.e. the distance term of the distance x direction ANOVA for an
interleaved session (direction_table.pkl), the pipeline's one-way ANOVA otherwise; 'bootstrap' =
the pipeline's bootstrap p < 0.05.

Writes disparity_vs_distance.png, disparity_range.png, disparity_by_modulation.png and
disparity_report.txt next to eye_table.pkl.
"""
import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.append(p)

from distance_eye_comparison import (DIST_CM, DIST_COLOR, EYE_COLOR, EYE_LABEL, Report, eye_devices,
                                     is_interleaved, results_dir)

rng = np.random.default_rng(0)


def left_angle(theta_deg, r_cm, s_cm):
    """direction of the ball (device angle theta, distance r) seen from the left eye, s cm nasal of the axis"""
    th = np.deg2rad(theta_deg)
    return np.rad2deg(np.arctan2(r_cm * np.sin(th) + s_cm, r_cm * np.cos(th)))


def distance_peaks(d, cells):
    """cells x distances: peak of each distance's trial-averaged curve, stimulus distances only"""
    rows = np.flatnonzero(np.isin(d.suite2p_id, cells))
    assert np.array_equal(d.suite2p_id[rows], cells), "cells must be in suite2p order"
    resp = d.response[rows]
    trials = d.get_condition_trials()
    dist = np.ravel(d.distance_sequence)[trials]
    # get_on_frames puts reverse sweeps (interleaved sessions) on the same angle axis
    X = np.stack([resp[:, d.get_on_frames(k)] for k in trials], axis=1)
    ds = np.unique(dist)
    return ds, np.stack([X[:, dist == k].mean(axis=1).max(axis=1) for k in ds], axis=1)


def weighted_distance(peaks, ds):
    w = peaks - peaks.min(axis=1, keepdims=True)
    cm = np.array([DIST_CM[int(k)] for k in ds], dtype=float)
    tot = w.sum(axis=1)
    return np.where(tot > 0, (w * cm).sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)


def _rank_corr(rx, ry):
    return np.corrcoef(rx, ry)[0, 1]


def _residual(r, rz):
    A = np.column_stack([np.ones_like(rz), rz])
    return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]


def spearman_test(x, y, z=None, n_perm=10000, n_boot=2000):
    """Spearman (partial on z if given): rho, cell-shuffle p (two-sided), bootstrap 95% CI, n"""
    ok = ~(np.isnan(x) | np.isnan(y)) if z is None else ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y = x[ok], y[ok]
    z = None if z is None else z[ok]

    def rho(xx, yy, zz):
        rx, ry = rankdata(xx), rankdata(yy)
        if zz is not None:
            rz = rankdata(zz)
            rx, ry = _residual(rx, rz), _residual(ry, rz)
        return _rank_corr(rx, ry)

    r = rho(x, y, z)
    null = np.array([rho(rng.permutation(x), y, z) for _ in range(n_perm)])
    p = (1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boot.append(rho(x[i], y[i], None if z is None else z[i]))
    return r, p, np.nanpercentile(boot, [2.5, 97.5]), len(x)


def predicted_distance(drf_rel, rf_right, ds, s_cm):
    r = np.array([DIST_CM[int(k)] for k in ds], dtype=float)
    th = rf_right[:, None]
    par = left_angle(th, r[None, :], s_cm) - th                        # cells x distances
    offset = drf_rel[:, None] + par - par.mean(axis=1, keepdims=True)
    return ds[np.argmin(np.abs(offset), axis=1)], par


def preferred_disparity(drf, rf_right, ds, s_cm):
    """
    Absolute geometry, no gap removed (depends on the left-eye offset s). Disparity = phi_L - theta:
    how much further toward + the left eye sees a point than the right eye (> 0 in front of the
    animal, larger = nearer). A cell's preferred disparity is where its two RFs cross:
    delta* = mean_d p(d) - dRF, p(d) = left-eye parallax at its right-eye RF and distance d (the
    pooled left RF sits about mean_d p(d) below the left-eye direction).
    Returns delta* (cells), p (cells x distances: the ball's disparity at each cell's RF),
    the ball's disparity per distance at the median RF, and that median RF.
    """
    r = np.array([DIST_CM[int(k)] for k in ds], dtype=float)
    th = rf_right[:, None]
    par = left_angle(th, r[None, :], s_cm) - th
    theta_med = float(np.median(rf_right))
    return par.mean(axis=1) - drf, par, left_angle(theta_med, r, s_cm) - theta_med, theta_med


def plot_disparity_range(delta, ball, theta_med, pref, ds, s_cm, out_dir, title, log):
    """
    Each cell's preferred disparity (see preferred_disparity) against the disparities the ball has
    at the tested distances, stacked by the distance each cell prefers with both eyes.
    """
    log(f"\npreferred disparity from the RF offsets (left eye {s_cm} cm, absolute geometry): "
        f"median {np.median(delta):+.1f} deg, IQR [{np.percentile(delta, 25):+.1f}, {np.percentile(delta, 75):+.1f}]")
    log("  ball disparity at the median RF (%+.0f deg): " % theta_med
        + ", ".join(f"{DIST_CM[int(k)]} cm: {b:.1f}" for k, b in zip(ds, ball)))
    log(f"  cells nearer than {DIST_CM[int(ds[0])]} cm (delta* > {ball[0]:.1f}): {np.mean(delta > ball[0]):.0%};  "
        f"within the tested range: {np.mean((delta <= ball[0]) & (delta >= ball[-1])):.0%};  "
        f"farther than {DIST_CM[int(ds[-1])]} cm: {np.mean((delta < ball[-1]) & (delta > 0)):.0%};  "
        f"RFs don't cross in front (delta* <= 0): {np.mean(delta <= 0):.0%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    edges = np.arange(np.floor(delta.min()) - 0.75, np.ceil(delta.max()) + 1.5, 1.5)
    ax.axvspan(ball[-1], ball[0], color="0.9", zorder=0, label="disparities the ball covers")
    ax.hist([delta[pref == k] for k in ds], bins=edges, stacked=True, edgecolor="white", linewidth=0.5,
            color=[DIST_COLOR[int(k)] for k in ds],
            label=[f"cells preferring {DIST_CM[int(k)]} cm (both eyes)" for k in ds], zorder=2)
    for k, b in zip(ds, ball):
        ax.axvline(b, color=DIST_COLOR[int(k)], linewidth=2, linestyle="--", zorder=3,
                   label=f"ball at {DIST_CM[int(k)]} cm ({b:.1f}°)")
    ax.axvline(0, color="0.4", linewidth=1, zorder=1)
    ax.set_xlabel("Disparity (deg): how much further toward + the left eye sees the point\n"
                  "← behind / diverging            nearer →", fontsize=11)
    ax.set_ylabel("Cells", fontsize=11)
    # top axis: distance at which the two RFs cross, for a cell at the median RF
    crossing = [3, 4, 5, 7, 10, 20]
    sec = ax.twiny()
    sec.set_xlim(ax.get_xlim())
    sec.set_xticks([left_angle(theta_med, D, s_cm) - theta_med for D in crossing])
    sec.set_xticklabels([f"{D}" for D in crossing], fontsize=9)
    sec.set_xlabel("RFs cross at (cm from the right eye)", fontsize=10)
    for a in (ax, sec):
        a.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title(f"{title}: each cell's preferred disparity vs the disparities tested (left eye {s_cm} cm)",
                 fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "disparity_range.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_modulation(delta, par, ball, wdist, pref, mod, ds, s_cm, out_dir, title, log, n_perm=10000,
                       mod_label="pipeline bootstrap p < 0.05"):
    """
    Preferred disparity for distance-modulated vs not distance-modulated cells (both eyes).
    Top: preferred-disparity histograms (stacked by preferred distance) with the disparities the
    ball covers. Bottom: preferred disparity vs preferred distance (jittered), with the distance
    pure geometry predicts: the tested distance whose ball disparity, at the cell's RF, is closest
    to the cell's preferred disparity.
    """
    pred = ds[np.argmin(np.abs(par - delta[:, None]), axis=1)]
    inside = (delta <= par[:, 0]) & (delta >= par[:, -1])
    groups = (("distance-modulated", mod), ("not distance-modulated", ~mod))

    log(f"\nby distance modulation (both eyes, {mod_label}); preferred disparity, absolute geometry")
    stats = {}
    for name, m in groups:
        d = delta[m]
        rho, p_rho, ci, n = spearman_test(d, wdist[m], n_perm=n_perm)
        agree = np.mean(pred[m] == pref[m])
        null = np.array([np.mean(rng.permutation(pred[m]) == pref[m]) for _ in range(n_perm)])
        p_agree = (1 + np.sum(null >= agree)) / (1 + n_perm)
        stats[name] = dict(n=int(m.sum()), rho=rho, p_rho=p_rho, ci=ci, agree=agree, chance=null.mean(), p_agree=p_agree)
        log(f"  {name:<23} n={m.sum():3d}  prefers 10/16/22/28 cm: {[int(np.sum(pref[m] == k)) for k in ds]}  "
            f"delta* median {np.median(d):+.1f} IQR [{np.percentile(d, 25):+.1f}, {np.percentile(d, 75):+.1f}]")
        log(f"  {'':<23} nearer than 10 cm {np.mean(d > par[m, 0]):.0%}, inside tested range {np.mean(inside[m]):.0%}, "
            f"farther than 28 cm {np.mean((d < par[m, -1]) & (d > 0)):.0%}, don't cross {np.mean(d <= 0):.0%}")
        log(f"  {'':<23} Spearman(delta*, weighted distance) rho={rho:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p_rho:.4f} "
            f"(prediction < 0);  geometry match {agree:.0%} (shuffled {null.mean():.0%}) p={p_agree:.4f}")
    table = [[np.sum(inside & mod), np.sum(~inside & mod)], [np.sum(inside & ~mod), np.sum(~inside & ~mod)]]
    p_fisher = fisher_exact(table).pvalue
    p_mw = mannwhitneyu(delta[mod], delta[~mod]).pvalue
    log(f"  inside the tested range, modulated vs not: Fisher p={p_fisher:.3g};  delta* distributions: Mann-Whitney p={p_mw:.3g}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    lo, hi = np.floor(delta.min()) - 1, np.ceil(delta.max()) + 1
    edges = np.arange(lo + 0.25, hi + 1.5, 1.5)
    grid = np.linspace(lo, hi, 2000)
    ball_pred = np.array([DIST_CM[int(k)] for k in ds[np.argmin(np.abs(ball[None, :] - grid[:, None]), axis=1)]])
    for col, (name, m) in enumerate(groups):
        st = stats[name]
        ax = axes[0, col]
        ax.axvspan(ball[-1], ball[0], color="0.9", zorder=0, label="disparities the ball covers")
        ax.hist([delta[m & (pref == k)] for k in ds], bins=edges, stacked=True, edgecolor="white", linewidth=0.5,
                color=[DIST_COLOR[int(k)] for k in ds],
                label=[f"prefers {DIST_CM[int(k)]} cm" for k in ds], zorder=2)
        for k, b in zip(ds, ball):
            ax.axvline(b, color=DIST_COLOR[int(k)], linewidth=1.5, linestyle="--", zorder=3)
        ax.axvline(0, color="0.4", linewidth=1, zorder=1)
        ax.set_ylabel("Cells", fontsize=11)
        ax.set_title(f"{name.capitalize()} (n={st['n']}): {np.mean(inside[m]):.0%} inside the tested range,\n"
                     f"{np.mean(delta[m] > par[m, 0]):.0%} nearer than 10 cm", fontsize=11, loc="left")
        if col == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")

        ax = axes[1, col]
        ax.axvspan(ball[-1], ball[0], color="0.9", zorder=0)
        ax.plot(grid, ball_pred, color="black", linestyle="--", linewidth=1.5, zorder=3,
                label="pure geometry: tested distance\nclosest to the cell's disparity")
        for k in ds:
            mk = m & (pref == k)
            ax.scatter(delta[mk], DIST_CM[int(k)] + rng.uniform(-1.6, 1.6, mk.sum()), s=18,
                       color=DIST_COLOR[int(k)], edgecolor="white", linewidth=0.4, zorder=2)
        ax.axvline(0, color="0.4", linewidth=1, zorder=1)
        ax.set_yticks([DIST_CM[int(k)] for k in ds])
        ax.set_yticklabels([f"{DIST_CM[int(k)]} cm" for k in ds])
        ax.set_ylabel("Preferred distance, both eyes", fontsize=11)
        ax.set_xlabel("Preferred disparity (deg)   ← behind / diverging      nearer →", fontsize=11)
        ax.set_title(f"Spearman ρ = {st['rho']:+.2f} [{st['ci'][0]:+.2f}, {st['ci'][1]:+.2f}], p = {st['p_rho']:.3f}"
                     f"  (prediction < 0)\ngeometry match {st['agree']:.0%} (shuffled {st['chance']:.0%}), "
                     f"p = {st['p_agree']:.3f}", fontsize=11, loc="left")
        if col == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{title}: preferred disparity vs distance, distance-modulated vs not (left eye {s_cm} cm)\n"
                 f"modulated cells inside the tested range vs not: Fisher p = {p_fisher:.3f};  "
                 f"preferred disparities differ: Mann-Whitney p = {p_mw:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "disparity_by_modulation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Interocular RF offset vs distance preference")
    ap.add_argument("--mouse", default="SAM047")
    ap.add_argument("--date", default="260930")
    ap.add_argument("--data-dir", default=r"D:/data-distance-coding-project/processed-data")
    ap.add_argument("--eye-offset", type=float, default=1.0,
                    help="left eye, cm nasal of the right eye (rotation axis); used for the gap-corrected match")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--modulation", choices=("anova", "bootstrap"), default="anova",
                    help="which test makes a cell distance-modulated (see the module docstring)")
    args = ap.parse_args()
    plt.switch_backend("Agg")

    out_dir = results_dir(args.mouse, args.date, args.data_dir)
    log = Report(os.path.join(out_dir, "disparity_report.txt"))
    title = f"{args.date}_{args.mouse}"
    log(f"{title}: interocular RF offset vs distance preference\nargs: {vars(args)}\n")

    t = pd.read_pickle(os.path.join(out_dir, "eye_table.pkl"))
    cells = t.index[t["resp_left"] & t["resp_right"] & t["resp_both"]].to_numpy()
    sub = t.loc[cells]
    drf = (sub["RF_right"] - sub["RF_left"]).to_numpy(dtype=float)
    gap = np.median(drf)
    drf_rel = drf - gap
    log(f"cells reliable to left, right and both: {len(cells)}")
    log(f"gap (median RF_right - RF_left): {gap:+.1f} deg; dRF_rel IQR "
        f"[{np.percentile(drf_rel, 25):+.1f}, {np.percentile(drf_rel, 75):+.1f}]")

    # distance preference per eye condition
    D = eye_devices(args.mouse, args.date, args.data_dir)
    pref, wdist, ds = {}, {}, None
    for eye in ("both", "left", "right"):
        ds_e, peaks = distance_peaks(D[eye], cells)
        assert ds is None or np.array_equal(ds, ds_e), "sessions show different distances"
        ds = ds_e
        pref[eye] = ds[peaks.argmax(axis=1)]
        wdist[eye] = weighted_distance(peaks, ds)
    if len(ds) < 2:
        raise SystemExit("needs sessions with several distances")
    log("preferred distance (both eyes), cells per distance: "
        + ", ".join(f"{DIST_CM[int(k)]} cm: {np.sum(pref['both'] == k)}" for k in ds))

    # rank tests
    log("\nSpearman(dRF_rel, response-weighted preferred distance); prediction: > 0 for both eyes only")
    res = {}
    for eye in ("both", "left", "right"):
        res[eye] = spearman_test(drf_rel, wdist[eye], n_perm=args.n_perm)
        r, p, ci, n = res[eye]
        log(f"  {eye:<5} rho={r:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  shuffle p={p:.4f}  n={n}")
    res["both, RF partialled out"] = spearman_test(drf_rel, wdist["both"], z=sub["RF_right"].to_numpy(float),
                                                   n_perm=args.n_perm)
    r, p, ci, n = res["both, RF partialled out"]
    log(f"  both, controlling for RF position: rho={r:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  shuffle p={p:.4f}")

    # both-eye pipeline stats (cached by distance_eye_comparison.py)
    d = D["both"]
    S = pd.read_pickle(os.path.join(d.data_dir, f"{d.session_name}_dataframe.pkl")).set_index("cell_id")
    b = pd.DataFrame(S.loc[cells, "binocular"].tolist(), index=cells)
    if args.modulation == "bootstrap":
        mod, mod_label = (b["bootstrap_pvalue"] < 0.05).to_numpy(), "pipeline bootstrap p < 0.05"
    elif is_interleaved(args.mouse, args.date, args.data_dir):
        dt = pd.read_pickle(os.path.join(out_dir, "direction_table.pkl"))
        mod, mod_label = (dt.loc[cells, "p2_both_distance"] < 0.05).to_numpy(), "distance x direction ANOVA, distance p < 0.05"
    else:
        mod, mod_label = (b["anova_pvalue"] < 0.05).to_numpy(), "pipeline one-way ANOVA p < 0.05"
    log(f"  distance-modulated ({mod_label}): {mod.sum()} of {len(mod)}")
    r, p, ci, n = spearman_test(drf_rel[mod], wdist["both"][mod], n_perm=args.n_perm)
    log(f"  both, distance-modulated cells only: rho={r:+.3f}  "
        f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  shuffle p={p:.4f}  n={n}")
    r, p, ci, n = spearman_test(drf_rel, b["distance"].to_numpy(float), n_perm=args.n_perm)
    log(f"  both, pipeline preferred distance instead: rho={r:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
        f"shuffle p={p:.4f}")

    groups = [drf_rel[pref["both"] == k] for k in ds if np.sum(pref["both"] == k) > 1]
    p_kw = kruskal(*groups).pvalue
    log(f"\ndRF_rel by preferred distance (both eyes), Kruskal-Wallis p={p_kw:.3g}; medians: "
        + ", ".join(f"{DIST_CM[int(k)]} cm: {np.median(drf_rel[pref['both'] == k]):+.1f}"
                    for k in ds if np.any(pref["both"] == k)))

    # gap-corrected match
    pred, par = predicted_distance(drf_rel, sub["RF_right"].to_numpy(float), ds, args.eye_offset)
    agree = np.mean(pred == pref["both"])
    null = np.array([np.mean(rng.permutation(pred) == pref["both"]) for _ in range(args.n_perm)])
    p_match = (1 + np.sum(null >= agree)) / (1 + args.n_perm)
    swing = par - par.mean(axis=1, keepdims=True)
    log(f"\ngap-corrected match (left eye {args.eye_offset} cm): parallax swing across distances "
        + ", ".join(f"{DIST_CM[int(k)]} cm: {np.median(swing[:, i]):+.1f}" for i, k in enumerate(ds))
        + " deg (median over cells)")
    log("  predicted preferred distance, cells: "
        + ", ".join(f"{DIST_CM[int(k)]} cm: {np.sum(pred == k)}" for k in ds))
    log(f"  agreement with the observed both-eyes preference: {agree:.0%} "
        f"(shuffled: {null.mean():.0%}, 95th pct {np.percentile(null, 95):.0%}), p={p_match:.4f}")
    conf = pd.crosstab(pd.Series(pred, name="predicted"), pd.Series(pref["both"], name="observed"))
    conf = conf.reindex(index=ds, columns=ds, fill_value=0)
    log(str(conf))

    # ---------------------------------------------------------------- figure
    cm_labels = [f"{DIST_CM[int(k)]} cm" for k in ds]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    ax = axes[0, 0]
    for k in ds:
        m = pref["both"] == k
        ax.scatter(drf_rel[m], wdist["both"][m], s=22, color=DIST_COLOR[int(k)], edgecolor="white", linewidth=0.4,
                   label=f"prefers {DIST_CM[int(k)]} cm (n={m.sum()})", zorder=2)
    edges = np.percentile(drf_rel, np.linspace(0, 100, 9))
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (drf_rel >= lo) & (drf_rel <= hi)
        mids.append(np.median(drf_rel[m]))
        meds.append(np.nanmedian(wdist["both"][m]))
    ax.plot(mids, meds, color="black", linewidth=2, marker="o", markersize=4, label="running median", zorder=3)
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1)
    r, p, ci, n = res["both"]
    ax.set_title(f"A. Both eyes: Spearman ρ = {r:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], p = {p:.3f}, n = {n}",
                 fontsize=11, loc="left")
    ax.set_xlabel("RF right − left, gap removed (deg)\n← crosses nearer      crosses farther →", fontsize=11)
    ax.set_ylabel("Preferred distance, both eyes\n(response-weighted, cm)", fontsize=11)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5, loc="upper left")

    ax = axes[0, 1]
    for i, k in enumerate(ds):
        v = drf_rel[pref["both"] == k]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=DIST_COLOR[int(k)], alpha=0.75, zorder=2)
        if len(v):
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", linewidth=2.5, zorder=3)
    ax.axhline(0, color="0.6", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels(cm_labels)
    ax.set_xlabel("Preferred distance, both eyes (peak response)", fontsize=11)
    ax.set_ylabel("RF right − left, gap removed (deg)", fontsize=11)
    ax.set_title(f"B. Disparity by preferred distance (Kruskal-Wallis p = {p_kw:.3f})", fontsize=11, loc="left")

    ax = axes[1, 0]
    rows = [("both", "Both eyes", EYE_COLOR["both"]),
            ("both, RF partialled out", "Both eyes,\nRF position\npartialled out", EYE_COLOR["both"]),
            ("left", "Left eye only\n(control)", EYE_COLOR["left"]),
            ("right", "Right eye only\n(control)", EYE_COLOR["right"])]
    for i, (key, label, color) in enumerate(rows):
        r, p, ci, n = res[key]
        ax.errorbar(i, r, yerr=[[r - ci[0]], [ci[1] - r]], fmt="o", color=color, capsize=4, markersize=7,
                    mfc="white" if "partialled" in key else color)
        ax.text(i, ci[1] + 0.02, f"p = {p:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="0.6", linestyle="--", linewidth=1)
    ax.set_ylim(top=max(res[key][2][1] for key, _, _ in rows) + 0.1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([lab for _, lab, _ in rows], fontsize=9.5)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylabel("Spearman ρ (95% CI)", fontsize=11)
    ax.set_title("C. Disparity vs preferred distance, per session\n(prediction: > 0 with both eyes only)",
                 fontsize=11, loc="left")

    ax = axes[1, 1]
    im = ax.imshow(conf.to_numpy(), cmap="Greys", aspect="equal")
    for (i, j), v in np.ndenumerate(conf.to_numpy()):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                color="white" if v > conf.to_numpy().max() * 0.55 else "black")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels(cm_labels)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels(cm_labels)
    ax.set_xlabel("Observed preferred distance (both eyes)", fontsize=11)
    ax.set_ylabel(f"Predicted from gap-corrected disparity\n(left eye {args.eye_offset} cm)", fontsize=11)
    ax.set_title(f"D. Gap-corrected match: {agree:.0%} agree (shuffled {null.mean():.0%}), p = {p_match:.3f}",
                 fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, label="Cells", shrink=0.8)

    for a in axes.flat[:3]:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(labelsize=10)
    fig.suptitle(f"{title}: does the interocular RF offset predict distance preference?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "disparity_vs_distance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    delta, par, ball, theta_med = preferred_disparity(drf, sub["RF_right"].to_numpy(float), ds, args.eye_offset)
    plot_disparity_range(delta, ball, theta_med, pref["both"], ds, args.eye_offset, out_dir, title, log)
    plot_by_modulation(delta, par, ball, wdist["both"], pref["both"], mod, ds, args.eye_offset, out_dir, title, log,
                       n_perm=args.n_perm, mod_label=mod_label)
    log(f"\nSaved: disparity_vs_distance.png, disparity_range.png, disparity_by_modulation.png in {out_dir}")


if __name__ == "__main__":
    main()
