"""Per-session analysis summary.

One tidy **per-cell** stats table per session (one row per cell, all cells), consolidating:

- **visual classification** (``VisualClassifier``): ``is_visual`` + light/dark reliability,
  null percentile, and significance flags.
- **RF localization** (``RFAnalyzer``, light): the robust two-peak preferred orientation
  ``rf_angle_deg`` and distance ``rf_dist_cm`` (+ widths, ``rf_R``, frontal/ghost/ambiguous
  flags, and the split-half ``rf_stability``), plus the single-bin ``argmax_*`` reference.
- **EBC pipeline** (``ce.ebcc``, light + dark): the pipeline's own ``ebc_is_ebc``, preferred
  angle/distance, tuning strength ``ebc_mrl``, correlation ``ebc_cc`` and criteria.

``keep`` is the analysis-final "visual" set: ``is_visual & rf_frontal & rf_stability < cut`` --
i.e. a cell counts as visual only if its RF is also frontal and **stable** across halves.

``build_session_summary`` returns the DataFrame; ``process_session`` also writes
``summary.pkl`` (and an updated per-cell ``visual_cells_rf.pdf``); ``summarize_all`` runs every
session in a folder and writes a combined master table.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from cell_explorer import CellExplorer
from visual_cells import VisualClassifier
from receptive_field import RFAnalyzer
from ego_reformat import _signed_angle_deg, SessionReformatter

# ebcc per-cell field -> summary column base (suffixed _light / _dark)
_EBC_FIELDS = {
    "ebcs_mask": "ebc_is_ebc",
    "mras_all_cells": "ebc_mra_deg",          # radians -> signed degrees
    "pref_dists_all_cells": "ebc_pref_dist_cm",
    "mrls_all_cells": "ebc_mrl",
    "ebr_cc_all_cells": "ebc_cc",
    "mrl_criterion_met": "ebc_mrl_crit",
    "cc_criterion_met": "ebc_cc_crit",
}


def _session_folder(session_id):
    """Output folder name for a session, e.g. ``DMM065_ltdk_03`` from its rec_name."""
    m = re.search(r"[A-Z]+\d+_ltdk_\w+", session_id)
    return m.group(0) if m else session_id


def _mouse_name(session_id):
    """Mouse id from a session rec_name, e.g. ``DMM065`` from ``251027_LDR_DMM065_ltdk_03``."""
    m = re.search(r"[A-Z]+\d+", str(session_id))
    return m.group(0) if m else "all"


def build_session_summary(source, stability=True, stability_cut=30.0, verbose=True):
    """Per-cell stats DataFrame for one session (see module docstring for columns).

    ``source`` is a ``CellExplorer`` or a path to an ``ebr_pipeline_*.pkl``.
    """
    ce = CellExplorer(source) if isinstance(source, (str, os.PathLike)) else source
    clf = VisualClassifier(ce)
    clf.classify(verbose=verbose)
    rf = RFAnalyzer(ce, clf)
    return _build(ce, clf, rf, stability, stability_cut, verbose)


def _build(ce, clf, rf, stability, stability_cut, verbose):
    rf_tab = rf.table(cells=np.arange(ce.n_cells), condition="light", stability=stability)
    vis = clf._last["df"]                                   # cell, is_visual, reliabilities...

    df = vis.merge(rf_tab.drop(columns=["light_reliability"], errors="ignore"), on="cell")
    for cond in ("light", "dark"):                          # EBC-pipeline metrics
        e = ce.ebcc[cond]
        for key, base in _EBC_FIELDS.items():
            v = np.asarray(e[key])
            df[f"{base}_{cond}"] = _signed_angle_deg(v) if key == "mras_all_cells" else v
    df.insert(0, "session_id", ce.session_id)

    stable = (df["rf_stability"] < stability_cut) if "rf_stability" in df else True
    df["keep"] = df["is_visual"].to_numpy() & df["rf_frontal"].to_numpy() & np.asarray(stable)
    if verbose:
        print(f"  summary: {len(df)} cells | visual {int(df.is_visual.sum())} | "
              f"frontal+visual {int((df.is_visual & df.rf_frontal).sum())} | "
              f"keep (visual+frontal+stable<{stability_cut:.0f}) {int(df.keep.sum())}", flush=True)
    return df


def _render_pdf(rf, df, path):
    """Per-cell review PDF over the visual cells (keep first, then by stability), each page the
    current RFAnalyzer.plot_cell with the cell's split-half stability + KEEP/DROP annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    vis = df[df["is_visual"]].sort_values(["keep", "rf_stability"],
                                          ascending=[False, True], na_position="last")
    with PdfPages(path) as pdf:
        for _, r in vis.iterrows():
            c = int(r["cell"])
            try:
                fig = rf.plot_cell(c)
                s = r.get("rf_stability", np.nan)
                stab = f"{s:.0f} deg" if np.isfinite(s) else "nan"
                keep = bool(r["keep"])
                fig.text(0.5, 0.005, f"split-half stability {stab}   |   "
                         f"{'KEEP (visual)' if keep else 'DROP (unstable/peripheral)'}",
                         ha="center", fontsize=11, color="green" if keep else "red")
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as ex:                         # one bad cell can't abort the PDF
                print(f"    cell {c} plot FAILED: {ex}", flush=True)
                plt.close("all")
    return int(vis.shape[0])


def process_session(source, out_dir, stability_cut=30.0, pdf=True, verbose=True):
    """Build the summary for one session and write, to ``out_dir``: ``summary.pkl``,
    ``plot_population.png`` (RF centers of the keep set), and (if ``pdf``) the updated
    ``visual_cells_rf.pdf``. Returns the DataFrame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ce = CellExplorer(source) if isinstance(source, (str, os.PathLike)) else source
    clf = VisualClassifier(ce)
    clf.classify(verbose=verbose)
    rf = RFAnalyzer(ce, clf)
    df = _build(ce, clf, rf, True, stability_cut, verbose)

    os.makedirs(out_dir, exist_ok=True)
    df.to_pickle(os.path.join(out_dir, "summary.pkl"))
    try:                                                # population over the keep (clean) cells
        pop = rf.plot_population(df[df["keep"]])
        pop.savefig(os.path.join(out_dir, "plot_population.png"), dpi=130, bbox_inches="tight")
        plt.close(pop)
    except Exception as ex:
        print(f"  plot_population FAILED: {ex}", flush=True)
        plt.close("all")
    if pdf:
        n = _render_pdf(rf, df, os.path.join(out_dir, "visual_cells_rf.pdf"))
        if verbose:
            print(f"  wrote summary.pkl + plot_population.png + visual_cells_rf.pdf "
                  f"({n} pages)", flush=True)
    return df


def summarize_all(ebr_dir, results_root, pattern="ebr_pipeline_*.pkl",
                  stability_cut=30.0, pdf=True, combined=True, skip_existing=False,
                  verbose=True):
    """Summarize every ``ebr_pipeline_*.pkl`` in ``ebr_dir`` -> per-session
    ``results_root/<MOUSE>_ltdk_<NN>/summary.pkl`` (+ ``visual_cells_rf.pdf`` if ``pdf``) and a
    combined ``results_root/<MOUSE>_all_sessions_summary.pkl`` (mouse id read from rec_name).

    ``skip_existing`` makes the (~1 hr) run resumable: a session whose ``summary.pkl`` already
    exists is reloaded from disk instead of recomputed (still folded into the combined master).
    The folder is derived from the pickle filename, so a skipped session is never loaded."""
    paths = sorted(glob.glob(os.path.join(ebr_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {ebr_dir}")
    all_df = []
    for i, p in enumerate(paths, 1):
        try:
            folder = _session_folder(os.path.basename(p))   # from filename: no pickle load to skip
            out_dir = os.path.join(results_root, folder)
            spath = os.path.join(out_dir, "summary.pkl")
            if skip_existing and os.path.exists(spath):
                print(f"[{i}/{len(paths)}] {folder} -- summary.pkl exists, loading", flush=True)
                all_df.append(pd.read_pickle(spath))
                continue
            print(f"[{i}/{len(paths)}] {folder}", flush=True)
            df = process_session(CellExplorer(p), out_dir,
                                 stability_cut=stability_cut, pdf=pdf, verbose=verbose)
            all_df.append(df)
        except Exception as ex:
            print(f"[{i}/{len(paths)}] {os.path.basename(p)} FAILED: {ex}", flush=True)
    if combined and all_df:
        master = pd.concat(all_df, ignore_index=True)
        os.makedirs(results_root, exist_ok=True)
        mouse = _mouse_name(master["session_id"].iloc[0])
        mpath = os.path.join(results_root, f"{mouse}_all_sessions_summary.pkl")
        master.to_pickle(mpath)
        print(f"DONE: {len(all_df)} sessions, {len(master)} cells -> {mpath}", flush=True)
    return all_df


# ----------------------------------------------------------------------------- visual table
_DMM065 = r"D:\data-distance-coding-project\mini2p-data\DMM065"
_EGO_PKL_DEFAULT = os.path.join(_DMM065, "DMM065_sessions_ego.pkl")
_VISUAL_PKL_DEFAULT = os.path.join(_DMM065, "DMM065_sessions_visual.pkl")
_RESULTS_DEFAULT = os.path.join(_DMM065, "results")
_WIDEFIELD_PKL_DEFAULT = os.path.join(_DMM065, "DMM065_widefield_transformations.pkl")
_SIGN_MAP_DEFAULT = os.path.join(_DMM065, "sign-map")

_WF_RETINO_COLS = ["session_id", "cell_id", "wf_x", "wf_y", "azi", "alt", "area"]

# column order for the combined visual table (any extra columns are appended after these)
_VISUAL_ORDER = (
    _WF_RETINO_COLS
    + ["is_visual", "light_reliability", "light_null_p", "light_sig",
       "dark_reliability", "dark_null_p", "dark_sig", "n_spk_light", "n_spk_dark"]
    + ["rf_angle_deg", "rf_dist_cm", "rf_ang_sd_deg", "rf_dist_sd_cm",
       "rf_ang_span_deg", "rf_dist_span_cm", "rf_area_bins", "rf_peak", "rf_R",
       "rf_frontal", "rf_global_angle_deg", "rf_ghost_angle_deg", "rf_ghost_dist_cm",
       "rf_ambiguous", "argmax_angle_deg", "argmax_dist_cm", "rf_stability"]
    + ["keep"]
)


def _wf_retino_recompute(sessions, ebr_dir, widefield_pkl, sign_map_dir, verbose=True):
    """``[session_id, cell_id, wf_x, wf_y, azi, alt, area]`` for ``sessions``, via
    ``SessionReformatter`` (the code that built the ego table) -- only its cheap per-cell
    widefield/retinotopy helpers, so no ego maps are rebuilt."""
    if not (ebr_dir and widefield_pkl and sign_map_dir):
        raise FileNotFoundError(
            "wf/retino missing for sessions " + ", ".join(map(str, sessions)) +
            "; pass ebr_dir + widefield_pkl + sign_map_dir to recompute")
    out = []
    for s in sessions:
        hits = glob.glob(os.path.join(ebr_dir, f"ebr_pipeline_*{s}*.pkl"))
        if not hits:
            raise FileNotFoundError(f"no ebr_pipeline pkl for session {s!r} in {ebr_dir}")
        r = SessionReformatter(hits[0], widefield_pkl, sign_map_dir=sign_map_dir)
        rows = []
        for cid in range(r.n_cells):
            x, y = r._wf_centroid(cid)
            azi, alt = r._retino_at(x, y)
            rows.append((r.session_id, cid, x, y, azi, alt, r._assign_area(x, y)))
        out.append(pd.DataFrame(rows, columns=_WF_RETINO_COLS))
        if verbose:
            print(f"  wf/retino {r.session_id} ({r.n_cells} cells)", flush=True)
    return pd.concat(out, ignore_index=True)


def _wf_retino_table(ego_pkl, sessions, ebr_dir=None, widefield_pkl=None,
                     sign_map_dir=None, verbose=True):
    """``[session_id, cell_id, wf_x, wf_y, azi, alt, area]`` for every requested session.

    Sliced from the existing ego pkl when it is present and readable (fast path); otherwise
    (or for any session missing there) recomputed from the source files via
    ``SessionReformatter`` -- e.g. when the ego pkl was written by a different numpy and can't
    be unpickled here.
    """
    frames, have = [], set()
    if ego_pkl and os.path.exists(ego_pkl):
        try:
            ego = pd.read_pickle(ego_pkl)
            if set(_WF_RETINO_COLS).issubset(ego.columns):
                frames.append(ego[_WF_RETINO_COLS])
                have = set(ego["session_id"].unique())
            elif verbose:
                print(f"  ego pkl lacks {set(_WF_RETINO_COLS) - set(ego.columns)}; recomputing",
                      flush=True)
        except Exception as ex:
            if verbose:
                print(f"  ego pkl unreadable ({type(ex).__name__}); recomputing wf/retino "
                      f"from source files", flush=True)
    missing = [s for s in sessions if s not in have]
    if missing:
        frames.append(_wf_retino_recompute(missing, ebr_dir, widefield_pkl, sign_map_dir, verbose))
    wf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_WF_RETINO_COLS)
    return wf[wf["session_id"].isin(sessions)].reset_index(drop=True)


def build_sessions_visual(results_root=_RESULTS_DEFAULT, ego_pkl=_EGO_PKL_DEFAULT,
                          out_path=_VISUAL_PKL_DEFAULT, drop_ebc=True,
                          pattern="*_ltdk_*/summary.pkl", ebr_dir=_DMM065,
                          widefield_pkl=_WIDEFIELD_PKL_DEFAULT, sign_map_dir=_SIGN_MAP_DEFAULT,
                          verbose=True):
    """Combine the per-session ``summary.pkl``s with each cell's widefield position + retinotopy
    into one per-cell table (``DMM065_sessions_visual.pkl``), the same one-row-per-cell shape as
    ``DMM065_sessions_ego.pkl``, for the ``WidefieldCellPlotter`` visual plots.

    The visual columns come from the summaries under ``results_root``; the widefield/retinotopy
    columns (``wf_x``, ``wf_y``, ``azi``, ``alt``, ``area``) are sliced from ``ego_pkl`` (already
    computed for every cell), recomputed via ``SessionReformatter`` only for any session missing
    there (needs ``ebr_dir`` + ``widefield_pkl`` + ``sign_map_dir``). With ``drop_ebc`` the EBC
    pipeline columns are dropped (recoverable from ``summary.pkl``); ego-map arrays are never
    included. Returns the merged DataFrame and writes ``out_path``.
    """
    paths = sorted(glob.glob(os.path.join(results_root, *pattern.split("/"))))
    if not paths:
        raise FileNotFoundError(f"no summaries matching {pattern!r} under {results_root}")
    vis = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    vis = vis.rename(columns={"cell": "cell_id"})
    if drop_ebc:
        vis = vis.drop(columns=[c for c in vis.columns if c.startswith("ebc_")], errors="ignore")

    wf = _wf_retino_table(ego_pkl, vis["session_id"].unique(), ebr_dir=ebr_dir,
                          widefield_pkl=widefield_pkl, sign_map_dir=sign_map_dir, verbose=verbose)

    out = wf.merge(vis, on=["session_id", "cell_id"], how="inner", validate="one_to_one")
    if len(out) != len(vis) and verbose:
        print(f"  WARNING: {len(vis) - len(out)} of {len(vis)} visual rows had no wf/retino match",
              flush=True)

    cols = [c for c in _VISUAL_ORDER if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_pickle(out_path)
    if verbose:
        print(f"visual table: {len(out)} cells / {out['session_id'].nunique()} sessions, "
              f"{out.shape[1]} cols -> {out_path}", flush=True)
        print(f"  is_visual {int(out['is_visual'].sum())} | keep {int(out['keep'].sum())} | "
              f"areas {dict(out['area'].value_counts())}", flush=True)
    return out
