"""Build every per-cell data table for one mini2p animal, end to end.

Given an animal id (e.g. ``DMM065``) and the standard per-animal data folder, this runs the
three existing builders in order and writes their outputs:

1. ego table       -> ``<ANIMAL>_sessions_ego.pkl``            (ego_reformat.build_all_sessions)
2. per-session     -> ``results/<ANIMAL>_ltdk_NN/summary.pkl`` (+ combined master)
   summaries          (session_summary.summarize_all; the slow ~1 hr stage)
3. sessions-visual -> ``<ANIMAL>_sessions_visual.pkl``         (session_summary.build_sessions_visual)

All paths are derived from the per-animal convention (verified for LDR004 / DMM065)::

    <data_dir>/
        ebr_pipeline_*.pkl                       # session inputs (Lucas pipeline)
        <ANIMAL>_widefield_transformations.pkl
        sign-map/   (retinotopy.npz, retinotopy_areas.pkl)
        results/
        <ANIMAL>_sessions_ego.pkl
        <ANIMAL>_sessions_visual.pkl

``data_dir`` defaults to ``D:\\data-distance-coding-project\\mini2p-data\\<ANIMAL>``; every path
can be overridden explicitly. A stage whose output already exists is skipped unless ``force`` is
set; the summaries stage additionally resumes per session (see ``summarize_all(skip_existing=)``).

Run with the interpreter whose numpy can read the pickles -- the notebook kernel (numpy 2.x);
``py -3.12`` (numpy 1.26) imports fine but may fail to unpickle the inputs.

Usage:
    python build_animal.py DMM065
    python build_animal.py DMM065 --steps summaries visual --force --no-pdf
    python build_animal.py LDR004 --data-dir D:\\...\\LDR004 --dry-run

    from build_animal import build_animal
    build_animal("DMM065")                      # all three stages, skip what exists
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# make the sibling modules importable regardless of CWD / how we're launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ego_reformat import build_all_sessions
from session_summary import summarize_all, build_sessions_visual

DATA_ROOT = r"D:\data-distance-coding-project\mini2p-data"
ALL_STEPS = ("ego", "summaries", "visual")


def _resolve_paths(animal, data_dir, sign_map_subdir, ebr_dir, widefield_pkl,
                   sign_map_dir, results_root, ego_pkl, visual_pkl):
    """Fill in every path from the per-animal convention, honoring explicit overrides."""
    base = data_dir or os.path.join(DATA_ROOT, animal)
    return {
        "base": base,
        "ebr_dir": ebr_dir or base,
        "widefield_pkl": widefield_pkl
            or os.path.join(base, f"{animal}_widefield_transformations.pkl"),
        "sign_map_dir": sign_map_dir or os.path.join(base, sign_map_subdir),
        "results_root": results_root or os.path.join(base, "results"),
        "ego_pkl": ego_pkl or os.path.join(base, f"{animal}_sessions_ego.pkl"),
        "visual_pkl": visual_pkl or os.path.join(base, f"{animal}_sessions_visual.pkl"),
    }


def _preflight(paths, steps):
    """Raise FileNotFoundError listing every missing input needed for the requested steps.

    Returns the number of ``ebr_pipeline_*.pkl`` files found.
    """
    problems = []
    n_ebr = len(glob.glob(os.path.join(paths["ebr_dir"], "ebr_pipeline_*.pkl")))
    if n_ebr == 0:
        problems.append(f"no ebr_pipeline_*.pkl in {paths['ebr_dir']}")
    if not os.path.exists(paths["widefield_pkl"]):
        problems.append(f"missing widefield pkl: {paths['widefield_pkl']}")
    for fname in ("retinotopy.npz", "retinotopy_areas.pkl"):
        if not os.path.exists(os.path.join(paths["sign_map_dir"], fname)):
            problems.append(f"missing {fname} in {paths['sign_map_dir']}")
    # the visual stage consumes the ego table; if we are not building it this run it must exist
    if "visual" in steps and "ego" not in steps and not os.path.exists(paths["ego_pkl"]):
        problems.append(f"ego table needed by 'visual' but missing: {paths['ego_pkl']} "
                        f"(add 'ego' to --steps)")
    # the visual stage also consumes the per-session summaries
    if "visual" in steps and "summaries" not in steps:
        if not glob.glob(os.path.join(paths["results_root"], "*_ltdk_*", "summary.pkl")):
            problems.append(f"no per-session summary.pkl under {paths['results_root']} "
                            f"(add 'summaries' to --steps)")
    if problems:
        raise FileNotFoundError(
            "build_animal preflight failed:\n  - " + "\n  - ".join(problems))
    return n_ebr


def build_animal(animal, data_dir=None, *, steps=ALL_STEPS, sign_map_subdir="sign-map",
                 force=False, pdf=True, stability_cut=30.0, angle_clip=120, window_deg=5,
                 dry_run=False, verbose=True, ebr_dir=None, widefield_pkl=None,
                 sign_map_dir=None, results_root=None, ego_pkl=None, visual_pkl=None):
    """Build the ego / summaries / visual tables for ``animal`` (see the module docstring).

    Parameters
    ----------
    animal : str
        Animal id (e.g. ``"DMM065"``); drives the derived paths and output filenames.
    data_dir : str, optional
        Per-animal data folder. Default ``<DATA_ROOT>/<animal>``.
    steps : tuple
        Which stages to run, in order; any subset of ``("ego", "summaries", "visual")``.
    sign_map_subdir : str
        Retinotopy subfolder under ``data_dir`` (default ``"sign-map"``).
    force : bool
        Rebuild a stage even if its output exists; also disables the per-session summary resume.
    pdf, stability_cut, angle_clip, window_deg :
        Forwarded to the underlying builders (``pdf``/``stability_cut`` -> ``summarize_all``;
        ``angle_clip``/``window_deg`` -> ``build_all_sessions``).
    dry_run : bool
        Resolve + validate paths, print them, and return without building anything.
    ebr_dir, widefield_pkl, sign_map_dir, results_root, ego_pkl, visual_pkl : str, optional
        Explicit path overrides; any left ``None`` is derived from the convention.

    Returns
    -------
    dict
        The resolved paths (keys ``base``, ``ebr_dir``, ``widefield_pkl``, ``sign_map_dir``,
        ``results_root``, ``ego_pkl``, ``visual_pkl``).
    """
    steps = tuple(steps)
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise ValueError(f"unknown step(s) {bad}; choose from {ALL_STEPS}")

    paths = _resolve_paths(animal, data_dir, sign_map_subdir, ebr_dir, widefield_pkl,
                           sign_map_dir, results_root, ego_pkl, visual_pkl)

    if verbose or dry_run:
        print(f"=== build_animal: {animal} ===")
        print(f"  data dir      : {paths['base']}")
        print(f"  ebr dir       : {paths['ebr_dir']}")
        print(f"  widefield pkl : {paths['widefield_pkl']}")
        print(f"  sign-map dir  : {paths['sign_map_dir']}")
        print(f"  results root  : {paths['results_root']}")
        print(f"  ego pkl       : {paths['ego_pkl']}")
        print(f"  visual pkl    : {paths['visual_pkl']}")
        print(f"  steps         : {', '.join(steps)}   force={force}  pdf={pdf}")

    n_ebr = _preflight(paths, steps)
    if verbose or dry_run:
        print(f"  preflight OK  : {n_ebr} ebr_pipeline pickles found")

    if dry_run:
        print("  dry run -- nothing built")
        return paths

    # 1) ego table -------------------------------------------------------------------
    if "ego" in steps:
        if os.path.exists(paths["ego_pkl"]) and not force:
            print(f"[ego] exists, skipping -> {paths['ego_pkl']}", flush=True)
        else:
            print("[ego] building ego table (build_all_sessions)...", flush=True)
            df = build_all_sessions(paths["ebr_dir"], paths["widefield_pkl"],
                                    sign_map_dir=paths["sign_map_dir"],
                                    angle_clip=angle_clip, window_deg=window_deg)
            df.to_pickle(paths["ego_pkl"])
            print(f"[ego] wrote {len(df)} cells -> {paths['ego_pkl']}", flush=True)

    # 2) per-session summaries (the slow stage; resumable unless force) ----------------
    if "summaries" in steps:
        print(f"[summaries] running summarize_all (skip_existing={not force}, pdf={pdf})...",
              flush=True)
        summarize_all(paths["ebr_dir"], paths["results_root"], stability_cut=stability_cut,
                      pdf=pdf, combined=True, skip_existing=not force, verbose=verbose)

    # 3) sessions-visual table --------------------------------------------------------
    if "visual" in steps:
        if os.path.exists(paths["visual_pkl"]) and not force:
            print(f"[visual] exists, skipping -> {paths['visual_pkl']}", flush=True)
        else:
            print("[visual] building sessions-visual table (build_sessions_visual)...",
                  flush=True)
            build_sessions_visual(results_root=paths["results_root"], ego_pkl=paths["ego_pkl"],
                                  out_path=paths["visual_pkl"], ebr_dir=paths["ebr_dir"],
                                  widefield_pkl=paths["widefield_pkl"],
                                  sign_map_dir=paths["sign_map_dir"], verbose=verbose)

    print(f"=== done: {animal} ===", flush=True)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the ego / summaries / visual tables for one mini2p animal.")
    ap.add_argument("animal", help="animal id, e.g. DMM065 (drives paths + output names)")
    ap.add_argument("--data-dir", default=None,
                    help=rf"per-animal data folder (default: {DATA_ROOT}\<ANIMAL>)")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS),
                    metavar="STEP", help="stages to run, in order (default: all three)")
    ap.add_argument("--sign-map-subdir", default="sign-map",
                    help="retinotopy subfolder under the data dir (default: sign-map)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild stages even if their output exists "
                         "(also disables the per-session summary resume)")
    ap.add_argument("--no-pdf", dest="pdf", action="store_false",
                    help="skip the per-cell visual_cells_rf.pdf render (faster summaries)")
    ap.add_argument("--stability-cut", type=float, default=30.0,
                    help="RF split-half stability cut (deg) for the keep set (default: 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + validate paths, print them, and exit without building")
    args = ap.parse_args(argv)

    build_animal(args.animal, data_dir=args.data_dir, steps=tuple(args.steps),
                 sign_map_subdir=args.sign_map_subdir, force=args.force, pdf=args.pdf,
                 stability_cut=args.stability_cut, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
