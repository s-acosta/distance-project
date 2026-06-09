"""Reformat ``ebr_pipeline_*.pkl`` outputs into a compact one-row-per-cell table.

Each ``ebr_pipeline`` pickle holds a large nested dict per recording. For downstream
work we usually only need, per cell:

- ``session_id``, ``cell_id``
- widefield coordinates ``wf_x`` / ``wf_y`` (centroid of the ROI mask)
- per condition (``light`` / ``dark``): normalized ego map, preferred distance,
  preferred angle, and whether the cell is an EBC in that condition
- ``general_ego_norm``: the combined ego map (taken directly from the pickle)
- optionally (when a ``sign_map_dir`` is supplied): ``azi`` / ``alt`` retinotopy and
  visual ``area`` sampled at the cell's widefield coordinate.

``SessionReformatter`` builds that table for one session; ``combine_sessions`` stacks
several sessions into a single DataFrame.

The per-condition ego maps are NOT stored in the pickle (only one combined ``ego_norm``
exists), so they are reconstructed here from the raw ray distances + thresholded spike
rates. The reconstruction reproduces the pipeline formula
``ego_norm = ego_spk / ego_occ`` where ``ego_spk`` is the per-bin sum of
``spk_rate_thr``; it omits the pipeline's tracking-quality filter and smoothing, so it
can differ from the stored combined map by ~1-2%.
"""

from __future__ import annotations

import glob
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path

# The stored azimuth map values are in a compressed/relative unit; the true visual
# azimuth spans this degree range. The azimuth map is linearly rescaled (its full
# min..max -> this range) so the ``azi`` column is in real degrees. Altitude is left as
# the raw map degrees.
AZI_DEG_RANGE = (-20.0, 130.0)


def _signed_angle_deg(rad):
    """Convert an upstream egocentric angle to a signed degree convention.

    Upstream angles are in radians, 0 = in front of the animal, increasing
    anti-clockwise. This returns degrees with front = 0, **right positive, left
    negative**, wrapped to ``(-180, 180]``.
    """
    deg = np.degrees(rad)
    return ((-deg + 180) % 360) - 180


def compute_ego_maps(df, settings, light_val):
    """Reconstruct per-cell normalized egocentric maps for one light/dark condition.

    ``ego_norm = ego_spk / ego_occ`` where ``ego_spk`` is the per-(ray, distance-bin)
    sum of ``spk_rate_thr`` and ``ego_occ`` the per-bin frame count, restricted to rows
    where ``df['light'] == light_val``.

    Returns an array of shape ``(n_cells, n_rays, n_dist)``; bins with zero occupancy are
    ``np.nan``.
    """
    n_rays = 360 // settings["ego_rays_deg"]
    n_dist = round(settings["ego_radius"] / settings["ego_bin_size"])
    n_cells = settings["nr_cells"]
    bin_size = settings["ego_bin_size"]
    radius = settings["ego_radius"]

    sel = (df["light"] == light_val).to_numpy()
    rd = np.stack(df["ray_distances"].values)[sel]  # (F, n_rays)
    sr = np.stack(
        [np.asarray(x, dtype=float) for x in df["spk_rate_thr"].values]
    )[sel]  # (F, n_cells)

    bin_idx = np.floor(rd / bin_size).astype(int)
    valid = (rd >= 0) & (rd < radius) & (bin_idx < n_dist)

    occ = np.zeros((n_rays, n_dist))
    spk = np.zeros((n_cells, n_rays, n_dist))
    for r in range(n_rays):
        m = valid[:, r]
        b = bin_idx[m, r]
        occ[r] = np.bincount(b, minlength=n_dist).astype(float)
        tmp = np.zeros((n_dist, n_cells))
        np.add.at(tmp, b, sr[m])  # sum spk_rate_thr per (dist_bin, cell)
        spk[:, r, :] = tmp.T

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(occ > 0, spk / occ, np.nan)


def preferred_from_map(ego, settings, clip_deg=120, window_deg=5):
    """Preferred angle/distance for one ``(n_rays, n_dist)`` ego map.

    Preferred angle = peak of the distance-collapsed angular profile (within
    +/- ``clip_deg`` of front; signed degrees, front=0, right +, left -). The distance
    curve is the mean (+/- SEM) over rays within +/- ``window_deg`` of that peak;
    preferred distance = its peak. Returns a dict with the angle/distance and the
    plotting helpers (raw-radian angle, normalized radius, distance mean/SEM curves).
    """
    ego = np.asarray(ego, dtype=float)
    n_rays, n_dist = ego.shape
    ray_signed = _signed_angle_deg(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    ang_mean = np.nanmean(ego, axis=1)                   # collapse over distance
    if clip_deg is not None:
        ang_mean = np.where(np.abs(ray_signed) <= clip_deg, ang_mean, np.nan)

    bin_size = settings["ego_bin_size"]
    nan_row = np.full(n_dist, np.nan)
    if np.all(np.isnan(ang_mean)):                       # no spikes in this condition
        return {"ray": 0, "ang_signed": np.nan, "ang_raw": np.nan, "ang_val": np.nan,
                "dbin": 0, "dist_m": np.nan, "dist_cm": np.nan, "r_norm": np.nan,
                "dist_mean": nan_row, "dist_sem": nan_row}

    pref_ray = int(np.nanargmax(ang_mean))
    pref_signed = float(ray_signed[pref_ray])

    # rays within +/- window_deg of the peak angle (circular difference)
    diff = (ray_signed - pref_signed + 180) % 360 - 180
    block = ego[np.abs(diff) <= window_deg]              # (k, n_dist)
    dist_mean = np.nanmean(block, axis=0)
    cnt = np.sum(np.isfinite(block), axis=0)
    with np.errstate(invalid="ignore"):
        dist_sem = np.nanstd(block, axis=0) / np.sqrt(cnt)
    pref_dbin = int(np.nanargmax(dist_mean))
    return {
        "ray": pref_ray,
        "ang_signed": pref_signed,
        "ang_raw": pref_ray * 2 * np.pi / n_rays,        # raw radians for polar marker
        "ang_val": float(ang_mean[pref_ray]),
        "dbin": pref_dbin,
        "dist_m": pref_dbin * bin_size,
        "dist_cm": pref_dbin * bin_size * 100,
        "r_norm": pref_dbin / (n_dist - 1),
        "dist_mean": dist_mean,
        "dist_sem": dist_sem,
    }


class SessionReformatter:
    """Build a per-cell summary table for a single ``ebr_pipeline`` recording."""

    # The "light" column in df is, counter-intuitively, 0 for the LIGHT condition and
    # 1 for the DARK condition: ebcc_outputs_light was computed on rows where
    # df['light'] == 0, and ebcc_outputs_dark on df['light'] == 1. Per-condition ego
    # maps must use the same split so they line up with the stored scalars/masks.
    COND_TO_LIGHTVAL = {"light": 0, "dark": 1}

    def __init__(self, ebr_pkl_path, widefield_pkl_path, session_id=None,
                 sign_map_dir=None):
        with open(ebr_pkl_path, "rb") as f:
            ebr = pickle.load(f)
        with open(widefield_pkl_path, "rb") as f:
            wf = pickle.load(f)

        self.settings = ebr["settings"]
        self.session_id = session_id or self.settings["rec_name"]

        self.df = ebr["df"]
        self.ego_norm_general = ebr["ego_norm"]  # object array, one (120,14) per cell
        self.ebcc = {
            "light": ebr["ebcc_outputs_light"],
            "dark": ebr["ebcc_outputs_dark"],
        }
        self.roi_coords_wf = wf["roi_coords_widefield"][self.session_id]
        self.wf_shape = wf["widefield_img"].shape  # (H, W) widefield frame of wf coords

        self.n_cells = self.settings["nr_cells"]
        self.n_rays = 360 // self.settings["ego_rays_deg"]
        self.n_dist = round(self.settings["ego_radius"] / self.settings["ego_bin_size"])

        self.has_retino = sign_map_dir is not None
        if self.has_retino:
            self._load_retino(sign_map_dir)

    def _load_retino(self, sign_map_dir):
        """Load retinotopy maps + area polygons and bring them into the widefield frame.

        Maps (azimuth/altitude) may be stored in the widefield image dimensions or in a
        smaller frame (e.g. 400x400). They are resized to the widefield frame so they can
        be indexed directly with wf_x/wf_y; area polygons are scaled by the same factor.
        """
        npz = np.load(os.path.join(sign_map_dir, "retinotopy.npz"), allow_pickle=True)
        with open(os.path.join(sign_map_dir, "retinotopy_areas.pkl"), "rb") as f:
            areas = pickle.load(f)

        h_wf, w_wf = self.wf_shape
        map_h, map_w = npz["azimuth_map"].shape
        # uniform scale from map space -> widefield space (1.0 if already matching)
        scale = w_wf / map_w

        # nearest keeps NaNs from spreading and matches direct-index sampling
        self.azi_map = cv2.resize(np.asarray(npz["azimuth_map"], dtype=float),
                                  (w_wf, h_wf), interpolation=cv2.INTER_NEAREST)
        self.alt_map = cv2.resize(np.asarray(npz["altitude_map"], dtype=float),
                                  (w_wf, h_wf), interpolation=cv2.INTER_NEAREST)

        # rescale azimuth from its full map range into real degrees (AZI_DEG_RANGE)
        amin, amax = np.nanmin(self.azi_map), np.nanmax(self.azi_map)
        lo, hi = AZI_DEG_RANGE
        if amax > amin:
            self.azi_map = lo + (self.azi_map - amin) / (amax - amin) * (hi - lo)

        self._area_paths = {
            name: Path(np.asarray(verts, dtype=float) * scale)
            for name, verts in areas.items()
        }

    # ------------------------------------------------------------------ helpers
    def _ego_maps(self, light_val):
        """Per-cell normalized ego maps for one condition (see ``compute_ego_maps``)."""
        return compute_ego_maps(self.df, self.settings, light_val)

    def _wf_centroid(self, cell_id):
        roi = self.roi_coords_wf[cell_id]
        return float(np.mean(roi["x_coords"])), float(np.mean(roi["y_coords"]))

    def _retino_at(self, x, y):
        """Sample (azi, alt) at a widefield coordinate; NaN if out of map bounds."""
        h, w = self.azi_map.shape
        xi, yi = int(x), int(y)
        if 0 <= yi < h and 0 <= xi < w:
            return float(self.azi_map[yi, xi]), float(self.alt_map[yi, xi])
        return np.nan, np.nan

    def _assign_area(self, x, y):
        """Return the visual area whose polygon contains (x, y), else 'None'."""
        for name, path in self._area_paths.items():
            if path.contains_point((x, y)):
                return name
        return "None"

    # -------------------------------------------------------------------- build
    def build(self, angle_clip=120, window_deg=5):
        """Return a per-cell ``pd.DataFrame`` for this session.

        Preferred angle/distance are computed from the reconstructed ego maps
        (``preferred_from_map``: peak of the angular profile within +/- ``angle_clip``,
        then peak of the distance curve averaged over +/- ``window_deg`` around it) rather
        than the stored pipeline stats. ``light_is_ebc``/``dark_is_ebc`` still come from
        the pipeline's classification. Distances are in cm; angles in signed degrees.
        """
        maps = {c: self._ego_maps(self.COND_TO_LIGHTVAL[c]) for c in ("light", "dark")}

        rows = []
        for cid in range(self.n_cells):
            wf_x, wf_y = self._wf_centroid(cid)
            lp = preferred_from_map(maps["light"][cid], self.settings, angle_clip, window_deg)
            dp = preferred_from_map(maps["dark"][cid], self.settings, angle_clip, window_deg)
            row = {
                "session_id": self.session_id,
                "cell_id": cid,
                "wf_x": wf_x,
                "wf_y": wf_y,
                "general_ego_norm": self.ego_norm_general[cid],
                "light_ego_norm": maps["light"][cid],
                "light_pref_dist": lp["dist_cm"],         # cm, from reconstructed map
                "light_pref_angle": lp["ang_signed"],     # signed deg, front=0, right +
                "light_is_ebc": bool(self.ebcc["light"]["ebcs_mask"][cid]),
                "dark_ego_norm": maps["dark"][cid],
                "dark_pref_dist": dp["dist_cm"],
                "dark_pref_angle": dp["ang_signed"],
                "dark_is_ebc": bool(self.ebcc["dark"]["ebcs_mask"][cid]),
            }
            if self.has_retino:
                azi, alt = self._retino_at(wf_x, wf_y)
                row["azi"] = azi
                row["alt"] = alt
                row["area"] = self._assign_area(wf_x, wf_y)
            rows.append(row)
        return pd.DataFrame(rows)


def combine_sessions(reformatters, **build_kwargs):
    """Build and concatenate per-cell tables from many ``SessionReformatter`` objects.

    Extra keyword args (e.g. ``angle_clip``, ``window_deg``) are forwarded to ``build``.
    Each row carries a ``session_id`` column identifying its source recording.
    """
    frames = [r.build(**build_kwargs) for r in reformatters]
    return pd.concat(frames, ignore_index=True)


def build_all_sessions(ebr_dir, widefield_pkl_path, sign_map_dir=None,
                       pattern="ebr_pipeline_*.pkl", angle_clip=120, window_deg=5):
    """Build and combine every session pickle in a directory into one table.

    Parameters
    ----------
    ebr_dir : str
        Directory holding the ``ebr_pipeline_*.pkl`` files (e.g. the LDR004 folder).
    widefield_pkl_path : str
        Path to the mouse's ``*_widefield_transformations.pkl`` (shared by all sessions).
    sign_map_dir : str, optional
        Directory with ``retinotopy.npz`` + ``retinotopy_areas.pkl``. When given, each
        cell also gets ``azi`` / ``alt`` / ``area`` columns.
    pattern : str
        Glob pattern for the session pickles.
    angle_clip, window_deg :
        Forwarded to ``build`` for the map-derived preferred angle/distance.

    Returns
    -------
    pd.DataFrame
        One row per cell across all sessions, with a ``session_id`` source column.
    """
    paths = sorted(glob.glob(os.path.join(ebr_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {ebr_dir}")
    reformatters = [
        SessionReformatter(p, widefield_pkl_path, sign_map_dir=sign_map_dir)
        for p in paths
    ]
    return combine_sessions(reformatters, angle_clip=angle_clip, window_deg=window_deg)
