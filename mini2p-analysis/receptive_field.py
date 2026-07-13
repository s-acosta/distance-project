"""Egocentric receptive-field location + size for visual cells.

For a visual cell the egocentric (occupancy-normalized) **light** map has a preferred wall
region ``(angle, distance)``. In a closed symmetric box that region comes as a **confound
pair**: the near wall the animal looks at *and* its far-wall ghost ~180 deg behind at the
complementary distance (their distances sum to the box size). ``RFAnalyzer`` per cell:

1. **Detects the two strongest, opposite 2D peaks** of the (smoothed) map (``_peaks``) -- the
   front-near field and the rear-far ghost (which a 1D distance-collapse would let outvote the
   front, since the ghost is a broad rear arc).
2. **Characterises each peak** with a half-max-weighted **centroid** (``_peak_rf`` /
   ``_marginal_center``, a generalization of ``ego_reformat.preferred_from_map``): angle =
   circular centroid of the angular profile over the peak's distance band + width + vector
   strength ``R``; distance = centroid over that angle's run + width (the per-cell
   distance-tuning width, the headline quantity for a distance-coding project).
3. **Confirms the confound** (~180 deg apart, distances summing to ``box_cm``) and reports the
   **frontal member** as the RF (``rf_frontal``), the other as the ghost (``rf_ghost_*``). A
   single rear peak stays rear (``rf_frontal=False``); a ~90 deg left/right mirror pair, which a
   symmetric box can't resolve, is flagged (``rf_ambiguous``).

Peak-seeded centroids (not a global marginal) on purpose: many cells are **distance-sharp but
angle-broad**, and the broad rear ghost would drag a single global centroid off the front
peak; seeding each centroid at its own 2D peak keeps front and back separate and stable.

The half-max run uses a **baseline** floor (a low percentile of the profile) so "half-max"
is measured above the cell's background, making widths comparable across cells with
different background rates (raise ``baseline_pct`` toward the median to tighten the field
around the peak).

Composes a ``CellExplorer`` (loaded arrays + ``_ego_maps`` + drawing helpers) and,
optionally, a ``VisualClassifier`` to default the analysis to the ``is_visual`` subset --
the same composition pattern as ``CellSimulator`` / ``VisualClassifier``. No existing files
are modified.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cell_explorer import CellExplorer, COND_TO_LIGHTVAL
from ego_reformat import compute_ego_maps, preferred_from_map, _signed_angle_deg


class RFAnalyzer:
    """Robust egocentric RF center + size for the visual cells of one session."""

    def __init__(self, source, classifier=None, clip_deg=None, frac=0.5,
                 baseline_pct=10, ego_radius=0.5, min_occ=0, smooth=0.5,
                 box_cm=None, ang_tol=40.0, fov=(-120.0, 120.0), frames=None):
        """
        Parameters
        ----------
        source : CellExplorer | str
            Existing ``CellExplorer`` (reuses its loaded arrays) or a path to an
            ``ebr_pipeline_*.pkl``.
        classifier : VisualClassifier, optional
            A *classified* ``VisualClassifier`` (``classify()`` already run). When given,
            ``table`` / ``plot_population`` default to its ``is_visual`` subset and carry
            ``light_reliability``.
        clip_deg : float | None
            Symmetric shortcut: if given, sets ``fov = (-clip_deg, +clip_deg)`` (overrides the
            asymmetric ``fov`` default). Prefer ``fov`` for a hemisphere-specific visual field.
        frac : float
            Half-max cutoff as a fraction of the baseline->peak response (0.5 = FWHM) used to
            pick each axis's run around its peak.
        baseline_pct : float
            Percentile of the (clipped) profile taken as the baseline floor for the half-max
            cutoff and the centroid weights. Higher -> tighter field around the peak (and a
            narrower angular run, hence a tighter set of rays for the distance profile).
        ego_radius : float | None
            Max egocentric distance (meters) used to build the maps. Default ``0.5`` (50 cm).
            ``None`` keeps the session's ``settings['ego_radius']`` (35 cm, which *discards*
            every ray whose boundary is farther than that); a larger value rebuilds the maps out
            to that radius -- ``ray_distances`` already store the true distance, so this recovers
            the far bins instead of dropping them (bins past the arena's reach stay ``np.nan``).
            Affects ``rf`` / ``table`` / ``plot_cell`` / ``plot_population`` (but not
            ``plot_cell_full``, which draws the explorer's map).
        min_occ : int
            Occupancy floor: every ``(ray, distance)`` bin sampled by fewer than ``min_occ``
            frames is set to ``np.nan`` before the RF is estimated, so sparsely-visited far
            bins (where ``spk/occ`` explodes from one or two frames) can't hijack the center.
            ``0`` (default) disables it. Occupancy is the trajectory's dwell, so the same mask
            applies to every cell.
        smooth : float | tuple
            Gaussian smoothing SD **in bins** applied to the ego map before the RF is computed
            (and shown in the figures), after the occupancy floor. Scalar -> same SD on both
            axes; a 2-tuple is ``(angular, distance)``. The angular axis wraps at the front
            seam; the distance axis does not. NaN bins (unvisited / floored) are ignored in the
            convolution and kept NaN, so smoothing never invents data. Default ``0.5`` bins;
            pass ``0`` to disable.
        box_cm : float | None
            Expected near+far distance sum of a front/back confound pair (the box size along
            heading; ~53 cm here). When set, a peak pair counts as a confound only if its
            distances sum to within 30% of ``box_cm`` (else the stronger peak is kept). ``None``
            (default) => skip the sum check (the ~180 deg separation alone confirms the pair).
            The unresolvable left/right case is flagged separately via ``rf_ambiguous`` (the
            chosen member sits near +/-90 deg).
        ang_tol : float
            Tolerance (deg) on the ~180 deg separation when testing whether two peaks form a
            confound pair (default 40; observed separations are 167-177 deg).
        fov : (lo, hi) | None
            Field-of-view window (signed deg, front=0, right +, left -) used to pick the real RF
            from a confound pair and to set ``rf_frontal`` (= RF inside the window). Default
            ``(-120, 120)``. The RF search is full-circle; for a front/back pair the member
            **inside** the window is kept (the rear ghost, |angle| > 120, is dropped); a lateral
            pair (both inside) is resolved to the **right** one -- the priority side of a
            **left-hemisphere (right-FOV)** recording -- and flagged ``rf_ambiguous``. ``None``
            => whole circle; right-hemisphere data needs the mirror priority. ``rf_frontal=False``
            marks RFs outside the window, droppable via ``table(frontal_only=True)``.
        frames : array | None
            Boolean mask over df rows; when given, the ego maps are built from only those frames
            (intersected with the condition). Used to recompute the RF on a subset of the session
            -- e.g. each half, for the split-half stability check in
            ``CellExplorer.compare_rf(..., split_half=True)``. ``None`` (default) = all frames.
        """
        # detect by path-ness, not isinstance(CellExplorer), so a reloaded class can't break it
        ce = CellExplorer(source) if isinstance(source, (str, os.PathLike)) else source
        self.ce = ce
        self.settings = ce.settings
        self.session_id = ce.session_id
        self.n_cells = ce.n_cells
        self.clf = classifier

        self.n_rays = 360 // ce.settings["ego_rays_deg"]
        # optional full-extent override: the default ego map drops rays past ego_radius;
        # raising it rebuilds the maps and recovers those far bins (cached in _ego_override).
        self.ego_radius = ce.settings["ego_radius"] if ego_radius is None else float(ego_radius)
        self.min_occ = int(min_occ)
        sm = (smooth, smooth) if np.isscalar(smooth) else tuple(smooth)
        self.smooth_sigma = (float(sm[0]), float(sm[1]))         # (angular, distance) bins
        self.n_dist = round(self.ego_radius / ce.settings["ego_bin_size"])
        self.bin_cm = ce.settings["ego_bin_size"] * 100
        self._ego_cache = {}                                     # condition -> computed ego map

        self.clip_deg = clip_deg
        self.frac = frac
        self.baseline_pct = baseline_pct
        self.box_cm = box_cm
        self.ang_tol = float(ang_tol)
        # FOV window (signed deg, front=0, right+). Asymmetric & right-biased by default for a
        # left-hemisphere (right visual field) recording; clip_deg, if given, makes it symmetric.
        if clip_deg is not None:
            self.fov = (-float(clip_deg), float(clip_deg))
        elif fov is not None:
            self.fov = (float(fov[0]), float(fov[1]))
        else:
            self.fov = (-180.0, 180.0)
        self.frames = None if frames is None else np.asarray(frames, dtype=bool)

        # geometry (raw theta: 0 = front, increasing anti-clockwise; matches _draw_ego)
        self.theta = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        self.ray_signed = _signed_angle_deg(self.theta)          # signed deg, front=0 right+
        self.dist_cm = np.arange(self.n_dist) * self.bin_cm      # bin d -> d*bin_size (cm)

    # ------------------------------------------------------------ visual subset
    def _visual_cells(self):
        if getattr(self.clf, "_last", None) is None:
            raise ValueError(
                "no visual subset available -- pass a classified VisualClassifier, "
                "or call with an explicit `cells` list")
        df = self.clf._last["df"]
        return df.loc[df["is_visual"], "cell"].to_numpy()

    # ----------------------------------------------------------- ego map source
    def _ego(self, condition):
        """Per-cell ego maps for ``condition``, honoring the ``ego_radius`` / ``min_occ``
        overrides.

        With neither override this is exactly ``ce._ego_maps(condition)`` (the default map
        cached on the explorer). Otherwise the maps are recomputed once -- at the requested
        radius, with every bin below the occupancy floor blanked to ``np.nan`` -- and cached.
        """
        if (self.ego_radius == self.ce.settings["ego_radius"] and not self.min_occ
                and not any(self.smooth_sigma) and self.frames is None):
            return self.ce._ego_maps(condition)
        if condition not in self._ego_cache:
            s = dict(self.ce.settings)
            s["ego_radius"] = self.ego_radius
            ego, occ = compute_ego_maps(
                self.ce.df, s, COND_TO_LIGHTVAL[condition], return_occ=True,
                frame_mask=self.frames)
            if self.min_occ:
                ego = np.where(occ[None] >= self.min_occ, ego, np.nan)
            if any(self.smooth_sigma):
                ego = self._smooth_map(ego)
            self._ego_cache[condition] = ego
        return self._ego_cache[condition]

    def _smooth_map(self, ego):
        """NaN-aware Gaussian smooth of an ``(n_cells, n_rays, n_dist)`` stack: the angular
        axis is circular (wrap-padded so the front seam is continuous), the distance axis is
        not. NaNs are ignored in the convolution (normalized convolution) and restored
        afterwards, so smoothing never fills a hole that had no data."""
        from scipy.ndimage import gaussian_filter

        s_ang, s_dist = self.smooth_sigma
        pad = int(np.ceil(4.0 * s_ang)) + 1
        valid = np.isfinite(ego)
        filled = np.where(valid, ego, 0.0)

        def _f(a):                                           # smooth rays+dist, not cells
            ap = np.concatenate([a[:, -pad:, :], a, a[:, :pad, :]], axis=1)   # wrap angular
            ap = gaussian_filter(ap, sigma=(0.0, s_ang, s_dist), mode="nearest")
            return ap[:, pad:-pad, :]

        num, den = _f(filled), _f(valid.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[~valid] = np.nan                                 # keep original holes
        return out

    # --------------------------------------------------------------- core: rf
    def rf(self, cell, condition="light"):
        """Robust egocentric RF (center + size + field) for one cell.

        Returns a dict with the scalar metrics (see ``table``) plus the field ``mask`` and
        the polar marker coords (``ang_rad`` / ``r_norm``) used by ``plot_cell``.
        """
        ego = np.asarray(self._ego(condition)[cell], dtype=float)
        return self._rf_from_map(ego)

    def _rf_from_map(self, ego):
        """Locate the RF as the visible (in-FOV) member of the confound pair.

        The closed box makes a near-wall cell fire for the near wall in front AND its far-wall
        ghost ~180 deg behind. Detect the two strongest, opposite 2D peaks, characterise each
        with the half-max centroid, confirm they form a confound pair (~180 deg apart at
        complementary distances), and report the member that falls inside the ``fov`` window as
        the RF (the other as the ghost) -- so a lateral mirror pair resolves to the visible side.
        """
        out = self._empty_rf()
        ego = np.asarray(ego, dtype=float)
        ego_nan = np.where(np.isfinite(ego), ego, np.nan)
        fov_lo, fov_hi = self.fov

        peaks = self._peaks(ego_nan)                             # 2D local maxima, by height
        if not peaks:
            return out
        out["rf_global_angle_deg"] = float(self.ray_signed[peaks[0][0]])

        P1 = self._peak_rf(ego_nan, peaks[0][1])                 # strongest peak
        if P1 is None:
            return out
        a1 = P1["rf_angle_deg"]

        # opposite peak: strongest local max whose angle is >90 deg from P1 and which refines to
        # a genuinely distinct (>45 deg) centre (so a single-peak cell collapses back to P1).
        P2 = None
        for i, j, _h in peaks:
            if abs((self.ray_signed[i] - a1 + 180) % 360 - 180) > 90:
                cand = self._peak_rf(ego_nan, j)
                if cand is not None and abs((cand["rf_angle_deg"] - a1 + 180) % 360 - 180) > 45:
                    P2 = cand
                    break

        sel, ghost, ambiguous = P1, None, False
        if P2 is not None:
            a2 = P2["rf_angle_deg"]
            dang = abs((a1 - a2 + 180) % 360 - 180)
            sum_ok = (abs(P1["rf_dist_cm"] + P2["rf_dist_cm"] - self.box_cm) <= 0.3 * self.box_cm
                      if self.box_cm else True)
            if abs(dang - 180.0) <= self.ang_tol and sum_ok:     # confirmed confound pair
                in1, in2 = (fov_lo <= a1 <= fov_hi), (fov_lo <= a2 <= fov_hi)
                if in1 != in2:                       # exactly one visible (front/back) -> take it
                    sel = P1 if in1 else P2
                elif in1 and in2:                    # both visible (lateral pair) -> right side
                    sel = P1 if a1 >= a2 else P2     # priority (left hemisphere = right FOV)
                    ambiguous = True                 # resolved by the hemisphere prior, flagged
                else:                                # neither visible -> the more central
                    cen = 0.5 * (fov_lo + fov_hi)
                    sel = P1 if abs(a1 - cen) <= abs(a2 - cen) else P2
                ghost = P2 if sel is P1 else P1
            # else: not a confirmed confound pair -> keep the strongest peak (P1)

        out.update({k: sel[k] for k in (
            "rf_angle_deg", "rf_dist_cm", "rf_ang_sd_deg", "rf_dist_sd_cm",
            "rf_ang_span_deg", "rf_dist_span_cm", "rf_area_bins", "rf_peak", "rf_R",
            "mask", "ang_rad", "r_norm")})
        out["rf_frontal"] = bool(fov_lo <= sel["rf_angle_deg"] <= fov_hi)
        out["rf_ambiguous"] = ambiguous
        if ghost is not None:
            out["rf_ghost_angle_deg"] = ghost["rf_angle_deg"]
            out["rf_ghost_dist_cm"] = ghost["rf_dist_cm"]
        return out

    def _peaks(self, ego_nan):
        """2D local maxima of the (smoothed) ego map, above the baseline percentile, sorted by
        height; angular axis wraps, distance does not. Returns ``[(ray, dbin, height), ...]``."""
        from scipy.ndimage import maximum_filter
        if not np.isfinite(ego_nan).any():
            return []
        m = np.nan_to_num(ego_nan, nan=-np.inf)
        mx = maximum_filter(m, size=(9, 3), mode=("wrap", "nearest"))
        base = np.nanpercentile(ego_nan, self.baseline_pct)
        ii, jj = np.where((m == mx) & (ego_nan > base))
        order = np.argsort(-ego_nan[ii, jj])
        return [(int(ii[k]), int(jj[k]), float(ego_nan[ii[k], jj[k]])) for k in order]

    def _peak_rf(self, ego_nan, seed_dbin):
        """Half-max RF metrics for the peak seeded at distance bin ``seed_dbin``: angle =
        circular half-max centroid of the angular profile over a small band around that bin;
        distance = centroid over that angle's run. Full circle (no clip). Dict or ``None``."""
        band = slice(max(0, seed_dbin - 1), seed_dbin + 2)
        with np.errstate(invalid="ignore"):
            aprof = np.nanmean(ego_nan[:, band], axis=1)
        ang = self._marginal_center(aprof, self.theta, circular=True)
        if ang is None:
            return None
        pref_rad, R, ang_sd, ang_run, ang_span = ang
        with np.errstate(invalid="ignore"):
            dprof = np.nanmean(ego_nan[ang_run], axis=0)
        dist = self._marginal_center(dprof, self.dist_cm, circular=False)
        if dist is None:
            return None
        pref_dist, _, dist_sd, dist_run, dist_span = dist
        mask = np.zeros((self.n_rays, self.n_dist), bool)
        mask[np.ix_(np.nonzero(ang_run)[0], np.nonzero(dist_run)[0])] = True
        with np.errstate(invalid="ignore"):
            peak = float(np.nanmax(np.where(mask, ego_nan, np.nan)))
        return {
            "rf_angle_deg": float(_signed_angle_deg(pref_rad)), "rf_dist_cm": pref_dist,
            "rf_ang_sd_deg": ang_sd, "rf_dist_sd_cm": dist_sd,
            "rf_ang_span_deg": ang_span, "rf_dist_span_cm": dist_span,
            "rf_area_bins": int(mask.sum()), "rf_peak": peak, "rf_R": R,
            "mask": mask, "ang_rad": pref_rad,
            "r_norm": (pref_dist / self.bin_cm) / (self.n_dist - 1),
        }

    def _marginal_center(self, profile, coords, circular):
        """Half-max-weighted center + width of one 1D marginal profile.

        ``coords`` are raw radians (``circular=True``, angle) or cm (distance). Returns
        ``(center, R, sd, run_mask, span)`` -- ``center`` in the coord's units, ``R`` the
        circular vector strength (NaN for distance), ``sd`` the rate-weighted width
        (deg / cm), ``run_mask`` the half-max run around the peak -- or ``None`` if the
        profile has no response above baseline.
        """
        finite = np.isfinite(profile)
        if not finite.any():
            return None
        b = float(np.nanpercentile(profile, self.baseline_pct))
        peak_idx = int(np.nanargmax(np.where(finite, profile, -np.inf)))
        peak = float(profile[peak_idx])
        if not np.isfinite(peak) or peak <= b:
            return None
        thresh = b + self.frac * (peak - b)
        supra = finite & (profile >= thresh)
        run = self._run_around(supra, peak_idx, circular)

        w = np.clip(profile[run] - b, 0.0, None)
        W = float(w.sum())
        if W <= 0:
            return None
        c = coords[run]
        if circular:
            cos_s, sin_s = float(np.sum(w * np.cos(c))), float(np.sum(w * np.sin(c)))
            center = float(np.arctan2(sin_s, cos_s) % (2 * np.pi))
            R = float(np.hypot(cos_s, sin_s) / W)
            sd = float(np.degrees(np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))))
            # contiguous run -> arc width = (#rays - 1) * deg/ray (wrap-safe, unlike max-min)
            span = float((int(run.sum()) - 1) * 360.0 / len(run))
        else:
            center = float(np.sum(w * c) / W)
            R = np.nan
            sd = float(np.sqrt(np.sum(w * (c - center) ** 2) / W))
            span = float(c.max() - c.min())
        return center, R, sd, run, span

    @staticmethod
    def _run_around(supra, peak_idx, circular):
        """Contiguous ``True`` run of ``supra`` containing ``peak_idx`` (wrap-aware when
        ``circular``)."""
        n = len(supra)
        run = np.zeros(n, bool)
        run[peak_idx] = True
        for step in (1, -1):                       # expand right, then left
            i = peak_idx
            while True:
                j = (i + step) % n if circular else i + step
                if (circular and j == peak_idx) or (not circular and not 0 <= j < n):
                    break
                if not supra[j]:
                    break
                run[j] = True
                i = j
        return run

    def _empty_rf(self):
        return {
            "rf_angle_deg": np.nan, "rf_dist_cm": np.nan,
            "rf_ang_sd_deg": np.nan, "rf_dist_sd_cm": np.nan,
            "rf_ang_span_deg": np.nan, "rf_dist_span_cm": np.nan,
            "rf_area_bins": 0, "rf_peak": np.nan, "rf_R": np.nan,
            "rf_frontal": False, "rf_global_angle_deg": np.nan,
            "rf_ghost_angle_deg": np.nan, "rf_ghost_dist_cm": np.nan, "rf_ambiguous": False,
            "mask": np.zeros((self.n_rays, self.n_dist), bool),
            "ang_rad": np.nan, "r_norm": np.nan,
        }

    def with_frames(self, frames):
        """A sibling analyzer with this one's config but built from only ``frames`` (boolean
        mask over df rows) -- used for the split-half stability metric."""
        return RFAnalyzer(self.ce, classifier=self.clf, frac=self.frac,
                          baseline_pct=self.baseline_pct, ego_radius=self.ego_radius,
                          min_occ=self.min_occ, smooth=self.smooth_sigma, box_cm=self.box_cm,
                          ang_tol=self.ang_tol, fov=self.fov, frames=frames)

    def _half_masks(self, condition):
        """Boolean masks for the two chronological halves of the condition's frames."""
        idx = np.where(self.ce.light_col == COND_TO_LIGHTVAL[condition])[0]
        F = len(self.ce.df)
        h1 = np.zeros(F, bool); h1[idx[:len(idx) // 2]] = True
        h2 = np.zeros(F, bool); h2[idx[len(idx) // 2:]] = True
        return h1, h2

    # ------------------------------------------------------------------ table
    def table(self, cells=None, condition="light", frontal_only=False, stability=False):
        """Per-cell RF metrics (one row per visual cell by default).

        ``cells=None`` uses the classifier's ``is_visual`` subset; pass a list to override.
        ``argmax_angle_deg`` / ``argmax_dist_cm`` are the ``preferred_from_map`` single-bin
        estimates, for comparison with the robust centroid. ``rf_global_angle_deg`` is the
        full-circle peak angle and ``rf_frontal`` whether the clipped RF is trustworthy (see
        ``_rf_from_map``); ``frontal_only=True`` drops the rows with ``rf_frontal == False``.
        ``stability=True`` adds ``rf_stability`` = the RF-angle shift (deg) between the two
        chronological halves -- small = a well-localized, reliable RF; large = no stable single
        location (recompute the RF on each half, so ~3x the cost).
        """
        if cells is None:
            cells = self._visual_cells()
        cells = np.atleast_1d(cells)
        cdf = self.clf._last["df"] if getattr(self.clf, "_last", None) is not None else None
        if stability:
            h1, h2 = self._half_masks(condition)
            ah1, ah2 = self.with_frames(h1), self.with_frames(h2)

        keys = ("rf_angle_deg", "rf_dist_cm", "rf_ang_sd_deg", "rf_dist_sd_cm",
                "rf_ang_span_deg", "rf_dist_span_cm", "rf_area_bins", "rf_peak", "rf_R",
                "rf_frontal", "rf_global_angle_deg", "rf_ghost_angle_deg", "rf_ghost_dist_cm",
                "rf_ambiguous")
        rows = []
        for c in cells:
            c = int(c)
            ego = np.asarray(self._ego(condition)[c], dtype=float)
            d = self._rf_from_map(ego)
            am = preferred_from_map(ego, self.settings, self.clip_deg)  # argmax reference
            row = {"cell": c}
            if cdf is not None:
                row["light_reliability"] = float(
                    cdf.loc[cdf["cell"] == c, "light_reliability"].iloc[0])
            row.update({k: d[k] for k in keys})
            row["argmax_angle_deg"] = am["ang_signed"]
            row["argmax_dist_cm"] = am["dist_cm"]
            if stability:
                a1 = ah1.rf(c, condition)["rf_angle_deg"]
                a2 = ah2.rf(c, condition)["rf_angle_deg"]
                row["rf_stability"] = (abs((a1 - a2 + 180) % 360 - 180)
                                       if np.isfinite(a1) and np.isfinite(a2) else np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        if frontal_only and "rf_frontal" in df:
            df = df[df["rf_frontal"]].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ plots
    @staticmethod
    def _signed_to_raw(signed_deg):
        """Signed degrees (front=0, right+) -> raw polar radians used by ``_draw_ego``."""
        return np.radians((-np.asarray(signed_deg, dtype=float)) % 360)

    def _outline(self, ax, mask, color="red", lw=1.6):
        """Outline a field mask on a polar ego axis (ray seam closed so frontal fields don't
        break at front)."""
        if not mask.any():
            return
        theta_w = np.concatenate([self.theta, [2 * np.pi]])
        r = np.linspace(0, 1, self.n_dist)
        mask_w = np.vstack([mask, mask[:1]]).astype(float)
        TH, R = np.meshgrid(theta_w, r)
        ax.contour(TH, R, mask_w.T, levels=[0.5], colors=color, linewidths=lw, zorder=4)

    def plot_cell(self, cell, condition="light", ego_cmap="Blues", hd_cmap="hsv",
                  figsize=(20, 4.8)):
        """trajectory + spikes (colored by head direction) | polar ego map (RF center + field
        outline + FOV bounds + ghost) | angular profile | distance profile, centroid +/- SD
        marked on each. Returns the figure."""
        ego = np.asarray(self._ego(condition)[cell], dtype=float)
        d = self._rf_from_map(ego)
        finite = ego[np.isfinite(ego)]
        vmax = float(np.percentile(finite, 99)) if finite.size else None

        fig = plt.figure(figsize=figsize)
        axT = fig.add_subplot(1, 4, 1)
        axP = fig.add_subplot(1, 4, 2, projection="polar")
        axA = fig.add_subplot(1, 4, 3)
        axD = fig.add_subplot(1, 4, 4)

        # trajectory + spikes colored by head direction (spike rate map)
        all_mag = self.ce.spk_mag[self.ce.spk[:, cell], cell]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0
        sc, n_spk = self.ce._draw_traj(axT, cell, condition, hd_cmap, 15, size_ref)
        axT.set_title(f"{condition} traj ({n_spk} spikes)")
        fig.colorbar(sc, ax=axT, fraction=0.046, pad=0.04).set_label("head dir (deg)")

        CellExplorer._draw_ego(axP, ego, ego_cmap, f"{condition} ego + RF", vmin=0,
                               vmax=vmax, pref=(d["ang_rad"], d["r_norm"]),
                               clip_deg=None)            # full circle so the ghost shows
        self._outline(axP, d["mask"])
        for b in self.fov:                               # FOV boundary radial lines (full map shown)
            tb = self._signed_to_raw(b)
            axP.plot([tb, tb], [0, 1], color="dimgray", ls="--", lw=1.0, zorder=3)
        if np.isfinite(d["rf_ghost_angle_deg"]):         # ghost = rejected confound member
            gr = (d["rf_ghost_dist_cm"] / self.bin_cm) / (self.n_dist - 1)
            axP.plot(self._signed_to_raw(d["rf_ghost_angle_deg"]), gr, marker="x",
                     color="gray", markersize=9, markeredgewidth=2, zorder=5)

        # angular profile (collapsed over distance) from the ego in scope so it honors an
        # ego_radius override; identical to ce._angular_profile when there is none. centroid +/- SD
        if (self.ego_radius == self.ce.settings["ego_radius"] and not self.min_occ
                and not any(self.smooth_sigma)):
            ang, mean, sem = self.ce._angular_profile(cell, condition)
        else:
            cnt = np.sum(np.isfinite(ego), axis=1)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(ego, axis=1)
                sem = np.nanstd(ego, axis=1) / np.sqrt(cnt)
            order = np.argsort(self.ray_signed)
            ang, mean, sem = self.ray_signed[order], mean[order], sem[order]
        axA.plot(ang, mean, color="goldenrod")
        axA.fill_between(ang, mean - sem, mean + sem, color="goldenrod", alpha=0.25)
        if np.isfinite(d["rf_angle_deg"]):
            axA.axvline(d["rf_angle_deg"], color="red", lw=1.5)
            axA.axvspan(d["rf_angle_deg"] - d["rf_ang_sd_deg"],
                        d["rf_angle_deg"] + d["rf_ang_sd_deg"], color="red", alpha=0.12)
        if np.isfinite(d["rf_ghost_angle_deg"]):
            axA.axvline(d["rf_ghost_angle_deg"], color="gray", ls="--", lw=1.2)   # ghost
        if self.fov != (-180.0, 180.0):                                          # FOV bounds
            for b in self.fov:
                axA.axvline(b, color="lightgray", lw=0.8)
        axA.set_xlim(-180, 180)
        axA.set_xlabel("egocentric angle (deg)"); axA.set_ylabel("mean response")
        axA.set_title("angular profile")

        # distance profile collapsed over the angular run (the same rays the center uses)
        rows = np.unique(np.nonzero(d["mask"])[0])
        if rows.size:
            block = ego[rows]
            with np.errstate(invalid="ignore"):
                dmean = np.nanmean(block, axis=0)
                dsem = np.nanstd(block, axis=0) / np.sqrt(np.sum(np.isfinite(block), axis=0))
        else:
            dmean = np.full(self.n_dist, np.nan); dsem = dmean
        pref_dbin = int(round(d["r_norm"] * (self.n_dist - 1))) if np.isfinite(d["r_norm"]) else 0
        self.ce._draw_distance(axD, dmean, np.nan_to_num(dsem), "goldenrod", pref_dbin,
                               title="distance profile")
        if np.isfinite(d["rf_dist_cm"]):
            axD.axvspan(max(0.0, d["rf_dist_cm"] - d["rf_dist_sd_cm"]),
                        d["rf_dist_cm"] + d["rf_dist_sd_cm"], color="red", alpha=0.12)

        tag = self._visual_tag(cell)
        flags = f" | frontal {'Y' if d['rf_frontal'] else 'N'}"
        if np.isfinite(d["rf_ghost_angle_deg"]):
            flags += f" | ghost {d['rf_ghost_angle_deg']:.0f} deg/{d['rf_ghost_dist_cm']:.0f} cm"
        if d["rf_ambiguous"]:
            flags += " | AMBIGUOUS"
        fig.suptitle(
            f"{self.session_id} - cell {cell}{tag}   "
            f"RF {d['rf_angle_deg']:.0f}+/-{d['rf_ang_sd_deg']:.0f} deg | "
            f"{d['rf_dist_cm']:.1f}+/-{d['rf_dist_sd_cm']:.1f} cm | R {d['rf_R']:.2f}{flags}")
        fig.tight_layout()
        return fig

    def plot_cell_figure(self, cell, condition="light", ego_cmap="viridis", hd_cmap="hsv",
                         s=15, figsize=(13, 4), angle_clip=120, normalize=True,
                         show_ghost=True, show_fov=True, show_center=False,
                         rf_color="lightgray",
                         titles=("spike map", "egocentric map", "distance profile")):
        """Figure-grade single-cell triptych with the robust RF drawn on, as in ``compare_rf``
        / ``RFAnalyzer.plot_cell``: spike map (HD-coloured, with a circular colorbar) |
        egocentric map (half-max RF field outline; centre marker off unless ``show_center``;
        clipped to the front +/- ``angle_clip`` deg) | distance profile over the RF's angular
        run (RF distance +/- SD
        shaded), for one ``condition`` ('light'/'dark'/'general').

        With ``normalize`` (default) the ego rate is divided by its peak over the visible
        (+/- ``angle_clip``) wedge, so the egocentric map and the distance profile share a 0..1
        normalized scale. ``angle_clip`` (default 120 deg; ``None`` = full circle) hides the
        rear of the map. Drawn on this analyzer's own ego map, so any ``ego_radius`` /
        ``min_occ`` / ``smooth`` / ``frames`` override is honoured and the RF stays aligned with
        it (the reason this lives here, not on ``CellExplorer.plot_cell_figure``). Returns the
        figure.
        """
        ce = self.ce
        ego = np.asarray(self._ego(condition)[cell], dtype=float)
        d = self._rf_from_map(ego)                       # RF is invariant to the global rescale
        finite_rf = bool(np.isfinite(d["rf_angle_deg"]))
        color = ce.COND_COLORS.get(condition)

        # normalize the ego rate by its peak over the visible (front +/- angle_clip) wedge, so
        # the egocentric map and the distance profile (built from it) share a 0..1 scale
        vis = (np.abs(self.ray_signed) <= angle_clip if angle_clip is not None
               else np.ones(self.n_rays, bool))
        with np.errstate(invalid="ignore"):
            peak = float(np.nanmax(ego[vis])) if np.isfinite(ego[vis]).any() else np.nan
        if normalize and np.isfinite(peak) and peak > 0:
            ego_disp, vmax, rate_label = ego / peak, 1.0, "normalized rate"
        else:
            finite = ego[np.isfinite(ego)]
            ego_disp = ego
            vmax = float(np.percentile(finite, 99)) if finite.size else None
            rate_label = "ego rate"

        fig = plt.figure(figsize=figsize)
        ax_spk = fig.add_subplot(1, 3, 1)
        ax_ego = fig.add_subplot(1, 3, 2, projection="polar")
        ax_dist = fig.add_subplot(1, 3, 3)

        # spike map: trajectory + spikes coloured by head direction, with a circular colorbar
        all_mag = ce.spk_mag[ce.spk[:, cell], cell]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0
        sc, n_spk = ce._draw_traj(ax_spk, cell, condition, hd_cmap, s, size_ref,
                                  frame_mask=self.frames)
        ax_spk.set_title(f"{titles[0]} ({n_spk} spikes)")
        if ce._hd_colorwheel(ax_spk, hd_cmap=hd_cmap) is None:       # mpl too old for a polar inset
            fig.colorbar(sc, ax=ax_spk, fraction=0.046, pad=0.04).set_label(
                "head direction (deg)")

        # ego map + half-max RF field outline (centre marker only if show_center), clipped
        mesh = CellExplorer._draw_ego(
            ax_ego, ego_disp, ego_cmap, titles[1], vmin=0, vmax=vmax,
            pref=((d["ang_rad"], d["r_norm"]) if (finite_rf and show_center) else None),
            clip_deg=angle_clip, pref_color=rf_color)
        if finite_rf:
            self._outline(ax_ego, d["mask"], color=rf_color)
        if show_fov:                                     # FOV bounds, only where inside the wedge
            for b in self.fov:
                if angle_clip is None or abs(b) < angle_clip:
                    tb = self._signed_to_raw(b)
                    ax_ego.plot([tb, tb], [0, 1], color="dimgray", ls="--", lw=1.0, zorder=3)
        if (show_ghost and np.isfinite(d["rf_ghost_angle_deg"])
                and (angle_clip is None or abs(d["rf_ghost_angle_deg"]) <= angle_clip)):
            gr = (d["rf_ghost_dist_cm"] / self.bin_cm) / (self.n_dist - 1)
            ax_ego.plot(self._signed_to_raw(d["rf_ghost_angle_deg"]), gr, marker="x",
                        color="gray", markersize=9, markeredgewidth=2, zorder=5)
        cb2 = fig.colorbar(mesh, ax=ax_ego, fraction=0.046, pad=0.08)
        cb2.set_label(rate_label)
        cb2.outline.set_visible(False)

        # distance profile over the RF's angular run (same rays + normalization as the map)
        rows = np.unique(np.nonzero(d["mask"])[0])
        if rows.size:
            block = ego_disp[rows]
            with np.errstate(invalid="ignore"):
                dmean = np.nanmean(block, axis=0)
                dsem = np.nanstd(block, axis=0) / np.sqrt(np.sum(np.isfinite(block), axis=0))
        else:
            dmean = np.full(self.n_dist, np.nan)
            dsem = dmean
        pref_dbin = int(round(d["r_norm"] * (self.n_dist - 1))) if finite_rf else 0
        ce._draw_distance(ax_dist, dmean, np.nan_to_num(dsem), color, pref_dbin,
                          title=titles[2])
        ax_dist.set_ylabel(rate_label)
        if finite_rf:
            ax_dist.axvspan(max(0.0, d["rf_dist_cm"] - d["rf_dist_sd_cm"]),
                            d["rf_dist_cm"] + d["rf_dist_sd_cm"], color=rf_color, alpha=0.35)
        ce._despine(ax_dist)

        if finite_rf:
            sub = (f"RF {d['rf_angle_deg']:.0f}+/-{d['rf_ang_sd_deg']:.0f} deg | "
                   f"{d['rf_dist_cm']:.1f}+/-{d['rf_dist_sd_cm']:.1f} cm | R {d['rf_R']:.2f}")
            if d["rf_ambiguous"]:
                sub += " | AMBIGUOUS"
        else:
            sub = "no RF"
        fig.suptitle(f"{self.session_id} - cell {cell}  ({condition})  {sub}")
        fig.tight_layout()
        return fig

    def plot_cell_full(self, cell, conditions=("light",), **kw):
        """The full ``CellExplorer.plot_cell`` page (trajectory | ego | angular | distance,
        light & dark) with the robust RF drawn into the listed ``conditions`` (default light)
        in place of the argmax markers. Full-circle (``angle_clip=None``) so rear RFs show.
        Returns the matplotlib figure."""
        if (self.ego_radius != self.ce.settings["ego_radius"] or self.min_occ
                or any(self.smooth_sigma)):
            raise ValueError(
                "plot_cell_full draws CellExplorer.plot_cell, which uses the session's raw, "
                f"unsmoothed {self.ce.settings['ego_radius'] * 100:.0f} cm map; an ego_radius "
                f"({self.ego_radius * 100:.0f} cm), min_occ ({self.min_occ}) or smooth "
                f"({self.smooth_sigma}) override would mismatch the overlaid RF. Use "
                "RFAnalyzer.plot_cell (self-contained) instead.")
        rf = {c: self.rf(cell, c) for c in conditions}
        return self.ce.plot_cell(cell, angle_clip=None, rf=rf, **kw)

    def _visual_tag(self, cell):
        if getattr(self.clf, "_last", None) is None:
            return ""
        cdf = self.clf._last["df"]
        vis = bool(cdf.loc[cdf["cell"] == cell, "is_visual"].iloc[0])
        return "  [VISUAL]" if vis else "  [not visual]"

    def plot_population(self, df=None, condition="light", figsize=(15, 5), cmap="viridis"):
        """RF centers across visual cells: polar scatter (color=R, size=reliability) |
        preferred-angle rose | preferred-distance histogram. Returns the figure."""
        if df is None:
            df = self.table(condition=condition)
        ok = np.isfinite(df["rf_angle_deg"]) & np.isfinite(df["rf_dist_cm"])
        sub = df[ok]

        fig = plt.figure(figsize=figsize)
        axP = fig.add_subplot(1, 3, 1, projection="polar")
        axA = fig.add_subplot(1, 3, 2)
        axD = fig.add_subplot(1, 3, 3)

        ang_rad = self._signed_to_raw(sub["rf_angle_deg"].to_numpy())
        r_norm = (sub["rf_dist_cm"].to_numpy() / self.bin_cm) / (self.n_dist - 1)
        size = (20 + 130 * np.clip(sub["light_reliability"].to_numpy(), 0, 1)
                if "light_reliability" in sub else 50)
        sc = axP.scatter(ang_rad, r_norm, c=sub["rf_R"].to_numpy(), s=size, cmap=cmap,
                         vmin=0, vmax=1, alpha=0.8, edgecolors="k", linewidths=0.3)
        axP.set_theta_zero_location("N")               # full circle, no clip (show every cell)
        for b in self.fov:                             # FOV bounds for reference only
            tb = self._signed_to_raw(b)
            axP.plot([tb, tb], [0, 1], color="dimgray", ls="--", lw=1.0)
        axP.set_yticklabels([])
        axP.set_title(f"RF centers (n={len(sub)})")
        fig.colorbar(sc, ax=axP, fraction=0.046, pad=0.10, label="vector strength R")

        axA.hist(sub["rf_angle_deg"], bins=np.arange(-180, 181, 15),
                 color="goldenrod", edgecolor="k")
        axA.axvline(0, color="gray", ls="--", lw=1)
        for b in self.fov:                             # FOV bounds for reference only
            axA.axvline(b, color="dimgray", ls="--", lw=1)
        axA.set_xlabel("RF angle (deg)"); axA.set_ylabel("cells")
        axA.set_title("preferred angle")

        axD.hist(sub["rf_dist_cm"],
                 bins=np.arange(0, self.n_dist * self.bin_cm + 1, self.bin_cm),
                 color="steelblue", edgecolor="k")
        axD.set_xlabel("RF distance (cm)"); axD.set_ylabel("cells")
        axD.set_title("preferred distance")

        fig.suptitle(f"{self.session_id} - visual-cell RF locations")
        fig.tight_layout()
        return fig
