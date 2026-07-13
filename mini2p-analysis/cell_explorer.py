"""Explore individual cells from a session ``ebr_pipeline_*.pkl``.

``CellExplorer`` loads one session pickle and produces per-cell figures. Each method
builds and returns a matplotlib figure (nothing is cached/saved). More plot types will
be added over time.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt

from ego_reformat import (compute_ego_maps, preferred_from_map, SessionReformatter,
                          _signed_angle_deg)

COND_TO_LIGHTVAL = SessionReformatter.COND_TO_LIGHTVAL  # {'light': 0, 'dark': 1}


class CellExplorer:
    """Per-cell visualizations for a single session recording."""

    COND_COLORS = {"light": "goldenrod", "dark": "navy", "general": "gray"}

    def __init__(self, ebr_pkl_path):
        with open(ebr_pkl_path, "rb") as f:
            ebr = pickle.load(f)
        self.settings = ebr["settings"]
        self.session_id = self.settings["rec_name"]
        self.df = ebr["df"]
        self.ego_norm_general = ebr["ego_norm"]  # combined map, kept for reference
        self.ebcc = {
            "light": ebr["ebcc_outputs_light"],
            "dark": ebr["ebcc_outputs_dark"],
        }

        # cached per-frame arrays
        self.spk = np.stack(self.df["spk_bool"].values)  # (F, n_cells) bool
        self.spk_mag = np.stack(  # (F, n_cells) thresholded spike rate (magnitude)
            [np.asarray(v, dtype=float) for v in self.df["spk_rate_thr"].values]
        )
        self.x = self.df["x"].to_numpy()
        self.y = self.df["y"].to_numpy()
        self.hd = self.df["hd_degrees"].to_numpy()
        self.light_col = self.df["light"].to_numpy()
        self.n_cells = self.settings["nr_cells"]

        self._ego_cache = {}  # condition -> (n_cells, n_rays, n_dist) data cache

    # ------------------------------------------------------------------ helpers
    def _ego_maps(self, condition):
        """Per-condition ego maps (recomputed once and cached)."""
        if condition not in self._ego_cache:
            self._ego_cache[condition] = compute_ego_maps(
                self.df, self.settings, COND_TO_LIGHTVAL[condition]
            )
        return self._ego_cache[condition]

    def _ego_for(self, condition, cell_id):
        """Ego field for a cell: 'light'/'dark' (recomputed) or 'general' (stored)."""
        if condition == "general":
            return self.ego_norm_general[cell_id]
        return self._ego_maps(condition)[cell_id]

    def _cell_stats(self, condition, cell_id):
        """(is_ebc, pref_angle_deg, pref_dist) for a cell in one condition."""
        out = self.ebcc[condition]
        is_ebc = bool(out["ebcs_mask"][cell_id])
        angle = _signed_angle_deg(out["mras_all_cells"][cell_id])  # signed degrees
        dist = float(out["pref_dists_all_cells"][cell_id])
        return is_ebc, angle, dist

    def _draw_traj(self, ax, cell_id, condition, hd_cmap, s, size_ref, frame_mask=None):
        """Trajectory (gray) with this condition's spikes colored by head direction; restricted
        to ``frame_mask`` (boolean over frames) when given -- e.g. one half of the session.

        Marker size scales with spike magnitude (``spk_rate_thr``), normalized by the
        shared ``size_ref`` so larger spikes appear as larger dots.
        """
        fm = None if frame_mask is None else np.asarray(frame_mask, dtype=bool)
        x, y = (self.x, self.y) if fm is None else (self.x[fm], self.y[fm])
        ax.plot(x, y, color="lightgray", lw=0.5, zorder=1)
        mask = self.spk[:, cell_id] & (self.light_col == COND_TO_LIGHTVAL[condition])
        if fm is not None:
            mask = mask & fm
        mag = self.spk_mag[mask, cell_id]
        sizes = s * np.clip(mag / size_ref, 0.1, 4.0)  # area scales with magnitude
        sc = ax.scatter(self.x[mask], self.y[mask], c=self.hd[mask], cmap=hd_cmap,
                        vmin=0, vmax=360, s=sizes, zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return sc, int(mask.sum())

    @staticmethod
    def _draw_ego(ax, ego, cmap, title, vmin=None, vmax=None, pref=None, clip_deg=None,
                  pref_color="red"):
        """Polar pcolormesh of a (n_rays, n_dist) ego field (0 deg up = in front).

        ``pref=(angle_rad, r_norm)`` draws an open circle (``pref_color``) at the preferred
        angle/distance.
        ``clip_deg`` restricts the visible wedge to +/- that many degrees of front (raw
        theta +/-clip, which equals the front +/-clip signed wedge). Returns the
        ``QuadMesh`` so callers can attach a colorbar.
        """
        n_rays, n_dist = ego.shape
        theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        r = np.linspace(0, 1, n_dist)
        theta_grid, r_grid = np.meshgrid(theta, r)
        mesh = ax.pcolormesh(theta_grid, r_grid, ego.T, cmap=cmap, vmin=vmin, vmax=vmax)
        if pref is not None and np.isfinite(pref[0]) and np.isfinite(pref[1]):
            ax.plot(pref[0], pref[1], marker="o", markersize=12, markeredgecolor=pref_color,
                    markerfacecolor="none", markeredgewidth=2, zorder=5)
        ax.set_theta_zero_location("N")
        if clip_deg is not None:
            ax.set_thetamin(-clip_deg)
            ax.set_thetamax(clip_deg)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(title)
        return mesh

    @staticmethod
    def _draw_field_outline(ax, mask, color="red", lw=1.6):
        """Overlay a half-max RF field outline on a polar ego axis; the ray seam at front is
        closed so a frontal field's contour doesn't break."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return
        n_rays, n_dist = mask.shape
        theta = np.concatenate([np.linspace(0, 2 * np.pi, n_rays, endpoint=False), [2 * np.pi]])
        r = np.linspace(0, 1, n_dist)
        mask_w = np.vstack([mask, mask[:1]]).astype(float)
        TH, R = np.meshgrid(theta, r)
        ax.contour(TH, R, mask_w.T, levels=[0.5], colors=color, linewidths=lw, zorder=4)

    def _angular_profile(self, cell_id, condition):
        """Return (angle_deg, mean, sem) of the response collapsed over distance.

        Angles are the signed convention (front=0, right +, left -), sorted -180..180.
        """
        ego = self._ego_for(condition, cell_id)          # (n_rays, n_dist)
        mean = np.nanmean(ego, axis=1)                    # collapse over distance
        cnt = np.sum(np.isfinite(ego), axis=1)
        with np.errstate(invalid="ignore"):
            sem = np.nanstd(ego, axis=1) / np.sqrt(cnt)
        ang = _signed_angle_deg(np.linspace(0, 2 * np.pi, ego.shape[0], endpoint=False))
        order = np.argsort(ang)
        return ang[order], mean[order], sem[order]

    def _preferred(self, cell_id, condition, clip_deg=None, window_deg=5):
        """Preferred angle/distance from the reconstructed map (see
        ``ego_reformat.preferred_from_map``). Same computation used to build the df, so
        the plot and the table agree."""
        ego = self._ego_for(condition, cell_id)
        return preferred_from_map(ego, self.settings, clip_deg, window_deg)

    def _draw_angular(self, ax, cell_id, condition, ylim=None, clip_deg=None, peak=None):
        ang, mean, sem = self._angular_profile(cell_id, condition)
        color = self.COND_COLORS.get(condition, None)
        ax.plot(ang, mean, color=color)
        ax.fill_between(ang, mean - sem, mean + sem, color=color, alpha=0.25)
        ax.axvline(0, color="lightgray", linestyle="--", lw=1)  # front
        if peak is not None and np.isfinite(peak[0]):
            ax.plot(peak[0], peak[1], "o", color=color, markersize=8,
                    markeredgecolor="black", zorder=5)  # preferred angle
        lim = clip_deg if clip_deg is not None else 180
        ax.set_xlim(-lim, lim)
        ax.set_xticks(np.arange(-lim, lim + 1, 60))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("egocentric angle (deg)")
        ax.set_ylabel("mean response")
        ax.set_title(f"{condition} angular response")

    def _draw_distance(self, ax, dist_mean, dist_sem, color, pref_dbin, ylim=None, title=""):
        """Response vs distance to boundary (cm), mean +/- SEM over the angle window."""
        bin_size_cm = self.settings["ego_bin_size"] * 100
        x = np.arange(len(dist_mean)) * bin_size_cm
        ax.plot(x, dist_mean, color=color)
        ax.fill_between(x, dist_mean - dist_sem, dist_mean + dist_sem,
                        color=color, alpha=0.25)
        if np.isfinite(dist_mean[pref_dbin]):
            ax.plot(x[pref_dbin], dist_mean[pref_dbin], "o", color=color, markersize=8,
                    markeredgecolor="black", zorder=5)  # preferred distance
            ax.axvline(x[pref_dbin], color="lightgray", linestyle="--", lw=1)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("distance to boundary (cm)")
        ax.set_ylabel("response")
        ax.set_title(title)

    @staticmethod
    def _despine(ax):
        """Hide the top/right spines for a clean, publication-style axis."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
    def _hd_colorwheel(parent_ax, hd_cmap="hsv", size=0.26, loc=(1.05, 0.04),
                       label="HD (deg)"):
        """Add a circular colorbar (hue wheel) for head direction just outside
        ``parent_ax`` -- the right legend for the angular HD variable the spike map encodes
        with ``hd_cmap``. ``loc`` is the inset's lower-left corner in parent-axes fractions, so
        the default (x>1) sits in the whitespace to the right of the (equal-aspect) box rather
        than over the trajectory. Returns the inset polar axes, or ``None`` if this matplotlib
        can't inset a polar axes (so the caller can fall back to a linear colorbar)."""
        try:
            cax = parent_ax.inset_axes([loc[0], loc[1], size, size], projection="polar")
        except Exception:
            return None
        theta = np.linspace(0, 2 * np.pi, 181)
        c = np.degrees(theta[:-1])[None, :]              # hue = direction in degrees (0..360)
        cax.pcolormesh(theta, [0.6, 1.0], c, cmap=hd_cmap, vmin=0, vmax=360, shading="flat")
        cax.set_theta_zero_location("N")
        cax.set_theta_direction(-1)                      # clockwise, compass-style
        cax.set_ylim(0, 1)
        cax.set_yticks([])
        cax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        cax.set_xticklabels(["0", "90", "180", "270"], fontsize=7)
        cax.tick_params(pad=-3)
        cax.grid(False)
        cax.spines["polar"].set_visible(False)
        if label:
            cax.set_title(label, fontsize=8, pad=2)
        return cax

    # -------------------------------------------------------------------- plots
    def plot_cell(self, cell_id, hd_cmap="hsv", ego_cmap="Blues", s=15, figsize=(18, 8),
                  angle_clip=None, window_deg=5, rf=None):
        """2x4 figure: rows = light (top) / dark (bottom); columns =
        trajectory+spikes (colored by head direction) | egocentric boundary field |
        response vs angle (collapsed over distance, mean +/- SEM) | response vs distance
        (cm) at the preferred angle, mean +/- SEM over +/- window_deg around it.

        Preferred angle = peak of the angular-response curve; preferred distance = peak
        of the windowed distance curve. These (not the stored pipeline stats) drive the
        red circle on the ratemap and the titles. ``angle_clip`` (deg) restricts the ego
        wedge and the angular x-axis to the front +/- that many degrees; default ``None``
        shows the **whole circle** (like ``compare_rf`` / ``RFAnalyzer.plot_cell``), so a
        rear field stays visible. ``window_deg`` (default 5) is the half-width of the
        angular window used for the distance curve's mean/SEM.

        ``rf`` (optional) is a ``{condition: RFAnalyzer.rf(...) dict}`` mapping; for each
        condition present, that row is drawn from the robust RF -- center marker + half-max
        field outline on the ego map, and the RF center +/- SD on the angular/distance panels
        (the distance curve over the RF's angular run) -- instead of the argmax preferred.
        Pass ``angle_clip=None`` alongside it to see full-circle / rear RFs. Returns the
        matplotlib figure.
        """
        ego = {c: self._ego_maps(c)[cell_id] for c in ("light", "dark")}
        prefs = {c: self._preferred(cell_id, c, angle_clip, window_deg)
                 for c in ("light", "dark")}

        # rays kept within +/- angle_clip of front (signed convention)
        n_rays = next(iter(ego.values())).shape[0]
        ray_signed = _signed_angle_deg(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
        keep = (np.abs(ray_signed) <= angle_clip) if angle_clip is not None else np.ones(n_rays, bool)

        # shared color scale across both maps (5th-95th pct of kept finite values)
        both = np.concatenate([m[keep][np.isfinite(m[keep])] for m in ego.values()])
        vmin, vmax = (np.percentile(both, [5, 95]) if both.size else (None, None))

        # shared y-axis for the two angular panels (over kept angles, comparable l/d)
        profiles = {c: self._angular_profile(cell_id, c) for c in ("light", "dark")}
        kept_band = (np.abs(ray_signed[np.argsort(ray_signed)]) <= angle_clip
                     if angle_clip is not None else np.ones(n_rays, bool))
        lo = min(np.nanmin((m - e)[kept_band]) for _, m, e in profiles.values())
        hi = max(np.nanmax((m + e)[kept_band]) for _, m, e in profiles.values())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ang_ylim = (lo - pad, hi + pad)

        # shared y-axis for the two distance panels (mean +/- SEM bands)
        dbands = []
        for p in prefs.values():
            dbands.append(p["dist_mean"] - p["dist_sem"])
            dbands.append(p["dist_mean"] + p["dist_sem"])
        dvals = np.concatenate([a[np.isfinite(a)] for a in dbands])
        if dvals.size:
            dlo, dhi = dvals.min(), dvals.max()
            dpad = 0.05 * (dhi - dlo) if dhi > dlo else 1.0
            dist_ylim = (dlo - dpad, dhi + dpad)
        else:
            dist_ylim = None

        # shared spike-size reference: 95th pct magnitude over all of this cell's spikes
        all_mag = self.spk_mag[self.spk[:, cell_id], cell_id]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0

        fig = plt.figure(figsize=figsize)
        for row, condition in enumerate(("light", "dark")):
            ax_traj = fig.add_subplot(2, 4, 4 * row + 1)
            ax_ego = fig.add_subplot(2, 4, 4 * row + 2, projection="polar")
            ax_ang = fig.add_subplot(2, 4, 4 * row + 3)
            ax_dist = fig.add_subplot(2, 4, 4 * row + 4)
            p = prefs[condition]

            sc, n_spk = self._draw_traj(ax_traj, cell_id, condition, hd_cmap, s, size_ref)
            ax_traj.set_title(f"{condition} trajectory ({n_spk} spikes)")
            cbar = fig.colorbar(sc, ax=ax_traj, fraction=0.046, pad=0.04)
            cbar.set_label("head direction (deg)")

            is_ebc = bool(self.ebcc[condition]["ebcs_mask"][cell_id])
            tag = "EBC" if is_ebc else "not EBC"
            color = self.COND_COLORS.get(condition)
            rfd = rf.get(condition) if rf else None
            use_rf = rfd is not None and np.isfinite(rfd.get("ang_rad", np.nan))

            if use_rf:
                self._draw_ego(
                    ax_ego, ego[condition], ego_cmap,
                    f"{condition}: {tag} | RF {rfd['rf_angle_deg']:.0f}°±{rfd['rf_ang_sd_deg']:.0f} "
                    f"| {rfd['rf_dist_cm']:.1f}±{rfd['rf_dist_sd_cm']:.1f} cm",
                    vmin=vmin, vmax=vmax, pref=(rfd["ang_rad"], rfd["r_norm"]), clip_deg=angle_clip,
                )
                self._draw_field_outline(ax_ego, rfd["mask"])

                self._draw_angular(ax_ang, cell_id, condition, ylim=ang_ylim,
                                   clip_deg=angle_clip, peak=None)
                ax_ang.axvline(rfd["rf_angle_deg"], color="red", lw=1.5, zorder=4)
                ax_ang.axvspan(rfd["rf_angle_deg"] - rfd["rf_ang_sd_deg"],
                               rfd["rf_angle_deg"] + rfd["rf_ang_sd_deg"],
                               color="red", alpha=0.12, zorder=0)

                n_dist = ego[condition].shape[1]
                rows = np.unique(np.nonzero(rfd["mask"])[0])
                if rows.size:
                    block = ego[condition][rows]
                    with np.errstate(invalid="ignore"):
                        dmean = np.nanmean(block, axis=0)
                        dsem = np.nanstd(block, axis=0) / np.sqrt(np.sum(np.isfinite(block), axis=0))
                else:
                    dmean = np.full(n_dist, np.nan); dsem = dmean
                dbin = int(np.clip(round(rfd["r_norm"] * (n_dist - 1)), 0, n_dist - 1))
                self._draw_distance(ax_dist, dmean, np.nan_to_num(dsem), color, dbin,
                                    ylim=None, title=f"{condition} dist over RF run")
                ax_dist.axvspan(max(0.0, rfd["rf_dist_cm"] - rfd["rf_dist_sd_cm"]),
                                rfd["rf_dist_cm"] + rfd["rf_dist_sd_cm"],
                                color="red", alpha=0.12, zorder=0)
            else:
                self._draw_ego(
                    ax_ego, ego[condition], ego_cmap,
                    f"{condition}: {tag} | angle {p['ang_signed']:.0f}° | dist {p['dist_cm']:.1f} cm",
                    vmin=vmin, vmax=vmax, pref=(p["ang_raw"], p["r_norm"]), clip_deg=angle_clip,
                )
                self._draw_angular(ax_ang, cell_id, condition, ylim=ang_ylim,
                                   clip_deg=angle_clip, peak=(p["ang_signed"], p["ang_val"]))
                self._draw_distance(ax_dist, p["dist_mean"], p["dist_sem"], color, p["dbin"],
                                    ylim=dist_ylim,
                                    title=f"{condition} dist @ {p['ang_signed']:.0f}° (±{window_deg}°)")

        fig.suptitle(f"{self.session_id} - cell {cell_id}")
        fig.tight_layout()
        return fig

    # ---------------------------------------------------- figure-grade single panels
    def plot_spikemap(self, cell_id, condition="light", hd_cmap="hsv", s=15,
                      figsize=(4.5, 4), colorbar=True, title=None):
        """Figure-grade spike map for one cell: trajectory (gray) with this condition's
        spikes coloured by head direction and sized by spike magnitude -- the same view as
        ``plot_cell``'s first column. Returns the figure."""
        all_mag = self.spk_mag[self.spk[:, cell_id], cell_id]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0
        fig, ax = plt.subplots(figsize=figsize)
        sc, n_spk = self._draw_traj(ax, cell_id, condition, hd_cmap, s, size_ref)
        if colorbar:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("head direction (deg)")
            cb.outline.set_visible(False)
        ax.set_title(title or f"{self.session_id} - cell {cell_id} "
                              f"({condition}, {n_spk} spikes)")
        fig.tight_layout()
        return fig

    def plot_ego(self, cell_id, condition="light", ego_cmap="Blues", angle_clip=None,
                 window_deg=5, figsize=(4.5, 4), colorbar=True, mark_pref=True,
                 title=None):
        """Figure-grade egocentric boundary map for one cell (front = up). The red ring
        marks the preferred angle/distance when ``mark_pref``; ``angle_clip`` (deg) limits
        the visible wedge (``None`` = full circle). Returns the figure."""
        ego = self._ego_for(condition, cell_id)
        fin = ego[np.isfinite(ego)]
        vmin, vmax = (np.percentile(fin, [5, 95]) if fin.size else (None, None))
        pref = None
        if mark_pref:
            p = self._preferred(cell_id, condition, angle_clip, window_deg)
            pref = (p["ang_raw"], p["r_norm"])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
        mesh = self._draw_ego(ax, ego, ego_cmap,
                              title or f"{self.session_id} - cell {cell_id} ({condition})",
                              vmin=vmin, vmax=vmax, pref=pref, clip_deg=angle_clip)
        if colorbar:
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.08)
            cb.set_label("ego rate")
            cb.outline.set_visible(False)
        fig.tight_layout()
        return fig

    def plot_distance(self, cell_id, condition="light", angle_clip=None, window_deg=5,
                      figsize=(4.5, 3.2), title=None):
        """Figure-grade distance-tuning profile for one cell: response vs distance to the
        boundary (cm) at the preferred angle, mean +/- SEM over +/- ``window_deg``. Returns
        the figure."""
        p = self._preferred(cell_id, condition, angle_clip, window_deg)
        color = self.COND_COLORS.get(condition)
        fig, ax = plt.subplots(figsize=figsize)
        self._draw_distance(
            ax, p["dist_mean"], p["dist_sem"], color, p["dbin"],
            title=title or f"{condition} dist @ {p['ang_signed']:.0f}° (±{window_deg}°)",
        )
        self._despine(ax)
        fig.tight_layout()
        return fig

    def plot_cell_figure(self, cell_id, condition="light", hd_cmap="hsv",
                         ego_cmap="Blues", s=15, angle_clip=None, window_deg=5,
                         figsize=(13, 4),
                         titles=("spike map", "egocentric map", "distance profile")):
        """Figure-grade single-cell triptych in one row -- spike map | egocentric map |
        distance profile -- for one ``condition`` ('light' / 'dark' / 'general').

        Reuses the same ``_draw_traj`` / ``_draw_ego`` / ``_draw_distance`` helpers as the
        standalone ``plot_spikemap`` / ``plot_ego`` / ``plot_distance`` (and as ``plot_cell``),
        so every panel matches its single-panel version. ``angle_clip`` limits the ego wedge
        and the preferred-angle search. Returns the figure.
        """
        color = self.COND_COLORS.get(condition)
        p = self._preferred(cell_id, condition, angle_clip, window_deg)
        ego = self._ego_for(condition, cell_id)
        fin = ego[np.isfinite(ego)]
        vmin, vmax = (np.percentile(fin, [5, 95]) if fin.size else (None, None))
        all_mag = self.spk_mag[self.spk[:, cell_id], cell_id]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0

        fig = plt.figure(figsize=figsize)
        ax_spk = fig.add_subplot(1, 3, 1)
        ax_ego = fig.add_subplot(1, 3, 2, projection="polar")
        ax_dist = fig.add_subplot(1, 3, 3)

        sc, n_spk = self._draw_traj(ax_spk, cell_id, condition, hd_cmap, s, size_ref)
        ax_spk.set_title(f"{titles[0]} ({n_spk} spikes)")
        cb = fig.colorbar(sc, ax=ax_spk, fraction=0.046, pad=0.04)
        cb.set_label("head direction (deg)")
        cb.outline.set_visible(False)

        mesh = self._draw_ego(ax_ego, ego, ego_cmap, titles[1], vmin=vmin, vmax=vmax,
                              pref=(p["ang_raw"], p["r_norm"]), clip_deg=angle_clip)
        cb2 = fig.colorbar(mesh, ax=ax_ego, fraction=0.046, pad=0.08)
        cb2.set_label("ego rate")
        cb2.outline.set_visible(False)

        self._draw_distance(ax_dist, p["dist_mean"], p["dist_sem"], color, p["dbin"],
                            title=titles[2])
        self._despine(ax_dist)

        fig.suptitle(f"{self.session_id} - cell {cell_id}  ({condition})")
        fig.tight_layout()
        return fig

    def compare_rf(self, cell_id, variants=None, condition="light", labels=None,
                   hd_cmap="hsv", ego_cmap="Blues", s=15, angle_clip=None, figsize=None,
                   split_half=False):
        """Stack one ``plot_cell``-style row per RFAnalyzer parameter setting to see how a
        knob shifts the egocentric RF of one cell (same ``condition`` -- light by default --
        in every row).

        Each row: trajectory + spikes | ego map with the RF center marker + half-max field
        outline | angular profile | distance profile over the RF's angular run, the RF center
        +/- SD marked. The ego map, profiles and RF are all recomputed at that row's
        parameters, so the panels match the RF drawn on them. Color scale and angular y-axis
        are shared across rows for like-for-like comparison.

        Parameters
        ----------
        cell_id : int
        variants : list[dict] | None
            RFAnalyzer keyword sets to compare, one row each. Default compares the distance
            cap ``[{"ego_radius": 0.35}, {"ego_radius": 0.50}]``; pass e.g.
            ``[{"min_occ": 0}, {"min_occ": 30}]`` for the occupancy floor, or mix knobs.
        condition : str
            "light" (default) or "dark"; the same for every row.
        labels : list[str] | None
            Per-row labels; default is a compact ``key=val`` repr of each variant.
        angle_clip : float | None
            Visible wedge half-width in degrees (None = full circle, so rear RFs show).
        split_half : bool
            If True, ignore ``variants`` and compare the RF on the two chronological halves of
            the condition's frames (a stability check: a reliable RF barely moves between halves,
            an unstable/broad one moves a lot). The suptitle reports the between-halves shift.

        Returns the matplotlib figure.
        """
        from receptive_field import RFAnalyzer  # local: avoids a cell_explorer<->rf cycle

        if split_half:                       # two chronological halves of the condition's frames
            idx = np.where(self.light_col == COND_TO_LIGHTVAL[condition])[0]
            h1 = np.zeros(len(self.df), bool); h1[idx[:len(idx) // 2]] = True
            h2 = np.zeros(len(self.df), bool); h2[idx[len(idx) // 2:]] = True
            variants = [{"frames": h1}, {"frames": h2}]
            if labels is None:
                labels = ["1st half", "2nd half"]
        if variants is None:
            variants = [{"ego_radius": 0.35}, {"ego_radius": 0.50}]
        variants = list(variants)
        if labels is None:
            labels = [", ".join(f"{k}={v}" for k, v in v_.items()) or "default"
                      for v_ in variants]

        # one analyzer + RF + (variant-parameter) ego map per row
        anas = [RFAnalyzer(self, **v_) for v_ in variants]
        rfs = [a.rf(cell_id, condition) for a in anas]
        egos = [a._ego(condition)[cell_id] for a in anas]

        def _angprof(e):
            with np.errstate(invalid="ignore"):
                m = np.nanmean(e, axis=1)
                c = np.sum(np.isfinite(e), axis=1)
                sm = np.nanstd(e, axis=1) / np.sqrt(np.where(c > 0, c, np.nan))
            a = _signed_angle_deg(np.linspace(0, 2 * np.pi, e.shape[0], endpoint=False))
            o = np.argsort(a)
            return a[o], m[o], sm[o]
        profs = [_angprof(e) for e in egos]

        lim = angle_clip if angle_clip is not None else 180
        # shared color scale (kept-wedge 5-95 pct) and angular y-axis across rows
        kept_vals, band = [], []
        for (a, m, sm), e in zip(profs, egos):
            inb = np.ones(len(a), bool) if angle_clip is None else np.abs(a) <= angle_clip
            band += [(m - sm)[inb], (m + sm)[inb]]
            ev = e if angle_clip is None else e[np.abs(_signed_angle_deg(
                np.linspace(0, 2 * np.pi, e.shape[0], endpoint=False))) <= angle_clip]
            kept_vals.append(ev[np.isfinite(ev)])
        allv = np.concatenate(kept_vals) if kept_vals else np.array([])
        vmin, vmax = (np.percentile(allv, [5, 95]) if allv.size else (None, None))
        bvals = np.concatenate([b[np.isfinite(b)] for b in band]) if band else np.array([])
        ang_ylim = None
        if bvals.size:
            lo, hi = float(bvals.min()), float(bvals.max())
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ang_ylim = (lo - pad, hi + pad)

        color = self.COND_COLORS.get(condition)
        all_mag = self.spk_mag[self.spk[:, cell_id], cell_id]
        size_ref = np.percentile(all_mag, 95) if all_mag.size else 1.0

        nrow = len(variants)
        fig = plt.figure(figsize=figsize or (18, 4 * nrow))
        for row, (lab, ana, rfd, ego, (ang, amean, asem)) in enumerate(
                zip(labels, anas, rfs, egos, profs)):
            ax_traj = fig.add_subplot(nrow, 4, 4 * row + 1)
            ax_ego = fig.add_subplot(nrow, 4, 4 * row + 2, projection="polar")
            ax_ang = fig.add_subplot(nrow, 4, 4 * row + 3)
            ax_dist = fig.add_subplot(nrow, 4, 4 * row + 4)
            finite_rf = np.isfinite(rfd.get("ang_rad", np.nan))

            sc, n_spk = self._draw_traj(ax_traj, cell_id, condition, hd_cmap, s, size_ref,
                                        frame_mask=ana.frames)
            ax_traj.set_title(f"{lab}\n{condition} traj ({n_spk} spikes)")
            fig.colorbar(sc, ax=ax_traj, fraction=0.046, pad=0.04).set_label("head dir (deg)")

            title = (f"RF {rfd['rf_angle_deg']:.0f}°±{rfd['rf_ang_sd_deg']:.0f} | "
                     f"{rfd['rf_dist_cm']:.1f}±{rfd['rf_dist_sd_cm']:.1f} cm"
                     if finite_rf else "no RF")
            if ana.fov != (-180.0, 180.0):
                title += f" | frontal {'Y' if rfd['rf_frontal'] else 'N'}"
            if rfd.get("rf_ambiguous"):
                title += " | AMBIG"
            self._draw_ego(ax_ego, ego, ego_cmap, title, vmin=vmin, vmax=vmax,
                           pref=((rfd["ang_rad"], rfd["r_norm"]) if finite_rf else None),
                           clip_deg=angle_clip)
            if finite_rf:
                self._draw_field_outline(ax_ego, rfd["mask"])
            if np.isfinite(rfd.get("rf_ghost_angle_deg", np.nan)):       # rejected confound member
                gr = (rfd["rf_ghost_dist_cm"] / ana.bin_cm) / (ana.n_dist - 1)
                ax_ego.plot(ana._signed_to_raw(rfd["rf_ghost_angle_deg"]), gr, marker="x",
                            color="gray", markersize=8, markeredgewidth=2, zorder=5)

            ax_ang.plot(ang, amean, color=color)
            ax_ang.fill_between(ang, amean - asem, amean + asem, color=color, alpha=0.25)
            ax_ang.axvline(0, color="lightgray", ls="--", lw=1)
            if finite_rf:
                ax_ang.axvline(rfd["rf_angle_deg"], color="red", lw=1.5, zorder=4)
                ax_ang.axvspan(rfd["rf_angle_deg"] - rfd["rf_ang_sd_deg"],
                               rfd["rf_angle_deg"] + rfd["rf_ang_sd_deg"],
                               color="red", alpha=0.12, zorder=0)
            ax_ang.set_xlim(-lim, lim)
            ax_ang.set_xticks(np.arange(-lim, lim + 1, 60))
            if ang_ylim is not None:
                ax_ang.set_ylim(ang_ylim)
            ax_ang.set_xlabel("egocentric angle (deg)")
            ax_ang.set_ylabel("mean response")
            ax_ang.set_title("angular profile")

            n_dist = ego.shape[1]
            rrows = np.unique(np.nonzero(rfd["mask"])[0])
            if rrows.size:
                block = ego[rrows]
                with np.errstate(invalid="ignore"):
                    dmean = np.nanmean(block, axis=0)
                    dsem = np.nanstd(block, axis=0) / np.sqrt(np.sum(np.isfinite(block), axis=0))
            else:
                dmean = np.full(n_dist, np.nan)
                dsem = dmean
            dbin = int(np.clip(round(rfd["r_norm"] * (n_dist - 1)), 0, n_dist - 1)) if finite_rf else 0
            self._draw_distance(ax_dist, dmean, np.nan_to_num(dsem), color, dbin,
                                title="dist over RF run")
            if finite_rf:
                ax_dist.axvspan(max(0.0, rfd["rf_dist_cm"] - rfd["rf_dist_sd_cm"]),
                                rfd["rf_dist_cm"] + rfd["rf_dist_sd_cm"],
                                color="red", alpha=0.12, zorder=0)

        sub = f"RF parameter sweep ({condition})"
        if split_half and len(rfs) == 2 and all(np.isfinite(r["rf_angle_deg"]) for r in rfs):
            dang = abs((rfs[0]["rf_angle_deg"] - rfs[1]["rf_angle_deg"] + 180) % 360 - 180)
            ddist = abs(rfs[0]["rf_dist_cm"] - rfs[1]["rf_dist_cm"])
            sub = f"split-half ({condition}):  Δangle {dang:.0f}°,  Δdist {ddist:.1f} cm"
        fig.suptitle(f"{self.session_id} - cell {cell_id} - {sub}")
        fig.tight_layout()
        return fig
