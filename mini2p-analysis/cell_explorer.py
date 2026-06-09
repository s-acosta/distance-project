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

    def _draw_traj(self, ax, cell_id, condition, hd_cmap, s, size_ref):
        """Full trajectory (gray) with this condition's spikes colored by head direction.

        Marker size scales with spike magnitude (``spk_rate_thr``), normalized by the
        shared ``size_ref`` so larger spikes appear as larger dots.
        """
        ax.plot(self.x, self.y, color="lightgray", lw=0.5, zorder=1)
        mask = self.spk[:, cell_id] & (self.light_col == COND_TO_LIGHTVAL[condition])
        mag = self.spk_mag[mask, cell_id]
        sizes = s * np.clip(mag / size_ref, 0.1, 4.0)  # area scales with magnitude
        sc = ax.scatter(self.x[mask], self.y[mask], c=self.hd[mask], cmap=hd_cmap,
                        vmin=0, vmax=360, s=sizes, zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return sc, int(mask.sum())

    @staticmethod
    def _draw_ego(ax, ego, cmap, title, vmin=None, vmax=None, pref=None, clip_deg=None):
        """Polar pcolormesh of a (n_rays, n_dist) ego field (0 deg up = in front).

        ``pref=(angle_rad, r_norm)`` draws an open circle at the preferred angle/distance.
        ``clip_deg`` restricts the visible wedge to +/- that many degrees of front (raw
        theta +/-clip, which equals the front +/-clip signed wedge).
        """
        n_rays, n_dist = ego.shape
        theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        r = np.linspace(0, 1, n_dist)
        theta_grid, r_grid = np.meshgrid(theta, r)
        ax.pcolormesh(theta_grid, r_grid, ego.T, cmap=cmap, vmin=vmin, vmax=vmax)
        if pref is not None and np.isfinite(pref[0]) and np.isfinite(pref[1]):
            ax.plot(pref[0], pref[1], marker="o", markersize=12, markeredgecolor="red",
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

    # -------------------------------------------------------------------- plots
    def plot_cell(self, cell_id, hd_cmap="hsv", ego_cmap="Blues", s=15, figsize=(18, 8),
                  angle_clip=120, window_deg=5):
        """2x4 figure: rows = light (top) / dark (bottom); columns =
        trajectory+spikes (colored by head direction) | egocentric boundary field |
        response vs angle (collapsed over distance, mean +/- SEM) | response vs distance
        (cm) at the preferred angle, mean +/- SEM over +/- window_deg around it.

        Preferred angle = peak of the angular-response curve; preferred distance = peak
        of the windowed distance curve. These (not the stored pipeline stats) drive the
        red circle on the ratemap and the titles. ``angle_clip`` (deg, default 120)
        restricts the ego wedge and the angular x-axis to the front +/- that many degrees
        (None = full circle). ``window_deg`` (default 5) is the half-width of the angular
        window used for the distance curve's mean/SEM. Returns the matplotlib figure.
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
            self._draw_ego(
                ax_ego, ego[condition], ego_cmap,
                f"{condition}: {tag} | angle {p['ang_signed']:.0f}° | dist {p['dist_cm']:.1f} cm",
                vmin=vmin, vmax=vmax, pref=(p["ang_raw"], p["r_norm"]), clip_deg=angle_clip,
            )

            self._draw_angular(ax_ang, cell_id, condition, ylim=ang_ylim,
                               clip_deg=angle_clip, peak=(p["ang_signed"], p["ang_val"]))

            self._draw_distance(ax_dist, p["dist_mean"], p["dist_sem"],
                                self.COND_COLORS.get(condition), p["dbin"], ylim=dist_ylim,
                                title=f"{condition} dist @ {p['ang_signed']:.0f}° (±{window_deg}°)")

        fig.suptitle(f"{self.session_id} - cell {cell_id}")
        fig.tight_layout()
        return fig
