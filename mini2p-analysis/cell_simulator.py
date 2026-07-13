"""Simulate an idealized egocentric boundary cell (EBC) and visualize it.

Given a preferred egocentric angle and distance, ``CellSimulator`` generates the spike
train a "perfect" boundary cell would emit from the recorded wall geometry
(``ray_distances``), bins it into an occupancy-normalized egocentric map with the same
formula real cells use (``ego_reformat.compute_ego_maps``), and plots it with the same
panels as ``cell_explorer.CellExplorer.plot_cell``.

The cell is pure geometry -- it depends only on where the walls are, not on luminance --
so its light and dark maps are identical by construction. That makes it a useful
ground-truth / null reference against the real recordings.

Tuning width is a single knob per axis:

- ``sigma_ang_deg = sigma_dist_cm = 0`` is the literal single-bin cell: it fires iff the
  wall on the preferred ray falls in the preferred distance bin.
- ``sigma > 0`` spreads the response over nearby rays/distances (a separable Gaussian
  kernel), giving a smooth EBC-like blob.
- ``dist_cm = None`` makes the cell **distance-invariant** -- a pure angular cell that
  fires for a boundary at the preferred direction at *any* distance, so its field is a
  radial spoke at theta rather than a localized blob.

Note the distinction the simulation makes visible: the *tuning* can be a single bin, but
the *measured* ego map is the tuning convolved with arena geometry -- bins that
co-occur with the preferred one (a square arena has strongly correlated boundaries)
pick up a background, so even a single-bin cell is not a clean delta in the map. The peak
still sits at the input (angle, distance).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ego_reformat import _signed_angle_deg, preferred_from_map
from cell_explorer import CellExplorer


class CellSimulator:
    """Generate and plot a synthetic EBC over a recorded session's wall geometry."""

    def __init__(self, source):
        """
        Parameters
        ----------
        source : CellExplorer | str
            An already-built ``CellExplorer`` (preferred -- reuses its loaded ``df`` /
            ``settings`` / ``x`` / ``y`` / ``hd`` arrays, no second pickle load) or a path
            to an ``ebr_pipeline_*.pkl`` (a ``CellExplorer`` is built from it).
        """
        ce = source if isinstance(source, CellExplorer) else CellExplorer(source)
        self.ce = ce
        self.settings = ce.settings
        self.session_id = ce.session_id
        self.x, self.y, self.hd = ce.x, ce.y, ce.hd
        self.light_col = ce.light_col

        s = ce.settings
        self.n_rays = 360 // s["ego_rays_deg"]
        self.n_dist = round(s["ego_radius"] / s["ego_bin_size"])
        self.bin_size = s["ego_bin_size"]            # m
        self.radius = s["ego_radius"]                # m

        # (F, n_rays) egocentric wall distances; ray angles in the signed convention
        # (front = 0, right +, left -) so a preferred angle maps straight onto them.
        self.ray_dist = np.stack(ce.df["ray_distances"].values).astype(float)
        ray_rad = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        self.ray_signed = _signed_angle_deg(ray_rad)  # (n_rays,) degrees

    # ------------------------------------------------------------ frame helper
    def _frames(self, condition):
        """Boolean frame mask for ``condition`` in {None, 'light', 'dark'} (None = all)."""
        if condition is None:
            return None
        return self.light_col == {"light": 0, "dark": 1}[condition]

    def _dbin(self, dist_cm):
        """Preferred distance bin index for a distance in cm (clipped to the map)."""
        return int(np.clip(np.floor((dist_cm / 100.0) / self.bin_size), 0, self.n_dist - 1))

    # --------------------------------------------------------------- generative
    def response(self, theta_deg, dist_cm, sigma_ang_deg=0.0, sigma_dist_cm=0.0,
                 condition=None):
        """Per-frame response of a cell tuned to ``(theta_deg, dist_cm)``.

        Separable angle x distance kernel summed over the rays that see a wall in range::

            response(t) = sum_r  Kang(phi_r - theta) * Kdist(rho_r - d)

        ``sigma_* = 0`` collapses the corresponding kernel to an indicator (nearest ray /
        preferred distance bin), so ``sigma_ang = sigma_dist = 0`` is the literal
        single-bin cell with ``response`` in ``{0, 1}``. Passing ``dist_cm=None`` makes the
        cell **distance-invariant**: any in-range wall at the preferred angle counts,
        regardless of distance (``sigma_dist_cm`` is then ignored). ``condition`` optionally
        restricts to ``'light'`` / ``'dark'`` frames.
        """
        rd = self.ray_dist if condition is None else self.ray_dist[self._frames(condition)]
        valid = (rd >= 0) & (rd < self.radius)          # (F, n_rays) wall in range
        safe = np.where(valid, rd, -1.0)                # avoid NaN in the floor/exp below

        # angle kernel over rays (n_rays,)
        dphi = (self.ray_signed - theta_deg + 180) % 360 - 180
        if sigma_ang_deg > 0:
            kang = np.exp(-dphi ** 2 / (2 * sigma_ang_deg ** 2))
        else:
            kang = np.zeros(self.n_rays)
            kang[int(np.argmin(np.abs(dphi)))] = 1.0    # nearest single ray

        # distance kernel per (frame, ray)
        if dist_cm is None:
            kdist = np.ones_like(safe)                  # distance-invariant: any wall in range
        elif sigma_dist_cm > 0:
            kdist = np.exp(-(safe - dist_cm / 100.0) ** 2 / (2 * (sigma_dist_cm / 100.0) ** 2))
        else:
            dbin = np.floor(safe / self.bin_size).astype(int)
            kdist = (dbin == self._dbin(dist_cm)).astype(float)  # preferred bin only

        resp = np.where(valid, kdist * kang[None, :], 0.0)
        return resp.sum(axis=1)

    def ego_map(self, response, condition=None):
        """Occupancy-normalized ``(n_rays, n_dist)`` map from a per-frame ``response``.

        Bins ``response`` by wall position with the same formula as
        ``ego_reformat.compute_ego_maps`` (``ego_norm = ego_spk / ego_occ``; NaN where a
        bin was never occupied).
        """
        rd = self.ray_dist if condition is None else self.ray_dist[self._frames(condition)]
        valid = (rd >= 0) & (rd < self.radius)
        safe = np.where(valid, rd, -1.0)
        bin_idx = np.floor(safe / self.bin_size).astype(int)
        valid &= bin_idx < self.n_dist

        occ = np.zeros((self.n_rays, self.n_dist))
        spk = np.zeros((self.n_rays, self.n_dist))
        for r in range(self.n_rays):
            m = valid[:, r]
            b = bin_idx[m, r]
            occ[r] = np.bincount(b, minlength=self.n_dist)[: self.n_dist]
            np.add.at(spk[r], b, response[m])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(occ > 0, spk / occ, np.nan)

    def simulate(self, theta_deg, dist_cm, sigma_ang_deg=0.0, sigma_dist_cm=0.0,
                 condition=None):
        """Return ``{'response', 'ego', 'pref'}`` for a tuned cell.

        ``pref`` is ``ego_reformat.preferred_from_map`` on the simulated map (clip off, so
        rear-preferring cells recover correctly) -- compare it against the input to confirm
        the round trip.
        """
        resp = self.response(theta_deg, dist_cm, sigma_ang_deg, sigma_dist_cm, condition)
        ego = self.ego_map(resp, condition)
        pref = preferred_from_map(ego, self.settings, clip_deg=None)
        return {"response": resp, "ego": ego, "pref": pref}

    # ---------------------------------------------------------------- plotting
    def _theta_to_polar(self, theta_deg, dist_cm):
        """Signed angle (deg) + distance (cm) -> (raw_radians, r_norm) for the polar marker
        (matches ``preferred_from_map``: raw radians, r_norm = dbin / (n_dist - 1))."""
        raw_rad = np.deg2rad((-theta_deg) % 360)
        return raw_rad, self._dbin(dist_cm) / (self.n_dist - 1)

    def plot(self, theta_deg, dist_cm, sigma_ang_deg=0.0, sigma_dist_cm=0.0,
             condition=None, clip_deg=120, s=15, figsize=(18, 4.4),
             ego_cmap="Blues", hd_cmap="hsv", window_deg=5):
        """1x4 figure mirroring one row of ``plot_cell``: trajectory+spikes | ego field |
        angular profile | distance profile. Red marks the input tuning, black-x the
        recovered tuning (they should coincide). Returns the matplotlib figure.
        """
        sim = self.simulate(theta_deg, dist_cm, sigma_ang_deg, sigma_dist_cm, condition)
        resp, ego, pref = sim["response"], sim["ego"], sim["pref"]

        fr = self._frames(condition)
        x = self.x if fr is None else self.x[fr]
        y = self.y if fr is None else self.y[fr]
        hd = self.hd if fr is None else self.hd[fr]

        fig = plt.figure(figsize=figsize)
        ax_traj = fig.add_subplot(1, 4, 1)
        ax_ego = fig.add_subplot(1, 4, 2, projection="polar")
        ax_ang = fig.add_subplot(1, 4, 3)
        ax_dist = fig.add_subplot(1, 4, 4)

        # 1) trajectory with simulated spikes colored by head direction
        ax_traj.plot(x, y, color="lightgray", lw=0.5, zorder=1)
        fire = resp > 0
        ref = np.percentile(resp[fire], 95) if fire.any() else 1.0
        sizes = s * np.clip(resp[fire] / ref, 0.1, 4.0)
        sc = ax_traj.scatter(x[fire], y[fire], c=hd[fire], cmap=hd_cmap, vmin=0, vmax=360,
                             s=sizes, zorder=2)
        ax_traj.set_aspect("equal"); ax_traj.set_xticks([]); ax_traj.set_yticks([])
        ax_traj.set_title(f"trajectory ({int(fire.sum())} spike frames)")
        cb = fig.colorbar(sc, ax=ax_traj, fraction=0.046, pad=0.04)
        cb.set_label("head direction (deg)")

        # 2) simulated ego field: red = input tuning, black x = recovered
        in_rad = np.deg2rad((-theta_deg) % 360)
        if dist_cm is None:                                  # distance-invariant -> red spoke
            CellExplorer._draw_ego(
                ax_ego, ego, ego_cmap,
                f"input {theta_deg:.0f}° / any distance\n"
                f"recovered {pref['ang_signed']:.0f}° / (flat)",
                clip_deg=clip_deg,
            )
            ax_ego.plot([in_rad, in_rad], [0, 1], color="red", lw=2, zorder=6)
        else:
            _, in_rnorm = self._theta_to_polar(theta_deg, dist_cm)
            CellExplorer._draw_ego(
                ax_ego, ego, ego_cmap,
                f"input {theta_deg:.0f}° / {dist_cm:.1f} cm\n"
                f"recovered {pref['ang_signed']:.0f}° / {pref['dist_cm']:.1f} cm",
                pref=(in_rad, in_rnorm), clip_deg=clip_deg,
            )
            if np.isfinite(pref["ang_raw"]):
                ax_ego.plot(pref["ang_raw"], pref["r_norm"], marker="x", markersize=11,
                            markeredgecolor="black", markeredgewidth=2, zorder=6)

        # 3) angular profile (collapsed over distance), input angle dashed
        ang_mean = np.nanmean(ego, axis=1)
        order = np.argsort(self.ray_signed)
        ax_ang.plot(self.ray_signed[order], ang_mean[order], color="navy")
        ax_ang.axvline(theta_deg, color="red", ls="--", lw=1)
        lim = clip_deg if clip_deg is not None else 180
        ax_ang.set_xlim(-lim, lim); ax_ang.set_xticks(np.arange(-lim, lim + 1, 60))
        ax_ang.set_xlabel("egocentric angle (deg)"); ax_ang.set_ylabel("mean response")
        ax_ang.set_title("angular profile")

        # 4) distance profile at the input angle (reuse CellExplorer._draw_distance)
        dphi = (self.ray_signed - theta_deg + 180) % 360 - 180
        block = ego[np.abs(dphi) <= window_deg]
        dmean = np.nanmean(block, axis=0)
        cnt = np.sum(np.isfinite(block), axis=0)
        with np.errstate(invalid="ignore"):
            dsem = np.nanstd(block, axis=0) / np.sqrt(cnt)
        if dist_cm is None:                                  # flat curve, no preferred dist
            bin_cm = self.bin_size * 100
            xd = np.arange(len(dmean)) * bin_cm
            ax_dist.plot(xd, dmean, color="navy")
            ax_dist.fill_between(xd, dmean - dsem, dmean + dsem, color="navy", alpha=0.25)
            ax_dist.set_xlabel("distance to boundary (cm)"); ax_dist.set_ylabel("response")
            ax_dist.set_title(f"distance @ {theta_deg:.0f}° (invariant — flat)")
        else:
            self.ce._draw_distance(ax_dist, dmean, dsem, "navy", self._dbin(dist_cm),
                                   title=f"distance @ {theta_deg:.0f}° (±{window_deg}°)")

        cond = condition or "all frames"
        dist_tag = "dist-invariant" if dist_cm is None else f"σ_dist={sigma_dist_cm} cm"
        fig.suptitle(f"{self.session_id} - simulated EBC  "
                     f"(σ_ang={sigma_ang_deg}°, {dist_tag}, {cond})")
        fig.tight_layout()
        return fig

    def plot_grid(self, thetas_deg, dists_cm, sigma_ang_deg=15.0, sigma_dist_cm=3.0,
                  condition=None, clip_deg=120, ego_cmap="Blues", panel=1.8):
        """Grid of simulated ego fields (rows = distances, cols = angles) to show how the
        egocentric field tiles across preferred tunings. Returns the matplotlib figure.
        """
        nr, nc = len(dists_cm), len(thetas_deg)
        fig, axes = plt.subplots(nr, nc, figsize=(panel * nc, panel * nr),
                                 subplot_kw={"projection": "polar"}, squeeze=False)
        for i, dcm in enumerate(dists_cm):
            for j, th in enumerate(thetas_deg):
                ego = self.simulate(th, dcm, sigma_ang_deg, sigma_dist_cm, condition)["ego"]
                dtag = "any" if dcm is None else f"{dcm:.0f} cm"
                CellExplorer._draw_ego(axes[i][j], ego, ego_cmap,
                                       f"{th:.0f}° / {dtag}", clip_deg=clip_deg)
        fig.suptitle(f"{self.session_id} - simulated EBC tiling "
                     f"(σ_ang={sigma_ang_deg}°, σ_dist={sigma_dist_cm} cm)")
        fig.tight_layout()
        return fig
