"""Align several animals' mini2p widefield maps into one common frame and pool their cells.

Mirrors radial-analysis ``SessionVisualizer.align_animals``: load each animal's visual
field-sign (VFS) map, fit a VFS -> reference affine by a brute-force rotation x scale x
translation search, apply that affine to the cells' widefield coordinates, and pool everything
into the reference animal's frame.

The one mini2p difference from radial: radial keeps a VFS in memory, mini2p stores it on disk in
``sign-map/vfs.mat`` (MATLAB variable ``vfs``) next to ``retinotopy.npz`` / ``reference.tif`` /
``retinotopy_areas.pkl``. Otherwise the alignment math is identical.

The pooled frame plugs straight into ``WidefieldCellPlotter`` (via ``make_pooled_plotter``), so
every per-cell / density plot -- including the radial distance colormap -- works across animals
with no extra plotting code::

    aligner = MultiAnimalAligner(all_animals_df)         # df spans several animals
    aligner.align(reference=None)                         # default ref = animal with most cells
    aligner.plot_alignment()                              # QC: each aligned VFS + Pearson r
    p = aligner.make_pooled_plotter()
    p.plot_cell_density(feature="distance")               # pooled, in the common frame
"""

from __future__ import annotations

import os
import re
import pickle

import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from ego_reformat import AZI_DEG_RANGE
from widefield_viz import WidefieldCellPlotter

# mirrors build_animal.DATA_ROOT (kept local to avoid importing build_animal's heavy chain)
DATA_ROOT = r"D:\data-distance-coding-project\mini2p-data"
WORK = 400  # common working resolution (px) for VFS matching + the pooled frame


def _animal_of(session_id):
    """Mouse id from a session rec_name (e.g. 'DMM065' from '251027_LDR_DMM065_ltdk_03')."""
    m = re.search(r"[A-Z]+\d+", str(session_id))
    return m.group(0) if m else "all"


class MultiAnimalAligner:
    """Align each animal's VFS to a reference animal's and pool all cells into that frame.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Multi-animal per-cell table (concatenated ``*_sessions_ego.pkl``) with ``session_id``,
        ``wf_x`` / ``wf_y`` and feature columns. An ``animal`` column is added if absent
        (derived from ``session_id``).
    data_root : str
        Root holding ``{animal}/{sign_map_subdir}/`` with ``vfs.mat`` + ``reference.tif`` +
        ``retinotopy_areas.pkl``. Defaults to ``DATA_ROOT``.
    sign_map_subdir : str
        Per-animal retinotopy subfolder (default ``"sign-map"``).
    work : int
        Working resolution for the common frame (default ``WORK``).
    """

    def __init__(self, cell_df, data_root=DATA_ROOT, sign_map_subdir="sign-map", work=WORK):
        self.df = cell_df.copy()
        if "animal" not in self.df.columns:
            self.df["animal"] = self.df["session_id"].map(_animal_of)
        self.data_root = data_root
        self.sign_map_subdir = sign_map_subdir
        self.work = int(work)
        self._aligned = False

    # ----------------------------------------------------------------- loading
    def _sign_map_dir(self, animal):
        return os.path.join(self.data_root, animal, self.sign_map_subdir)

    def _load_animal(self, animal):
        """Load one animal's VFS (resized to the work frame), widefield width, and area polys.

        Returns ``dict(vfs_work, wf_w, areas_native, w_v)``. ``vfs.mat`` holds the VFS in its
        native retinotopy resolution (same frame as the area polygons); the widefield frame
        (``reference.tif``) shares that field of view at higher resolution, so a cell's
        ``wf_x``/``wf_y`` scales into the work frame by ``work / wf_w``.
        """
        d = self._sign_map_dir(animal)
        vfs = np.asarray(loadmat(os.path.join(d, "vfs.mat"))["vfs"], dtype=np.float32)
        vfs = np.nan_to_num(vfs)
        ref = tifffile.imread(os.path.join(d, "reference.tif"))
        with open(os.path.join(d, "retinotopy_areas.pkl"), "rb") as f:
            areas = pickle.load(f)
        vfs_work = cv2.resize(vfs, (self.work, self.work), interpolation=cv2.INTER_LINEAR)
        return dict(vfs_work=vfs_work, wf_w=int(ref.shape[1]),
                    areas_native=areas, w_v=int(vfs.shape[1]))

    # ------------------------------------------------- alignment primitives
    # (local copy of radial-analysis' sign-mapping affine search)
    @staticmethod
    def _prep(img, sigma):
        img = np.nan_to_num(np.asarray(img, dtype=np.float32))
        if img.ndim > 2:
            img = img.mean(axis=0)
        if sigma and sigma > 0:
            img = gaussian_filter(img, sigma=sigma)
        img = img - img.mean()
        s = img.std()
        return img / s if s > 0 else img

    @staticmethod
    def _affine(center, rot, scale, dx=0.0, dy=0.0):
        M = cv2.getRotationMatrix2D(center, float(rot), float(scale))
        M[0, 2] += dx
        M[1, 2] += dy
        return M

    @staticmethod
    def _warp(img, M, out_shape):
        h, w = out_shape
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def _best_shift(ref_norm, moving_norm, max_translation):
        pad = int(max_translation)
        ref_pad = cv2.copyMakeBorder(ref_norm, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=0)
        res = cv2.matchTemplate(ref_pad, moving_norm, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_loc[0] - pad, max_loc[1] - pad, max_val

    # -------------------------------------------------------------------- align
    def align(self, reference=None, vfs_sigma=2.0, rotation_degs=None, scale_factors=None,
              max_translation=120, verbose=True):
        """Fit a VFS->reference affine per animal and pool all cells into one frame.

        Parameters
        ----------
        reference : str or None
            Animal id whose VFS defines the common frame. Defaults to the animal with the most
            cells (a well-sampled target). Every other animal is aligned to it.
        vfs_sigma : float
            Gaussian sigma used to normalise both VFS maps before matching.
        rotation_degs, scale_factors : array-like or None
            Search ranges. Default rotation +/-15 deg (1 deg step), scale 0.8-1.2 (0.05 step).
        max_translation : int
            Max pixel shift searched by template matching (in the work frame).

        Returns
        -------
        dict
            ``animal -> {M, dx, dy, rotation_deg, scale_factor, pearson_r, n_cells}``.

        Also writes ``ref_x`` / ``ref_y`` (common-frame coords) onto ``self.df``.
        """
        if rotation_degs is None:
            rotation_degs = np.arange(-15, 15.1, 1.0)
        if scale_factors is None:
            scale_factors = np.arange(0.8, 1.21, 0.05)

        animals = sorted(self.df["animal"].dropna().unique())
        if not animals:
            raise ValueError("No animals in cell_df.")

        loaded = {}
        for animal in animals:
            try:
                loaded[animal] = self._load_animal(animal)
            except Exception as e:                       # missing/garbled sign-map -> skip
                if verbose:
                    print(f"  [skip] {animal}: could not load VFS ({e})")
        if not loaded:
            raise RuntimeError("No animals had a loadable vfs.mat.")

        if reference is None:                            # default: best-sampled animal
            counts = self.df[self.df["animal"].isin(loaded)]["animal"].value_counts()
            reference = counts.idxmax()
        if reference not in loaded:
            raise ValueError(f"Reference '{reference}' not loadable. Options: {list(loaded)}")

        W = self.work
        ref_shape = (W, W)
        center = (W / 2.0, W / 2.0)
        self._reference_animal = reference
        self._reference_vfs = loaded[reference]["vfs_work"]
        ref_norm = self._prep(self._reference_vfs, vfs_sigma)

        # reference area polygons -> work frame (native -> work scale)
        s_ref = W / loaded[reference]["w_v"]
        self._reference_areas = {
            name: np.asarray(v, dtype=float) * s_ref
            for name, v in loaded[reference]["areas_native"].items()
        }

        self._transforms, self._aligned_vfs = {}, {}
        self.df["ref_x"] = np.nan
        self.df["ref_y"] = np.nan

        for animal, info in loaded.items():
            if animal == reference:
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                best = dict(dx=0.0, dy=0.0, rotation_deg=0.0, scale_factor=1.0, pearson_r=1.0)
            else:
                mov_norm = self._prep(info["vfs_work"], vfs_sigma)
                best_corr, best_params = -np.inf, None
                for rot in rotation_degs:
                    for sc in scale_factors:
                        moving = self._warp(mov_norm, self._affine(center, rot, sc), ref_shape)
                        dx, dy, corr = self._best_shift(ref_norm, moving, max_translation)
                        if corr > best_corr:
                            best_corr, best_params = corr, (dx, dy, float(rot), float(sc))
                dx, dy, rot, sc = best_params
                M = self._affine(center, rot, sc, dx, dy)
                best = dict(dx=float(dx), dy=float(dy), rotation_deg=rot,
                            scale_factor=sc, pearson_r=float(best_corr))

            self._aligned_vfs[animal] = self._warp(info["vfs_work"], M, ref_shape)

            # cells: widefield px -> work px (scale by work/wf_w), then through the affine
            adf = self.df[self.df["animal"] == animal]
            s = W / info["wf_w"]
            pts = np.column_stack([adf["wf_x"].to_numpy(float) * s,
                                   adf["wf_y"].to_numpy(float) * s,
                                   np.ones(len(adf))])
            ref_xy = pts @ M.T                           # N x 2
            self.df.loc[adf.index, "ref_x"] = ref_xy[:, 0]
            self.df.loc[adf.index, "ref_y"] = ref_xy[:, 1]

            best["M"] = M
            best["n_cells"] = int(len(adf))
            self._transforms[animal] = best
            if verbose:
                tag = " (reference)" if animal == reference else ""
                print(f"  {animal}{tag}: rot={best['rotation_deg']:.1f} "
                      f"scale={best['scale_factor']:.2f} dx={best['dx']:.0f} "
                      f"dy={best['dy']:.0f} r={best['pearson_r']:.3f} n_cells={best['n_cells']}")

        self._aligned = True
        if verbose:
            print(f"\nAligned {len(self._transforms)} animals to '{reference}' "
                  f"(work frame {W}x{W}).")
        return self._transforms

    def _require_aligned(self):
        if not self._aligned:
            raise RuntimeError("Call align(...) before pooled / QC plots.")

    # ------------------------------------------------------------ pooled output
    def make_pooled_plotter(self, white_bg=True, frame_on="cells"):
        """A ``WidefieldCellPlotter`` over the pooled frame (cells from every aligned animal).

        ``ref_x`` / ``ref_y`` become ``wf_x`` / ``wf_y`` and the area polygons are the
        reference's -- so every plot method (incl. ``plot_cell_density(feature="distance")``
        with the radial colormap) works pooled.

        ``white_bg`` blanks the image background (white). ``frame_on`` sets the view:
        ``"cells"`` (default) tight-frames where the pooled cells actually are (avoids the empty
        margins of the full anatomical frame); ``"areas"`` frames on the visual-area boundaries
        (radial-analysis look); ``None`` leaves the per-call ``zoom`` in charge.
        """
        self._require_aligned()
        pooled = self.df[self.df["ref_x"].notna()].copy()
        pooled["wf_x"] = pooled["ref_x"]
        pooled["wf_y"] = pooled["ref_y"]
        plotter = WidefieldCellPlotter.from_components(
            pooled, reference=self._reference_vfs, area_polys=self._reference_areas,
            retino_range={"azi": AZI_DEG_RANGE})
        plotter.white_bg = white_bg
        if frame_on == "cells":
            plotter.view_extent = plotter._cell_view_extent(0.08)
        elif frame_on == "areas":
            plotter.view_extent = plotter._area_view_extent(0.10)
        else:
            plotter.view_extent = None
        return plotter

    # ----------------------------------------------------------------- QC plots
    def _overlay_areas(self, ax, color="black", lw=2.5):
        for name, verts in self._reference_areas.items():
            ax.add_patch(Polygon(verts, closed=True, fill=False,
                                 edgecolor=color, linewidth=lw, alpha=0.85))

    def plot_alignment(self, cmap="jet"):
        """Reference areas overlaid on each animal's aligned VFS (+ Pearson r) — fusion QC."""
        self._require_aligned()
        animals = list(self._aligned_vfs)
        n = len(animals) + 1
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        a = 1.0 / len(animals)
        for animal in animals:
            axes[0].imshow(self._aligned_vfs[animal], cmap=cmap, alpha=a, origin="upper")
        axes[0].set_title(f"All aligned VFS (n={len(animals)})")
        axes[0].axis("off")
        for i, animal in enumerate(animals, start=1):
            axes[i].imshow(self._aligned_vfs[animal], cmap=cmap, origin="upper")
            self._overlay_areas(axes[i])
            r = self._transforms[animal]["pearson_r"]
            axes[i].set_title(f"{animal}  r={r:.2f}")
            axes[i].axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        return fig, axes

    def plot_mean_vfs(self, cmap="jet", show_areas=True):
        """Mean composite VFS across aligned animals — sharp borders = good fusion."""
        self._require_aligned()
        mean_vfs = np.nanmean(np.stack(list(self._aligned_vfs.values())), axis=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mean_vfs, cmap=cmap, origin="upper")
        if show_areas:
            self._overlay_areas(ax)
        ax.set_title(f"Mean composite VFS (n={len(self._aligned_vfs)})")
        ax.axis("off")
        return fig, ax, mean_vfs

    def plot_pooled_cells(self, s=6, alpha=0.6, ax=None, show_areas=True):
        """Pooled cells in the common frame, colored by animal — checks coverage/overlap."""
        self._require_aligned()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()
        ax.imshow(self._reference_vfs, cmap="gray", origin="upper", alpha=0.6)
        df = self.df[self.df["ref_x"].notna()]
        cmap = plt.get_cmap("tab20")
        for i, animal in enumerate(sorted(df["animal"].unique())):
            sub = df[df["animal"] == animal]
            ax.scatter(sub["ref_x"], sub["ref_y"], s=s, alpha=alpha,
                       color=cmap(i % 20), label=animal, edgecolors="none")
        if show_areas:
            self._overlay_areas(ax)
        ax.legend(markerscale=2, fontsize=8, loc="upper right")
        ax.set_title(f"Pooled cells (n_animals={df['animal'].nunique()})")
        ax.axis("off")
        return fig, ax
