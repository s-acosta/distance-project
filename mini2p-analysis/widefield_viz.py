"""Plot reformatted EBC cells on the widefield reference image.

Inspired by ``radial-analysis/src/viz/SessionVisualizer.py``
(``MultiSessionVisualizer.plot_cell_features``): a grayscale widefield background
with cells scattered at their widefield coordinates and colored by a feature, with
non-classified cells optionally greyed out.

Takes the DataFrame produced by ``ego_reformat.SessionReformatter.build`` (or the
concatenation of several) — it only needs the ``wf_x`` / ``wf_y`` columns plus whatever
feature column you want to color by.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import tifffile
from scipy.stats import linregress

from ego_reformat import AZI_DEG_RANGE


def _make_angle_cmap(n=200):
    """Diverging colormap for signed angles, built from two separate arms that meet
    on a shared blue at the centre (0 deg / front).

    - positive (0 -> +180): viridis, blue/teal near 0 -> yellow at +180
    - negative (-180 -> 0): blue/teal near 0 -> purple -> red at -180

    Both arms share the same blue (viridis at 0.25) at the centre so the transition
    through 0 is seamless.
    """
    pos = plt.colormaps["viridis"](np.linspace(0.25, 1.0, n))  # blue/teal -> yellow
    neg_map = mcolors.LinearSegmentedColormap.from_list(
        "neg_arm", ["#31688e", "#762a83", "#b2182b"]  # blue/teal -> purple -> red
    )
    neg = neg_map(np.linspace(0.0, 1.0, n))  # blue -> red
    combined = np.vstack((neg[::-1], pos))   # -180 red ... 0 blue ... +180 yellow
    return mcolors.ListedColormap(combined, name="angle_split")


ANGLE_CMAP = _make_angle_cmap()  # opt-in via cmap=ANGLE_CMAP; viridis (ANGLE_RANGE) is the default

# Default viridis range (deg) for the egocentric RF angle, matching the radial-analysis RF
# coloring (plain viridis over a fixed range: front=0 near the low end, rightward -> yellow).
ANGLE_RANGE = (-20.0, 100.0)

# The four stimulus distances (cm) in the mini2p radial paradigm: distance index i -> cm.
STIM_DISTANCES_CM = (10.0, 16.0, 22.0, 28.0)

# Default range (cm) for the egocentric RF distance: the near..far stimulus distances, chosen
# so the four stimulus distances land exactly on the four DISTANCE_CMAP anchors.
DISTANCE_RANGE = (STIM_DISTANCES_CM[0], STIM_DISTANCES_CM[-1])

# Distance colormap copied from radial-analysis (utils.setup_plot_style -> distance_cmap): an
# evenly-spaced ramp orange (near) -> green -> blue -> purple (far). Over DISTANCE_RANGE the
# stimulus distances 10/16/22/28 cm map onto these four anchors, matching radial's per-distance
# colors (index 0->orange, 1->green, 2->blue, 3->purple).
DISTANCE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "distance", ["orange", "#35b779", "#31688e", "#440154"])


class WidefieldCellPlotter:
    """Scatter per-cell features over the widefield reference image."""

    def __init__(self, cell_df, reference_path, sign_map_dir=None):
        """
        Parameters
        ----------
        cell_df : pd.DataFrame
            Per-cell table with ``wf_x`` / ``wf_y`` (widefield-frame centroids) plus
            feature columns, e.g. from ``SessionReformatter.build``.
        reference_path : str
            Path to the widefield reference image, e.g.
            ``...\\sign-map\\reference.tif``.
        sign_map_dir : str, optional
            Directory with ``retinotopy_areas.pkl`` + ``retinotopy.npz``. When given,
            visual-area boundaries can be drawn on top (``show_areas=True``) and
            ``azi``/``alt`` default to the full retinotopy map range.
        """
        self.df = cell_df
        self.reference = tifffile.imread(reference_path)
        self.area_polys = None
        self.retino_range = {}  # {'azi': (lo, hi), 'alt': (lo, hi)} full map range
        if sign_map_dir is not None:
            self._load_sign_map(sign_map_dir)

    def _load_sign_map(self, sign_map_dir):
        """Load area polygons (scaled to widefield frame) and the azi/alt map ranges."""
        with open(os.path.join(sign_map_dir, "retinotopy_areas.pkl"), "rb") as f:
            areas = pickle.load(f)
        npz = np.load(os.path.join(sign_map_dir, "retinotopy.npz"), allow_pickle=True)
        map_w = npz["azimuth_map"].shape[1]
        scale = self.reference.shape[1] / map_w  # map space -> widefield frame
        self.area_polys = {
            name: np.asarray(v, dtype=float) * scale for name, v in areas.items()
        }
        self.retino_range = {
            # azi is rescaled to real degrees in the dataframe (see ego_reformat)
            "azi": AZI_DEG_RANGE,
            "alt": (float(np.nanmin(npz["altitude_map"])), float(np.nanmax(npz["altitude_map"]))),
        }

    @classmethod
    def from_components(cls, cell_df, reference, area_polys=None, retino_range=None):
        """Build a plotter from in-memory arrays instead of file paths.

        Mirrors what ``__init__`` + ``_load_sign_map`` set up, for callers that already hold
        the pieces in memory -- e.g. ``MultiAnimalAligner.make_pooled_plotter`` passing a pooled
        multi-animal frame, the reference VFS image, and area polygons already in that frame.

        Parameters
        ----------
        cell_df : pd.DataFrame
            Per-cell table with ``wf_x`` / ``wf_y`` in ``reference``'s pixel frame.
        reference : np.ndarray
            Background image (e.g. the reference animal's VFS at the pooled resolution).
        area_polys : dict, optional
            ``{area_name: (N, 2) verts}`` in the same frame; enables ``show_areas``.
        retino_range : dict, optional
            ``{'azi': (lo, hi), 'alt': (lo, hi)}`` default ranges for retinotopy features.
        """
        self = cls.__new__(cls)
        self.df = cell_df
        self.reference = np.asarray(reference)
        self.area_polys = area_polys
        self.retino_range = retino_range or {}
        self.white_bg = False        # set True by MultiAnimalAligner.make_pooled_plotter
        self.view_extent = None      # [xmin,xmax,ymax,ymin] area-boundary frame for pooled view
        return self

    # ------------------------------------------------------------------ helpers
    def _draw_areas(self, ax, color, lw, label):
        for name, poly in self.area_polys.items():
            ax.add_patch(Polygon(poly, closed=True, fill=False,
                                 edgecolor=color, linewidth=lw, alpha=0.9))
            if label:
                c = poly.mean(axis=0)
                ax.text(c[0], c[1], name, color=color, fontsize=9,
                        ha="center", va="center")
    def _show_background(self, ax):
        img = self.reference
        vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
        ax.imshow(
            img, cmap="gray", vmin=vmin, vmax=vmax,
            extent=[0, img.shape[1], img.shape[0], 0], origin="upper",
        )

    def _feature_style(self, values, is_angle, vmin, vmax, cmap):
        """Return a (norm, cmap) pair for the given feature values.

        Angles (signed degrees, front=0, right positive, left negative) use plain ``viridis``
        over a fixed range -- default ``ANGLE_RANGE`` (-20..100 deg), matching the
        radial-analysis RF coloring -- so the scale is comparable across plots; pass ``vmin`` /
        ``vmax`` to override (or ``cmap=ANGLE_CMAP`` for the old blue-centred split map). Other
        features default to ``viridis`` over their data range.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if cmap is None:
            cmap = "viridis"
        if is_angle:
            lo = ANGLE_RANGE[0] if vmin is None else vmin
            hi = ANGLE_RANGE[1] if vmax is None else vmax
            return mcolors.Normalize(vmin=lo, vmax=hi), cmap
        lo = vmin if vmin is not None else np.nanmin(vals)
        hi = vmax if vmax is not None else np.nanmax(vals)
        return mcolors.Normalize(vmin=lo, vmax=hi), cmap

    def _zoom_to_cells(self, ax, pad):
        x = np.asarray(self.df["wf_x"], dtype=float)
        y = np.asarray(self.df["wf_y"], dtype=float)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.max() + pad, y.min() - pad)  # inverted y (origin='upper')

    def _area_view_extent(self, pad_frac=0.10):
        """Square view extent centered on the area polygons (+``pad_frac`` padding), as
        ``[xmin, xmax, ymax, ymin]`` (origin='upper'). The boundaries -- not the cells -- define
        a stable anatomical frame, used for the pooled multi-animal view (cf. radial-analysis).
        Returns ``None`` if no area polygons are loaded.
        """
        verts = [np.asarray(v, float) for v in (self.area_polys or {}).values()
                 if np.asarray(v, float).ndim == 2 and len(v)]
        if not verts:
            return None
        allv = np.vstack(verts)
        xmin, ymin = np.nanmin(allv, axis=0)
        xmax, ymax = np.nanmax(allv, axis=0)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        half = 0.5 * max(xmax - xmin, ymax - ymin, 1.0) * (1.0 + pad_frac)
        return [cx - half, cx + half, cy + half, cy - half]

    def _cell_view_extent(self, pad_frac=0.08):
        """Tight view extent around the cells (``wf_x`` / ``wf_y``) with ``pad_frac`` padding,
        as ``[xmin, xmax, ymax, ymin]`` (origin='upper'). Unlike ``_area_view_extent`` this
        frames where the data actually is, so sparsely-covered corners of the anatomical frame
        don't leave the figure half-empty. Returns ``None`` if there are no finite coordinates.
        """
        x = np.asarray(self.df["wf_x"], dtype=float)
        y = np.asarray(self.df["wf_y"], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            return None
        x, y = x[m], y[m]
        px = (x.max() - x.min()) * pad_frac or 1.0
        py = (y.max() - y.min()) * pad_frac or 1.0
        return [x.min() - px, x.max() + px, y.max() + py, y.min() - py]

    # -------------------------------------------------------------------- plot
    # condition-prefixed per-EBC features -> "{condition}_pref_{suffix}"
    _FEATURE_COL = {"distance": "pref_dist", "angle": "pref_angle"}
    # condition-independent retinotopy features -> direct column
    _RETINO_FEATURES = {"azi": "azimuth", "alt": "altitude"}
    # visual-pipeline columns preferred for angle/distance when present in the df
    _VISUAL_COL = {"angle": "rf_angle_deg", "distance": "rf_dist_cm"}

    def _resolve_feature(self, feature, condition):
        """``(column, cbar_label, feat_label, is_retino)`` for a feature name.

        ``angle``/``distance`` prefer the **visual-pipeline** columns (``rf_angle_deg`` /
        ``rf_dist_cm``) when the DataFrame has them, else fall back to the EBC pref columns
        (``{condition}_pref_*``). ``azi``/``alt`` are the retinotopy columns; any raw column
        name in the DataFrame is also accepted.
        """
        if feature in self._FEATURE_COL:
            vcol = self._VISUAL_COL.get(feature)
            if vcol and vcol in self.df.columns:
                return vcol, f"visual {feature}", f"visual {feature}", False
            col = f"{condition}_{self._FEATURE_COL[feature]}"
            return col, f"{condition} pref {feature}", f"pref {feature}", False
        if feature in self._RETINO_FEATURES:
            lab = self._RETINO_FEATURES[feature]
            return feature, lab, lab, True
        if feature in self.df.columns:
            return feature, feature, feature, False
        raise ValueError(
            f"feature must be distance/angle/azi/alt or a df column, got {feature!r}")

    def plot_cells(self, condition="light", feature="distance", grey_unclassified=None,
                   classified_col=None, ax=None, cmap=None, vmin=None, vmax=None,
                   zoom=True, pad=80, s=25, title=None,
                   show_areas=True, area_color="black", area_lw=1.0, label_areas=True):
        """Plot cells over the widefield, colored by a per-condition feature.

        Parameters
        ----------
        condition : {'light', 'dark'}
            Which condition's columns to use (only relevant for ``distance``/``angle``
            and for the EBC greying of ``azi``/``alt``).
        feature : {'distance', 'angle', 'azi', 'alt'} or a column name
            Which feature to color by. ``angle``/``distance`` use the **visual-pipeline**
            columns ``rf_angle_deg`` / ``rf_dist_cm`` when present (else the EBC pref columns
            ``{condition}_pref_*``); ``azi``/``alt`` are the retinotopy columns; any raw column
            name in the DataFrame is also accepted.
        grey_unclassified : bool, optional
            If True, cells not in ``classified_col`` are drawn grey and only the classified
            ones are colored. Defaults to True for ``distance``/``angle`` and False for
            ``azi``/``alt`` (retinotopy is shown for all cells).
        classified_col : str, optional
            Boolean column used for the greying/highlight. Defaults to ``"keep"`` when the
            DataFrame has it (the visual set), else ``"{condition}_is_ebc"``.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Creates a new figure if omitted.
        cmap, vmin, vmax : optional
            Color scaling overrides. Defaults are feature-aware: for ``angle``, ``viridis``
            over the fixed ``ANGLE_RANGE`` (-20..100 deg, like radial-analysis); for
            ``distance``, ``viridis`` over the data range.
        zoom : bool
            Zoom to the cell bounding box (cells occupy a small part of the 2048x2048
            reference). Set False to show the whole reference.
        pad, s : float
            Zoom padding (pixels) and scatter marker size.
        title : str, optional
            Plot title. Defaults to ``"{condition} EBCs - pref {feature}"``.
        show_areas : bool
            Draw visual-area boundaries on top (requires the plotter to have been
            created with ``sign_map_dir``). No-op if no polygons were loaded.
        area_color, area_lw :
            Boundary line color and width.
        label_areas : bool
            Annotate each area with its name at the polygon centroid.

        Returns
        -------
        fig, ax
        """
        if condition not in ("light", "dark"):
            raise ValueError(f"condition must be 'light' or 'dark', got {condition!r}")

        col, cbar_label, feat_label, is_retino = self._resolve_feature(feature, condition)
        if classified_col is None:
            classified_col = "keep" if "keep" in self.df.columns else f"{condition}_is_ebc"
        is_angle = "angle" in col  # signed-degree column -> viridis over ANGLE_RANGE
        if grey_unclassified is None:
            grey_unclassified = not is_retino  # grey unclassified for non-retino features

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()

        self._show_background(ax)

        df = self.df
        if grey_unclassified:
            mask = df[classified_col].to_numpy(dtype=bool)
            bg = df[~mask]
            ax.scatter(bg["wf_x"], bg["wf_y"], c="lightgray", s=s, alpha=0.5,
                       edgecolor="black", linewidth=0.2)
            fg = df[mask]
        else:
            fg = df

        # retinotopy features default to the full map range (azi already in real deg)
        if is_retino and feature in self.retino_range:
            rlo, rhi = self.retino_range[feature]
            vmin = rlo if vmin is None else vmin
            vmax = rhi if vmax is None else vmax

        fg_vals = np.asarray(fg[col], dtype=float)
        norm, cmap = self._feature_style(fg_vals, is_angle, vmin, vmax, cmap)
        sc = ax.scatter(fg["wf_x"], fg["wf_y"], c=fg_vals,
                        cmap=cmap, norm=norm, s=s, alpha=1.0,
                        edgecolor="black", linewidth=0.2)
        unit = " (deg)" if (is_angle or is_retino) else (" (cm)" if feature == "distance" else "")
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=cbar_label + unit)

        if show_areas and self.area_polys is not None:
            self._draw_areas(ax, area_color, area_lw, label_areas)

        if zoom:
            self._zoom_to_cells(ax, pad)

        if title is None:
            if is_retino and not grey_unclassified:
                title = feat_label  # e.g. "azimuth" over all cells
            else:
                scope = {"keep": "keep (visual)",
                         f"{condition}_is_ebc": f"{condition} EBCs"}.get(
                             classified_col, classified_col) if grey_unclassified else "cells"
                title = f"{scope} - {feat_label}"
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    # ----------------------------------------------------------------- density
    def plot_cell_density(self, feature="keep", condition="light", classified_col=None,
                          n_bins=25, min_count=1, outline_coverage=True, ax=None, cmap=None,
                          vmin=None, vmax=None, zoom=True, pad=80, show_areas=True,
                          area_color="black", area_lw=1.0, label_areas=True, frame=True,
                          boundary_lw=1.5, title=None):
        """2D binned density of a feature over the widefield (cf. radial-analysis
        ``MultiSessionVisualizer.plot_cell_density``).

        Cells are binned into an ``n_bins`` x ``n_bins`` grid over their bounding box and each
        bin colored, depending on ``feature``, by:

        - ``"count"`` -> the **number of cells** in the bin (raw density).
        - a **boolean** column (e.g. ``"keep"`` (default), ``"is_visual"``, ``"rf_frontal"``,
          ``"{condition}_is_ebc"``) -> the **percentage** of the bin's cells that are True (the
          spatial density of that population).
        - a **value** feature (``angle``/``distance``/``azi``/``alt`` or any numeric column) ->
          the **mean** of that feature over the bin's cells. Non-retinotopy value features are
          restricted to ``classified_col`` (default ``"keep"``, since the RF is only defined for
          the visual set); angle uses ``viridis`` over ``ANGLE_RANGE`` (linear mean -- fine for
          the frontal cluster, not a circular mean).

        Bins with fewer than ``min_count`` cells render **white**. ``outline_coverage`` (default
        True; every mode except raw ``count``) then draws a gray staircase along the bin edges of
        the ``min_count``-covered region, so those white bins read against a clear coverage
        boundary -- for a % map or a mean map alike, e.g. distance bins with too few cells to
        average (cf. radial-analysis ``plot_cell_density``). Distance uses ``DISTANCE_CMAP`` over
        ``DISTANCE_RANGE`` (the 10/16/22/28 cm stimulus distances land on radial's four distance
        colors); angle uses ``ANGLE_RANGE``. All overridden by ``vmin``/``vmax``/``cmap``.

        For a figure-grade panel pass ``frame=False`` (drop the axes box) and
        ``label_areas=False`` (drop the area names). Returns ``(fig, ax)``.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()
        if getattr(self, "white_bg", False):
            ax.set_facecolor("white")                          # pooled: blank/white background
        else:
            self._show_background(ax)

        df = self.df
        x = np.asarray(df["wf_x"], dtype=float)
        y = np.asarray(df["wf_y"], dtype=float)
        h, w = self.reference.shape[:2]
        xe = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)   # bin over the cells' bbox
        ye = np.linspace(np.nanmin(y), np.nanmax(y), n_bins + 1)
        count_all, _, _ = np.histogram2d(x, y, bins=[xe, ye])

        is_angle = is_distance = False
        is_count = feature in ("count", "density")
        if is_count:
            grid, count_grid = count_all, count_all
            feat_label, cbar_label = "cell count", "cell count"
            if cmap is None:
                cmap = "magma"
        else:
            col, cbar_label, feat_label, is_retino = self._resolve_feature(feature, condition)
            if df[col].dtype == bool:                          # fraction (%) True per bin
                hit, _, _ = np.histogram2d(x, y, bins=[xe, ye],
                                           weights=df[col].to_numpy(dtype=float))
                with np.errstate(invalid="ignore", divide="ignore"):
                    grid = np.where(count_all > 0, hit / count_all * 100.0, np.nan)
                count_grid = count_all
                cbar_label = feat_label = f"% {col}"
                if cmap is None:
                    cmap = "viridis"
                if vmin is None:
                    vmin = 0.0
            else:                                              # mean feature per bin
                vals = df[col].to_numpy(dtype=float)
                sel = np.isfinite(vals)
                if classified_col is None and not is_retino:
                    classified_col = ("keep" if "keep" in df.columns else
                                      f"{condition}_is_ebc" if f"{condition}_is_ebc" in df.columns
                                      else None)
                if classified_col is not None:
                    sel = sel & df[classified_col].to_numpy(dtype=bool)
                ssum, _, _ = np.histogram2d(x[sel], y[sel], bins=[xe, ye], weights=vals[sel])
                count_grid, _, _ = np.histogram2d(x[sel], y[sel], bins=[xe, ye])
                with np.errstate(invalid="ignore", divide="ignore"):
                    grid = np.where(count_grid > 0, ssum / count_grid, np.nan)
                is_angle = "angle" in col
                is_distance = "dist" in col
                cbar_label = f"mean {cbar_label}"
                if cmap is None:
                    cmap = DISTANCE_CMAP if is_distance else "viridis"

        if is_angle:                                           # viridis over the fixed RF-angle range
            if vmin is None:
                vmin = ANGLE_RANGE[0]
            if vmax is None:
                vmax = ANGLE_RANGE[1]
        if is_distance:                                        # DISTANCE_CMAP over the stimulus-distance range (cm)
            if vmin is None:
                vmin = DISTANCE_RANGE[0]
            if vmax is None:
                vmax = DISTANCE_RANGE[1]

        grid = np.ma.masked_where((count_grid < min_count) | ~np.isfinite(grid), grid)

        cmap = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
        cmap.set_bad("white")                                  # under-min_count bins render white
        mesh = ax.pcolormesh(xe, ye, grid.T, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading="flat", alpha=0.85)
        unit = " (deg)" if is_angle else (" (cm)" if is_distance else "")
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02, label=cbar_label + unit)
        if is_distance:                                        # tick the four stimulus distances
            cbar.set_ticks(list(STIM_DISTANCES_CM))

        if outline_coverage and not is_count:
            # gray staircase along the bin edges of the region with >= min_count *total* cells --
            # the same coverage the % maps use (count_all, the population denominator), so a mean
            # map (e.g. distance) shows white where cells were sampled but too few had a defined
            # value (cf. radial plot_cell_density outline_coverage).
            covered = count_all >= min_count
            nx, ny = covered.shape
            segments = []
            for i in range(nx):
                for j in range(ny):
                    if not covered[i, j]:
                        continue
                    x0, x1 = xe[i], xe[i + 1]
                    y0, y1 = ye[j], ye[j + 1]
                    if i == 0 or not covered[i - 1, j]:
                        segments.append([(x0, y0), (x0, y1)])
                    if i == nx - 1 or not covered[i + 1, j]:
                        segments.append([(x1, y0), (x1, y1)])
                    if j == 0 or not covered[i, j - 1]:
                        segments.append([(x0, y0), (x1, y0)])
                    if j == ny - 1 or not covered[i, j + 1]:
                        segments.append([(x0, y1), (x1, y1)])
            if segments:
                ax.add_collection(LineCollection(segments, colors="gray", linewidths=boundary_lw))

        if show_areas and self.area_polys is not None:
            self._draw_areas(ax, area_color, area_lw, label_areas)
        ext = getattr(self, "view_extent", None)
        if ext is not None:                                    # pooled: area-boundary frame wins
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
        elif zoom:
            self._zoom_to_cells(ax, pad)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)                                  # inverted y (origin='upper')
        ax.set_aspect("equal")                                 # square pixels (imshow used to force this)
        ax.set_title(title if title is not None else f"{feat_label} density", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        if not frame:                                          # figure-grade: drop the axes box
            for spine in ax.spines.values():
                spine.set_visible(False)
        return fig, ax

    # -------------------------------------------------------------- histogram
    def _distance_values(self, df, feature="distance", condition="light", classified_col=None):
        """Finite preferred-distance values for ``df``, restricted to ``classified_col`` (default
        ``"keep"`` when present, since the RF is only defined for the visual set)."""
        col, _, _, _ = self._resolve_feature(feature, condition)
        if classified_col is None:
            classified_col = "keep" if "keep" in df.columns else None
        if classified_col is not None and classified_col in df.columns:
            df = df[df[classified_col].to_numpy(dtype=bool)]
        vals = np.asarray(df[col], dtype=float)
        return vals[np.isfinite(vals)]

    def plot_distance_histogram(self, n_bins=20, feature="distance", condition="light",
                                classified_col=None, hist_range=None, density=False,
                                cmap=None, vmin=None, vmax=None, mark_stim=True,
                                ax=None, edgecolor="black", title=None):
        """Histogram of preferred (RF) distance, each bar colored by the distance colormap.

        Whatever ``n_bins`` you pick, bar *color* encodes distance: a smooth ``DISTANCE_CMAP``
        gradient spread uniformly between ``vmin`` and ``vmax`` (default ``DISTANCE_RANGE`` =
        10..28 cm), so the bars mirror the spatial distance maps. Bar *heights* are the count (or
        fraction with ``density=True``); only the *color* encodes distance.

        ``hist_range`` sets the binned span (default: the data range), independent of the color
        scaling. Distance is restricted to ``classified_col`` (default ``"keep"`` when present),
        since the RF is only defined there. Returns ``(fig, ax)``.
        """
        vals = self._distance_values(self.df, feature, condition, classified_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
        else:
            fig = ax.get_figure()
        if len(vals) == 0:
            ax.set_title("no cells with a defined RF distance", fontsize=12)
            return fig, ax

        rng = hist_range if hist_range is not None else (float(vals.min()), float(vals.max()))
        counts, edges = np.histogram(vals, bins=n_bins, range=rng, density=density)
        centers = 0.5 * (edges[:-1] + edges[1:])

        cmap = DISTANCE_CMAP if cmap is None else cmap
        lo = DISTANCE_RANGE[0] if vmin is None else vmin
        hi = DISTANCE_RANGE[1] if vmax is None else vmax
        colors = cmap(mcolors.Normalize(vmin=lo, vmax=hi)(centers))   # smooth gradient, vmin..vmax

        ax.bar(centers, counts, width=np.diff(edges), color=colors,
               edgecolor=edgecolor, linewidth=0.6, align="center")
        if mark_stim:
            ax.set_xticks(list(STIM_DISTANCES_CM))               # the discrete stimulus distances
        ax.set_xlabel("preferred distance (cm)")
        ax.set_ylabel("fraction of cells" if density else "cell count")
        ax.set_title(title if title is not None else f"preferred distance  (n={len(vals)})",
                     fontsize=14)
        return fig, ax

    def plot_distance_histograms_by_area(self, areas=None, n_bins=20, ncols=3, min_cells=0,
                                         sharex=True, sharey=False, figsize=None, **kwargs):
        """One ``plot_distance_histogram`` per visual area (uses the ``area`` column).

        ``areas`` selects/orders the panels (default: every area present except ``"None"``, in the
        canonical V1, LM, RL, AM, PM, AL order); areas with fewer than ``min_cells`` cells are
        dropped. A common ``hist_range`` (the pooled data range) is
        applied across panels -- unless you pass one -- so the bins line up for comparison; pass
        ``density=True`` (and ``sharey=True``) to compare shapes rather than counts. Extra kwargs
        (``hist_range``, ``vmin``, ``vmax``, ``density``, ``feature`` ...) are forwarded to
        ``plot_distance_histogram``. Returns ``(fig, axes)``.
        """
        df = self.df
        if "area" not in df.columns:
            raise ValueError("cell table has no 'area' column")
        feat = kwargs.get("feature", "distance")
        cond = kwargs.get("condition", "light")
        cc = kwargs.get("classified_col", None)
        order = ["V1", "LM", "RL", "AM", "PM", "AL", "LI", "POR", "None"]
        if areas is None:
            present = [a for a in df["area"].dropna().unique() if a != "None"]
            areas = sorted(present, key=lambda a: order.index(a) if a in order else len(order))
        # cells per area (keep-restricted, finite distance); drop areas below min_cells
        counts = {a: len(self._distance_values(df[df["area"] == a], feat, cond, cc)) for a in areas}
        areas = [a for a in areas if counts[a] >= min_cells]
        if not len(areas):
            raise ValueError(f"no areas with >= {min_cells} cells (counts: {counts})")
        if kwargs.get("hist_range") is None:                   # shared bins across panels
            allvals = self._distance_values(df, feat, cond, cc)
            kwargs["hist_range"] = ((float(allvals.min()), float(allvals.max()))
                                    if len(allvals) else None)
        kwargs.pop("title", None)

        nrows = int(np.ceil(len(areas) / ncols))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey,
                                 figsize=figsize or (4.5 * ncols, 3.0 * nrows))
        axes = axes.ravel()
        for ax, area in zip(axes, areas):
            sub_df = df[df["area"] == area]
            self.from_components(sub_df, reference=None).plot_distance_histogram(
                n_bins=n_bins, ax=ax, **kwargs)
            ax.set_title(f"{area}  (n={counts[area]})", fontsize=12)
        for ax in axes[len(areas):]:
            ax.axis("off")
        fig.tight_layout()
        return fig, axes

    # -------------------------------------------------------------- regression
    def _resolve_col(self, feature, condition):
        """Map a feature name to a (column, label). ``angle``/``distance`` prefer the visual
        columns (``rf_angle_deg``/``rf_dist_cm``) when present, else the EBC pref columns; also
        accepts ``azi``/``alt`` or any raw column name in the DataFrame."""
        if feature in self._FEATURE_COL:
            vcol = self._VISUAL_COL.get(feature)
            if vcol and vcol in self.df.columns:
                return vcol, f"visual {feature}"
            return f"{condition}_{self._FEATURE_COL[feature]}", f"{condition} pref {feature}"
        if feature in self._RETINO_FEATURES:
            return feature, self._RETINO_FEATURES[feature]
        if feature in self.df.columns:
            return feature, feature
        raise ValueError(f"unknown feature/column {feature!r}")

    def scatter_regression(self, x="azi", y="angle", condition="light", ebc_only=True,
                           classified_col=None, ax=None, color="tab:blue", s=18, annotate=True):
        """Scatter two per-cell features and fit a linear regression.

        Parameters
        ----------
        x, y : str
            Features to plot (x default ``azi``, y default ``angle`` ->
            ``{condition}_pref_angle``). Accepts ``distance``/``angle``/``azi``/``alt``
            or any raw column name.
        condition : {'light', 'dark'}
            Used to resolve ``distance``/``angle`` columns and the EBC filter.
        ebc_only : bool
            Restrict to the classified set (``classified_col``; default True).
        classified_col : str, optional
            Boolean column for the restriction. Defaults to ``"keep"`` when present (the
            visual set), else ``"{condition}_is_ebc"``.
        ax, color, s, annotate :
            Plot axis, point color/size, and whether to annotate r / p / slope.

        Returns
        -------
        fig, ax, result
            ``result`` is the ``scipy.stats.linregress`` output (slope, intercept,
            rvalue, pvalue, stderr).

        Note
        ----
        Angles are circular; this is an ordinary least-squares fit, so interpret r/slope
        with that caveat (use it as a first-pass trend, not a circular-stats test).
        """
        xcol, xlab = self._resolve_col(x, condition)
        ycol, ylab = self._resolve_col(y, condition)

        cc = classified_col or ("keep" if "keep" in self.df.columns else f"{condition}_is_ebc")
        d = self.df
        if ebc_only:
            d = d[d[cc].to_numpy(dtype=bool)]
        xv = np.asarray(d[xcol], dtype=float)
        yv = np.asarray(d[ycol], dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[m], yv[m]

        res = linregress(xv, yv)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        ax.scatter(xv, yv, s=s, color=color, alpha=0.6, edgecolor="black", linewidth=0.2)
        xs = np.array([xv.min(), xv.max()])
        ax.plot(xs, res.intercept + res.slope * xs, color="black", lw=2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        scope = cc if ebc_only else "cells"
        ax.set_title(f"{ylab} vs {xlab}  ({scope}, n={len(xv)})", fontsize=13)
        if annotate:
            ax.text(0.04, 0.96,
                    f"r = {res.rvalue:.2f}\np = {res.pvalue:.2g}\nslope = {res.slope:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        return fig, ax, res
