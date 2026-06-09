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


ANGLE_CMAP = _make_angle_cmap()


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

        Angles are signed degrees (front=0, right positive, left negative), drawn with
        the blue-centred split colormap: positive arm = viridis (blue->yellow),
        negative arm = blue->red, meeting on blue at 0. The limits default to the 95th
        percentile of each direction (so outliers don't dominate), with 0 pinned to the
        blue centre via a TwoSlopeNorm.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if is_angle:
            if cmap is None:
                cmap = ANGLE_CMAP
            pos, neg = vals[vals > 0], vals[vals < 0]
            hi = vmax if vmax is not None else (
                np.percentile(pos, 95) if pos.size else 1.0)
            lo = vmin if vmin is not None else (
                -np.percentile(np.abs(neg), 95) if neg.size else -1.0)
            norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            return norm, cmap
        if cmap is None:
            cmap = "viridis"
        lo = vmin if vmin is not None else np.nanmin(vals)
        hi = vmax if vmax is not None else np.nanmax(vals)
        return mcolors.Normalize(vmin=lo, vmax=hi), cmap

    def _zoom_to_cells(self, ax, pad):
        x = np.asarray(self.df["wf_x"], dtype=float)
        y = np.asarray(self.df["wf_y"], dtype=float)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.max() + pad, y.min() - pad)  # inverted y (origin='upper')

    # -------------------------------------------------------------------- plot
    # condition-prefixed per-EBC features -> "{condition}_pref_{suffix}"
    _FEATURE_COL = {"distance": "pref_dist", "angle": "pref_angle"}
    # condition-independent retinotopy features -> direct column
    _RETINO_FEATURES = {"azi": "azimuth", "alt": "altitude"}

    def plot_cells(self, condition="light", feature="distance", grey_unclassified=None,
                   ax=None, cmap=None, vmin=None, vmax=None,
                   zoom=True, pad=80, s=12, title=None,
                   show_areas=True, area_color="black", area_lw=1.0, label_areas=True):
        """Plot cells over the widefield, colored by a per-condition feature.

        Parameters
        ----------
        condition : {'light', 'dark'}
            Which condition's columns to use (only relevant for ``distance``/``angle``
            and for the EBC greying of ``azi``/``alt``).
        feature : {'distance', 'angle', 'azi', 'alt'}
            Which feature to color by. ``distance``/``angle`` map to
            ``{condition}_pref_dist`` / ``{condition}_pref_angle``; ``azi``/``alt`` are
            the condition-independent retinotopy columns. EBC membership uses
            ``{condition}_is_ebc``.
        grey_unclassified : bool, optional
            If True, cells not classified as EBC in this condition are drawn grey and
            only the EBCs are colored. Defaults to True for ``distance``/``angle`` and
            False for ``azi``/``alt`` (retinotopy is shown for all cells).
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Creates a new figure if omitted.
        cmap, vmin, vmax : optional
            Color scaling overrides. Defaults are feature-aware: for ``angle``, the
            blue-centred split colormap with limits at the 95th percentile of each
            direction (0 pinned to blue); for ``distance``, ``viridis`` over the data
            range.
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

        if feature in self._FEATURE_COL:
            col = f"{condition}_{self._FEATURE_COL[feature]}"
            cbar_label = f"{condition} pref {feature}"
            feat_label = f"pref {feature}"
            is_retino = False
        elif feature in self._RETINO_FEATURES:
            col = feature  # direct column, not condition-prefixed
            cbar_label = feat_label = self._RETINO_FEATURES[feature]
            is_retino = True
        else:
            raise ValueError(
                f"feature must be one of distance/angle/azi/alt, got {feature!r}")

        classified_col = f"{condition}_is_ebc"
        is_angle = feature == "angle"
        if grey_unclassified is None:
            grey_unclassified = not is_retino  # grey non-EBC for EBC features only

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
                scope = "EBCs" if grey_unclassified else "cells"
                title = f"{condition} {scope} - {feat_label}"
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    # -------------------------------------------------------------- regression
    def _resolve_col(self, feature, condition):
        """Map a feature name to a (column, label). Accepts 'distance'/'angle'
        (condition-prefixed), 'azi'/'alt', or a raw column name in the DataFrame."""
        if feature in self._FEATURE_COL:
            return f"{condition}_{self._FEATURE_COL[feature]}", f"{condition} pref {feature}"
        if feature in self._RETINO_FEATURES:
            return feature, self._RETINO_FEATURES[feature]
        if feature in self.df.columns:
            return feature, feature
        raise ValueError(f"unknown feature/column {feature!r}")

    def scatter_regression(self, x="azi", y="angle", condition="light", ebc_only=True,
                           ax=None, color="tab:blue", s=18, annotate=True):
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
            Restrict to ``{condition}`` EBCs (default True).
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

        d = self.df
        if ebc_only:
            d = d[d[f"{condition}_is_ebc"].to_numpy(dtype=bool)]
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
        scope = "EBCs" if ebc_only else "cells"
        ax.set_title(f"{ylab} vs {xlab}  ({condition} {scope}, n={len(xv)})", fontsize=13)
        if annotate:
            ax.text(0.04, 0.96,
                    f"r = {res.rvalue:.2f}\np = {res.pvalue:.2g}\nslope = {res.slope:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        return fig, ax, res
