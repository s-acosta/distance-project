import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")                       # repo root: path_setup, utils
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project\disparity-analysis")    # so `src.*` = disparity's own
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils.helpers import setup_plot_style
from src.analysis.CellAnalysis import SessionAnalyzer

setup_plot_style()

_UNSET = object()   # sentinel: "not passed, use the instance default"


class SessionVisualizer:
    """
    Session-level summaries for a disparity (left/right eye) session.

    Prints reliable-cell counts (per eye and both) and draws the population
    plots that used to live in the back half of the notebook: ODI histogram,
    per-eye RF histograms, left-vs-right RF scatter, and RF-disparity histogram.

    The per-cell statistics DataFrame (from ``SessionAnalyzer``) is the expensive
    part, so it is computed once and cached. Pass a precomputed ``stats_df`` to
    skip recomputation.

    ``corr_threshold`` sets a default ``mean_corr`` floor applied on top of the
    ``reliable`` flag everywhere a reliable-cell set is used (summaries and all
    population plots). Every method that consumes it also takes its own
    ``corr_threshold`` to override the default for that one call — pass ``None``
    there to disable filtering just for that call.

    Examples
    --------
    >>> from src.data.MonocularDevice import MonocularDevice
    >>> from src.analysis.CellAnalysis import SessionAnalyzer
    >>> from src.viz.SessionVisualizer import SessionVisualizer
    >>> D = MonocularDevice(mouse_name='SAM044', record_date='260502', experiment_name='goggles_right-left')
    >>> df = SessionAnalyzer(D).analyze_session_stats()   # compute once
    >>> sv = SessionVisualizer(D, stats_df=df, corr_threshold=0.75)
    >>> sv.print_summary()
    >>> sv.plot_summary()
    >>> sv.plot_location_map(corr_threshold=0.5)   # looser floor, this call only
    """

    EYE_COLOR = {'left': 'red', 'right': 'blue'}

    # left / right / both-eye marker colors for the field map
    FIELD_COLOR = {'left': 'red', 'right': 'blue', 'both': 'purple'}

    def __init__(self, device, stats_df=None, n_iterations=200, corr_threshold=None):
        self.device = device
        self.analyzer = SessionAnalyzer(device)
        self.n_iterations = n_iterations
        self.stats_df = stats_df
        self.corr_threshold = corr_threshold
        self._suite2p = None
        setup_plot_style()

    # ----------------------------------------------------------------- stats

    def compute_stats(self, force=False):
        """Return the per-cell stats DataFrame, computing+caching on first use."""
        if self.stats_df is None or force:
            print(f"Computing session stats ({len(self.device.suite2p_id)} cells, "
                  f"n_iterations={self.n_iterations})...")
            self.stats_df = self.analyzer.analyze_session_stats(n_iterations=self.n_iterations)
        return self.stats_df

    def _resolve_threshold(self, corr_threshold):
        """``_UNSET`` -> the instance default; anything else (incl. ``None``) passes through."""
        return self.corr_threshold if corr_threshold is _UNSET else corr_threshold

    def _reliable_mask(self, df, prefix, corr_threshold):
        """Boolean mask: ``{prefix}reliable``, optionally AND ``{prefix}mean_corr >= corr_threshold``."""
        mask = df[f'{prefix}reliable'].astype(bool)
        if corr_threshold is not None:
            mask &= df[f'{prefix}mean_corr'] >= corr_threshold
        return mask

    def reliable_sets(self, corr_threshold=_UNSET):
        """
        Sets of reliable cell ids. For a both-eye session returns
        right / left / both / right_only / left_only / either / none; for a
        single-eye session returns reliable / none. Always includes 'all'.

        ``corr_threshold`` adds a ``mean_corr`` floor on top of each reliability
        flag (defaults to the instance's ``corr_threshold``; pass an explicit
        value, or ``None`` to disable filtering, to override it for this call).
        """
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        all_cells = set(int(c) for c in df['cell'])

        if self.device.single_eye:
            reliable = set(int(c) for c in df.loc[self._reliable_mask(df, '', corr_threshold), 'cell'])
            return {'all': all_cells, 'reliable': reliable, 'none': all_cells - reliable}

        right = set(int(c) for c in df.loc[self._reliable_mask(df, 'right_', corr_threshold), 'cell'])
        left = set(int(c) for c in df.loc[self._reliable_mask(df, 'left_', corr_threshold), 'cell'])
        both = right & left
        return {
            'all': all_cells,
            'right': right, 'left': left, 'both': both,
            'right_only': right - both, 'left_only': left - both,
            'either': right | left, 'none': all_cells - (right | left),
        }

    # ------------------------------------------- suite2p field-of-view helpers

    def _load_suite2p(self):
        """Lazy-load the session's suite2p ``stat`` + ``ops`` (cell footprints and
        the reference FOV image), cached after first use."""
        if self._suite2p is None:
            plane = os.path.join(self.device.suite2p_dir, 'plane0')
            self._suite2p = {
                'stat': np.load(os.path.join(plane, 'stat.npy'), allow_pickle=True),
                'ops': np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item(),
            }
        return self._suite2p

    def _fov_image(self):
        """Background field-of-view image (registration reference, mean fallback)."""
        ops = self._load_suite2p()['ops']
        return ops['refImg'] if 'refImg' in ops else ops['meanImg']

    def _cell_xy(self, cell):
        """(x, y) pixel centroid of a cell's footprint (non-overlapping pixels)."""
        stat = self._load_suite2p()['stat'][cell]
        keep = ~stat['overlap']
        if not keep.any():                      # cell buried under overlaps
            keep = np.ones_like(stat['overlap'], dtype=bool)
        return stat['xpix'][keep].mean(), stat['ypix'][keep].mean()

    @staticmethod
    def _pct(k, n):
        return f"{100 * k / n:5.1f}%" if n else "  n/a"

    def print_summary(self, corr_threshold=_UNSET):
        """Print reliable-cell counts for the session (per eye and both).

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering).
        """
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        n = len(df)
        sets = self.reliable_sets(corr_threshold)
        line = "=" * 50

        print(line)
        print(f" Session: {self.device.session_name}")
        print(f" Eyes shown: {', '.join(self.device.present_eyes)}")
        print(f" Total cells: {n}")
        if corr_threshold is not None:
            print(f" mean_corr floor: {corr_threshold}")
        print("-" * 50)
        if self.device.single_eye:
            r = len(sets['reliable'])
            print(f" {self.device.eye.capitalize()+'-eye reliable':22s}: {r:4d}  ({self._pct(r, n)})")
            print(f" {'Not reliable':22s}: {len(sets['none']):4d}  ({self._pct(len(sets['none']), n)})")
        else:
            print(f" {'Right reliable':22s}: {len(sets['right']):4d}  ({self._pct(len(sets['right']), n)})")
            print(f" {'Left reliable':22s}: {len(sets['left']):4d}  ({self._pct(len(sets['left']), n)})")
            print(f" {'Both eyes reliable':22s}: {len(sets['both']):4d}  ({self._pct(len(sets['both']), n)})")
            print("-" * 50)
            print(f" {'Right-only':22s}: {len(sets['right_only']):4d}")
            print(f" {'Left-only':22s}: {len(sets['left_only']):4d}")
            print(f" {'Either eye':22s}: {len(sets['either']):4d}  ({self._pct(len(sets['either']), n)})")
            print(f" {'Neither':22s}: {len(sets['none']):4d}  ({self._pct(len(sets['none']), n)})")
        print(line)
        return sets

    def print_params(self):
        """
        Print this session's stimulus parameters as logged in its .mat file (the
        ``opts`` block the presentation script was run with), plus the per-eye
        deg->pixel calibration fits when present — everything needed to rerun
        the same stimulus with matching settings.
        """
        params = self.device.parameters
        line = "=" * 50

        print(line)
        print(f" Session: {self.device.session_name}")
        print(" Stimulus parameters:")
        print("-" * 50)
        for key in sorted(params):
            print(f" {key:22s}: {params[key]}")

        left_fit = getattr(self.device, 'left_fit', None)
        right_fit = getattr(self.device, 'right_fit', None)
        if left_fit is not None or right_fit is not None:
            print("-" * 50)
            print(" Eye calibration fits (deg -> px, [slope, intercept]):")
            print(f" {'left_fit':22s}: {left_fit}")
            print(f" {'right_fit':22s}: {right_fit}")
        print(line)
        return params

    # ----------------------------------------------------------- draw helpers

    def _draw_odi(self, ax, corr_threshold=_UNSET):
        df, sets = self.compute_stats(), self.reliable_sets(corr_threshold)
        odis = df.loc[df['cell'].isin(sets['either']), 'ODI'].dropna()
        ax.hist(odis, bins=15, edgecolor='black', alpha=0.8, color='mediumpurple')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('ODI  (<0 left, >0 right)')
        ax.set_ylabel('Number of cells')
        ax.set_title(f'Ocular dominance (either-reliable, n={len(odis)})')

    def _draw_rf_overlay(self, ax, corr_threshold=_UNSET):
        """Reliable RF positions across the visual field: left (red, negative px),
        right (blue, positive px) overlaid on one signed pixel axis."""
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        left_px = df.loc[self._reliable_mask(df, 'left_', corr_threshold), 'left_px'].dropna()
        right_px = df.loc[self._reliable_mask(df, 'right_', corr_threshold), 'right_px'].dropna()
        ax.hist(left_px, bins=np.linspace(-120, 0, 30), color='red', edgecolor='black',
                alpha=0.7, label=f'Left (n={len(left_px)})')
        ax.hist(right_px, bins=np.linspace(0, 120, 30), color='blue', edgecolor='black',
                alpha=0.7, label=f'Right (n={len(right_px)})')
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('RF position (pixels)')
        ax.set_ylabel('Number of cells')
        ax.set_title('Reliable RF positions')
        ax.legend(fontsize=12, frameon=False)

    def _draw_rf_single(self, ax, corr_threshold=_UNSET):
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        eye = self.device.eye
        px = df.loc[self._reliable_mask(df, '', corr_threshold), 'px'].dropna()
        ax.hist(px, bins=30, color=self.EYE_COLOR[eye], edgecolor='black', alpha=0.8)
        ax.set_xlabel('RF position (pixels)')
        ax.set_ylabel('Number of cells')
        ax.set_title(f'{eye.capitalize()}-eye RF (reliable, n={len(px)})')

    def _draw_scatter(self, ax, corr_threshold=_UNSET):
        df, sets = self.compute_stats(), self.reliable_sets(corr_threshold)
        sub = df[df['cell'].isin(sets['both'])][['left_px', 'right_px']].dropna()
        ax.scatter(sub['left_px'], sub['right_px'], edgecolor='black', alpha=0.8)
        if len(sub) >= 2:
            slope, intercept, r, p, _ = stats.linregress(sub['left_px'], sub['right_px'])
            x = np.linspace(sub['left_px'].min(), sub['left_px'].max(), 100)
            ax.plot(x, slope * x + intercept, color='red', linewidth=2,
                    label=f'slope={slope:.2f}, r²={r**2:.2f}, p={p:.3f}')
            ax.legend(fontsize=12, frameon=False)
        ax.set_xlabel('Left-eye RF (px)')
        ax.set_ylabel('Right-eye RF (px)')
        ax.set_title(f'Left vs Right RF (both-reliable, n={len(sub)})')

    def _draw_difference(self, ax, corr_threshold=_UNSET):
        df, sets = self.compute_stats(), self.reliable_sets(corr_threshold)
        sub = df[df['cell'].isin(sets['both'])][['right_px', 'left_px']].dropna().copy()
        sub['px_diff'] = sub['right_px'] - sub['left_px']
        ax.hist(sub['px_diff'], bins=30, edgecolor='black', alpha=0.8, color='teal')
        if len(sub):
            med = float(sub['px_diff'].median())
            ax.axvline(med, color='red', linestyle='--', linewidth=1.5, label=f'median={med:.1f}')
            ax.legend(fontsize=12)
        ax.set_xlabel('RF difference (right - left, px)')
        ax.set_ylabel('Number of cells')
        ax.set_title(f'RF disparity (both-reliable, n={len(sub)})')

    def _require_both_eyes(self, what):
        if self.device.single_eye:
            raise ValueError(f"{what} needs both eyes; this session shows only {self.device.present_eyes}.")

    # --------------------------------------------------------- public plots

    def plot_odi_histogram(self, corr_threshold=_UNSET):
        """Histogram of ocular-dominance index for either-reliable cells (both-eye).

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering)."""
        self._require_both_eyes("ODI histogram")
        fig, ax = plt.subplots(figsize=(7, 4))
        self._draw_odi(ax, corr_threshold)
        plt.tight_layout()
        return fig

    def plot_rf_histograms(self, corr_threshold=_UNSET):
        """Per-eye reliable-RF position histograms (one axis per eye for both-eye
        sessions, a single axis for single-eye).

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering)."""
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        if self.device.single_eye:
            fig, ax = plt.subplots(figsize=(7, 4))
            self._draw_rf_single(ax, corr_threshold)
            plt.tight_layout()
            return fig

        left_px = df.loc[self._reliable_mask(df, 'left_', corr_threshold), 'left_px'].dropna()
        right_px = df.loc[self._reliable_mask(df, 'right_', corr_threshold), 'right_px'].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes[0].hist(left_px, bins=np.linspace(-120, 0, 30), color='red', edgecolor='black', alpha=0.8)
        axes[0].set_title(f'Left-eye (n={len(left_px)})')
        axes[0].set_xlabel('Pixels')
        axes[0].set_ylabel('Number of cells')
        axes[1].hist(right_px, bins=np.linspace(0, 120, 30), color='blue', edgecolor='black', alpha=0.8)
        axes[1].set_title(f'Right-eye (n={len(right_px)})')
        axes[1].set_xlabel('Pixels')
        plt.tight_layout()
        return fig

    def plot_rf_scatter(self, corr_threshold=_UNSET):
        """Left-eye vs right-eye RF position for both-reliable cells, with a fit.

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering)."""
        self._require_both_eyes("RF scatter")
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_scatter(ax, corr_threshold)
        plt.tight_layout()
        return fig

    def plot_rf_difference(self, corr_threshold=_UNSET):
        """Histogram of right-minus-left RF position for both-reliable cells.

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering)."""
        self._require_both_eyes("RF difference")
        fig, ax = plt.subplots(figsize=(7, 4))
        self._draw_difference(ax, corr_threshold)
        plt.tight_layout()
        return fig

    def plot_corr_scatter(self, corr_threshold=_UNSET, cell_size=60):
        """
        Every cell's split-half reliability, left-eye mean corr (y) vs
        right-eye mean corr (x), colored by which eye(s) currently count as
        reliable — same classification and palette as :meth:`plot_location_map`:
        left (red), right (blue), both (purple), neither (gray). Dashed
        reference lines mark ``corr_threshold`` on each axis, so you can see
        the whole population's spread relative to the current cutoff and how
        much margin cells have around it.

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering and hides the reference
        lines).
        """
        self._require_both_eyes("corr scatter")
        corr_threshold = self._resolve_threshold(corr_threshold)
        df = self.compute_stats()
        sets = self.reliable_sets(corr_threshold)

        fig, ax = plt.subplots(figsize=(7, 7))

        groups = [
            ('left', sets['left_only'], self.FIELD_COLOR['left']),
            ('right', sets['right_only'], self.FIELD_COLOR['right']),
            ('both', sets['both'], self.FIELD_COLOR['both']),
            ('neither', sets['none'], 'lightgray'),
        ]
        for label, cells, color in groups:
            sub = df[df['cell'].isin(cells)]
            faint = label == 'neither'
            ax.scatter(sub['right_mean_corr'], sub['left_mean_corr'], c=color,
                       edgecolor='black', linewidth=0.4,
                       s=cell_size * 0.5 if faint else cell_size,
                       alpha=0.5 if faint else 0.85,
                       label=f'{label} (n={len(cells)})')

        if corr_threshold is not None:
            ax.axvline(corr_threshold, color='gray', linestyle='--', linewidth=1)
            ax.axhline(corr_threshold, color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel('Right-eye mean corr')
        ax.set_ylabel('Left-eye mean corr')
        ax.set_title(f'{self.device.session_name} — split-half reliability by eye')
        ax.legend(loc='best', frameon=False, fontsize=11)
        plt.tight_layout()
        return fig

    def plot_location_map(self, ax=None, cell_size=60, show_unresponsive=True, corr_threshold=_UNSET):
        """
        Scatter accepted cells on the imaging field of view, colored by which eye
        reliably drives them: left (red), right (blue), both (purple). Mirrors
        radial ``SessionVisualizer.plot_location_map``. Single-eye sessions color
        reliable cells in that eye's color. Cells reliable to neither eye are
        drawn small and gray when ``show_unresponsive`` is set.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. A new 9x9 figure is created when omitted.
        cell_size : int, optional
            Marker size for responsive cells (unresponsive cells are smaller).
        show_unresponsive : bool, optional
            Whether to also plot cells reliable to neither eye. Default True.
        corr_threshold : float, optional
            Overrides the instance's default mean_corr floor for this call only
            (``None`` disables filtering).

        Returns
        -------
        matplotlib.figure.Figure
        """
        sets = self.reliable_sets(corr_threshold)
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        bg = self._fov_image()
        ax.imshow(bg, cmap='gray', origin='upper',
                  vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99),
                  extent=[0, bg.shape[1], bg.shape[0], 0])

        if self.device.single_eye:
            groups = [(self.device.eye, sets['reliable'], self.FIELD_COLOR[self.device.eye])]
        else:
            groups = [
                ('left', sets['left_only'], self.FIELD_COLOR['left']),
                ('right', sets['right_only'], self.FIELD_COLOR['right']),
                ('both', sets['both'], self.FIELD_COLOR['both']),
            ]
        if show_unresponsive:
            groups.append(('none', sets['none'], 'lightgray'))

        for label, cells, color in groups:
            if cells:
                xy = np.array([self._cell_xy(c) for c in cells])
                xs, ys = xy[:, 0], xy[:, 1]
            else:
                xs, ys = [], []
            faint = label == 'none'
            ax.scatter(xs, ys, c=color, edgecolor='black', linewidth=0.4,
                       s=cell_size * 0.35 if faint else cell_size,
                       alpha=0.5 if faint else 0.95,
                       label=f'{label} (n={len(cells)})')

        ax.invert_xaxis()
        ax.invert_yaxis()                       # match radial FOV orientation
        ax.axis('off')
        ax.set_title(f'{self.device.session_name} — cells in field')
        ax.legend(loc='upper right', frameon=True, fontsize=11)
        plt.tight_layout()
        return fig

    def plot_summary(self, print_counts=True, corr_threshold=_UNSET):
        """
        Print the reliable-cell summary and draw a single multi-panel figure:
        ODI, RF positions, left-vs-right scatter, and RF disparity (both-eye); or
        the single RF histogram (single-eye). Returns the figure.

        ``corr_threshold`` overrides the instance's default mean_corr floor for
        this call only (``None`` disables filtering).
        """
        if print_counts:
            self.print_summary(corr_threshold)

        if self.device.single_eye:
            fig, ax = plt.subplots(figsize=(7, 4))
            self._draw_rf_single(ax, corr_threshold)
            fig.suptitle(self.device.session_name, fontsize=16)
            plt.tight_layout()
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self._draw_odi(axes[0, 0], corr_threshold)
        self._draw_scatter(axes[0, 1], corr_threshold)
        self._draw_rf_overlay(axes[1, 0], corr_threshold)
        self._draw_difference(axes[1, 1], corr_threshold)
        fig.suptitle(self.device.session_name, fontsize=18)
        plt.tight_layout()
        return fig
