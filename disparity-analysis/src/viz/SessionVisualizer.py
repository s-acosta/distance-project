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


class SessionVisualizer:
    """
    Session-level summaries for a disparity (left/right eye) session.

    Prints reliable-cell counts (per eye and both) and draws the population
    plots that used to live in the back half of the notebook: ODI histogram,
    per-eye RF histograms, left-vs-right RF scatter, and RF-disparity histogram.

    The per-cell statistics DataFrame (from ``SessionAnalyzer``) is the expensive
    part, so it is computed once and cached. Pass a precomputed ``stats_df`` to
    skip recomputation.

    Examples
    --------
    >>> from src.data.MonocularDevice import MonocularDevice
    >>> from src.analysis.CellAnalysis import SessionAnalyzer
    >>> from src.viz.SessionVisualizer import SessionVisualizer
    >>> D = MonocularDevice(mouse_name='SAM044', record_date='260502', experiment_name='goggles_right-left')
    >>> df = SessionAnalyzer(D).analyze_session_stats()   # compute once
    >>> sv = SessionVisualizer(D, stats_df=df)
    >>> sv.print_summary()
    >>> sv.plot_summary()
    """

    EYE_COLOR = {'left': 'red', 'right': 'blue'}

    # left / right / both-eye marker colors for the field map
    FIELD_COLOR = {'left': 'red', 'right': 'blue', 'both': 'purple'}

    def __init__(self, device, stats_df=None, n_iterations=200):
        self.device = device
        self.analyzer = SessionAnalyzer(device)
        self.n_iterations = n_iterations
        self.stats_df = stats_df
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

    def reliable_sets(self):
        """
        Sets of reliable cell ids. For a both-eye session returns
        right / left / both / right_only / left_only / either / none; for a
        single-eye session returns reliable / none. Always includes 'all'.
        """
        df = self.compute_stats()
        all_cells = set(int(c) for c in df['cell'])

        if self.device.single_eye:
            reliable = set(int(c) for c in df.loc[df['reliable'], 'cell'])
            return {'all': all_cells, 'reliable': reliable, 'none': all_cells - reliable}

        right = set(int(c) for c in df.loc[df['right_reliable'], 'cell'])
        left = set(int(c) for c in df.loc[df['left_reliable'], 'cell'])
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

    def print_summary(self):
        """Print reliable-cell counts for the session (per eye and both)."""
        df = self.compute_stats()
        n = len(df)
        sets = self.reliable_sets()
        line = "=" * 50

        print(line)
        print(f" Session: {self.device.session_name}")
        print(f" Eyes shown: {', '.join(self.device.present_eyes)}")
        print(f" Total cells: {n}")
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

    # ----------------------------------------------------------- draw helpers

    def _draw_odi(self, ax):
        df, sets = self.compute_stats(), self.reliable_sets()
        odis = df.loc[df['cell'].isin(sets['either']), 'ODI'].dropna()
        ax.hist(odis, bins=15, edgecolor='black', alpha=0.8, color='mediumpurple')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('ODI  (<0 left, >0 right)')
        ax.set_ylabel('Number of cells')
        ax.set_title(f'Ocular dominance (either-reliable, n={len(odis)})')

    def _draw_rf_overlay(self, ax):
        """Reliable RF positions across the visual field: left (red, negative px),
        right (blue, positive px) overlaid on one signed pixel axis."""
        df = self.compute_stats()
        left_px = df.loc[df['left_reliable'], 'left_px'].dropna()
        right_px = df.loc[df['right_reliable'], 'right_px'].dropna()
        ax.hist(left_px, bins=np.linspace(-120, 0, 30), color='red', edgecolor='black',
                alpha=0.7, label=f'Left (n={len(left_px)})')
        ax.hist(right_px, bins=np.linspace(0, 120, 30), color='blue', edgecolor='black',
                alpha=0.7, label=f'Right (n={len(right_px)})')
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('RF position (pixels)')
        ax.set_ylabel('Number of cells')
        ax.set_title('Reliable RF positions')
        ax.legend(fontsize=12, frameon=False)

    def _draw_rf_single(self, ax):
        df = self.compute_stats()
        eye = self.device.eye
        px = df.loc[df['reliable'], 'px'].dropna()
        ax.hist(px, bins=30, color=self.EYE_COLOR[eye], edgecolor='black', alpha=0.8)
        ax.set_xlabel('RF position (pixels)')
        ax.set_ylabel('Number of cells')
        ax.set_title(f'{eye.capitalize()}-eye RF (reliable, n={len(px)})')

    def _draw_scatter(self, ax):
        df, sets = self.compute_stats(), self.reliable_sets()
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

    def _draw_difference(self, ax):
        df, sets = self.compute_stats(), self.reliable_sets()
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

    def plot_odi_histogram(self):
        """Histogram of ocular-dominance index for either-reliable cells (both-eye)."""
        self._require_both_eyes("ODI histogram")
        fig, ax = plt.subplots(figsize=(7, 4))
        self._draw_odi(ax)
        plt.tight_layout()
        return fig

    def plot_rf_histograms(self):
        """Per-eye reliable-RF position histograms (one axis per eye for both-eye
        sessions, a single axis for single-eye)."""
        df = self.compute_stats()
        if self.device.single_eye:
            fig, ax = plt.subplots(figsize=(7, 4))
            self._draw_rf_single(ax)
            plt.tight_layout()
            return fig

        left_px = df.loc[df['left_reliable'], 'left_px'].dropna()
        right_px = df.loc[df['right_reliable'], 'right_px'].dropna()
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

    def plot_rf_scatter(self):
        """Left-eye vs right-eye RF position for both-reliable cells, with a fit."""
        self._require_both_eyes("RF scatter")
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_scatter(ax)
        plt.tight_layout()
        return fig

    def plot_rf_difference(self):
        """Histogram of right-minus-left RF position for both-reliable cells."""
        self._require_both_eyes("RF difference")
        fig, ax = plt.subplots(figsize=(7, 4))
        self._draw_difference(ax)
        plt.tight_layout()
        return fig

    def plot_location_map(self, ax=None, cell_size=60, show_unresponsive=True):
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

        Returns
        -------
        matplotlib.figure.Figure
        """
        sets = self.reliable_sets()
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

    def plot_summary(self, print_counts=True):
        """
        Print the reliable-cell summary and draw a single multi-panel figure:
        ODI, RF positions, left-vs-right scatter, and RF disparity (both-eye); or
        the single RF histogram (single-eye). Returns the figure.
        """
        if print_counts:
            self.print_summary()

        if self.device.single_eye:
            fig, ax = plt.subplots(figsize=(7, 4))
            self._draw_rf_single(ax)
            fig.suptitle(self.device.session_name, fontsize=16)
            plt.tight_layout()
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self._draw_odi(axes[0, 0])
        self._draw_scatter(axes[0, 1])
        self._draw_rf_overlay(axes[1, 0])
        self._draw_difference(axes[1, 1])
        fig.suptitle(self.device.session_name, fontsize=18)
        plt.tight_layout()
        return fig
