import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")                       # repo root: path_setup, utils
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project\disparity-analysis")    # so `src.*` = disparity's own
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import setup_plot_style
from src.analysis.CellAnalysis import CellAnalyzer
from src.viz._base import CellVizBase

setup_plot_style()


class CellVisualizer(CellVizBase):
    """
    Per-cell plots for a disparity (left/right eye) session.

    Pass a precomputed ``stats_df`` (from ``SessionAnalyzer.analyze_session_stats``)
    to reuse its cached ``left_mean_corr``/``right_mean_corr`` instead of
    recomputing split-half reliability per cell; omitted, it's computed live.

    Examples
    --------
    >>> from src.data.MonocularDevice import MonocularDevice
    >>> from src.viz.CellVisualizer import CellVisualizer
    >>> D = MonocularDevice(mouse_name='SAM045', record_date='260609', experiment_name='goggles_left')
    >>> viz = CellVisualizer(D)
    >>> viz.make_eye_plot(cell=116, eye='left')          # single eye
    >>> viz.make_comparison_plot(cell=348)               # both eyes (if present)
    """

    EYE_COLOR = {'left': 'red', 'right': 'blue'}
    EYE_CMAP = {'left': 'Reds', 'right': 'Blues'}

    def __init__(self, device, stats_df=None, elevation=None):
        super().__init__(device)
        self.analyzer = CellAnalyzer(device)
        self.stats_df = stats_df
        self.elevation = elevation
        if getattr(device, 'has_elevation', False) and elevation is None:
            print(f"[CellVisualizer] warning: {device.session_name} swept elevations "
                  f"{device.elevations.tolist()} px, but elevation= wasn't set -- every trace/RF "
                  "below will pool ALL of them together. Pass elevation=<value> (e.g. 120.0) to "
                  "restrict to one condition.")

    def _cond(self):
        """The elevation condition to forward to trace/RF calls, if one was set."""
        return {} if self.elevation is None else {'elevation': self.elevation}

    def _mean_corr(self, cell, eye):
        """That eye's mean split-half corr for a cell: from ``stats_df`` if cached, else live."""
        if self.stats_df is not None:
            row = self.stats_df.loc[self.stats_df['cell'] == cell]
            if len(row):
                return float(row[f'{eye}_mean_corr'].iloc[0])
        return self.analyzer.analyze_cell_reliability(cell, eye=eye, **self._cond())['mean_corr']

    def make_comparison_plot(self, cell, fig=None):
        """
        Four panels: left-eye trial heatmap, right-eye trial heatmap, and the two
        mean±SEM tuning curves overlaid (red=left, blue=right) vs pixels and vs
        angle, with the RF position annotated per eye. Requires both eyes.

        Pass ``fig`` (a Figure or SubFigure) to draw into it instead of creating a
        new figure (used by the combined monocular+binocular view).
        """
        if not all(e in self.device.present_eyes for e in ('left', 'right')):
            raise ValueError(
                f"make_comparison_plot needs both eyes; this session has {self.device.present_eyes}. "
                "Use make_eye_plot(cell, eye=...) instead."
            )

        vmin, vmax = self._vlimits(cell)
        cond = self._cond()

        left_traces = self.device.get_all_traces(cell, eye='left', **cond)
        right_traces = self.device.get_all_traces(cell, eye='right', **cond)
        left_mean, left_sem = self.device.get_all_traces(cell, eye='left', mean_trace=True, **cond)
        right_mean, right_sem = self.device.get_all_traces(cell, eye='right', mean_trace=True, **cond)

        left_corr = self._mean_corr(cell, 'left')
        right_corr = self._mean_corr(cell, 'right')

        own_fig = fig is None
        if own_fig:
            fig = plt.figure(figsize=(22, 5))
        axes = fig.subplots(1, 4)

        ax = axes[0]
        ax.pcolor(left_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['left'])
        ax.set_xlabel('Time (frames)', fontsize=20)
        ax.set_ylabel('Trial', fontsize=20)
        ax.set_title(f'Left Eye (corr: {left_corr:.2f})', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax = axes[1]
        ax.pcolor(right_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['right'])
        ax.set_xlabel('Time (frames)', fontsize=20)
        ax.set_ylabel('Trial', fontsize=20)
        ax.set_title(f'Right Eye (corr: {right_corr:.2f})', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)

        left_rf_idx, _ = self.device.find_rf_position(cell, eye='left', **cond)
        right_rf_idx, _ = self.device.find_rf_position(cell, eye='right', **cond)

        # --- panel 3: tuning vs screen pixels (left on the negative axis) ---
        ax = axes[2]
        left_px = self.device.get_signed_screen_range('left')     # left -> negative side
        right_px = self.device.get_signed_screen_range('right')   # right -> positive side

        ax.fill_between(left_px, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
        ax.plot(left_px, left_mean, linewidth=2, color='red')
        ax.fill_between(right_px, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
        ax.plot(right_px, right_mean, linewidth=2, color='blue')

        ax.set_xlim(left_px.min(), right_px.max())
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('Pixels', fontsize=24)
        ax.set_ylabel('Activity (z-score)', fontsize=20)
        ax.set_title('Mean Trace ± SEM (px)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(
            handles=[ax.lines[0], ax.lines[1]],
            labels=[f'Left Eye (RF: {left_px[left_rf_idx]:.1f} px)',
                    f'Right Eye (RF: {right_px[right_rf_idx]:.1f} px)'],
            fontsize=16, loc='best', frameon=False,
        )

        # --- panel 4: tuning vs visual angle (shared axis for both eyes) ---
        ax = axes[3]
        if self.device.has_angle_range:
            ang = self.device.get_angle_range()

            ax.fill_between(ang, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
            ax.plot(ang, left_mean, linewidth=2, color='red')
            ax.fill_between(ang, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
            ax.plot(ang, right_mean, linewidth=2, color='blue')

            ax.set_xlim(ang[0], ang[-1])
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel('Angle (deg)', fontsize=24)
            ax.set_ylabel('Activity (z-score)', fontsize=20)
            ax.set_title('Mean Trace ± SEM (deg)', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.legend(
                handles=[ax.lines[0], ax.lines[1]],
                labels=[f'Left Eye (RF: {ang[left_rf_idx]:.1f}°)',
                        f'Right Eye (RF: {ang[right_rf_idx]:.1f}°)'],
                fontsize=16, loc='best', frameon=False,
            )
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No angle calibration\n(legacy pixel-format session)',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)

        if own_fig:
            fig.suptitle(f'Cell {cell}', fontsize=20)
            fig.tight_layout()
        return fig

    def make_elevation_comparison_plot(self, cell, fig=None):
        """
        Every swept elevation for one cell, both eyes, in a single figure.

        Panels 1/2: left/right trial heatmaps, with each elevation's trials
        stacked as its own block -- ascending bottom-to-top, so the highest
        elevation's trials sit at the top (matplotlib's default pcolor
        orientation) and the lowest at the bottom. Blocks are separated by a
        white line and y-tick-labeled with their elevation.

        Panel 3: mean +/- SEM tuning vs screen pixels, one line per elevation
        per eye, shaded light (lowest elevation) to dark (highest) -- red
        shades for left, blue for right.

        Requires both eyes and a session recorded with an elevation sweep
        (``device.has_elevation``). Ignores ``self.elevation`` if set on this
        instance -- this plot is specifically the cross-elevation view, so it
        always uses every elevation the device swept.

        Pass ``fig`` (a Figure or SubFigure) to draw into it instead of
        creating a new figure.
        """
        if not all(e in self.device.present_eyes for e in ('left', 'right')):
            raise ValueError(
                f"make_elevation_comparison_plot needs both eyes; this session has "
                f"{self.device.present_eyes}."
            )
        if not self.device.has_elevation:
            raise ValueError(
                f"{self.device.session_name} has no elevation sweep -- there's only one "
                "condition, so there's nothing to compare across elevations. Use "
                "make_comparison_plot() instead."
            )

        elevations = np.sort(self.device.elevations)
        vmin, vmax = self._vlimits(cell)

        own_fig = fig is None
        if own_fig:
            fig = plt.figure(figsize=(20, 6))
        axes = fig.subplots(1, 3)

        def _stacked_heatmap(ax, eye):
            blocks = [self.device.get_all_traces(cell, eye=eye, elevation=float(e)) for e in elevations]
            ax.pcolor(np.vstack(blocks), vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP[eye])

            boundaries = np.cumsum([0] + [len(b) for b in blocks])
            for b in boundaries[1:-1]:
                ax.axhline(b, color='white', linewidth=1.5)
            centers = (boundaries[:-1] + boundaries[1:]) / 2
            ax.set_yticks(centers)
            ax.set_yticklabels([f'{e:.0f}px' for e in elevations], fontsize=14)
            ax.set_xlabel('Time (frames)', fontsize=20)
            ax.set_title(f'{eye.capitalize()} Eye — all elevations', fontsize=22)
            ax.tick_params(axis='x', which='major', labelsize=16)

        _stacked_heatmap(axes[0], 'left')
        _stacked_heatmap(axes[1], 'right')

        # --- panel 3: mean trace vs pixels, one shaded line per (eye, elevation) ---
        ax = axes[2]
        left_px = self.device.get_signed_screen_range('left')
        right_px = self.device.get_signed_screen_range('right')
        cmap_left, cmap_right = plt.get_cmap(self.EYE_CMAP['left']), plt.get_cmap(self.EYE_CMAP['right'])
        shades = np.linspace(0.35, 0.95, len(elevations))   # skip near-white low end for visibility

        for shade, elev in zip(shades, elevations):
            left_mean, _ = self.device.get_all_traces(cell, eye='left', elevation=float(elev), mean_trace=True)
            right_mean, _ = self.device.get_all_traces(cell, eye='right', elevation=float(elev), mean_trace=True)
            ax.plot(left_px, left_mean, color=cmap_left(shade), linewidth=2, label=f'L {elev:.0f}px')
            ax.plot(right_px, right_mean, color=cmap_right(shade), linewidth=2, label=f'R {elev:.0f}px')

        ax.set_xlim(left_px.min(), right_px.max())
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('Pixels', fontsize=20)
        ax.set_ylabel('Activity (z-score)', fontsize=18)
        ax.set_title('Mean Trace by elevation', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=9, ncol=2, frameon=False, loc='best')

        if own_fig:
            fig.suptitle(f'Cell {cell} — {self.device.session_name}', fontsize=20)
            fig.tight_layout()
        return fig

    def make_eye_plot(self, cell, eye=None):
        """
        Two panels for a single eye: trial heatmap and the mean±SEM tuning curve
        with the RF pixel position annotated. ``eye`` defaults to the session's
        eye (``device.eye``).
        """
        if eye is None:
            eye = self.device.eye
        if eye is None:
            raise ValueError("Specify eye='left' or eye='right' for a both-eye session.")

        vmin, vmax = self._vlimits(cell)
        cond = self._cond()

        traces = self.device.get_all_traces(cell, eye=eye, **cond)
        mean_trace, sem_trace = self.device.get_all_traces(cell, eye=eye, mean_trace=True, **cond)

        color = self.EYE_COLOR[eye]
        cmap = self.EYE_CMAP[eye]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        ax = axes[0]
        ax.pcolor(traces, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel('Time (frames)', fontsize=20)
        ax.set_ylabel('Trial', fontsize=20)
        ax.set_title(f'{eye.capitalize()} Eye', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax = axes[1]
        rng = self.device.get_screen_range(eye)
        rf_idx, _ = self.device.find_rf_position(cell, eye=eye, **cond)
        rf_px = rng[rf_idx]

        ax.fill_between(rng, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=color)
        ax.plot(rng, mean_trace, linewidth=2, color=color)
        ax.set_xlim(rng[0], rng[-1])
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('|Pixels|', fontsize=24)
        ax.set_ylabel('Activity (z-score)', fontsize=20)
        ax.set_title('Mean Trace ± SEM', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend([ax.lines[0]], [f'{eye.capitalize()} Eye (RF: {rf_px:.1f} px)'],
                  fontsize=16, loc='best', frameon=False)

        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout()
        return fig

    def save_all_cells(self, out_dir=None, kind='comparison', eye=None, dpi=150, cells=None):
        """
        Render and save a figure for every cell in the session, or a specific
        subset.

        Parameters
        ----------
        out_dir : str, optional
            Output directory (default: ``device.results_dir``).
        kind : {'comparison', 'eye'}
            'comparison' uses :meth:`make_comparison_plot` (both eyes); 'eye'
            uses :meth:`make_eye_plot` for ``eye`` (defaults to ``device.eye``).
        eye : str, optional
            Eye to use when ``kind == 'eye'``.
        cells : iterable of int, optional
            Which cells to render (e.g. a reliable-cell set from
            ``SessionVisualizer.reliable_sets()``). Defaults to every cell in
            the session (``device.suite2p_id``).
        """
        out_dir = out_dir or self.device.results_dir
        os.makedirs(out_dir, exist_ok=True)
        cells = self.device.suite2p_id if cells is None else sorted(cells)

        for cell in cells:
            cell = int(cell)
            if kind == 'comparison':
                fig = self.make_comparison_plot(cell)
            elif kind == 'eye':
                fig = self.make_eye_plot(cell, eye=eye)
            else:
                raise ValueError("kind must be 'comparison' or 'eye'")

            fig.savefig(os.path.join(out_dir, f'cell_{cell}.png'), dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        print(f'Saved {len(cells)} figures to: {out_dir}')
