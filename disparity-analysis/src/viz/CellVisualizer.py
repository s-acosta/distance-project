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

setup_plot_style()


class CellVisualizer:
    """
    Per-cell plots for a disparity (left/right eye) session.

    Examples
    --------
    >>> from src.data.DisparityDevice import DisparityDevice
    >>> from src.viz.CellVisualizer import CellVisualizer
    >>> D = DisparityDevice(mouse_name='SAM045', record_date='260609', experiment_name='goggles_left')
    >>> viz = CellVisualizer(D)
    >>> viz.make_eye_plot(cell=116, eye='left')          # single eye
    >>> viz.make_comparison_plot(cell=348)               # both eyes (if present)
    """

    EYE_COLOR = {'left': 'red', 'right': 'blue'}
    EYE_CMAP = {'left': 'Reds', 'right': 'Blues'}

    def __init__(self, device):
        self.device = device
        self.analyzer = CellAnalyzer(device)

    def _vlimits(self, cell):
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        vmin = np.percentile(self.device.response[idx, :], 1)
        vmax = np.percentile(self.device.response[idx, :], 99)
        return vmin, vmax

    def make_comparison_plot(self, cell):
        """
        Three panels: left-eye trial heatmap, right-eye trial heatmap, and the
        two mean±SEM tuning curves overlaid (red=left, blue=right) with the RF
        pixel position annotated per eye. Requires both eyes to be present.
        """
        if not all(e in self.device.present_eyes for e in ('left', 'right')):
            raise ValueError(
                f"make_comparison_plot needs both eyes; this session has {self.device.present_eyes}. "
                "Use make_eye_plot(cell, eye=...) instead."
            )

        vmin, vmax = self._vlimits(cell)

        left_traces = self.device.get_all_traces(cell, eye='left')
        right_traces = self.device.get_all_traces(cell, eye='right')
        left_mean, left_sem = self.device.get_all_traces(cell, eye='left', mean_trace=True)
        right_mean, right_sem = self.device.get_all_traces(cell, eye='right', mean_trace=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.pcolor(left_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['left'])
        ax.set_xlabel('Time (frames)', fontsize=20)
        ax.set_ylabel('Trial', fontsize=20)
        ax.set_title('Left Eye', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax = axes[1]
        ax.pcolor(right_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['right'])
        ax.set_xlabel('Time (frames)', fontsize=20)
        ax.set_ylabel('Trial', fontsize=20)
        ax.set_title('Right Eye', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax = axes[2]
        left_range = self.device.get_screen_range('left')    # unsigned pixel magnitude
        right_range = self.device.get_screen_range('right')

        left_rf_idx, _ = self.device.find_rf_position(cell, eye='left')
        right_rf_idx, _ = self.device.find_rf_position(cell, eye='right')
        left_rf_px = left_range[left_rf_idx]
        right_rf_px = right_range[right_rf_idx]

        ax.fill_between(left_range, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
        ax.plot(left_range, left_mean, linewidth=2, color='red')
        ax.fill_between(right_range, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
        ax.plot(right_range, right_mean, linewidth=2, color='blue')

        ax.set_xlim(right_range[0], right_range[-1])
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('|Pixels|', fontsize=24)
        ax.set_ylabel('Activity (z-score)', fontsize=20)
        ax.set_title('Mean Trace ± SEM', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if len(ax.lines) >= 2:
            ax.legend(
                handles=[ax.lines[0], ax.lines[1]],
                labels=[f'Left Eye (RF: {-left_rf_px:.1f} px)', f'Right Eye (RF: {right_rf_px:.1f} px)'],
                fontsize=16, loc='best', frameon=False,
            )

        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout()
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

        traces = self.device.get_all_traces(cell, eye=eye)
        mean_trace, sem_trace = self.device.get_all_traces(cell, eye=eye, mean_trace=True)

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
        rf_idx, _ = self.device.find_rf_position(cell, eye=eye)
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

    def save_all_cells(self, out_dir=None, kind='comparison', eye=None, dpi=150):
        """
        Render and save a figure for every cell in the session.

        Parameters
        ----------
        out_dir : str, optional
            Output directory (default: ``device.results_dir``).
        kind : {'comparison', 'eye'}
            'comparison' uses :meth:`make_comparison_plot` (both eyes); 'eye'
            uses :meth:`make_eye_plot` for ``eye`` (defaults to ``device.eye``).
        eye : str, optional
            Eye to use when ``kind == 'eye'``.
        """
        out_dir = out_dir or self.device.results_dir
        os.makedirs(out_dir, exist_ok=True)

        for cell in self.device.suite2p_id:
            cell = int(cell)
            if kind == 'comparison':
                fig = self.make_comparison_plot(cell)
            elif kind == 'eye':
                fig = self.make_eye_plot(cell, eye=eye)
            else:
                raise ValueError("kind must be 'comparison' or 'eye'")

            fig.savefig(os.path.join(out_dir, f'cell_{cell}.png'), dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        print(f'Saved {len(self.device.suite2p_id)} figures to: {out_dir}')
