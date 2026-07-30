import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.helpers import setup_plot_style
from src.viz._base import CellVizBase, distance_palette

setup_plot_style()


class DisparityCellVisualizer(CellVizBase):
    """
    Per-cell plots for a binocular disparity session.

    Both eyes are shown at once, so responses are conditioned on disparity, and
    everything is drawn in purple (both eyes). Requires a ``DisparityDevice``.

    Examples
    --------
    >>> from src.data.DisparityDevice import DisparityDevice
    >>> from src.viz.DisparityCellVisualizer import DisparityCellVisualizer
    >>> D = DisparityDevice(mouse_name='SAM045', record_date='260713',
    ...                         experiment_name='goggles-disparity')
    >>> DisparityCellVisualizer(D).make_disparity_plot(cell=8)
    """

    def _disparity_colors(self):
        """One distinct color per disparity, matching the distance-analysis palette."""
        return distance_palette(self.device.n_disparities)

    def _disparity_cmaps(self):
        """A white -> color colormap per disparity (heatmap intensity ramp)."""
        return [LinearSegmentedColormap.from_list(f'disp{i}', ['white', c])
                for i, c in enumerate(self._disparity_colors())]

    def make_disparity_plot(self, cell, units='deg', fig=None):
        """
        Two wide rows: the top row is one raw trial heatmap per disparity (in that
        disparity's color intensity); the bottom row spans the same width with
        three summaries: mean vs right-eye position (the fixed/absolute reference),
        mean vs left-eye position (disparity-shifted), and a boxplot of the
        per-trial peak amplitude by disparity.

        Parameters
        ----------
        units : {'deg', 'px'}
            x-axis units for the two mean overlays: visual angle (default) or
            screen pixels (right eye fixed, left eye disparity-aware). The
            heatmaps and the boxplot are unaffected.
        fig : Figure or SubFigure, optional
            Draw into this instead of creating a new figure (used by the combined
            monocular+binocular view).
        """
        if units not in ('deg', 'px'):
            raise ValueError("units must be 'deg' or 'px'")

        vmin, vmax = self._vlimits(cell)
        disparities = self.device.disparities          # sorted ascending
        colors = self._disparity_colors()
        cmaps = self._disparity_cmaps()
        angle = self.device.get_angle_range()
        n = len(disparities)

        # mean +/- sem per disparity (shared by the two overlays)
        means = {d: self.device.get_all_traces(cell, disparity=d, mean_trace=True) for d in disparities}
        ylo = min((m - s).min() for m, s in means.values())
        yhi = max((m + s).max() for m, s in means.values())
        pad = 0.05 * (yhi - ylo)
        ylo, yhi = ylo - pad, yhi + pad

        # top row: n heatmaps (3 cols each); bottom row: 3 summaries (n cols each)
        ncols = 3 * n
        own_fig = fig is None
        if own_fig:
            fig = plt.figure(figsize=(4.8 * n, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, ncols, height_ratios=[1, 1.15])

        # --- top row: one colored raw-trace heatmap per disparity ---
        for i, (d, cmap) in enumerate(zip(disparities, cmaps)):
            ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
            ax.pcolor(self.device.get_all_traces(cell, disparity=d),
                      vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f'Disparity {d:+d}°', fontsize=18)
            ax.set_xlabel('Time (frames)', fontsize=13)
            if i == 0:
                ax.set_ylabel('Trial', fontsize=14)

        # --- bottom row: three full-width summaries ---
        ax_right = fig.add_subplot(gs[1, 0:n])
        ax_left = fig.add_subplot(gs[1, n:2 * n])
        ax_box = fig.add_subplot(gs[1, 2 * n:3 * n])

        # x-values per eye: angle, or the eye's screen pixels (left is disparity-aware)
        def x_right(d):
            return self.device.get_px_range('right', d) if units == 'px' else angle
        def x_left(d):
            return self.device.get_px_range('left', d) if units == 'px' else angle - d

        for d, color in zip(disparities, colors):
            m, s = means[d]
            xr, xl = x_right(d), x_left(d)
            ax_right.plot(xr, m, color=color, linewidth=2, label=f'{d:+d}°')
            ax_right.fill_between(xr, m - s, m + s, color=color, alpha=0.12)
            ax_left.plot(xl, m, color=color, linewidth=2, label=f'{d:+d}°')
            ax_left.fill_between(xl, m - s, m + s, color=color, alpha=0.12)
        unit_lbl = 'pixel' if units == 'px' else 'angle (deg)'
        for ax, title, xlabel in ((ax_right, f'Mean vs right-eye {unit_lbl} (absolute)', f'Right-eye {unit_lbl}'),
                                  (ax_left, f'Mean vs left-eye {unit_lbl}', f'Left-eye {unit_lbl}')):
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=13)
            ax.set_ylabel('Activity (z)', fontsize=13)
            ax.set_ylim(ylo, yhi)
        ax_right.legend(title='disp', fontsize=9, title_fontsize=10, frameon=False)

        # peak-amplitude boxplot per disparity (distance-plot style)
        rf_values = [self.device.find_rf_values(cell, disparity=d) for d in disparities]
        bp = ax_box.boxplot(rf_values, patch_artist=True,
                            labels=[f'{d:+d}' for d in disparities], showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color('black')
                item.set_linewidth(1)
        for box in bp['boxes']:
            box.set_facecolor('none')
        for i, vals in enumerate(rf_values):
            x = np.random.normal(i + 1, 0.05, size=len(vals))
            ax_box.scatter(x, vals, alpha=0.8, color=colors[i], s=30)
        ax_box.set_title('Peak amplitude', fontsize=14)
        ax_box.set_xlabel('Disparity (deg)', fontsize=13)
        ax_box.set_ylabel('Activity (z)', fontsize=13)

        if own_fig:
            fig.suptitle(f'Cell {cell}', fontsize=20)
        return fig

    def rf_table(self, cell, do_print=True):
        """
        Where the cell fires per disparity: RF angle (deg) and the left/right
        screen pixel the stimulus is at there, plus the peak response.
        """
        angle = self.device.get_angle_range()
        rows = []
        for d in self.device.disparities:
            rf_idx, v = self.device.find_rf_position(cell, disparity=d)
            lpx, rpx = self.device.find_rf_px(cell, d)
            rows.append({
                'disparity': d,
                'rf_deg': round(float(angle[rf_idx]), 1),
                'left_px': round(lpx, 1),
                'right_px': round(rpx, 1),
                'response': round(float(v), 2),
            })
        df = pd.DataFrame(rows)
        if do_print:
            print(f'Cell {cell} — RF location by disparity:')
            print(df.to_string(index=False))
        return df

    def save_all_cells(self, out_dir=None, dpi=150):
        """Render and save the disparity figure for every cell in the session."""
        out_dir = out_dir or self.device.results_dir
        os.makedirs(out_dir, exist_ok=True)

        for cell in self.device.suite2p_id:
            fig = self.make_disparity_plot(int(cell))
            fig.savefig(os.path.join(out_dir, f'cell_{int(cell)}.png'), dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        print(f'Saved {len(self.device.suite2p_id)} figures to: {out_dir}')
