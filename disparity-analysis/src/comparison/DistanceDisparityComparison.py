"""
Cross-experiment per-cell comparison: distance (radial) vs disparity (goggles).

Takes the two already-built device instances and draws their per-cell figures
together. It imports only numpy/matplotlib and never touches either experiment's
`src` package, so the two `src` namespaces never collide in one kernel.

Cell identity (`suite2p_id`) only corresponds when both sessions come from the
same suite2p split; `__init__` warns otherwise and `common_cells()` gives the
safe intersection.
"""
import numpy as np
import matplotlib.pyplot as plt


class DistanceDisparityComparison:

    EYE_COLOR = {'left': 'red', 'right': 'blue'}
    EYE_CMAP = {'left': 'Reds', 'right': 'Blues'}

    def __init__(self, distance_device, disparity_device):
        """
        Parameters
        ----------
        distance_device : RadialDevice
            Device for the radial / distance session.
        disparity_device : DisparityDevice
            Device for the goggles / disparity session.
        """
        self.distance = distance_device
        self.disparity = disparity_device

        if not np.array_equal(self.distance.suite2p_id, self.disparity.suite2p_id):
            print(
                "[warning] distance and disparity sessions have different suite2p_id arrays "
                "-> cell IDs may not refer to the same neuron. Use common_cells() for the "
                "overlap; results are only meaningful for a same-suite2p-run split pair."
            )

        self.distance_colors = distance_device.colors  # radial palette (5 entries: dist 0-3 + control)

    # ------------------------------------------------------------------ helpers

    def common_cells(self):
        """Sorted cell IDs present in both sessions."""
        return np.intersect1d(self.distance.suite2p_id, self.disparity.suite2p_id)

    @staticmethod
    def _vlimits(device, cell):
        idx = np.where(device.suite2p_id == cell)[0][0]
        return np.percentile(device.response[idx, :], [1, 99])

    # ------------------------------------------------- disparity (top) drawing

    def _draw_disparity_comparison(self, ax_left, ax_right, ax_tuning, cell):
        """Port of disparity CellVisualizer.make_comparison_plot into 3 given axes."""
        D = self.disparity
        vmin, vmax = self._vlimits(D, cell)
        both = ('left' in D.present_eyes) and ('right' in D.present_eyes)

        if both:
            left_traces = D.get_all_traces(cell, eye='left')
            right_traces = D.get_all_traces(cell, eye='right')
            left_mean, left_sem = D.get_all_traces(cell, eye='left', mean_trace=True)
            right_mean, right_sem = D.get_all_traces(cell, eye='right', mean_trace=True)

            ax_left.pcolor(left_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['left'])
            ax_left.set_title('Left Eye', fontsize=18)
            ax_left.set_xlabel('Time (frames)'); ax_left.set_ylabel('Trial')

            ax_right.pcolor(right_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['right'])
            ax_right.set_title('Right Eye', fontsize=18)
            ax_right.set_xlabel('Time (frames)'); ax_right.set_ylabel('Trial')

            left_range = D.get_screen_range('left')
            right_range = D.get_screen_range('right')
            left_rf_idx, _ = D.find_rf_position(cell, eye='left')
            right_rf_idx, _ = D.find_rf_position(cell, eye='right')
            left_rf_px = left_range[left_rf_idx]
            right_rf_px = right_range[right_rf_idx]

            ax_tuning.fill_between(left_range, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
            ax_tuning.plot(left_range, left_mean, linewidth=2, color='red')
            ax_tuning.fill_between(right_range, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
            ax_tuning.plot(right_range, right_mean, linewidth=2, color='blue')
            ax_tuning.set_xlim(right_range[0], right_range[-1])
            ax_tuning.set_ylim(vmin, vmax)
            ax_tuning.set_xlabel('|Pixels|'); ax_tuning.set_ylabel('Activity (z-score)')
            ax_tuning.set_title('Mean Trace ± SEM', fontsize=18)
            if len(ax_tuning.lines) >= 2:
                ax_tuning.legend(
                    handles=[ax_tuning.lines[0], ax_tuning.lines[1]],
                    labels=[f'Left Eye (RF: {-left_rf_px:.1f} px)', f'Right Eye (RF: {right_rf_px:.1f} px)'],
                    fontsize=12, loc='best', frameon=False,
                )
        else:
            eye = D.present_eyes[0]
            traces = D.get_all_traces(cell, eye=eye)
            mean_trace, sem_trace = D.get_all_traces(cell, eye=eye, mean_trace=True)

            ax_left.pcolor(traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP[eye])
            ax_left.set_title(f'{eye.capitalize()} Eye', fontsize=18)
            ax_left.set_xlabel('Time (frames)'); ax_left.set_ylabel('Trial')

            ax_right.axis('off')  # no second eye

            rng = D.get_screen_range(eye)
            rf_idx, _ = D.find_rf_position(cell, eye=eye)
            color = self.EYE_COLOR[eye]
            ax_tuning.fill_between(rng, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=color)
            ax_tuning.plot(rng, mean_trace, linewidth=2, color=color)
            ax_tuning.set_xlim(rng[0], rng[-1]); ax_tuning.set_ylim(vmin, vmax)
            ax_tuning.set_xlabel('|Pixels|'); ax_tuning.set_ylabel('Activity (z-score)')
            ax_tuning.set_title('Mean Trace ± SEM', fontsize=18)
            ax_tuning.legend([ax_tuning.lines[0]], [f'{eye.capitalize()} Eye (RF: {rng[rf_idx]:.1f} px)'],
                             fontsize=12, loc='best', frameon=False)

    # ------------------------------------------------- distance (bottom) drawing

    def _draw_distance_columns(self, ax_heat, ax_polar, ax_tuning, cell, monocular=None):
        """Port of radial CellVisualizer.plot_all_traces_colormap_distances cols 0-2."""
        R = self.distance
        idx = np.where(R.suite2p_id == cell)[0][0]

        all_traces = R.get_all_traces(cell, distance=None, monocular=monocular)
        vmin, vmax = np.percentile(all_traces, [1, 99])

        ang = np.ravel(R.angle_range).astype(float)

        # --- col 0: all-trials heatmap across distances 0..4 (incl. control) ---
        repeat_traces = []
        for dist in [0, 1, 2, 3, 4]:
            for repeat in R.get_condition_trials(monocular=monocular, distance=dist):
                repeat_traces.append(R.get_on_trace(idx, repeat))
        ax_heat.pcolormesh(np.array(repeat_traces), cmap='viridis', vmin=vmin, vmax=vmax)
        ax_heat.set_xlabel('Time (frames)'); ax_heat.set_ylabel('Trials')
        ax_heat.set_title('All Traces', fontsize=18)

        # --- col 1: polar mean-traces (distances 0..3 as rings) ---
        n_theta = len(ang)
        theta_edges = np.linspace(np.radians(ang[0]), np.radians(ang[-1]), n_theta + 1)
        r_edges = np.linspace(0, 4, 5)
        theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
        Z = np.vstack([
            np.ravel(R.get_all_traces(cell, distance=d, monocular=monocular, mean_trace=True)[0])
            for d in [0, 1, 2, 3]
        ])
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_thetamin(ang[0])
        ax_polar.set_thetamax(ang[-1])
        ax_polar.pcolormesh(theta_grid, r_grid, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax_polar.set_title('Mean Traces (polar)', fontsize=18)

        # --- col 2: cartesian mean ± SEM per distance ---
        for dist in [0, 1, 2, 3, 4]:
            mean_d, sem_d = R.get_all_traces(cell, distance=dist, monocular=monocular, mean_trace=True)
            mean_d, sem_d = np.ravel(mean_d), np.ravel(sem_d)
            label = 'control' if dist == 4 else f'Distance {dist}'
            ax_tuning.plot(ang, mean_d, color=self.distance_colors[dist], label=label, linewidth=3)
            ax_tuning.fill_between(ang, mean_d - sem_d, mean_d + sem_d, color=self.distance_colors[dist], alpha=0.2)
        ax_tuning.axvline(x=30, color='lightgray', linestyle='--', linewidth=2)

        # mark the cell's RF position (peak of the overall mean across distances)
        overall_mean = np.ravel(R.get_all_traces(cell, distance=None, monocular=monocular, mean_trace=True)[0])
        rf_deg = float(ang[int(np.argmax(overall_mean))])
        ax_tuning.axvline(rf_deg, color='black', linestyle=':', linewidth=2, label=f'RF: {rf_deg:.1f}°')

        ax_tuning.set_xlabel('Degrees (deg)'); ax_tuning.set_ylabel(R.response_mode)
        ax_tuning.set_title('Mean Traces', fontsize=18)
        ax_tuning.set_ylim(vmin, vmax)
        ax_tuning.legend(fontsize=10, frameon=False)

    # ------------------------------------------------------------- public plots

    def plot_cell(self, cell, monocular=None):
        """
        2 x 3 comparison figure for one cell:

        - Row 0 (goggles / disparity) = make_comparison_plot:
          left-eye heatmap | right-eye heatmap | mean±SEM eye tuning.
        - Row 1 (distance / radial) = first 3 columns of
          plot_all_traces_colormap_distances:
          all-trials heatmap | polar mean-traces | per-distance tuning.

        `monocular` selects the radial condition (None = pooled). Returns the figure.
        """
        fig = plt.figure(figsize=(18, 10))

        ax_dl = fig.add_subplot(2, 3, 1)
        ax_dr = fig.add_subplot(2, 3, 2)
        ax_dt = fig.add_subplot(2, 3, 3)
        ax_rh = fig.add_subplot(2, 3, 4)
        ax_rp = fig.add_subplot(2, 3, 5, projection='polar')
        ax_rt = fig.add_subplot(2, 3, 6)

        self._draw_disparity_comparison(ax_dl, ax_dr, ax_dt, cell)
        self._draw_distance_columns(ax_rh, ax_rp, ax_rt, cell, monocular=monocular)

        ax_dl.annotate('Goggles (disparity)', xy=(0, 1.18), xycoords='axes fraction',
                       fontsize=20, fontweight='bold')
        ax_rh.annotate('Distance (radial)', xy=(0, 1.18), xycoords='axes fraction',
                       fontsize=20, fontweight='bold')

        fig.suptitle(f'Cell {cell}', fontsize=24)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_cell_tuning(self, cell, monocular=None):
        """Lighter 1 x 2: disparity eye-tuning | distance per-distance tuning."""
        D, R = self.disparity, self.distance
        fig, (axD, axR) = plt.subplots(1, 2, figsize=(13, 5))

        # disparity eye tuning
        vmin, vmax = self._vlimits(D, cell)
        for eye in D.present_eyes:
            mean_e, sem_e = D.get_all_traces(cell, eye=eye, mean_trace=True)
            rng = D.get_screen_range(eye, signed=True)
            axD.plot(rng, mean_e, color=self.EYE_COLOR[eye], label=f'{eye} eye', linewidth=2)
            axD.fill_between(rng, mean_e - sem_e, mean_e + sem_e, color=self.EYE_COLOR[eye], alpha=0.2)
        axD.set_xlabel('Screen position (px)'); axD.set_ylabel('Activity (z-score)')
        axD.set_title('Goggles (disparity)', fontsize=18); axD.set_ylim(vmin, vmax)
        axD.legend(frameon=False)

        # distance per-distance tuning
        ang = np.ravel(R.angle_range).astype(float)
        dv_min, dv_max = np.percentile(R.get_all_traces(cell, distance=None, monocular=monocular), [1, 99])
        for dist in [0, 1, 2, 3]:
            mean_d, sem_d = R.get_all_traces(cell, distance=dist, monocular=monocular, mean_trace=True)
            mean_d, sem_d = np.ravel(mean_d), np.ravel(sem_d)
            axR.plot(ang, mean_d, color=self.distance_colors[dist], label=f'Distance {dist}', linewidth=2)
            axR.fill_between(ang, mean_d - sem_d, mean_d + sem_d, color=self.distance_colors[dist], alpha=0.2)
        axR.set_xlabel('Degrees (deg)'); axR.set_ylabel(R.response_mode)
        axR.set_title('Distance (radial)', fontsize=18); axR.set_ylim(dv_min, dv_max)
        axR.legend(frameon=False)

        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return fig
