"""
Cross-experiment per-cell visualization across up to three experiments run on
the same suite2p split: radial (distance), the per-eye monocular goggles
experiment, and the binocular disparity-stimulus experiment.

Takes the already-built device instances and draws their per-cell figures
together. It imports only numpy/matplotlib and never touches either experiment's
`src` package, so the `radial-analysis/src` and `disparity-analysis/src`
namespaces never collide in one kernel. (The disparity color palette is inlined
below for the same reason — it is NOT imported from `src.viz._base`.)

This is the visualization mirror of ``CrossExperimentAnalysis`` (which does the
pkl-based reliability selection). Cell identity (`suite2p_id`) only corresponds
when the sessions come from the same suite2p split; `__init__` warns otherwise
and `common_cells()` gives the safe intersection.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# Disparity palette, inlined from src.viz._base (kept import-free on purpose):
# orange, green, blue, dark-purple (grey control dropped for the continuous scale).
DISTANCE_COLORS = ['#FFA500', '#35b779', '#31688e', '#440154', 'grey']
DISTANCE_CMAP = LinearSegmentedColormap.from_list('distance', DISTANCE_COLORS[:-1])


def distance_palette(n):
    """``n`` colors sampled evenly along the distance colormap scale."""
    return [DISTANCE_CMAP(x) for x in np.linspace(0, 1, n)]


class CrossExperimentVisualization:

    EYE_COLOR = {'left': 'red', 'right': 'blue'}
    EYE_CMAP = {'left': 'Reds', 'right': 'Blues'}

    def __init__(self, distance_device, monocular_device, disparity_device=None):
        """
        Parameters
        ----------
        distance_device : RadialDevice, optional
            Device for the radial / distance session. Pass None to compare just
            the two goggles experiments (monocular + disparity) — the radial
            band is then omitted wherever it would otherwise appear (e.g.
            ``make_combined_plot``).
        monocular_device : MonocularDevice
            Device for the per-eye monocular goggles session.
        disparity_device : DisparityDevice, optional
            Device for the binocular disparity-stimulus session. When given,
            ``make_combined_plot`` can stack all three experiments per cell.
        """
        self.distance = distance_device
        self.monocular = monocular_device
        self.disparity = disparity_device

        # Fail fast on a swapped-device call. The monocular slot needs eye
        # conditioning (present_eyes); the disparity slot needs disparities.
        # Without this the mistake surfaces deep inside a plot as an opaque
        # AttributeError, and it is an easy one to make since the two goggles
        # devices construct identically and MonocularDevice/DisparityDevice are a
        # renamed pair (a stale kernel can even leave an object of the wrong one).
        self._check_device('monocular_device', monocular_device, needs='present_eyes',
                           wrong='disparities')
        if disparity_device is not None:
            self._check_device('disparity_device', disparity_device, needs='disparities',
                               wrong='present_eyes')

        if distance_device is not None:
            others = [('monocular', monocular_device)]
            if disparity_device is not None:
                others.append(('disparity', disparity_device))
            for name, dev in others:
                if not np.array_equal(self.distance.suite2p_id, dev.suite2p_id):
                    print(
                        f"[warning] distance and {name} sessions have different suite2p_id arrays "
                        "-> cell IDs may not refer to the same neuron. Use common_cells() for the "
                        "overlap; results are only meaningful for a same-suite2p-run split."
                    )
        elif disparity_device is not None and not np.array_equal(
                monocular_device.suite2p_id, disparity_device.suite2p_id):
            print(
                "[warning] monocular and disparity sessions have different suite2p_id arrays "
                "-> cell IDs may not refer to the same neuron. Use common_cells() for the "
                "overlap; results are only meaningful for a same-suite2p-run split."
            )

        # radial palette (5 entries: dist 0-3 + control); unavailable without a distance device
        self.distance_colors = distance_device.colors if distance_device is not None else None

    @staticmethod
    def _check_device(arg, device, needs, wrong):
        """Raise if ``device`` lacks the capability its slot requires."""
        if hasattr(device, needs):
            return
        cls = type(device).__name__
        hint = (f" It looks like a disparity session (has '{wrong}'); the arguments "
                "may be swapped." if hasattr(device, wrong) else "")
        raise TypeError(
            f"{arg} is a {cls}, which has no '{needs}' — that slot needs the "
            f"{'monocular' if needs == 'present_eyes' else 'disparity'} session.{hint} "
            "Argument order is (distance, monocular, disparity). If you just renamed "
            "the device classes, restart the kernel so stale instances are rebuilt.")

    # ------------------------------------------------------------------ helpers

    def common_cells(self):
        """Sorted cell IDs present in all provided sessions."""
        if self.distance is not None:
            ids = np.intersect1d(self.distance.suite2p_id, self.monocular.suite2p_id)
        else:
            ids = np.unique(self.monocular.suite2p_id)
        if self.disparity is not None:
            ids = np.intersect1d(ids, self.disparity.suite2p_id)
        return ids

    @staticmethod
    def _vlimits(device, cell):
        idx = np.where(device.suite2p_id == cell)[0][0]
        return np.percentile(device.response[idx, :], [1, 99])

    @staticmethod
    def _no_angle_range(ax_angle):
        """Blank the angle axis for a legacy pixel-format session (no start_deg/end_deg
        logged) instead of raising — see ``GogglesDevice.get_angle_range``."""
        ax_angle.axis('off')
        ax_angle.text(0.5, 0.5, 'No angle calibration\n(legacy pixel-format session)',
                      ha='center', va='center', fontsize=11, transform=ax_angle.transAxes)

    # ------------------------------------------------ monocular (top) drawing

    def _draw_monocular_comparison(self, ax_left, ax_right, ax_px, cell, ax_angle=None,
                                   fontscale=1.0):
        """
        Port of monocular CellVisualizer.make_comparison_plot into given axes:
        left-eye heatmap, right-eye heatmap, mean±SEM tuning vs SIGNED screen
        pixels (left on the negative axis, right on the positive), and — when
        ``ax_angle`` is given — a second mean±SEM tuning vs visual angle (deg).
        red = left eye, blue = right eye.
        """
        def fs(size):
            return size * fontscale

        D = self.monocular
        vmin, vmax = self._vlimits(D, cell)
        both = ('left' in D.present_eyes) and ('right' in D.present_eyes)

        if both:
            left_traces = D.get_all_traces(cell, eye='left')
            right_traces = D.get_all_traces(cell, eye='right')
            left_mean, left_sem = D.get_all_traces(cell, eye='left', mean_trace=True)
            right_mean, right_sem = D.get_all_traces(cell, eye='right', mean_trace=True)

            ax_left.pcolor(left_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['left'])
            ax_left.set_title('Left Eye', fontsize=fs(18))
            ax_left.set_xlabel('Time (frames)'); ax_left.set_ylabel('Trial')

            ax_right.pcolor(right_traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP['right'])
            ax_right.set_title('Right Eye', fontsize=fs(18))
            ax_right.set_xlabel('Time (frames)'); ax_right.set_ylabel('Trial')

            left_rf_idx, _ = D.find_rf_position(cell, eye='left')
            right_rf_idx, _ = D.find_rf_position(cell, eye='right')

            # --- tuning vs SIGNED screen pixels (left negative, right positive) ---
            left_px = D.get_signed_screen_range('left')
            right_px = D.get_signed_screen_range('right')
            ax_px.fill_between(left_px, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
            ax_px.plot(left_px, left_mean, linewidth=2, color='red')
            ax_px.fill_between(right_px, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
            ax_px.plot(right_px, right_mean, linewidth=2, color='blue')
            ax_px.set_xlim(left_px.min(), right_px.max())
            ax_px.set_ylim(vmin, vmax)
            ax_px.set_xlabel('Pixels'); ax_px.set_ylabel('Activity (z-score)')
            ax_px.set_title('Mean Trace ± SEM (px)', fontsize=fs(18))
            odi = self._ocular_dominance(cell)
            odi_h = [Line2D([], [], linestyle='none')] if odi is not None else []
            odi_l = [f'ODI {odi:+.2f}'] if odi is not None else []

            ax_px.legend(handles=[ax_px.lines[0], ax_px.lines[1]] + odi_h,
                         labels=[f'Left Eye (RF: {left_px[left_rf_idx]:.1f} px)',
                                 f'Right Eye (RF: {right_px[right_rf_idx]:.1f} px)'] + odi_l,
                         fontsize=fs(11), loc='best', frameon=False)

            # --- tuning vs visual angle (shared axis for both eyes) ---
            if ax_angle is not None and D.has_angle_range:
                ang = D.get_angle_range()
                ax_angle.fill_between(ang, left_mean - left_sem, left_mean + left_sem, alpha=0.2, color='red')
                ax_angle.plot(ang, left_mean, linewidth=2, color='red')
                ax_angle.fill_between(ang, right_mean - right_sem, right_mean + right_sem, alpha=0.2, color='blue')
                ax_angle.plot(ang, right_mean, linewidth=2, color='blue')
                ax_angle.set_xlim(ang[0], ang[-1])
                ax_angle.set_ylim(vmin, vmax)
                ax_angle.set_xlabel('Angle (deg)'); ax_angle.set_ylabel('Activity (z-score)')
                ax_angle.set_title('Mean Trace ± SEM (deg)', fontsize=fs(18))
                ax_angle.legend(handles=[ax_angle.lines[0], ax_angle.lines[1]] + odi_h,
                                labels=[f'Left Eye (RF: {ang[left_rf_idx]:.1f}°)',
                                        f'Right Eye (RF: {ang[right_rf_idx]:.1f}°)'] + odi_l,
                                fontsize=fs(11), loc='best', frameon=False)
            elif ax_angle is not None:
                self._no_angle_range(ax_angle)
        else:
            eye = D.present_eyes[0]
            traces = D.get_all_traces(cell, eye=eye)
            mean_trace, sem_trace = D.get_all_traces(cell, eye=eye, mean_trace=True)
            color = self.EYE_COLOR[eye]

            ax_left.pcolor(traces, vmin=vmin, vmax=vmax, cmap=self.EYE_CMAP[eye])
            ax_left.set_title(f'{eye.capitalize()} Eye', fontsize=fs(18))
            ax_left.set_xlabel('Time (frames)'); ax_left.set_ylabel('Trial')

            ax_right.axis('off')  # no second eye

            rf_idx, _ = D.find_rf_position(cell, eye=eye)
            px = D.get_signed_screen_range(eye)
            ax_px.fill_between(px, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=color)
            ax_px.plot(px, mean_trace, linewidth=2, color=color)
            ax_px.set_ylim(vmin, vmax)
            ax_px.set_xlabel('Pixels'); ax_px.set_ylabel('Activity (z-score)')
            ax_px.set_title('Mean Trace ± SEM (px)', fontsize=fs(18))
            ax_px.legend([ax_px.lines[0]], [f'{eye.capitalize()} Eye (RF: {px[rf_idx]:.1f} px)'],
                         fontsize=fs(11), loc='best', frameon=False)

            if ax_angle is not None and D.has_angle_range:
                ang = D.get_angle_range()
                ax_angle.fill_between(ang, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=color)
                ax_angle.plot(ang, mean_trace, linewidth=2, color=color)
                ax_angle.set_xlim(ang[0], ang[-1]); ax_angle.set_ylim(vmin, vmax)
                ax_angle.set_xlabel('Angle (deg)'); ax_angle.set_ylabel('Activity (z-score)')
                ax_angle.set_title('Mean Trace ± SEM (deg)', fontsize=fs(18))
                ax_angle.legend([ax_angle.lines[0]], [f'{eye.capitalize()} Eye (RF: {ang[rf_idx]:.1f}°)'],
                                fontsize=fs(11), loc='best', frameon=False)
            elif ax_angle is not None:
                self._no_angle_range(ax_angle)

    # ------------------------------------------------- distance (bottom) drawing

    def _draw_distance_columns(self, ax_heat, ax_polar, ax_tuning, cell, monocular=None,
                               ax_box=None, fontscale=1.0):
        """
        Port of radial CellVisualizer.plot_all_traces_colormap_distances cols 0-2
        (all-trials heatmap | polar mean-traces | per-distance tuning). When
        ``ax_box`` is given, also draw col 3: the per-distance peak-amplitude
        boxplot (distances 0-3 + control), matching the radial plotting class.
        """
        def fs(size):
            return size * fontscale

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
        ax_heat.set_title('All Traces', fontsize=fs(18))

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
        ax_polar.set_title('Mean Traces (polar)', fontsize=fs(18))

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
        ax_tuning.set_title('Mean Traces', fontsize=fs(18))
        ax_tuning.set_ylim(vmin, vmax)
        ax_tuning.legend(fontsize=fs(10), frameon=False)

        # --- col 3 (optional): per-distance peak-amplitude boxplot ---
        if ax_box is not None:
            rf_values = [R.find_rf_values(cell, distance=d, monocular=monocular) for d in [0, 1, 2, 3, 4]]
            bp = ax_box.boxplot(rf_values, patch_artist=True,
                                labels=['0', '1', '2', '3', 'control'], showfliers=False)
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
            for box in bp['boxes']:
                box.set_facecolor('none')
            for i, vals in enumerate(rf_values):
                x = np.random.normal(i + 1, 0.04, size=len(vals))
                ax_box.scatter(x, vals, alpha=0.8, color=self.distance_colors[i], s=40)
            ax_box.set_title('Peak amplitude', fontsize=fs(18))
            ax_box.set_xlabel('Distance'); ax_box.set_ylabel(R.response_mode)
            ax_box.set_ylim(vmin, vmax)

    # ---------------------------------------------- disparity (binocular) drawing

    # ------------------------------------------- monocular vs binocular integration

    def _ocular_dominance(self, cell):
        """
        ODI from the two monocular peak responses, (|R| - |L|) / (|R| + |L|).

        Positive = right-eye dominated, 0 = balanced, negative = left. None for a
        single-eye session. Same definition as ``CellAnalyzer``, recomputed here
        rather than read from a stats frame so the figure needs no cached df.
        """
        Dm = self.monocular
        if Dm is None or not all(e in Dm.present_eyes for e in ('left', 'right')):
            return None
        r = abs(Dm.find_rf_position(cell, eye='right')[1])
        l = abs(Dm.find_rf_position(cell, eye='left')[1])
        return (r - l) / (r + l) if (r + l) else None

    def _monocular_peak(self, cell):
        """
        The cell's best monocular peak response: the reference the binocular
        response is judged against.

        The *better* eye, not the sum. Beating the sum would test linear
        summation; beating the better eye is the usual binocular-interaction
        question, and it is the reading that survives a strongly dominated cell.
        """
        Dm = self.monocular
        if Dm is None or not Dm.present_eyes:
            return None
        return max(Dm.find_rf_position(cell, eye=e)[1] for e in Dm.present_eyes)

    def _facilitation_index(self, binocular, monocular):
        """
        (B - M) / (B + M): >0 the binocular stimulus beats the better eye alone,
        <0 it is suppressed below it.

        None when the two sum to zero or less. These are z-scored responses, so a
        weakly driven cell can put the denominator at or below zero, where the
        ratio is not interpretable rather than merely noisy.

        Deliberately a plain method rather than a staticmethod: ``%autoreload 2``
        cannot re-bind a *newly added* staticmethod onto a class that is already
        live in the kernel, so it arrives as a bare function and ``self`` becomes
        its first argument. A normal method reloads cleanly.
        """
        if monocular is None or binocular is None:
            return None
        total = binocular + monocular
        return (binocular - monocular) / total if total > 0 else None

    def _monocular_rf(self, cell, eye, units):
        """
        One eye's RF position from the monocular session, in the disparity plot's
        own units, or None when the two can't be compared.

        Screen pixels are shared by both sessions whatever format they were
        recorded in. Degrees are not: a legacy pixel-format monocular session has
        no angle calibration, so there is no degree to place the marker at.
        """
        Dm = self.monocular
        if Dm is None or eye not in Dm.present_eyes:
            return None
        rf_idx, _ = Dm.find_rf_position(cell, eye=eye)
        if units == 'px':
            return float(Dm.get_signed_screen_range(eye)[rf_idx])
        if Dm.has_angle_range:
            return float(Dm.get_angle_range()[rf_idx])
        return None

    def _draw_disparity(self, subfig, cell, units='deg', fontscale=1.0):
        """
        Port of disparity DisparityCellVisualizer.make_disparity_plot into a
        passed SubFigure: a top row of one raw-trial heatmap per disparity and a
        bottom row of three summaries (mean vs right-eye x, mean vs left-eye x,
        per-disparity peak-amplitude boxplot).
        """
        def fs(size):
            return size * fontscale

        if units not in ('deg', 'px'):
            raise ValueError("units must be 'deg' or 'px'")
        Dp = self.disparity
        disparities = Dp.disparities                 # sorted ascending
        n = Dp.n_disparities
        colors = distance_palette(n)
        cmaps = [LinearSegmentedColormap.from_list(f'disp{i}', ['white', c])
                 for i, c in enumerate(colors)]
        angle = Dp.get_angle_range()
        vmin, vmax = self._vlimits(Dp, cell)

        # mean +/- sem per disparity (shared by the two overlays)
        means = {d: Dp.get_all_traces(cell, disparity=d, mean_trace=True) for d in disparities}
        ylo = min((mn - sm).min() for mn, sm in means.values())
        yhi = max((mn + sm).max() for mn, sm in means.values())
        pad = 0.05 * (yhi - ylo)
        ylo, yhi = ylo - pad, yhi + pad

        # peak frame + height per disparity: the frame places the RF-shift readout
        # in each mean panel, the height feeds the facilitation index
        rf = {d: Dp.find_rf_position(cell, disparity=d) for d in disparities}
        mono_peak = self._monocular_peak(cell)
        fi = {d: self._facilitation_index(rf[d][1], mono_peak) for d in disparities}

        # Nested subfigures rather than one 3n-column gridspec. A gridspec whose
        # axes span many columns collapses under constrained_layout when it sits
        # inside a SubFigure, and matplotlib then drops the layout for the WHOLE
        # figure, silently overlapping every band -- not just this one.
        hm_sf, sum_sf = subfig.subfigures(2, 1, height_ratios=[1, 1.15])
        heat_axes = np.atleast_1d(hm_sf.subplots(1, n))

        # --- top row: one colored raw-trace heatmap per disparity ---
        for i, (d, cmap) in enumerate(zip(disparities, cmaps)):
            ax = heat_axes[i]
            ax.pcolor(Dp.get_all_traces(cell, disparity=d), vmin=vmin, vmax=vmax, cmap=cmap)
            title = f'Disparity {d:+d}°'
            if fi[d] is not None:
                title += f'\nFI {fi[d]:+.2f}'      # vs the better eye alone
            ax.set_title(title, fontsize=fs(16))
            ax.set_xlabel('Time (frames)', fontsize=fs(12))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if i == 0:
                ax.set_ylabel('Trial', fontsize=fs(13))
            else:
                # every heatmap shares the trial axis; repeating its tick labels
                # seven times is what starves the constrained layout of width
                ax.tick_params(labelleft=False)

        # --- bottom row: three full-width summaries ---
        ax_right, ax_left, ax_box = sum_sf.subplots(1, 3)

        def x_right(d):
            return Dp.get_px_range('right', d) if units == 'px' else angle
        def x_left(d):
            return Dp.get_px_range('left', d) if units == 'px' else angle - d

        handles = {'right': [], 'left': []}
        for d, color in zip(disparities, colors):
            mn, sm = means[d]
            xr, xl = x_right(d), x_left(d)
            handles['right'].append(ax_right.plot(xr, mn, color=color, linewidth=2)[0])
            ax_right.fill_between(xr, mn - sm, mn + sm, color=color, alpha=0.12)
            handles['left'].append(ax_left.plot(xl, mn, color=color, linewidth=2)[0])
            ax_left.fill_between(xl, mn - sm, mn + sm, color=color, alpha=0.12)
        unit_lbl = 'pixel' if units == 'px' else 'angle (deg)'
        for ax, title, xlabel in ((ax_right, f'Mean vs right-eye {unit_lbl} (absolute)', f'Right-eye {unit_lbl}'),
                                  (ax_left, f'Mean vs left-eye {unit_lbl}', f'Left-eye {unit_lbl}')):
            ax.set_title(title, fontsize=fs(13))
            ax.set_xlabel(xlabel, fontsize=fs(12))
            ax.set_ylabel('Activity (z)', fontsize=fs(12))
            ax.set_ylim(ylo, yhi)

        # Mark where this cell fired when that eye was driven alone, and label each
        # disparity with how far its binocular peak sits from that marker in this
        # eye's frame. The disparity that reads ~0 in BOTH panels is the one putting
        # the stimulus on the cell's RF in both eyes at once.
        # Needs the two sessions in a common frame: pixels always are, degrees only
        # if the monocular session was angle-calibrated too.
        unit_short = 'px' if units == 'px' else '°'
        for ax, eye in ((ax_right, 'right'), (ax_left, 'left')):
            pos = self._monocular_rf(cell, eye, units)
            labels = [f'{d:+d}°' for d in disparities]

            if pos is not None:
                ax.axvline(pos, color='gray', linestyle=':', linewidth=fs(1.4), zorder=0)
                ax.annotate(f'mono RF {pos:.0f}', xy=(pos, 1.0), xycoords=('data', 'axes fraction'),
                            xytext=(fs(3), fs(-2)), textcoords='offset points',
                            color='gray', fontsize=fs(9), ha='left', va='top')
                labels = [f'{lab}  ΔRF {float(xf[rf[d][0]]) - pos:+.0f}'
                          for lab, d, xf in zip(labels, disparities,
                                                (x_right(d) if eye == 'right' else x_left(d)
                                                 for d in disparities))]

            ax.legend(handles[eye], labels,
                      title=f'disp  (ΔRF in {unit_short})' if pos is not None else 'disp',
                      fontsize=fs(9), title_fontsize=fs(10), frameon=False)

        # peak-amplitude boxplot per disparity (distance-plot style)
        rf_values = [Dp.find_rf_values(cell, disparity=d) for d in disparities]
        bp = ax_box.boxplot(rf_values, patch_artist=True,
                            tick_labels=[f'{d:+d}' for d in disparities], showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color('black')
                item.set_linewidth(1)
        for box in bp['boxes']:
            box.set_facecolor('none')
        for i, vals in enumerate(rf_values):
            x = np.random.normal(i + 1, 0.05, size=len(vals))
            ax_box.scatter(x, vals, alpha=0.8, color=colors[i], s=30)
        ax_box.set_title('Peak amplitude', fontsize=fs(13))
        ax_box.set_xlabel('Disparity (deg)', fontsize=fs(12))
        ax_box.set_ylabel('Activity (z)', fontsize=fs(12))

    # ------------------------------------------------------------- public plots

    def plot_cell(self, cell, monocular=None):
        """
        2 x 3 comparison figure for one cell (distance + monocular goggles):

        - Row 0 (monocular goggles) = make_comparison_plot:
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

        self._draw_monocular_comparison(ax_dl, ax_dr, ax_dt, cell)
        self._draw_distance_columns(ax_rh, ax_rp, ax_rt, cell, monocular=monocular)

        ax_dl.annotate('Monocular (goggles)', xy=(0, 1.18), xycoords='axes fraction',
                       fontsize=20, fontweight='bold')
        ax_rh.annotate('Distance (radial)', xy=(0, 1.18), xycoords='axes fraction',
                       fontsize=20, fontweight='bold')

        fig.suptitle(f'Cell {cell}', fontsize=24)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # the project style, which every size in this figure is scaled from
    _BASE_FONT_RC = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 12, 'figure.titlesize': 22}

    def make_combined_plot(self, cell, units='deg', radial_condition=None, fontscale=1.8):
        """
        One figure stacking the available experiments for a single cell, top to
        bottom: monocular (per-eye goggles), binocular disparity stimulus, and —
        only when this instance was built with a distance device — radial
        (distance) as the last row (with its per-distance peak-amplitude bars).
        Without a distance device the figure is just the two goggles bands.
        Mirrors the two-experiment ``CombinedCellVisualizer.make_combined_plot``.

        Parameters
        ----------
        units : {'deg', 'px'}
            x-axis units for the disparity band's mean overlays.
        radial_condition : optional
            Forwarded to the radial band as the ``RadialDevice`` ``monocular``
            condition selector (None = pooled). Ignored when there is no
            distance device. This is the radial experiment's own
            monocular/binocular condition, distinct from the 'monocular'
            (per-eye goggles) experiment.
        fontscale : float
            Multiplies every font size. This figure is ~34 inches wide, several
            times the width of any single band drawn on its own, so whatever
            displays it shrinks it to fit and type sized for a normal plot comes
            out unreadable. Scaling the type is the only lever: shrinking the
            canvas instead makes matplotlib drop the constrained layout and
            overlap every panel. Applied two ways, because the bands set sizes
            two ways -- an rc_context for the axis labels and ticks, which are
            left at the project style, and a ``fontscale`` argument for the
            titles and legends, which name their size explicitly.
        """
        scaled = {k: v * fontscale for k, v in self._BASE_FONT_RC.items()}
        with plt.rc_context(scaled):
            return self._build_combined_plot(cell, units, radial_condition, fontscale)

    def _build_combined_plot(self, cell, units, radial_condition, fontscale):
        """Body of :meth:`make_combined_plot`, called inside its rc_context."""
        if self.disparity is None:
            raise ValueError(
                "make_combined_plot needs the disparity experiment; construct with "
                "CrossExperimentVisualization(distance, monocular, disparity).")

        include_distance = self.distance is not None
        n = self.disparity.n_disparities

        # bands, top -> bottom: monocular, disparity, radial (radial last, per request).
        # The cell title goes on the figure itself; it used to live in a thin extra
        # subfigure, but a subfigure holding no axes collapses to zero height and
        # takes the whole constrained layout down with it, overlapping every band.
        if include_distance:
            fig = plt.figure(figsize=(max(20, 4.8 * n), 26), constrained_layout=True)
            top, mid, bottom = fig.subfigures(
                3, 1, height_ratios=[1.0, 1.9, 1.1], hspace=0.05)
        else:
            fig = plt.figure(figsize=(max(20, 4.8 * n), 19), constrained_layout=True)
            top, mid = fig.subfigures(2, 1, height_ratios=[1.0, 1.9], hspace=0.05)
        fig.suptitle(f'Cell {cell}', fontsize=30, fontweight='bold')

        # --- band 1: monocular (per-eye goggles): heatmaps + px + angle tuning ---
        ax_left, ax_right, ax_px, ax_ang = top.subplots(1, 4)
        self._draw_monocular_comparison(ax_left, ax_right, ax_px, cell, ax_angle=ax_ang,
                                        fontscale=fontscale)
        top.suptitle('Monocular (goggles) — left vs right eye', fontsize=26, fontweight='bold')

        # --- band 2: binocular disparity stimulus ---
        self._draw_disparity(mid, cell, units=units, fontscale=fontscale)
        mid.suptitle('Disparity (both eyes)', fontsize=26, fontweight='bold')

        # --- band 3 (optional): radial (distance) — last row, incl. per-distance bars ---
        if include_distance:
            gs = bottom.add_gridspec(1, 4)
            ax_heat = bottom.add_subplot(gs[0, 0])
            ax_polar = bottom.add_subplot(gs[0, 1], projection='polar')
            ax_tuning = bottom.add_subplot(gs[0, 2])
            ax_box = bottom.add_subplot(gs[0, 3])
            self._draw_distance_columns(ax_heat, ax_polar, ax_tuning, cell,
                                        monocular=radial_condition, ax_box=ax_box,
                                        fontscale=fontscale)
            bottom.suptitle('Distance (radial)', fontsize=26, fontweight='bold')

        return fig

    def plot_cell_tuning(self, cell, monocular=None):
        """Lighter 1 x 2: monocular eye-tuning | distance per-distance tuning."""
        D, R = self.monocular, self.distance
        fig, (axD, axR) = plt.subplots(1, 2, figsize=(13, 5))

        # monocular eye tuning
        vmin, vmax = self._vlimits(D, cell)
        for eye in D.present_eyes:
            mean_e, sem_e = D.get_all_traces(cell, eye=eye, mean_trace=True)
            rng = D.get_signed_screen_range(eye)
            axD.plot(rng, mean_e, color=self.EYE_COLOR[eye], label=f'{eye} eye', linewidth=2)
            axD.fill_between(rng, mean_e - sem_e, mean_e + sem_e, color=self.EYE_COLOR[eye], alpha=0.2)
        axD.set_xlabel('Screen position (px)'); axD.set_ylabel('Activity (z-score)')
        axD.set_title('Monocular (goggles)', fontsize=18); axD.set_ylim(vmin, vmax)
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
