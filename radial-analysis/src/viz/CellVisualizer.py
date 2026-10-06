import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt
from src.utils import setup_plot_style
from src.analysis.CellAnalysis import CellAnalyzer
colors = setup_plot_style()

class CellVisualizer:

    def __init__(self, device):
        """
        Initialize CellVisualizer with separate data and analysis classes.

        Parameters
        ----------
        device : RadialDevice
            RadialDevice instance containing data access methods
        """
        self.device = device
        self.analyzer = CellAnalyzer(device)
        self.colors = ['#FFA500', '#35b779', '#31688e', '#440154', 'grey']

    def plot_stimulus_response(self, repeats=10, cell = None):
        """
        Plot the stimulus response for a given cell.

        Parameters:
        -----------
        fig_path : str, optional
            Path to save the figure. If provided, the figure will be saved.
        n_repeats : int or list, optional
            Number of repeats to plot or range of repeats [start, end] to plot.
            Default is 10 (plots first 10 repeats).

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.

        cell : int, optional
            The cell index to plot. If provided, the cell index will be used to plot the stimulus response.
        """
        fig = plt.figure(figsize=(14,4))

        if cell is not None:
            idx = np.where(self.device.suite2p_id == cell)[0][0]

        # Handle both single number and range inputs for n_repeats
        if isinstance(repeats, (list, tuple)) and len(repeats) == 2:
            repeats = np.arange(repeats[0], repeats[1])
        else:
            repeats = np.arange(repeats)

        repeat_start = self.device.trial_start_time[repeats[0]]*1000
        repeat_end = self.device.trial_end_time[repeats[-1]]*1000

        repeats_frames = (self.device.frames_time > repeat_start) & (self.device.frames_time < repeat_end)

        ax = plt.gca()

        if cell is None:
            ax.plot(self.device.frames_time[repeats_frames]/1000, self.device.angle_interp[repeats_frames], lw=2, color='black')
        else:
            ax.plot(self.device.frames_time[repeats_frames]/1000, self.device.response[idx, repeats_frames], lw=1.5, color='black')

        for rep in repeats:
            on_start, on_end = self.device.get_on_times(rep)
            if self.device.monocular_sequence[rep] == 1:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.device.distance_sequence[rep]], alpha=0.5)
            else:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.device.distance_sequence[rep]], alpha=0.5, hatch='///')

        #if cell is not None, set the ylim to the 99 and 1 percentiles of the response
        if cell is not None:
            ax.set_ylim(np.min(self.device.response[idx, :]), np.max(self.device.response[idx, :]))

        # make a legend for the colors and history/no history
        legend_elements = []
        for i in [0,1,2,3]:
            # No history patch
            patch_mono = plt.Rectangle((0,0), 1, 1, fc=colors[i], label=f'Distance {i} (Monocular)')
            patch_bino = plt.Rectangle((0,0), 1, 1, fc=colors[i], hatch='///', lw = 2, label=f'Distance {i} (Binocular)')
            legend_elements.append(patch_mono)
            legend_elements.append(patch_bino)

        # Only create one legend and keep it
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc='gray', label='Off Time'))
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xlabel('Time (s)')
        if cell is None:
            plt.ylabel('Angle (deg)')
            plt.title('Stimulus Response')
        else:
            plt.ylabel(self.device.response_mode)
            plt.title(f'Stimulus Response for Cell {cell}')

        plt.tight_layout()

    def plot_all_traces_colormap(self, cell, pre_time=2500, post_time=1000, distance=None, monocular=None, history=None):
        """
        Create a colormap visualization of neural traces for a given cell.

        Parameters:
        -----------
        cell : int
            The cell ID to plot.
        pre_time : int, optional
            Time before stimulus onset in ms. Default is 2000.
        post_time : int, optional
            Time after stimulus offset in ms. Default is 2000.
        distance : int or list, optional
            Distance(s) to plot. If None, all distances are plotted.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """

        # function will be two columns, one row.
        # first column pcolormesh of all traces
        # second column plot of mean traces
        idx = np.where(self.device.suite2p_id == cell)[0][0]
        # Handle the case of a single distance (1D array of axes)
        if distance == None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 4),
                               gridspec_kw={'width_ratios': [2, 1]})
            axs = np.array([axs])
        else:
            fig, axs = plt.subplots(len(distance), 2, figsize=(14, len(distance)*3.5),
                               gridspec_kw={'width_ratios': [2, 1]})

        vmin, vmax = np.percentile(self.device.response[idx], [1, 99])

        if distance is None:
            distance = [None]

        for i, dist in enumerate(distance):

            repeat_indices = self.device.get_condition_trials(monocular=monocular, distance=dist)

            repeat_traces = []
            for repeat in repeat_indices:
                on_start, on_end = self.device.get_on_times(repeat)
                frames = (self.device.frames_time > (on_start-pre_time)) & (self.device.frames_time < (on_end+post_time))
                trace_length = self.device.on_length + int((pre_time + post_time)/100)

                # Make sure we don't exceed the available frames
                trace = self.device.response[idx, frames]
                if len(trace) >= trace_length:
                    repeat_traces.append(trace[:trace_length].T)

            if not repeat_traces:
                continue

            repeat_traces = np.array(repeat_traces)

            # Plot heatmap
            if self.device.response_mode == 'spikes-zscore' or self.device.response_mode == 'spikes':
                cmap = 'viridis'
            elif self.device.response_mode == 'dFF' or self.device.response_mode == 'dFF-zscore':
                cmap = 'viridis'

            axs[i, 0].pcolormesh(repeat_traces, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[i, 0].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].axvline(x=(pre_time/100+self.device.on_length), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].set_xlabel('Time (frames)')
            axs[i, 0].set_ylabel('Trials')
            axs[i, 0].set_title(f'All Traces')

            repeat_traces = []
            for repeat in repeat_indices:
                on_start, on_end = self.device.get_on_times(repeat)
                frames = (self.device.frames_time > (on_start)) & (self.device.frames_time < (on_end))
                trace_length = self.device.on_length
                repeat_traces.append(self.device.response[idx, frames][:trace_length].T)

            repeat_traces = np.array(repeat_traces)
            repeat_traces_mean = np.mean(repeat_traces, axis=0)
            repeat_traces_sem = np.std(repeat_traces, axis=0)/np.sqrt(len(repeat_traces))

            # Plot mean trace
            if dist is not None:
                plot_color = self.colors[dist]
            else:
                plot_color = 'black'


            axs[i, 1].plot(np.linspace(-30, 60, len(repeat_traces_mean)), repeat_traces_mean, color=plot_color, label=f'Distance {dist}', linewidth=3)
            axs[i, 1].set_xlabel('Angle (deg)')
            axs[i, 1].set_ylabel(self.device.response_mode)
            axs[i, 1].set_title(f'Mean Traces ')
            axs[i, 1].fill_between(np.linspace(-30, 60, len(repeat_traces_mean)),
                                    repeat_traces_mean - repeat_traces_sem,
                                    repeat_traces_mean + repeat_traces_sem,
                                    color=plot_color, alpha=0.2)
            axs[i, 1].set_ylim(vmin, vmax)
            axs[i, 1].legend()

        # # add suptitle to the figure
        # fig.suptitle(f'Cell {cell} - {monocular_condition_name} - {history_condition_name}', fontsize=20)
        plt.tight_layout()

        return fig

    def plot_all_traces_colormap_distances(self, cell):
        """
        Create a colormap visualization of neural traces for a given cell, with an additional column for stats text.

        Parameters:
        -----------
        cell : int
            The cell ID to plot.
        pre_time : int, optional
            Time before stimulus onset in ms. Default is 2000.
        post_time : int, optional
            Time after stimulus offset in ms. Default is 2000.
        conditions : list, optional
            Conditions to plot. If None, all conditions are plotted.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """

        has_monocular, has_binocular = self.analyzer._get_available_conditions()

        conditions = []
        if has_monocular:
            conditions.append({'monocular': True,  'title': 'monocular'})
        if has_binocular:
            conditions.append({'monocular': False, 'title': 'binocular'})

        n_rows = len(conditions)

        # Create figure with a row for each condition, 5 columns per row (add one for stats text)
        fig, axs = plt.subplots(n_rows, 5, figsize=(22, n_rows * 4),
                               gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.2]},
                               squeeze=False)
        # remove the placeholder axis for the polar subplot
        for r in range(n_rows):
            axs[r, 1].remove()

        # Adjust spacing between columns
        plt.subplots_adjust(wspace=0.4, hspace=0.3)

        results = self.analyzer.analyze_cell_stats(cell)
        reliable = any(results[c['title']]['reliable'] for c in conditions)

        all_traces = self.device.get_all_traces(cell, distance=None, monocular=None, mean_trace=False)
        vmin, vmax = np.percentile(all_traces, [1, 99])

        # Process each condition
        for c_idx, condition in enumerate(conditions):
            self._draw_condition_row(fig, axs, c_idx, cell, condition, results, reliable, vmin, vmax)

        # Add suptitle to the figure
        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout()

        return fig

    def _draw_condition_row(self, fig, axs, row_idx, cell, condition, results, reliable, vmin, vmax, row_label=None,
                            show_condition=True):
        """
        Draw one condition's row (all-traces heatmap, polar mean plot, tuning curve,
        RF boxplot, stats text) into an existing axes grid built like the one in
        `plot_all_traces_colormap_distances`.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure the axes belong to (needed to add the polar subplot).
        axs : np.ndarray
            2D array of axes, shape (n_rows, 5), with column 1 of `row_idx` already removed.
        row_idx : int
            Row of `axs` to draw into.
        cell : int
            Cell ID to plot.
        condition : dict
            {'monocular': bool, 'title': str} — which condition this row represents.
        results : dict
            Output of `self.analyzer.analyze_cell_stats(cell)`.
        reliable : bool
            Whether the cell is reliable in any condition (shared across a session's rows).
        vmin, vmax : float
            Shared color/y-axis limits for this row.
        row_label : str, optional
            Prefix added to panel titles (e.g. a session name), for stacking rows from
            multiple sessions in one figure. Omit for single-session behavior.
        show_condition : bool, optional
            If False and `row_label` is given, panel titles show only `row_label`
            (e.g. sessions that differ by eye but are all flagged binocular).
        """

        # Get cell index
        idx = np.where(self.device.suite2p_id == cell)[0][0]

        # Define distances to plot
        distances_with_control = [0, 1, 2, 3, 4]
        distances_without_control = [0, 1, 2, 3]

        if not row_label:
            row_title = condition['title']
        elif show_condition:
            row_title = f'{row_label} - {condition["title"]}'
        else:
            row_title = row_label

        results_condition = results[condition['title']]

        repeat_traces = []

        for i, dist in enumerate(distances_with_control):
            repeat_indices = self.device.get_condition_trials(
                monocular=condition['monocular'],
                distance=dist
            )

            for repeat in repeat_indices:
                repeat_traces.append(self.device.get_on_trace(idx, repeat))


            repeats_distance_mean, repeats_distance_sem = self.device.get_all_traces(cell, distance=dist, monocular=condition['monocular'], mean_trace=True)

            axs[row_idx, 2].plot(self.device.angle_range, repeats_distance_mean,
                              color=self.colors[dist],
                              label=f'Distance {dist}',
                              linewidth=3)
            axs[row_idx, 2].fill_between(
                self.device.angle_range.flatten(),
                repeats_distance_mean - repeats_distance_sem,
                repeats_distance_mean + repeats_distance_sem,
                color=self.colors[dist], alpha=0.2
            )

        axs[row_idx, 0].pcolormesh(repeat_traces, cmap='viridis', vmin=vmin, vmax=vmax)

        axs[row_idx, 0].set_xlabel('Time (frames)')
        axs[row_idx, 0].set_ylabel('Trials')
        axs[row_idx, 0].set_title(f'{row_title} - All Traces', fontsize=14)

        # radial plot
        n_theta = len(self.device.angle_range)
        theta_edges = np.linspace(np.radians(self.device.angle_range[0]), np.radians(self.device.angle_range[-1]), n_theta + 1)
        r_edges = np.linspace(0, len(distances_without_control), len(distances_without_control) + 1)
        theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)

        matrices = [self.device.get_all_traces(cell=cell, monocular=condition['monocular'], distance=dist, mean_trace=True)[0] for dist in [0, 1, 2, 3]]
        Z = np.vstack(matrices)

        # # Make this axis polar
        # Create a polar axis in place of the corresponding cartesian axis
        ax_polar = plt.subplot2grid(axs.shape, (row_idx, 1), projection='polar', fig=fig)
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_thetamin(self.device.angle_range[0])
        ax_polar.set_thetamax(self.device.angle_range[-1])
        # Plot on the polar axis
        pcm = ax_polar.pcolormesh(theta_grid, r_grid, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        # Optionally, copy over the title
        ax_polar.set_title(f'{row_title} - Mean Traces', fontsize=14)

        # Add axis for second subplot
        axs[row_idx, 2].set_xlabel('Degrees (deg)')
        axs[row_idx, 2].set_ylabel(self.device.response_mode)
        axs[row_idx, 2].set_title(f'{row_title} - Mean Traces', fontsize=14)
        # vertical line at this condition's RF (peak of the mean trace over all distances)
        axs[row_idx, 2].axvline(x=results_condition['RF'], color='lightgray', linestyle='--', linewidth=2)
        axs[row_idx, 2].set_ylim(vmin, vmax)

        # boxplot
        rf_values = [self.device.find_rf_values(cell, distance=distance, monocular=condition['monocular']) for distance in distances_with_control]
        rf_position = self.device.find_rf_position(cell, monocular=condition['monocular'])

        bp = axs[row_idx, 3].boxplot(rf_values, patch_artist=True, labels=distances_without_control + ['control'], showfliers=False)

        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color('black')
                item.set_linewidth(1)
        for box in bp['boxes']:
            box.set_facecolor('none')

        for i, rf_vals in enumerate(rf_values):
            x = np.random.normal(i+1, 0.04, size=len(rf_vals))
            axs[row_idx, 3].scatter(x, rf_vals, alpha=0.8, color=self.colors[i], s=40)

        axs[row_idx, 3].set_xlabel('Distance')
        axs[row_idx, 3].set_ylabel(self.device.response_mode)
        axs[row_idx, 3].set_ylim(vmin, vmax)

        bootstrap_modulated = reliable and results_condition['bootstrap_pvalue'] < 0.05
        anova_modulated = reliable and results_condition['anova_pvalue'] < 0.05

        # --- Stats text column ---
        stats_lines = []

        stats_lines.append(f"Bootstrap modulated: {bootstrap_modulated}")
        stats_lines.append(f"Anova modulated: {anova_modulated}")
        stats_lines.append(f"")

        stats_lines.append(f"Reliable General: {reliable}")
        stats_lines.append(f"Reliable Condition: {results_condition['reliable']}")
        stats_lines.append(f"Correlation: {results_condition['reliable_corr']:.2f}")
        stats_lines.append(f"Bootstrap value: {results_condition['bootstrap_pvalue']:.3f}")
        stats_lines.append(f"Anova pvalue: {results_condition['anova_pvalue']:.3f}")
        stats_lines.append(f"RF: {results_condition['RF']:.2f}")
        stats_lines.append(f"Distance: {np.mean(results_condition['distance']):.2f}")
        stats_lines.append(f"Consistency: {results_condition['consistency']:.2f}%")
        stats_lines.append(f"DTI: {results_condition['DTI']:.2f}")

        stats_text = "\n".join(stats_lines)
        axs[row_idx, 4].axis('off')
        axs[row_idx, 4].text(0, 1, stats_text, va='top', ha='left', fontsize=16, family='monospace')

    def plot_bootstrap_modulation(self, cell):

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for row, (condition_name, monocular) in enumerate([('monocular', True), ('binocular', False)]):

            results = self.analyzer.bootstrap_modulation(cell = cell, monocular = monocular)

            # Plot histogram of preferred distance
            axes[row, 0].hist(results['preferred_distance'], bins=np.arange(5) - 0.5, edgecolor='black', alpha=0.7,
                                weights=np.ones_like(results['preferred_distance']) * 100 / len(results['preferred_distance']))
            axes[row, 0].set_xlabel('Preferred Distance')
            axes[row, 0].set_ylabel('Percentage (%)')
            axes[row, 0].set_title(f'{condition_name}: Preferred Distance', pad=15)
            axes[row, 0].set_xticks([0, 1, 2, 3])
            axes[row, 0].set_ylim(0, 100)

            # Plot histogram of least preferred distance
            axes[row, 1].hist(results['least_preferred_distance'], bins=np.arange(5) - 0.5, edgecolor='black', alpha=0.7,
                                weights=np.ones_like(results['least_preferred_distance']) * 100 / len(results['least_preferred_distance']))
            axes[row, 1].set_xlabel('Least Preferred Distance')
            axes[row, 1].set_ylabel('Percentage (%)')
            axes[row, 1].set_title(f'{condition_name}: Least Preferred Distance', pad=15)
            axes[row, 1].set_xticks([0, 1, 2, 3])
            axes[row, 1].set_ylim(0, 100)

            # Plot histogram of modulation index
            axes[row, 2].hist(results['modulation_index'], bins=np.arange(-1, 1, 0.1), edgecolor='black', alpha=0.7)
            axes[row, 2].set_xlim(-1, 1)
            axes[row, 2].set_xlabel('Modulation Index')
            axes[row, 2].set_ylabel('Count')
            axes[row, 2].set_title(f'{condition_name}: Modulation Index', pad=15)
            axes[row, 2].axvline(0, color='red', linestyle='--')

            # Stats text panel
            axes[row, 3].axis('off')
            stats_text = f"{condition_name} Statistics\n\n"
            stats_text += f"Modulated: {'Yes' if results['bootstrap_modulated'] else 'No'}\n"
            stats_text += f"p-value: {results['bootstrap_p_value']:.4f}\n\n"
            stats_text += f"Mean Pref. Dist: {np.mean(results['preferred_distance']):.2f}\n"
            # Calculate consistency: % of times the most frequent preferred distance occurs
            unique_vals, counts = np.unique(results['preferred_distance'], return_counts=True)
            max_count = np.max(counts)
            consistency = (max_count / len(results['preferred_distance'])) * 100
            stats_text += f"Pref. Dist. Consistency: {consistency:.1f}%\n"
            stats_text += f"Mean Mod. Index: {np.mean(results['modulation_index']):.4f}\n"

            axes[row, 3].text(0, 0.5, stats_text, fontsize=14, verticalalignment='center',
                                family='monospace')

        fig.suptitle(f'Cell {cell} Bootstrap Analysis', fontsize=24)
        plt.tight_layout()
        plt.show()

    def plot_reliability_analysis(self, cell):
        """
        Plot reliability analysis results for both monocular and binocular conditions.
        Shows correlation distributions and RF values comparison with control.

        Parameters
        ----------
        cell : int
            Cell ID to analyze
        """

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for row, (condition_name, monocular) in enumerate([('Monocular', True), ('Binocular', False)]):

            # Run reliability analysis
            results = self.analyzer.analyze_cell_reliability(cell=cell, monocular=monocular, n_iterations=100)

            # Get raw correlation data for plotting
            corr, corr_shifted = self.analyzer.compute_reliability(cell, distance=None, monocular=monocular, n_iterations=100)

            corr_control, corr_shifted_control = self.analyzer.compute_reliability(cell, distance=4, monocular=monocular, n_iterations=100)

            # Plot 1: Stimulus correlation distribution
            axes[row, 0].hist(corr_shifted, bins=np.arange(-1, 1, 0.05), alpha=0.5, label='Shifted (null)',
                            edgecolor='black', color='lightgray')
            axes[row, 0].hist(corr, bins=np.arange(-1, 1, 0.05), alpha=1, label='Stimulus',
                            edgecolor='black', color='lightblue')
            axes[row, 0].axvline(np.mean(corr), color='darkblue', linestyle='--',
                                linewidth=2.5, label=f'Mean: {np.mean(corr):.3f}')
            axes[row, 0].axvline(np.percentile(corr_shifted, 99), color='crimson',
                                linestyle='--', linewidth=2.5, label='99th %ile (null)')
            axes[row, 0].set_xlabel('Correlation', fontsize=14)
            axes[row, 0].set_ylabel('Count', fontsize=14)
            axes[row, 0].set_title(f'{condition_name}: Stimulus Reliability',
                                pad=15, fontsize=16)
            axes[row, 0].legend(fontsize=12, framealpha=0.9)
            axes[row, 0].grid(alpha=0.3, linestyle=':', linewidth=0.5)
            axes[row, 0].spines['top'].set_visible(False)
            axes[row, 0].spines['right'].set_visible(False)

            # Plot 2: Control correlation distribution
            axes[row, 1].hist(corr_shifted_control, bins=np.arange(-1, 1, 0.1), alpha=0.5, label='Shifted (null)',
                            edgecolor='black', color='lightgray')
            axes[row, 1].hist(corr_control, bins=np.arange(-1, 1, 0.1), alpha=0.7, label='Control',
                            edgecolor='black', color='coral')
            axes[row, 1].axvline(np.mean(corr_control), color='darkorange', linestyle='--',
                                linewidth=2.5, label=f'Mean: {np.mean(corr_control):.3f}')
            axes[row, 1].axvline(np.percentile(corr_shifted_control, 99), color='crimson',
                                linestyle='--', linewidth=2.5, label='99th %ile (null)')
            axes[row, 1].set_xlabel('Correlation', fontsize=14)
            axes[row, 1].set_ylabel('Count', fontsize=14)
            axes[row, 1].set_title(f'{condition_name}: Control Reliability',
                                pad=15, fontsize=16)
            axes[row, 1].legend(fontsize=12, framealpha=0.9)
            axes[row, 1].grid(alpha=0.3, linestyle=':', linewidth=0.5)
            axes[row, 1].spines['top'].set_visible(False)
            axes[row, 1].spines['right'].set_visible(False)

            # Plot 3: RF values comparison (distances vs control)
            rf_values_distances = [self.device.find_rf_values(cell, distance=d, monocular=monocular) for d in [0, 1, 2, 3]]
            rf_values_control = self.device.find_rf_values(cell, distance=4, monocular=monocular)

            # Create boxplot for distances
            bp = axes[row, 2].boxplot(rf_values_distances, positions=[0, 1, 2, 3], widths=0.6,
                                    patch_artist=True, showfliers=False)

            # Style the boxplot
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.7)

            # Add scatter points for distances
            for i, rf_values in enumerate(rf_values_distances):
                x = np.random.normal(i, 0.04, size=len(rf_values))
                axes[row, 2].scatter(x, rf_values, alpha=0.5, s=20, color='darkblue')

            # Create boxplot for control
            bp_control = axes[row, 2].boxplot([rf_values_control], positions=[4], widths=0.6,
                                            patch_artist=True, showfliers=False)

            # Style the control boxplot
            for patch in bp_control['boxes']:
                patch.set_facecolor('coral')
                patch.set_alpha(0.7)

            # Add scatter points for control
            x_control = np.random.normal(4, 0.04, size=len(rf_values_control))
            axes[row, 2].scatter(x_control, rf_values_control, alpha=0.5, s=20, color='darkorange')

            axes[row, 2].set_xlabel('Distance')
            axes[row, 2].set_ylabel('RF Value')
            axes[row, 2].set_title(f'{condition_name}: RF Values vs Control', pad=15)
            axes[row, 2].set_xticks([0, 1, 2, 3, 4])
            axes[row, 2].set_xticklabels(['0', '1', '2', '3', 'Control'])

            # Plot 4: Statistics text panel
            axes[row, 3].axis('off')
            stats_text = f"{condition_name} Statistics\n\n"
            stats_text += f"Reliable: {'Yes' if results['reliable'] else 'No'}\n\n"
            stats_text += f"Stimulus Reliable: {'Yes' if results['stimulus_reliable'] else 'No'}\n"
            stats_text += f"Mean Corr: {results['mean_corr']:.4f}\n"
            stats_text += f"99th %ile (null): {np.percentile(corr_shifted, 99):.4f}\n\n"
            stats_text += f"Control Reliable: {'Yes' if results['control_reliable'] else 'No'}\n"
            stats_text += f"Control Higher: {'Yes' if results['control_higher'] else 'No'}\n"

            # Add count of how many distances have lower RF values than control
            control_comparison = [np.mean(rf_values_control) > np.mean(rf_val) for rf_val in rf_values_distances]
            n_lower = sum(control_comparison)
            stats_text += f"Distances < Control: {n_lower}/4\n"

            axes[row, 3].text(0, 0.5, stats_text, fontsize=14, verticalalignment='center',
                            family='monospace')

        fig.suptitle(f'Cell {cell} Reliability Analysis', fontsize=24)
        plt.tight_layout()
        return fig
