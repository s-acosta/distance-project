import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from src.utils import setup_plot_style, import_vars_from_matlab, create_new_result_dir
colors = setup_plot_style()
from path_setup import setup_project_path
from src.ActivityPipelines import ActivityLoader
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import scipy.stats

class RadialDevice:

    def __init__(self, session_path, response_mode='dFF', tau=None):

        self.session_path = session_path
        self.results_path = create_new_result_dir(self.session_path, r'results')
        self.stimulus_path = os.path.join(self.session_path, r'stimulus')
        self.activity_path = os.path.join(self.session_path, r'suite2p\plane0')

        self.response_mode = response_mode

        self._load_activity(tau)
        self._load_stimulus_variables()
        
        if self.response_mode == 'dFF':
            self.response = self.dFF
            self.response = gaussian_filter1d(self.response, 1.5, axis=1)
        elif self.response_mode == 'spikes':
            self.response = self.spikes
            # do z-score
            self.response = (self.response - np.mean(self.response, axis=1, keepdims=True)) / np.std(self.response, axis=1, keepdims=True)
            self.response = gaussian_filter1d(self.response, 1.5, axis=1)
        else:
            raise Exception('Unknown activity mode')
        
        self._synchronize_stimulus_and_activity()
        self._set_on_length()

    def _load_stimulus_container(self):
        
        # check files inside stimulus_path
        filelist = [file for file in os.listdir(self.stimulus_path) if file.endswith(("_stimulus.mat"))]
        if len(filelist) == 0:
            # raise error
            raise Exception('No stimulus file found')
        elif len(filelist) > 1:
            raise Exception('More than one stimulus file found')
        else:
            print('Stimulus file found')
            return io.loadmat(self.stimulus_path + '/' + str(filelist[0]))
        
    def _load_stimulus_variables(self):

        container = self._load_stimulus_container()
        locals_dict = import_vars_from_matlab(container, 'stim')

        self.angle = locals_dict['angle']
        self.time = locals_dict['time']
        self.trial_sequence = locals_dict['trial_sequence']
        self.off_start_time = locals_dict['off_start']
        self.trial_start_time = locals_dict['trial_start_time']
        self.trial_end_time = locals_dict['trial_end_time']

        try:
            self.monocular_sequence = locals_dict['monocular_sequence']
        except:
            self.monocular_sequence = np.zeros_like(self.trial_sequence)
        
    def _load_activity(self, tau=None):

        try:
            D = ActivityLoader(self.session_path, tau)
            self.frames_time = D.frames_time
            self.dFF = D.dFF
            self.spikes = D.spikes
            self.suite2p_id = D.suite2p_id
            print("Successfully loaded activity data")
        except Exception as e:
            print(f"Error loading activity data: {e}")
            raise

    def _synchronize_stimulus_and_activity(self):

        self.response = self.response[:,self.frames_time <= 1000*self.trial_end_time[-1]]
        self.time = np.linspace(0, self.trial_end_time[-1]*1000, len(self.angle))
        self.frames_time = self.frames_time[self.frames_time <= 1000*self.trial_end_time[-1]]
        self.angle_interp = np.interp(self.frames_time, self.time, self.angle)

        print('Synchronized stimulus and activity')

    def _set_on_length(self):

        on_frames_experiment = []
        for repeat in range(len(self.trial_sequence)):
            on_start, on_end = self.get_on_times(repeat)
            on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)
            on_frames_experiment.append(on_frames)

        # Count the number of frames for each length
        frame_counts = {}
        for frames in on_frames_experiment:
            length = np.sum(frames)
            frame_counts[length] = frame_counts.get(length, 0) + 1
        
        # Print the distribution of frame lengths
        print("Repeat stimulus on length distribution:")
        for length, count in sorted(frame_counts.items()):
            print(f"    {count} repeats have length {length}")

        # set the on length to the shortest length
        self.on_length = min(frame_counts.keys())
        print(f'Set stimulus on length: {self.on_length}')
        self.angle_range = np.linspace(-30, 60, self.on_length)

    def get_on_times(self, repeat):
        """
        Returns the start and end times of the stimulus presentation for a specific trial repeat.
        
        Parameters
        ----------
        repeat : int
            The trial repeat index to get stimulus presentation times for
            
        Returns
        -------
        tuple
            (on_start, on_end) where:
            - on_start: timestamp (ms) when the stimulus begins to be presented
            - on_end: timestamp (ms) when the stimulus presentation ends, including any hold time
        """

        on_start = 1000*self.trial_start_time[repeat] + 2500
        on_end = 1000*self.trial_end_time[repeat]

        return on_start, on_end

    def get_on_trace(self, idx, repeat):

        on_start, on_end = self.get_on_times(repeat)
        on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)

        if self.on_length is not None:
            on_frames = self.response[idx, on_frames][:self.on_length]
        else:
            on_frames = self.response[idx, on_frames]

        return on_frames
        
    def get_condition_trials(self, monocular=None, distance=None):

        # find the indices of repeats that match the criteria
        if monocular is not None:
            monocular_indices = np.where(self.monocular_sequence == monocular)[0]
        else:
            monocular_indices = np.arange(len(self.trial_sequence))

        if distance is not None:
            distance_indices = np.where(self.trial_sequence == distance)[0]
        else:
            distance_indices = np.arange(len(self.trial_sequence))

        # do intersection
        repeat_indices = np.intersect1d(monocular_indices, distance_indices)

        # get the intersection of the indices
        return repeat_indices
    
    def get_all_traces(self, cell, distance=None, monocular=None, mean_trace=False):

        idx = np.where(self.suite2p_id == cell)[0][0]

        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance)

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        if mean_trace:
            return np.mean(all_traces, axis=0), np.std(all_traces, axis=0)/np.sqrt(all_traces.shape[0])
        else:
            return all_traces

    def analyze_cell_reliability(self, cell, n_iterations=200, distance=None, monocular=None):
        """
        Analyzes the reliability of a cell's response by comparing split-half correlations.
        
        Args:
            cell: Cell ID to analyze
            n_iterations: Number of iterations for reliability calculation
            distance: Optional distance condition to filter trials
            monocular: Optional monocular condition to filter trials
            
        Returns:
            tuple: (is_reliable, reliability, corr_shifted, corr) where is_reliable indicates if the cell
                   is considered reliable based on correlation analysis
        """
        
        # Get traces for the specified conditions
        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance)
        idx = np.where(self.suite2p_id == cell)[0][0]

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        n_repeats = all_traces.shape[0]
        n_halves = int(np.ceil(n_repeats / 2))
        halves_vector = np.concatenate((np.full(n_halves, True), np.full(n_halves, False)))

        corr = []
        corr_shifted = []
        for i in range(n_iterations):
            # Randomly split trials into two halves
            shuffled_halves = sorted(halves_vector, key=lambda k: np.random.random())[:n_repeats]

            first_half_traces = all_traces[shuffled_halves, :]
            second_half_traces = all_traces[np.invert(shuffled_halves), :]

            # Ensure both halves have same length
            if len(first_half_traces[0]) != len(second_half_traces[0]):
                if len(first_half_traces) > len(second_half_traces):
                    first_half_traces = first_half_traces[:len(second_half_traces[0]), :]
                else:
                    second_half_traces = second_half_traces[:len(first_half_traces[0]), :]

            # Calculate mean traces and correlation
            first_half_mean = np.mean(first_half_traces, axis=0)
            second_half_mean = np.mean(second_half_traces, axis=0)
            corr.append(scipy.stats.pearsonr(first_half_mean, second_half_mean)[0])

            # Calculate shifted correlations for null distribution
            first_half_shifted = []
            second_half_shifted = []
            for trace in first_half_traces:
                circular_shift = np.random.randint(-180, 180)
                first_half_shifted.append(np.roll(trace, circular_shift))
            for trace in second_half_traces:
                circular_shift = np.random.randint(-180, 180)
                second_half_shifted.append(np.roll(trace, circular_shift))

            first_half_shifted_mean = np.mean(first_half_shifted, axis=0)
            second_half_shifted_mean = np.mean(second_half_shifted, axis=0)
            corr_shifted.append(scipy.stats.pearsonr(first_half_shifted_mean, second_half_shifted_mean)[0])

        # Calculate reliability metrics
        reliability = np.mean(corr)
        is_reliable = reliability > np.percentile(corr_shifted, 99)

        return is_reliable, reliability, corr_shifted, corr
    
    def analyze_cell_responsiveness(self, cell, distance=None, monocular=None, history=None, threshold=1):
        """
        Analyzes if a cell is responsive to stimuli under a specific condition.
        
        Args:
            cell: Cell index to analyze
            distance: Distance condition to analyze (None for all distances)
            
        Returns:
            tuple: (is_responsive, p_value, on_max, off_max) where is_responsive is True if the cell is 
                   significantly more active during ON periods (p < 0.05 and ON median > OFF median)
        """
        
        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance)
        
        # Calculate medians for this condition
        on_max = []
        off_max = []

        idx = np.where(self.suite2p_id == cell)[0][0]

        # get the mean trace for this cell
        mean_trace = self.get_all_traces(cell, distance=distance, mean_trace=True, monocular=monocular)[0]
        max_pos = np.argmax(mean_trace)

        for repeat in repeat_indices:
            on_start, on_end = self.get_on_times(repeat)

            on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)   
            off_frames = (self.frames_time > on_start) & (self.frames_time < on_end)

            on_trace = self.response[idx, on_frames]
            off_trace = self.response[idx, off_frames]

            # Take mean of a 5-frame window centered around max_pos
            window_start = max(0, max_pos - 5)
            window_end = min(len(on_trace), max_pos + 5)
            on_max.append(np.mean(on_trace[window_start:window_end]))
            off_max.append(np.mean(off_trace))
        
        # Determine if cell is responsive (p < 0.05 and median ON > median OFF)
        # Ensure max_pos isn't the first or last index to avoid window issues
        if self.response_mode == 'spikes-zscore' or self.response_mode == 'spikes':
            _, p_val = ttest_ind(on_max, off_max)
            is_responsive = (p_val < 0.05) and (np.median(on_max)>np.median(off_max)) and (max_pos < len(on_trace) - 1) and (max_pos > 0)
            return is_responsive, p_val, on_max, off_max
        else:
            is_reliable, reliability, _, _ = self.analyze_cell_reliability(cell, n_iterations=100, distance=None, monocular=monocular, history=history)
            is_responsive = (max_pos > 0) and is_reliable and (np.median(on_max)>np.median(off_max))
            return is_responsive, reliability, on_max, off_max

    def fit_gaussian(self, cell, distance=None, monocular=None, history=None):

        def gaussian(x, a, mu, sigma, baseline):
            return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + baseline
        
        mean_trace = self.get_all_traces(cell, distance=distance, monocular=monocular, history=history, mean_trace=True)[0]
        x = np.arange(len(mean_trace))

        p0 = [np.max(mean_trace) - np.min(mean_trace), np.argmax(mean_trace), 5, np.min(mean_trace)]
        
        try:
            popt, _ = curve_fit(gaussian, x, mean_trace, p0=p0)
            a, mu, sigma, baseline = popt

            # Calculate R-squared
            y_fit = gaussian(x, a, mu, sigma, baseline)
            residuals = mean_trace - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mean_trace - np.mean(mean_trace))**2)
            r_squared = 1 - (ss_res / ss_tot)
        except:
            mu = np.nan
            r_squared = 0
        
        return mu, r_squared
    
    def analyze_cell_modulation(self, cell, monocular=None):

        window_size = 3

        distances = np.unique(self.trial_sequence)
        # get the mean trace for this cell
        mean_trace = self.get_all_traces(cell, mean_trace=True, monocular=monocular)[0]

        # find the receptive fied for this cell
        rf_idx = np.argmax(mean_trace)
        rf_position = self.angle_range[rf_idx]

        rf_values = []
        for distance in distances:
            on_traces = self.get_all_traces(cell,  
                                     distance=distance,
                                     monocular=monocular,
                                     )
            # Find the max value within window +5 -5 frames from rf_index
            window_start = max(0, rf_idx - window_size)
            window_end = min(on_traces.shape[1], rf_idx + window_size + 1)
            on_traces_max = np.mean(on_traces[:, window_start:window_end], axis=1)
            rf_values.append(on_traces_max)

        f_stat, p_value = f_oneway(*rf_values)
        # preffered distance is the one with the highest mean response
        # Handle case where rf_values may have different lengths
        means = np.array([np.mean(values) for values in rf_values])
        
        # Split trials into two halves for cross-validation
        half_rf_values = []
        for values in rf_values:
            n_trials = len(values)
            # Create a random permutation of indices
            random_indices = np.random.permutation(n_trials)
            half_idx = n_trials // 2
            # Use the randomized indices to split the data
            first_half = values[random_indices[:half_idx]]
            second_half = values[random_indices[half_idx:]]
            half_rf_values.append([first_half, second_half])
        
        # Use first half to determine preferred distance
        first_half_means = np.array([np.mean(values[0]) for values in half_rf_values])
        preferred_distance = np.argmax(first_half_means)
        least_preferred_distance = np.argmin(first_half_means)
        
        # Use second half to calculate modulation strength
        preferred_mean = np.mean(half_rf_values[preferred_distance][1])
        least_preferred_mean = np.mean(half_rf_values[least_preferred_distance][1])
        
        # Calculate modulation strength using the second half data
        modulation_strength = (np.abs(preferred_mean) - np.abs(least_preferred_mean)) / (np.abs(preferred_mean) + np.abs(least_preferred_mean) + 1e-10)  # Avoid division by zero

        return p_value, preferred_distance, modulation_strength, rf_values, rf_position
    
    def analyze_condition_stats(self, monocular=None):
        
        # Define condition name based on monocular and history options
        if monocular is None:
            condition_name = 'all_conditions'
        elif monocular:
            condition_name = 'monocular'
        else:
            condition_name = 'binocular'
            
        # Initialize empty lists to collect data for each cell
        cell_data = []
        
        # Analyze each cell
        for cell in self.suite2p_id:
            
            # Analyze condition
            is_reliable, reliability, _, _ = self.analyze_cell_reliability(cell, monocular=monocular)
            p_value, pref_distance, mod_strength, _, rf_position = self.analyze_cell_modulation(cell, monocular=monocular)
            
            # Create a row for this cell
            cell_row = {
                'cell_id': cell,
                'is_reliable': is_reliable,
                'reliable': reliability,
                'modulated': np.bool(p_value < 0.05),
                'modulated_p': p_value,
                'mod_strength': mod_strength,
                'pref_distance': pref_distance,
                'rf_position': rf_position,
                'condition': condition_name
            }
            
            cell_data.append(cell_row)
        
        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)
        
        # Calculate summary statistics
        n_cells = len(stats_df)
        n_responsive = stats_df['is_reliable'].sum()
        n_modulated = stats_df['modulated'].sum()

        # Print summary statistics
        print(f'Condition: {condition_name}')
        print(f"Total cells: {n_cells}")
        print(f"Responsive cells: {n_responsive} ({n_responsive / n_cells * 100:.1f}%)")
        print(f"Modulated cells: {n_modulated} ({n_modulated / n_responsive * 100:.1f}%)")
        
        return stats_df
    
class RadialPlots:

    def __init__(self, radial_class):
        self.R = radial_class
        self.colors = ['orange', '#35b779', '#31688e', '#440154']
        pass

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
            idx = np.where(self.R.suite2p_id == cell)[0][0]
    
        # Handle both single number and range inputs for n_repeats
        if isinstance(repeats, (list, tuple)) and len(repeats) == 2:
            repeats = np.arange(repeats[0], repeats[1])
        else:
            repeats = np.arange(repeats)

        repeat_start = self.R.trial_start_time[repeats[0]]*1000
        repeat_end = self.R.trial_end_time[repeats[-1]]*1000

        repeats_frames = (self.R.frames_time > repeat_start) & (self.R.frames_time < repeat_end)

        ax = plt.gca()

        if cell is None:
            ax.plot(self.R.frames_time[repeats_frames]/1000, self.R.angle_interp[repeats_frames], lw=2, color='black')
        else:
            ax.plot(self.R.frames_time[repeats_frames]/1000, self.R.response[idx, repeats_frames], lw=1.5, color='black')

        for rep in repeats:
            on_start, on_end = self.R.get_on_times(rep)
            if self.R.monocular_sequence[rep] == 1:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.R.trial_sequence[rep]], alpha=0.5)
            else:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.R.trial_sequence[rep]], alpha=0.5, hatch='///')

        #if cell is not None, set the ylim to the 99 and 1 percentiles of the response
        if cell is not None:
            ax.set_ylim(np.min(self.R.response[idx, :]), np.max(self.R.response[idx, :]))

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
            plt.ylabel(self.R.response_mode)
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
        idx = np.where(self.R.suite2p_id == cell)[0][0]
        # Handle the case of a single distance (1D array of axes)
        if distance == None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 4), 
                               gridspec_kw={'width_ratios': [2, 1]})
            axs = np.array([axs])
        else:
            fig, axs = plt.subplots(len(distance), 2, figsize=(14, len(distance)*3.5), 
                               gridspec_kw={'width_ratios': [2, 1]})
       
        vmin, vmax = np.percentile(self.R.response[idx], [1, 99])

        if distance is None:
            distance = [None]
        
        for i, dist in enumerate(distance):

            repeat_indices = self.R.get_condition_trials(monocular=monocular, distance=dist)

            repeat_traces = []
            for repeat in repeat_indices:
                on_start, on_end = self.R.get_on_times(repeat)
                frames = (self.R.frames_time > (on_start-pre_time)) & (self.R.frames_time < (on_end+post_time))
                trace_length = self.R.on_length + int((pre_time + post_time)/100)
                
                # Make sure we don't exceed the available frames
                trace = self.R.response[idx, frames]
                if len(trace) >= trace_length:
                    repeat_traces.append(trace[:trace_length].T)
            
            if not repeat_traces:
                continue
                
            repeat_traces = np.array(repeat_traces)
            
            # Plot heatmap
            if self.R.response_mode == 'spikes-zscore' or self.R.response_mode == 'spikes':
                cmap = 'viridis'
            elif self.R.response_mode == 'dFF' or self.R.response_mode == 'dFF-zscore':
                cmap = 'viridis'

            axs[i, 0].pcolormesh(repeat_traces, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[i, 0].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].axvline(x=(pre_time/100+self.R.on_length), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].set_xlabel('Time (frames)')
            axs[i, 0].set_ylabel('Trials')
            axs[i, 0].set_title(f'All Traces')

            repeat_traces = []
            for repeat in repeat_indices:
                on_start, on_end = self.R.get_on_times(repeat)
                frames = (self.R.frames_time > (on_start)) & (self.R.frames_time < (on_end))
                trace_length = self.R.on_length
                repeat_traces.append(self.R.response[idx, frames][:trace_length].T)

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
            axs[i, 1].set_ylabel(self.R.response_mode)
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
        Create a colormap visualization of neural traces for a given cell.
        
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
        
        # Get cell index
        idx = np.where(self.R.suite2p_id == cell)[0][0]
        
        # Define distances to plot
        distance = [0, 1, 2, 3]
        
        # Create figure with a row for each condition, 2 columns per row
        fig, axs = plt.subplots(2, 4, figsize=(15, 8), 
                               gridspec_kw={'width_ratios': [1, 1, 1, 1]})
        # remove axis ticks from second subplot
        axs[0, 1].remove()
        axs[1, 1].remove()
        
        conditions = [
            {'monocular': True,  'title': 'Monocular'},
            {'monocular': False, 'title': 'Binocular'},
        ]
       
        all_traces = self.R.get_all_traces(cell, distance=None, monocular=None, mean_trace=False)
        vmin, vmax = np.percentile(all_traces, [1, 99])

        # Process each condition
        for c_idx, condition in enumerate(conditions):
            repeat_traces = []
            repeats_per_distance = []
            
            for i, dist in enumerate(distance):
               
                repeat_indices = self.R.get_condition_trials(
                    monocular=condition['monocular'], 
                    distance=dist
                )

                for repeat in repeat_indices:
                    repeat_traces.append(self.R.get_on_trace(idx, repeat))
                

                repeats_distance_mean, repeats_distance_sem = self.R.get_all_traces(cell, distance=dist, monocular=condition['monocular'], mean_trace=True)

                axs[c_idx, 2].plot(self.R.angle_range, repeats_distance_mean, 
                                  color=self.colors[dist], 
                                  label=f'Distance {dist}', 
                                  linewidth=3)
                axs[c_idx, 2].fill_between(
                    self.R.angle_range, 
                    repeats_distance_mean - repeats_distance_sem, 
                    repeats_distance_mean + repeats_distance_sem, 
                    color=self.colors[dist], alpha=0.2
                )


            # Add horizontal lines to mark different distances
            for i, n_repeats in enumerate(repeats_per_distance):
                if i > 0:
                    y_pos = np.sum(repeats_per_distance[:i])
                    axs[c_idx, 0].axhline(y=y_pos, color=self.colors[i-1], linestyle='--', linewidth=2)
            
            # Plot heatmap
            if self.R.response_mode == 'spikes-zscore' or self.R.response_mode == 'spikes':
                cmap = 'viridis'
            elif self.R.response_mode == 'dFF' or self.R.response_mode == 'dFF-zscore':
                cmap = 'viridis'

            axs[c_idx, 0].pcolormesh(repeat_traces, cmap=cmap, vmin=vmin, vmax=vmax)

            axs[c_idx, 0].set_xlabel('Time (frames)')
            axs[c_idx, 0].set_ylabel('Trials')
            axs[c_idx, 0].set_title(f'{condition["title"]} - All Traces', fontsize=14)

            # radial plot
            n_theta = len(self.R.angle_range)
            theta_edges = np.linspace(np.radians(self.R.angle_range[0]), np.radians(self.R.angle_range[-1]), n_theta + 1)
            r_edges = np.linspace(0, len(distance), len(distance) + 1)
            theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
            
            matrices = [self.R.get_all_traces(cell=cell, monocular=condition['monocular'], distance=dist, mean_trace=True)[0] for dist in distance]
            Z = np.vstack(matrices)

            # # Make this axis polar
            # Create a polar axis in place of the corresponding cartesian axis
            ax_polar = plt.subplot2grid(axs.shape, (c_idx, 1), projection='polar', fig=fig)
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            ax_polar.set_thetamin(-30)
            ax_polar.set_thetamax(60)
            # Plot on the polar axis
            pcm = ax_polar.pcolormesh(theta_grid, r_grid, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Optionally, copy over the title
            ax_polar.set_title(f'{condition["title"]} - Mean Traces', fontsize=14)
            
            # Add axis for second subplot
            axs[c_idx, 2].set_xlabel('Degrees (deg)')
            axs[c_idx, 2].set_ylabel(self.R.response_mode)
            axs[c_idx, 2].set_title(f'{condition["title"]} - Mean Traces', fontsize=14)
            # add vertical line at 30 degrees
            axs[c_idx, 2].axvline(x=30, color='lightgray', linestyle='--', linewidth=2)
            axs[c_idx, 2].set_ylim(vmin, vmax)

            # boxplot
            p_value, _, _, rf_values, _ = self.R.analyze_cell_modulation(cell, monocular=condition['monocular'])

            bp = axs[c_idx, 3].boxplot(rf_values, patch_artist=True, labels=distance, showfliers=False)
            
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
            for box in bp['boxes']:
                box.set_facecolor('none')

            for i, rf_vals in enumerate(rf_values):
                x = np.random.normal(i+1, 0.04, size=len(rf_vals))
                axs[c_idx, 3].scatter(x, rf_vals, alpha=0.8, color=colors[i], s=40)

            # # Check if p_value is significant and if the cell is responsive for the monocular condition
            if p_value < 0.05 and self.R.analyze_cell_reliability(cell, monocular=condition['monocular'])[0]:
                bar_height = vmax * 0.8
                axs[c_idx, 3].plot([1, len(distance)], [bar_height, bar_height], 'k-', linewidth=1.5)
                axs[c_idx, 3].text((1 + len(distance))/2, bar_height*1.02, '*', ha='center', va='bottom')

            axs[c_idx, 3].set_title(f'p={p_value:.3f}', fontsize=14)
            axs[c_idx, 3].set_xlabel('Distance')
            axs[c_idx, 3].set_ylabel(self.R.response_mode)
            axs[c_idx, 3].set_ylim(vmin, vmax)


        # Add suptitle to the figure
        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout()

        return fig
    
    def plot_distance_and_rf_comparison(self, cell, conditions=None):
        
        """
        Plot distance traces and RF values across distances for all conditions.
        
        Parameters:
        -----------
        cell : int
            The cell ID to analyze.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """

        idx = np.where(self.L.suite2p_id == cell)[0][0]

        if conditions is None:
            conditions = [
                {'history': False, 'monocular': True, 'title': 'No History (Monocular)'},
                {'history': False, 'monocular': False, 'title': 'No History (Binocular)'},
            ]
        
    
        # Create figure with 2 rows, 4 columns
        fig = plt.figure(figsize=(4*len(conditions), 8))
        gs = fig.add_gridspec(2, len(conditions))
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(conditions))]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(len(conditions))]
        
        plt.suptitle(f'Cell {cell}', fontsize=24)
        
        colors = setup_plot_style()
        distances = [0, 1, 2, 3]

        all_traces = self.L.get_all_traces(cell, distance=None, monocular=None, history=None, mean_trace=False).flatten()
        
        # Calculate global min/max for top row
        vmin_top = np.percentile(all_traces, 0)
        vmax_top = np.percentile(all_traces, 100)

        vmin_bottom = np.min(all_traces)
        vmax_bottom = np.max(all_traces)

        # Plot both rows
        for j, condition in enumerate(conditions):

            monocular_condition_name = 'monocular' if condition['monocular'] else ''
            history_condition_name = 'history' if condition['history'] else 'no_history'

            # Top row - distance traces
            ax_top = axes_top[j]
            for i, distance in enumerate(distances):
                mean_trace, sem_trace = self.L.get_all_traces(cell, distance=distance,
                                                    monocular=condition['monocular'],
                                                    history=condition['history'],
                                                    mean_trace=True)
                
                ax_top.plot(mean_trace.T, color=colors[i], linewidth=2, label=f'Distance {distance}')
                ax_top.fill_between(range(len(mean_trace)), 
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color=colors[i], alpha=0.2)
            
            ax_top.set_xlabel('Time (frames)')
            ax_top.set_ylabel(self.L.response_mode)
            ax_top.set_title(f'{monocular_condition_name} - {history_condition_name}')
            # ax_top.legend()
            ax_top.set_ylim(vmin_top, vmax_top)

            # Bottom row - RF values
            ax_bottom = axes_bottom[j]
            ax_bottom.set_ylim(vmin_bottom, vmax_bottom)

            p_value, _, _, rf_values = self.L.analyze_cell_modulation(cell, monocular=condition['monocular'], history=condition['history'])

            bp = ax_bottom.boxplot(rf_values, patch_artist=True, labels=distances, showfliers=False)
            
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
            for box in bp['boxes']:
                box.set_facecolor('none')

            for i, rf_vals in enumerate(rf_values):
                x = np.random.normal(i+1, 0.04, size=len(rf_vals))
                ax_bottom.scatter(x, rf_vals, alpha=0.8, color=colors[i], s=40)

            # indicate rf_idx with a vertical line
            # get the mean trace for this cell
            rf_idx = np.argmax(self.L.get_all_traces(cell, mean_trace=True, monocular=None)[0])
            ax_top.axvline(x=rf_idx, color='grey', linestyle='--', linewidth=1.5)

            # # Check if p_value is significant and if the cell is responsive for the monocular condition
            if p_value < 0.05 and self.L.analyze_cell_responsiveness(cell, monocular=condition['monocular'], history=condition['history'])[0]:
                bar_height = vmax_bottom * 0.95
                ax_bottom.plot([1, len(distances)], [bar_height, bar_height], 'k-', linewidth=1.5)
                ax_bottom.text((1 + len(distances))/2, bar_height*1.02, '*', ha='center', va='bottom')

            ax_bottom.set_title(f'p={p_value:.3f}')
            ax_bottom.set_xlabel('Distance')
            ax_bottom.set_ylabel(self.L.response_mode)
            

        plt.tight_layout()
        return fig
    
class RadialSummaries():

    def __init__(self, radial_class, radial_stats):
        self.R = radial_class
        self.radial_stats = radial_stats
        self.session_name = self.R.session_path.split('/')[-1]
        self.colors = setup_plot_style()

        self.responsive_cells, self.modulated_cells = self.find_modulated_cells()
        self.non_modulated_cells = self.find_non_modulated_cells()

    # def find_responsive_cells(self):

    #     # find the cells that are responsive in any condition
    #     responsive_cells = []
    #     for stat_df in self.linear_stats:
    #         responsive_cells.append(stat_df[stat_df['responsive']==True]['cell_id'].values)
    #     responsive_cells = np.concatenate(responsive_cells)
    #     responsive_cells = np.unique(responsive_cells)

    #     return responsive_cells

    def find_non_modulated_cells(self):
        non_modulated_cells = {}
        for i, stat_df in enumerate(self.radial_stats):
            condition_name = stat_df['condition'].iloc[0] if not stat_df.empty else f"condition_{i}"
            # Get cells that are responsive but not modulated
            non_modulated_cells[condition_name] = stat_df[(stat_df['reliable']==True) & (stat_df['modulated']==False)]['cell_id'].values

        # Find cells that are non-modulated in all conditions
        all_conditions = set.intersection(*[set(cells) for cells in non_modulated_cells.values()]) if non_modulated_cells else set()
        non_modulated_cells = np.array(list(all_conditions))

        return non_modulated_cells
    
    def find_modulated_cells(self):

        # find the cells that are modulated in any condition
        modulated_cells = {}
        responsive_cells = {}
        for i, stat_df in enumerate(self.radial_stats):
            condition_name = stat_df['condition'].iloc[0] if not stat_df.empty else f"condition_{i}"
            modulated_cells[condition_name] = stat_df[(stat_df['modulated']==True) & (stat_df['is_reliable']==True)]['cell_id'].values
            responsive_cells[condition_name] = stat_df[stat_df['is_reliable']==True]['cell_id'].values

        # # Find cells that are modulated in all conditions
        # all_conditions = set.intersection(*[set(cells) for cells in modulated_cells.values()]) if modulated_cells else set()
        # modulated_cells['all_conditions'] = np.array(list(all_conditions))

        # # Find cells that are responsive in all conditions
        # all_conditions = set.intersection(*[set(cells) for cells in responsive_cells.values()]) if responsive_cells else set()
        # responsive_cells['all_conditions'] = np.array(list(all_conditions))
              
        # find the non-modulated, that is the responsive cells that are not in any categorymodulated cells
        non_modulated = np.setdiff1d(responsive_cells['all_conditions'], 
                                   np.concatenate([cells for cells in modulated_cells.values()]))
        # modulated_cells['non_modulated'] = non_modulated
        
        return responsive_cells, modulated_cells

    def plot_modulation_bars(self):
        
        # Create and configure bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(self.session_name, fontsize=16)

        labels = list(self.modulated_cells.keys())
        responsive_counts = [len(self.responsive_cells[label]) for label in labels]
        modulated_counts = [len(self.modulated_cells[label]) for label in labels]
        percentages = [mod/resp*100 for mod, resp in zip(modulated_counts, responsive_counts)]

        # Create stacked bar chart
        bars = ax.bar(labels, responsive_counts, color='lightgray', label='Responsive')
        modulated_bars = ax.bar(labels, modulated_counts, color='#4CAF50', label='Modulated')
        
        # Add value labels on top of bars
        for bar, percentage, count, resp_count in zip(bars, percentages, modulated_counts, responsive_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%\n({count}/{resp_count})',
                   ha='center', va='bottom', fontsize=12,
                   )

        # Customize the plot
        ax.set_ylabel('Number of Cells', fontsize=14)
        ax.set_ylim(0, max(responsive_counts) * 1.2)  # Add 20% space for labels
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

        return fig
    
    def plot_responsive_bars(self):

        # Create and configure bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(self.session_name, fontsize=16)

        labels = list(self.modulated_cells.keys())
        responsive_counts = [len(self.responsive_cells[label]) for label in labels]
        total_counts = [len(self.R.suite2p_id) for label in labels]

        # Create stacked bar chart
        bars = ax.bar(labels, total_counts, color='lightgray', label='Total')
        responsive_bars = ax.bar(labels, responsive_counts, color='blue', label='Responsive')

        # Add value labels on top of bars
        for bar, count, total_count in zip(bars, responsive_counts, total_counts):
            height = bar.get_height()
            percentage = (count/total_count)*100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%\n({count}/{total_count})',
                   ha='center', va='bottom', fontsize=12,
                   )

        # Customize the plot
        ax.set_ylabel('Number of Responsive Cells', fontsize=14)
        ax.set_ylim(0, total_counts[0] * 1.2)  # Add 20% space for labels
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

        return fig

        
    def plot_responsive_pie_chart(self):

        # Create a larger figure with more space for the pie charts
        fig, axes = plt.subplots(1, 1+len(self.radial_stats), figsize=(12, 2*len(self.radial_stats)))

        # first axes will be the total responsive pie chart
        # Extract unique responsive cells across conditions
        responsive_cells = []
        for i, stat_df in enumerate(self.radial_stats):
            responsive = set(stat_df.loc[stat_df['is_reliable'] == True, 'cell_id'].values)
            responsive_cells.append(np.array(list(responsive)))

        # concatenate the arrays
        responsive_cells = np.concatenate(responsive_cells)
        # find the unique cells
        responsive_cells = np.unique(responsive_cells)

        axes[0].pie([len(responsive_cells), len(self.R.suite2p_id) - len(responsive_cells)], 
                        labels=[f'Responsive ({len(responsive_cells)}/{len(self.R.suite2p_id)})', f'Non-reliable ({len(self.R.suite2p_id) - len(responsive_cells)}/{len(self.R.suite2p_id)})'],
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        startangle=90,
                        shadow=True,
                        textprops={'fontsize': 16})
        axes[0].set_title('Either condition', fontsize=20)

        # Data for pie charts from stats DataFrame
        responsive_counts = []
        non_responsive_counts = []

        for i, df in enumerate(self.radial_stats):

            condition_name = df['condition'][0]
            responsive_count = df['is_reliable'].sum()
            total_count = len(df)
            non_responsive_count = total_count - responsive_count
            
            responsive_counts.append(responsive_count)
            non_responsive_counts.append(non_responsive_count)
            
            axes[i+1].pie([responsive_count, non_responsive_count], 
                        labels=[f'Responsive ({responsive_count}/{total_count})', f'Non-reliable ({non_responsive_count}/{total_count})'],
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        startangle=90,
                        shadow=True,
                        textprops={'fontsize': 16})
            axes[i+1].set_title(f'{condition_name}', fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)  # Add more space between subplots
        plt.show()

        return fig

    def plot_modulation_strength(self):

        # Create a boxplot of modulation strength for monocular, binocular, both conditions, and non-modulated
        n_conditions = len(self.modulated_cells.keys()) + 1
        fig, ax = plt.subplots(figsize=(2*n_conditions, 6))

        # Prepare data for boxplot
        data_for_boxplot = []
        box_labels = []
        box_colors = ['#FF9999', '#9999FF', '#CC99CC', '#CCCCCC']  # Predefined colors

        conditions = list(self.modulated_cells.keys()) + ['non_modulated']
        cells_to_plot = [self.modulated_cells[condition] for condition in conditions[:-1]] + [self.non_modulated_cells]

        # for every condition in modulated_cells, get the modulation strength
        for i, condition in enumerate(conditions):
            print(condition)
            cells_strenght = []
            for cell in cells_to_plot[i]:

                if condition == 'non_modulated' or condition == 'all_conditions':
                    strength = self.radial_stats[-1].loc[self.radial_stats[-1]['cell_id'] == cell, 'mod_strength'].values[0]
                else:
                    strength = self.radial_stats[i].loc[self.radial_stats[i]['cell_id'] == cell, 'mod_strength'].values[0]
                
                cells_strenght.append(strength)
            
            data_for_boxplot.append(np.array(cells_strenght))
            box_labels.append(condition)

        bp = ax.boxplot(data_for_boxplot, patch_artist=True, labels=box_labels, showfliers=False)

        # Customize boxplot appearance
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=box_colors[i % len(box_colors)], alpha=0.7)
            
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        # Add individual points with jitter
        for i, data in enumerate(data_for_boxplot):
            # Create jitter effect
            x = np.random.normal(i+1, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.7, s=40, color=box_colors[i % len(box_colors)], edgecolor='black')

        # Add labels and title
        ax.set_xlabel('Condition', fontsize=16)
        ax.set_xticklabels(box_labels, fontsize=12)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Modulation Strength')
        ax.set_title('Modulation Strength by Condition')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show plot
        plt.tight_layout()
        plt.show()

        return fig

    def plot_preferred_distance(self):

        # plot a barplot of the preferred distance distribution for each condition
        # check number of conditions and make figure accordingly
        n_conditions = len(self.modulated_cells.keys())
        fig, ax = plt.subplots(figsize=(3*n_conditions, 6))

        # Prepare data for barplot
        # Get preferred distances for each condition
        distances = [0, 1, 2, 3]
        conditions = list(self.modulated_cells.keys()) + ['non_modulated']
        cells_to_plot = [self.modulated_cells[condition] for condition in conditions[:-1]] + [self.non_modulated_cells]
        
        # Initialize lists to store data
        all_pref_distances = []
        
        # Process each condition from modulated_cells
        for i, condition in enumerate(conditions):
            cells_pref_distances = []
            for cell in cells_to_plot[i]:
                if condition == 'non_modulated' or condition == 'all_conditions':
                    pref_distance = self.radial_stats[-1].loc[self.radial_stats[-1]['cell_id'] == cell, 'pref_distance'].values[0]
                else:
                    pref_distance = self.radial_stats[i].loc[self.radial_stats[i]['cell_id'] == cell, 'pref_distance'].values[0]
                
                cells_pref_distances.append(pref_distance)
            
            all_pref_distances.append(np.array(cells_pref_distances))
        
        # Count cells for each distance in each condition
        counts_by_distance = []
        for pref_distances in all_pref_distances:
            counts = [sum(pref_distances == d) for d in distances]
            counts_by_distance.append(counts)
        
        # Set width of bars
        bar_width = 0.2
        r1 = np.arange(len(conditions))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]
        
        # Create grouped bar plot
        for i, distance in enumerate(distances):
            position = [r1, r2, r3, r4][i]
            counts = [counts_by_distance[j][i] for j in range(len(conditions))]
            ax.bar(position, counts, width=bar_width, color=self.colors[i], 
                    edgecolor='black', label=f'Distance {distance}')
        
        # Add labels and title
        ax.set_ylabel('Number of Cells')
        ax.set_xlabel('Condition')
        ax.set_xticklabels(conditions, fontsize=12)
        ax.set_title('Distribution of Preferred Distances by Condition')
        ax.set_xticks([r + 1.5 * bar_width for r in range(len(conditions))])
        ax.legend()
        
        # Add cell counts as text above each bar
        for i, distance in enumerate(distances):
            position = [r1, r2, r3, r4][i]
            counts = [counts_by_distance[j][i] for j in range(len(conditions))]
            for j, count in enumerate(counts):
                if count > 0:
                    ax.text(position[j], count + 0.3, f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

        return fig
    

    def plot_receptive_field_map(self, condition):

        cells_to_plot = self.responsive_cells[condition]
        # Filter stats for given condition
        stats_df = self.radial_stats['condition'==condition]

        # Load Suite2p outputs
        stat = np.load(self.R.activity_path + r'\stat.npy', allow_pickle=True)
        ops = np.load(self.R.activity_path + r'\ops.npy', allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        n_angles = len(self.R.angle_range)
        cmap = plt.get_cmap('viridis', n_angles)
        angle_colors = [cmap(i) for i in range(n_angles)]

        # Scatter plot each pixel
        for cell_id in cells_to_plot:
            idx = np.where(self.R.suite2p_id == cell_id)[0][0]
            ypix = stat[idx]['ypix'][~stat[idx]['overlap']].mean()
            xpix = stat[idx]['xpix'][~stat[idx]['overlap']].mean()

            rf_location = stats_df.loc[stats_df['cell_id'] == cell_id, 'rf_position'].values[0]

            # Map rf_location (an angle) to the closest angle in self.R.angle_range, then use that index for color
            angle_diffs = np.abs(np.array(self.R.angle_range) - rf_location)
            color_idx = int(np.argmin(angle_diffs))
            color = angle_colors[color_idx]
            plt.scatter(xpix, ypix, c=[color], s=100, alpha=1, edgecolor='black', linewidth=1)




        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        
    
    def plot_location_map(self, condition):

        # Plot all responsive cells using scatter at pixel locations
        cells_to_plot = self.responsive_cells[condition]

        # Filter stats for given condition
        stats_df = self.radial_stats['condition'==condition]

        # Load Suite2p outputs
        stat = np.load(self.R.activity_path + r'\stat.npy', allow_pickle=True)
        ops = np.load(self.R.activity_path + r'\ops.npy', allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        # Scatter plot each pixel
        for cell_id in cells_to_plot:
            idx = np.where(self.R.suite2p_id == cell_id)[0][0]
            ypix = stat[idx]['ypix'][~stat[idx]['overlap']].mean()
            xpix = stat[idx]['xpix'][~stat[idx]['overlap']].mean()

            is_modulated = stats_df.loc[stats_df['cell_id'] == cell_id, 'modulated'].values[0]

            # if cell is modulated, plot in green, if not, plot in blue
            if is_modulated:
                plt.scatter(xpix, ypix, c='yellow', s=100, alpha=1)
            else:
                plt.scatter(xpix, ypix, c='blue', s=100, alpha=1)

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

        return fig




