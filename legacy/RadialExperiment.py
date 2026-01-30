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
from scipy.stats import f_oneway, ttest_ind, linregress, binned_statistic_2d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import scipy.stats
import cv2
from matplotlib.widgets import Slider
import pickle
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Polygon
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap

class RadialDevice:

    def __init__(self, data_dir=None, response_mode='spikes', tau=None,
                 mouse_name=None, record_date=None, experiment_name=None,
                 base_data_dir=r'D:/data-distance-coding-project/processed-data'):
        """
        Initialize RadialDevice.
        
        Can be called two ways:
        1. Direct path: RadialDevice(data_dir='path/to/data')
        2. Session info: RadialDevice(mouse_name='SAM041', record_date='251120', 
                                      experiment_name='tiling_4_120um')
        
        Parameters:
        -----------
        data_dir : str, optional
            Direct path to data directory
        response_mode : str
            'spikes' or other response mode
        mouse_name : str, optional
            Mouse identifier (e.g., 'SAM041')
        record_date : str, optional
            Recording date (e.g., '251120')
        experiment_name : str, optional
            Experiment name (e.g., 'tiling_4_120um')
        base_data_dir : str
            Base directory for processed data (only used with session info)
        """
        
        # Build data_dir from session info if provided
        if data_dir is None and all([mouse_name, record_date, experiment_name]):
            self.mouse_name = mouse_name
            self.record_date = record_date
            self.experiment_name = experiment_name
            self.session_name = f'{record_date}_{mouse_name}_{experiment_name}'
            data_dir = os.path.join(base_data_dir, self.session_name)
            print(f"[ok] Loading session: {self.session_name}")
        elif data_dir is None:
            raise ValueError(
                "Must provide either:\n"
                "  - data_dir, OR\n"
                "  - mouse_name, record_date, and experiment_name"
            )
        
        # Verify directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory does not exist: {data_dir}")
        
        self.data_dir = data_dir
        self.suite2p_dir = os.path.join(data_dir, 'suite2p')
        self.stimulus_dir = os.path.join(data_dir, 'stimulus')
        self.results_dir = create_new_result_dir(data_dir, 'results')

        self.colors = setup_plot_style()
    
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
        
        self._set_on_length()

    def _load_stimulus_container(self):
        
        # check files inside stimulus_path
        filelist = [file for file in os.listdir(self.stimulus_dir) if file.endswith(("_stimulus.mat"))]
        if len(filelist) == 0:
            # raise error
            raise Exception('No stimulus file found')
        elif len(filelist) > 1:
            raise Exception('More than one stimulus file found')
        else:
            print('Stimulus file found')
            return io.loadmat(self.stimulus_dir + '/' + str(filelist[0]))
        
    def _load_stimulus_variables(self):

        container = self._load_stimulus_container()
        locals_dict = import_vars_from_matlab(container, 'distance_stim')

        self.distances_showed = locals_dict['distances']
        self.time_per_trial = locals_dict['time_per_trial']
        self.n_repeats = locals_dict['n_repeats']
        self.start_angle = locals_dict['start_angle']
        self.end_angle = locals_dict['end_angle']
        self.off_time = locals_dict['off_time']
        self.raise_time = locals_dict['raise_time']
        self.lower_time = locals_dict['lower_time']
        self.forward_speed = locals_dict['forward_speed']
        self.return_speed = locals_dict['return_speed']
        self.monocular_sequence = locals_dict['monocular_sequence']
        self.distance_sequence = locals_dict['distance_sequence']
        self.off_start = locals_dict['off_start']
        self.on_start = locals_dict['on_start']
        self.on_end = locals_dict['on_end']

        try:
            self.monocular_sequence = locals_dict['monocular_sequence']
        except:
            self.monocular_sequence = np.zeros_like(self.distance_sequence)
        
    def _load_activity(self, tau=None):

        try:
            D = ActivityLoader(self.data_dir, tau)
            self.frames_time = D.frames_time
            self.dFF = D.dFF
            self.spikes = D.spikes
            self.suite2p_id = D.suite2p_id
            print("Successfully loaded activity data")
        except Exception as e:
            print(f"Error loading activity data: {e}")
            raise

    def _set_on_length(self):

        on_frames_experiment = []
        for repeat in range(len(self.distance_sequence)):
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
        self.angle_range = np.linspace(self.start_angle, self.end_angle, int(self.on_length))

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

        on_start = 1000*self.on_start[repeat] 
        on_end = 1000*self.on_end[repeat]

        return on_start, on_end

    def get_on_trace(self, idx, repeat):

        on_start, on_end = self.get_on_times(repeat)
        on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)

        if self.on_length is not None:
            on_frames = self.response[idx, on_frames][:self.on_length]
        else:
            on_frames = self.response[idx, on_frames]

        return on_frames

    def get_repeat_trace(self, cell, repeat):

        idx = np.where(self.suite2p_id == cell)[0][0]
        
        return self.get_on_trace(idx, repeat)
        
    def get_condition_trials(self, monocular=None, distance=None):

        control_indices = np.where(self.distance_sequence == 4)[0]

        # find the indices of repeats that match the criteria
        if monocular is not None:
            monocular_indices = np.where(self.monocular_sequence == monocular)[0]
        else:
            monocular_indices = np.arange(len(self.distance_sequence))

        if distance is not None:
            distance_indices = np.where(self.distance_sequence == distance)[0]
        else:
            distance_indices = np.arange(len(self.distance_sequence))
            distance_indices = np.setdiff1d(distance_indices, control_indices)

        return  np.intersect1d(monocular_indices, distance_indices)
    
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
        
    def find_rf_position(self, cell, distance=None, monocular=None):
        
        mean_trace = self.get_all_traces(cell, distance=distance, monocular=monocular, mean_trace=True)[0]

        # find the receptive field for this cell
        rf_idx = np.argmax(mean_trace)
        rf_position = self.angle_range[rf_idx][0]

        return rf_position
        
    def find_rf_values(self, cell, distance=None, monocular=None):

        rf_position = self.find_rf_position(cell, distance=None, monocular=monocular)
        rf_idx = np.where(self.angle_range == rf_position)[0][0]
        
        window_size = 5
        # get the mean trace for this cell
        
        on_traces = self.get_all_traces(cell,  
                                    distance=distance,
                                    monocular=monocular,
                                    )
        window_start = max(0, rf_idx - window_size)
        window_end = min(on_traces.shape[1], rf_idx + window_size + 1)
        rf_values = np.mean(on_traces[:, window_start:window_end], axis=1)
        
        return rf_values
    
    def compute_reliability(self, cell, distance=None, monocular=None, n_iterations=100):

        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance)
        idx = np.where(self.suite2p_id == cell)[0][0]

        all_traces = np.array([self.get_on_trace(idx, repeat).T for repeat in repeat_indices])
        
        n_repeats, n_time = all_traces.shape
        n_halves = int(np.ceil(n_repeats / 2))

        corr = np.zeros(n_iterations)
        corr_shifted = np.zeros(n_iterations)

        for i in range(n_iterations):
            # Fast shuffle and split
            shuffled_idx = np.random.permutation(n_repeats)
            first_idx = shuffled_idx[:n_halves]
            second_idx = shuffled_idx[n_halves:]

            # Mean traces
            first_mean = all_traces[first_idx].mean(axis=0)
            second_mean = all_traces[second_idx].mean(axis=0)
            corr[i] = np.corrcoef(first_mean, second_mean)[0, 1]

            # Shifted version
            shifts_first = np.random.randint(-180, 180, size=len(first_idx))
            shifts_second = np.random.randint(-180, 180, size=len(second_idx))
            
            time_idx = np.arange(n_time)
            first_shifted = all_traces[first_idx][
                np.arange(len(first_idx))[:, None],
                (time_idx - shifts_first[:, None]) % n_time
            ]
            second_shifted = all_traces[second_idx][
                np.arange(len(second_idx))[:, None],
                (time_idx - shifts_second[:, None]) % n_time
            ]

            corr_shifted[i] = np.corrcoef(
                first_shifted.mean(axis=0),
                second_shifted.mean(axis=0)
            )[0, 1]

        return corr, corr_shifted

    def analyze_cell_reliability(self, cell, n_iterations=100, corr_threshold=None, distance=None, monocular=None):
        """
        Analyzes the reliability of a cell's response by comparing split-half correlations.
        
        Args:
            cell: Cell ID to analyze
            n_iterations: Number of iterations for reliability calculation
            distance: Optional distance condition to filter trials
            monocular: Optional monocular condition to filter trials
            control: Use control trials condition to filter trials
        Returns:
            tuple: (is_reliable, reliability, corr_shifted, corr) where is_reliable indicates if the cell
                   is considered reliable based on correlation analysis
        """

        corr, corr_shifted = self.compute_reliability(cell, distance=distance, monocular=monocular, n_iterations=n_iterations)
        stimulus_reliable = np.mean(corr) > np.percentile(corr_shifted, 99)

        corr_control, corr_shifted_control = self.compute_reliability(cell, distance=4, monocular=monocular, n_iterations=n_iterations)
        control_reliable = np.mean(corr_control) > np.percentile(corr_shifted_control, 99.9)

        rf_values_distances = [np.mean(self.find_rf_values(cell, distance=distance, monocular=monocular)) for distance in [0, 1, 2, 3]]
        rf_values_control = np.mean(self.find_rf_values(cell, distance=4, monocular=monocular))

        # make sure that rf_values_distances are higher than rf_values_control, at least for three distances
        control_comparison = [rf_values_control > rf_values for rf_values in rf_values_distances]
        control_higher = np.sum(control_comparison) > 1

        if corr_threshold is not None:
            reliable = stimulus_reliable and not (control_higher) and np.mean(corr) > corr_threshold
        else:
            reliable = stimulus_reliable and not (control_higher)

        results = {
            'reliable': reliable,
            'stimulus_reliable': stimulus_reliable,
            'control_reliable': control_reliable,
            'control_higher': control_higher,
            'mean_corr': np.mean(corr)
        }

        return results

    def bootstrap_modulation(self, cell, monocular=None, window_size=5, n_iterations=200):

        """
        Analyze the distribution of modulation indices for a cell using bootstrap resampling.
        
        Parameters
        ----------
        cell : int
            Cell ID to analyze
        monocular : bool, optional
            Whether to analyze monocular condition (default: False)
        window_size : int, optional
            Size of window around RF peak (default: 5)
        n_iterations : int, optional
            Number of bootstrap iterations (default: 1000)
            
        Returns
        -------
        dict with keys:
            - preferred_distance: list of preferred distances per iteration
            - least_preferred_distance: list of least preferred distances per iteration
            - modulation_index: list of modulation indices from each iteration
            - p_value: 1 - (percentage above zero / 100)
            - modulated: True if percentage_above_zero >= 95
        """
        # Get mean trace and find RF peak window
        mean_trace = self.get_all_traces(cell, distance=None, monocular=monocular, mean_trace=True)[0]
        rf_idx = np.argmax(mean_trace)
        window_slice = slice(max(0, rf_idx - window_size), min(len(mean_trace), rf_idx + window_size + 1))

        # Get number of repeats per distance and validate
        n_repeats_per_distance = [
            self.get_all_traces(cell, distance=d, monocular=monocular, mean_trace=False).shape[0]
            for d in range(4)
        ]
        
        if len(set(n_repeats_per_distance)) != 1:
            print(f"Warning: Cell {cell} has different numbers of repeats per distance: {n_repeats_per_distance}")
            print("Using the minimum number of repeats across all distances.")

        n_repeats = min(n_repeats_per_distance)
        n_halves = (n_repeats + 1) // 2  # Equivalent to ceil(n_repeats / 2)

        # Pre-compute windowed mean traces for all distances: shape (4, n_repeats)
        trace_values = np.array([
            self.get_all_traces(cell, distance=d, monocular=monocular, mean_trace=False)[:n_repeats, window_slice].mean(axis=1)
            for d in range(4)
        ])

        # Pre-generate all random permutations at once: shape (n_iterations, n_repeats)
        halves_matrix = np.array([
            np.random.permutation(np.concatenate([np.ones(n_halves, dtype=bool), np.zeros(n_halves, dtype=bool)]))[:n_repeats]
            for _ in range(n_iterations)
        ])

        # Vectorized computation of mean values for both halves across all iterations
        # trace_values: (4, n_repeats), halves_matrix: (n_iterations, n_repeats)
        # Result shapes: (n_iterations, 4)
        mean_first_half = np.array([
            trace_values[:, mask].mean(axis=1) for mask in halves_matrix
        ])
        mean_second_half = np.array([
            trace_values[:, ~mask].mean(axis=1) for mask in halves_matrix
        ])

        # Find preferred/least preferred distances from first half
        preferred_distance = mean_first_half.argmax(axis=1)
        least_preferred_distance = mean_first_half.argmin(axis=1)

        # Calculate modulation index using second half values
        # Index into second half using the distances determined from first half
        iter_indices = np.arange(n_iterations)
        pref_values = np.abs(mean_second_half[iter_indices, preferred_distance])
        least_values = np.abs(mean_second_half[iter_indices, least_preferred_distance])
        
        modulation_indices = (pref_values - least_values) / (pref_values + least_values + 1e-10)

        unique_vals, counts = np.unique(preferred_distance, return_counts=True)
        preferred_distance_consistency = (np.max(counts) / len(preferred_distance)) * 100

        # Calculate statistics
        percentage_above_zero = (modulation_indices > 0).mean() * 100

        return {
            'rf_position': self.angle_range[rf_idx],
            'preferred_distance': preferred_distance.tolist(),
            'least_preferred_distance': least_preferred_distance.tolist(),
            'modulation_index': modulation_indices.tolist(),
            'bootstrap_p_value': 1 - (percentage_above_zero / 100),
            'preferred_distance_consistency': preferred_distance_consistency,
            'bootstrap_modulated': percentage_above_zero >= 95
        }

    def analyze_cell_modulation(self, cell, monocular=None, n_iterations=200):

        results = self.bootstrap_modulation(cell, monocular=monocular, n_iterations=n_iterations)

        # calculate the f_statistic and p_value
        rf_values = [self.find_rf_values(cell, distance=distance, monocular=monocular) for distance in [0, 1, 2, 3, 4]]
        distances_rf = [rf_values[i] for i in range(len(rf_values)-1)]
        f_stat, p_value = f_oneway(*distances_rf)
        results['anova_p_value'] = p_value
        results['anova_modulated'] = p_value < 0.05
        
        return results
    
    def analyze_ocular_dominance(self, cell):

        monocular_mean = self.get_all_traces(cell, distance=None, monocular=True, mean_trace=True)[0]
        binocular_mean = self.get_all_traces(cell, distance=None, monocular=False, mean_trace=True)[0]

        receptive_field_position = self.find_rf_position(cell, distance=None, monocular=None)
        receptive_field_position_index = np.where(self.angle_range == receptive_field_position)[0][0]

        window_size = 10
        window_start = max(0, receptive_field_position_index - window_size)
        window_end = min(len(monocular_mean), receptive_field_position_index + window_size + 1)
        monocular_mean_rf = np.mean(monocular_mean[window_start:window_end])
        binocular_mean_rf = np.mean(binocular_mean[window_start:window_end])
    
        ocular_dominance_index = (binocular_mean_rf - monocular_mean_rf) / (np.abs(binocular_mean_rf) + np.abs(monocular_mean_rf) + 1e-10)

        return ocular_dominance_index

    def analyze_cell_stats(self, cell, n_iterations=200):

        rf = self.find_rf_position(cell, distance=None, monocular=None)
        ODI = self.analyze_ocular_dominance(cell)

        rf_values = [np.mean(self.find_rf_values(cell, distance=i)) for i in range(4)]
        all_distance = np.argmax(rf_values)
        
        results = {
            'cell_id': cell,
            'ODI': ODI.tolist(),
            'RF': rf.tolist(),
            'distance': all_distance.tolist(),
            'monocular': {},
            'binocular': {},
        }

        for condition, monocular in [('monocular', True), ('binocular', False)]:
            
            modulation_results = self.analyze_cell_modulation(cell, monocular=monocular)
            reliable_results = self.analyze_cell_reliability(cell, monocular=monocular, n_iterations=n_iterations)

            results[condition] = {
                'RF': modulation_results['rf_position'][0].tolist(),
                'reliable': reliable_results['reliable'],
                'reliable_corr': reliable_results['mean_corr'].tolist(),
                'bootstrap_pvalue': modulation_results['bootstrap_p_value'].tolist(),
                'anova_pvalue': modulation_results['anova_p_value'].tolist(),
                'consistency': modulation_results['preferred_distance_consistency'].tolist(),
                'DTI': np.mean(modulation_results['modulation_index']).tolist(),
                'distance': np.mean(modulation_results['preferred_distance']).tolist(),
            }
        
        return results

    def analyze_session_stats(self, n_iterations=200):

        # Initialize empty lists to collect data for each cell
        cell_data = []
        
        # Analyze each cell
        for cell in self.suite2p_id:

            print(f'Analyzing cell {cell} of {self.suite2p_id[-1]}', end='\r', flush=True)

            results = self.analyze_cell_stats(cell, n_iterations=n_iterations)
            cell_data.append(results)
        
        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)
        
        return stats_df

class SingleCellPlots:

    def __init__(self, radial_class):
        self.R = radial_class
        self.colors = ['#FFA500', '#35b779', '#31688e', '#440154', 'grey']
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
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.R.distance_sequence[rep]], alpha=0.5)
            else:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.R.distance_sequence[rep]], alpha=0.5, hatch='///')

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
        
        # Get cell index
        idx = np.where(self.R.suite2p_id == cell)[0][0]
        
        # Define distances to plot
        distances_with_control = [0, 1, 2, 3, 4]
        distances_without_control = [0, 1, 2, 3]

        # Create figure with a row for each condition, 5 columns per row (add one for stats text)
        fig, axs = plt.subplots(2, 5, figsize=(22, 8), 
                               gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.2]})
        # remove axis ticks from second subplot
        axs[0, 1].remove()
        axs[1, 1].remove()

        # Adjust spacing between columns
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        
        conditions = [
            {'monocular': True,  'title': 'monocular'},
            {'monocular': False, 'title': 'binocular'},
        ]

        results = self.R.analyze_cell_stats(cell)
        reliable = results['monocular']['reliable'] or results['binocular']['reliable']

        all_traces = self.R.get_all_traces(cell, distance=None, monocular=None, mean_trace=False)
        vmin, vmax = np.percentile(all_traces, [1, 99])

        # Process each condition
        for c_idx, condition in enumerate(conditions):

            results_condition = results[condition['title']]

            repeat_traces = []
            repeats_per_distance = []
            
            for i, dist in enumerate(distances_with_control):
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
                    self.R.angle_range.flatten(), 
                    repeats_distance_mean - repeats_distance_sem, 
                    repeats_distance_mean + repeats_distance_sem, 
                    color=self.colors[dist], alpha=0.2
                )

            axs[c_idx, 0].pcolormesh(repeat_traces, cmap='viridis', vmin=vmin, vmax=vmax)

            axs[c_idx, 0].set_xlabel('Time (frames)')
            axs[c_idx, 0].set_ylabel('Trials')
            axs[c_idx, 0].set_title(f'{condition["title"]} - All Traces', fontsize=14)

            # radial plot
            n_theta = len(self.R.angle_range)
            theta_edges = np.linspace(np.radians(self.R.angle_range[0]), np.radians(self.R.angle_range[-1]), n_theta + 1)
            r_edges = np.linspace(0, len(distances_without_control), len(distances_without_control) + 1)
            theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
            
            matrices = [self.R.get_all_traces(cell=cell, monocular=condition['monocular'], distance=dist, mean_trace=True)[0] for dist in [0, 1, 2, 3]]
            Z = np.vstack(matrices)

            # # Make this axis polar
            # Create a polar axis in place of the corresponding cartesian axis
            ax_polar = plt.subplot2grid(axs.shape, (c_idx, 1), projection='polar', fig=fig)
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            ax_polar.set_thetamin(self.R.angle_range[0])
            ax_polar.set_thetamax(self.R.angle_range[-1])
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
            rf_values = [self.R.find_rf_values(cell, distance=distance, monocular=condition['monocular']) for distance in distances_with_control]
            rf_position = self.R.find_rf_position(cell, monocular=condition['monocular'])
            
            bp = axs[c_idx, 3].boxplot(rf_values, patch_artist=True, labels=distances_without_control + ['control'], showfliers=False)
            
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
            for box in bp['boxes']:
                box.set_facecolor('none')

            for i, rf_vals in enumerate(rf_values):
                x = np.random.normal(i+1, 0.04, size=len(rf_vals))
                axs[c_idx, 3].scatter(x, rf_vals, alpha=0.8, color=self.colors[i], s=40)

            # # # Check if p_value is significant and if the cell is responsive for the monocular condition
            # responsive, responsive_visually, responsive_control, responsive_control_higher, responsive_corr = self.R.analyze_cell_reliability(cell, monocular=condition['monocular'])
            # if modulated and responsive_visually:
            #     bar_height = vmax * 0.8
            #     axs[c_idx, 3].plot([1, len(distances_with_control)], [bar_height, bar_height], 'k-', linewidth=1.5)
            #     axs[c_idx, 3].text((1 + len(distances_with_control))/2, bar_height*1.02, '*', ha='center', va='bottom')

            # axs[c_idx, 3].set_title(f'modulated={modulated}', fontsize=14)
            axs[c_idx, 3].set_xlabel('Distance')
            axs[c_idx, 3].set_ylabel(self.R.response_mode)
            axs[c_idx, 3].set_ylim(vmin, vmax)

            

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
            axs[c_idx, 4].axis('off')
            axs[c_idx, 4].text(0, 1, stats_text, va='top', ha='left', fontsize=16, family='monospace')

        # Add suptitle to the figure
        fig.suptitle(f'Cell {cell}', fontsize=20)
        plt.tight_layout()

        return fig
    
    def plot_bootstrap_modulation(self, cell):

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for row, (condition_name, monocular) in enumerate([('monocular', True), ('binocular', False)]):
            
            results = self.R.bootstrap_modulation(cell = cell, monocular = monocular)
        
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
            results = self.R.analyze_cell_reliability(cell=cell, monocular=monocular, n_iterations=100)

            # Get raw correlation data for plotting
            corr, corr_shifted = self.R.compute_reliability(cell, distance=None, monocular=monocular, n_iterations=100)
            
            corr_control, corr_shifted_control = self.R.compute_reliability(cell, distance=4, monocular=monocular, n_iterations=100)

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
            rf_values_distances = [self.R.find_rf_values(cell, distance=d, monocular=monocular) for d in [0, 1, 2, 3]]
            rf_values_control = self.R.find_rf_values(cell, distance=4, monocular=monocular)

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

class SingleSessionPlots():

    def __init__(self, session_df, session_class=None):
        
        self.session_df = session_df
        self.R = session_class
    
        if self.R is not None:
            self.session_name = self.R.data_dir.split('\\')[-1]
        else:
            self.session_name = 'Multi-Session'

        self.distance_colors = setup_plot_style()
        self.modulation_colors = {
            'monocular': 'coral',
            'binocular': 'lightblue',
            'both': 'purple',
            'none': 'gray'
        }
    
        self.reliable_classes, self.modulated_classes = self.identify_cells()

    def identify_cells(self, threshold=0.85):

        reliable = self.session_df[
            (self.session_df['monocular'].apply(lambda x: x['reliable']) == True) &
            (self.session_df['binocular'].apply(lambda x: x['reliable']) == True) &
            ((self.session_df['monocular'].apply(lambda x: x['reliable_corr']) >= threshold) |
            (self.session_df['binocular'].apply(lambda x: x['reliable_corr']) >= threshold))
        ]['cell_id'].values

        not_reliable = np.setdiff1d(self.session_df['cell_id'].values, reliable)

        # Filter for cells that are in the reliable set AND meet the monocular/binocular criteria
        monocular_only = self.session_df[
            (self.session_df['cell_id'].isin(reliable)) &
            (self.session_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
            (self.session_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05)
        ]['cell_id'].values

        binocular_only = self.session_df[
            (self.session_df['cell_id'].isin(reliable)) &
            (self.session_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05) &
            (self.session_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
        ]['cell_id'].values

        both = self.session_df[
            (self.session_df['cell_id'].isin(reliable)) &
            (self.session_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05) &
            (self.session_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) < 0.05)
        ]['cell_id'].values

        none = self.session_df[
            (self.session_df['cell_id'].isin(reliable)) &
            (self.session_df['monocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05) &
            (self.session_df['binocular'].apply(lambda x: x['bootstrap_pvalue']) > 0.05)
        ]['cell_id'].values

        reliable_classes = {
            'reliable': reliable,
            'not_reliable': not_reliable,
        }

        modulated_classes = {
            'binocular': binocular_only,
            'monocular': monocular_only,
            'both': both,
            'none': none
        }

        total_cells = len(self.session_df)
        print(f"Reliable cells: {len(reliable)} ({len(reliable)/total_cells*100:.1f}%)")
        print(f"Not reliable cells: {len(not_reliable)} ({len(not_reliable)/total_cells*100:.1f}%)")
        print(f"Monocular cells: {len(monocular_only)} ({len(monocular_only)/len(reliable)*100:.1f}%)")
        print(f"Binocular cells: {len(binocular_only)} ({len(binocular_only)/len(reliable)*100:.1f}%)")
        print(f"Both cells: {len(both)} ({len(both)/len(reliable)*100:.1f}%)")
        print(f"None cells: {len(none)} ({len(none)/len(reliable)*100:.1f}%)")

        return reliable_classes, modulated_classes

    def extract_condition_feature(self, cell, condition=None, feature='RF'):
        
        if isinstance(cell, int):
            cell = [cell]

        if condition == 'monocular' or condition == 'binocular':
            feature_values = self.session_df.loc[self.session_df['cell_id'].isin(cell)][condition].apply(lambda x: x[feature]).values

        elif condition == 'both' or condition == 'none':
            monocular_values = self.session_df.loc[self.session_df['cell_id'].isin(cell)]['monocular'].apply(lambda x: x[feature]).values
            binocular_values = self.session_df.loc[self.session_df['cell_id'].isin(cell)]['binocular'].apply(lambda x: x[feature]).values
            feature_values = (monocular_values + binocular_values) / 2
        
        return feature_values

    def extract_general_feature(self, cell, feature='RF'):

        if isinstance(cell, int):
            cell = [cell]

        feature_values = self.session_df.loc[self.session_df['cell_id'].isin(cell)][feature].values
        return feature_values

    def plot_single_cells(self):

        results_dir = os.path.join(self.R.data_dir, 'results')
        CellPlots = SingleCellPlots(self.R)

        for condition in self.modulated_classes.keys():

            print(f'Plotting {condition} cells')
            figure_directory = create_new_result_dir(results_dir, f'{condition}')
            for cell in self.cell_classes[condition]:
                fig = CellPlots.plot_all_traces_colormap_distances(cell=cell)
                fig.savefig(os.path.join(figure_directory, f'{cell}.png'))
                plt.close(fig)
        
    def plot_responsive_bars(self):

        # Create and configure bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ['coral', 'lightgreen', 'blue']

        responsive_cells_set = {}
        for i, stat_df in enumerate(self.session_df):
            condition_name = stat_df['condition'].iloc[0] if not stat_df.empty else f"condition_{i}"
            responsive_cells_set[condition_name] = set(stat_df[stat_df['responsive'] == True]['cell_id'].values)

        total_cells = len(self.R.suite2p_id)
        bar_width = 0.75

        # Plot for each condition
        for i, cond in enumerate(responsive_cells_set.keys()):
            responsive_cells = responsive_cells_set[cond]
            n_resp = len(responsive_cells)
            percentage = (n_resp/total_cells)*100 

            # Plot total cells as gray bar
            ax.bar(cond, 100, color='lightgray', width=bar_width, label='Total Cells' if i == 0 else "")

            # Plot responsive cells as colored bar
            ax.bar(cond, percentage, color=bar_colors[i], width=bar_width)
            # Show both percentage and absolute number
            ax.text(cond, percentage, f'{percentage:.1f}%\n({n_resp}/{total_cells})', ha='center', va='bottom', fontsize=12)

        # Plot for 'both' (intersection)
        responsive_cells = self.find_responsive_cells()
        n_resp_both = len(responsive_cells)
        percentage = (n_resp_both/total_cells)*100

        ax.bar('both', 100, color='lightgray', width=bar_width, label='Total Cells' if len(responsive_cells_set) == 0 else "")
        ax.bar('both', percentage, color=bar_colors[2], alpha=0.5, width=bar_width)
        ax.text('both', percentage, f'{percentage:.1f}%\n({n_resp_both}/{total_cells})', ha='center', va='bottom', fontsize=12)

        # remove y axis altogether   
        ax.set_yticklabels([])
        ax.set_yticks([])

        # remove spines
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # add title with less padding so they appear closer
        fig.suptitle('Responsive Cells', fontsize=18, color='black', y=0.94)
        ax.set_title(self.session_name, fontsize=14, color='grey', pad=2)

        plt.tight_layout()

        return fig

    def plot_modulation_bars(self):
        
        # Create and configure bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bar_colors = ['coral', 'lightgreen', 'blue', 'gray']

        labels = list(self.modulated_classes.keys()) 
        modulated_counts = [len(self.modulated_classes[label]) for label in labels]
        total_reliable = len(self.reliable_classes['reliable'])
        percentages = [count/total_reliable*100 for count in modulated_counts]
        
        bar_width = 0.75

        # Plot bars for each modulation class
        for i, label in enumerate(labels):
            # Plot total reliable cells as gray bar (100%)
            ax.bar(label, 100, color='lightgray', width=bar_width, label='Reliable Cells' if i == 0 else "")
            
            # Plot modulated cells as colored bar
            ax.bar(label, percentages[i], color=self.modulation_colors[label], width=bar_width)
            
            # Show both percentage and absolute number
            ax.text(label, percentages[i], f'{percentages[i]:.1f}%\n({modulated_counts[i]}/{total_reliable})', 
                   ha='center', va='bottom', fontsize=12)

        # remove y axis
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # add title with less padding so they appear closer
        fig.suptitle('Modulated Cells', fontsize=18, color='black', y=0.94)
        ax.set_title(self.session_name, fontsize=14, color='grey', pad=2)
        
        plt.tight_layout()

        return fig

    def plot_modulation_index(self):

        # Create a boxplot of modulation strength for monocular, binocular, both conditions, and non-modulated
        n_conditions = len(self.modulated_classes.keys())
        fig, ax = plt.subplots(figsize=(2*n_conditions, 6))

        # Prepare data for boxplot
        data_for_boxplot = []
        box_labels = []

        conditions = list(self.modulated_classes.keys())
        cells_to_plot = [self.modulated_classes[condition] for condition in conditions]

        # for every condition in modulated_cells, get the modulation strength
        for i, condition in enumerate(conditions):
            print(condition)
            DTI_condition = self.extract_condition_feature(cell=cells_to_plot[i], condition=condition, feature='DTI')
            
            data_for_boxplot.append(np.array(DTI_condition))
            box_labels.append(condition)

        bp = ax.boxplot(data_for_boxplot, patch_artist=True, labels=box_labels, showfliers=False)

        # Customize boxplot appearance
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=self.modulation_colors[conditions[i]], alpha=0.7)
            
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        # Add individual points with jitter
        for i, data in enumerate(data_for_boxplot):
            # Create jitter effect
            x = np.random.normal(i+1, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.7, s=40, color=self.modulation_colors[conditions[i]], edgecolor='black')

        # Add labels and title
        ax.set_xlabel('Condition', fontsize=16)
        ax.set_xticklabels(box_labels, fontsize=12)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Modulation Strength')

        # add title with less padding so they appear closer
        fig.suptitle('Modulation Index by Condition', fontsize=18, color='black', y=0.94)
        ax.set_title(self.session_name, fontsize=14, color='grey', pad=2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show plot
        plt.tight_layout()

        return fig

    def plot_distance_bars(self):

        fig, axes = plt.subplots(nrows=1, ncols=len(self.modulated_classes.keys()), figsize=(4*len(self.modulated_classes.keys()), 5))

        counts_dict = {
            'monocular': [],
            'binocular': [],
            'both': [],
            'none': []
        }
        
        for i, condition in enumerate(self.modulated_classes.keys()):
            modulated_cells = self.modulated_classes[condition]
            
            if condition == 'both' or condition == 'none':
                all_pref_distances = self.extract_general_feature(cell=modulated_cells, feature='distance')
            else:
                all_pref_distances = self.extract_condition_feature(cell=modulated_cells, condition=condition, feature='distance')
            
            all_pref_distance = [int(np.round(distance)) for distance in all_pref_distances]
            counts = [np.sum(np.array(all_pref_distance) == d) for d in [0, 1, 2, 3]]
            
            counts_dict[condition] = counts
            
            axes[i].bar([0, 1, 2, 3], counts, width=0.9, color=self.distance_colors, edgecolor='black')

        for i, condition in enumerate(self.modulated_classes.keys()):
            counts = counts_dict[condition]
            total = sum(counts)
            
            # Add percentage labels on top of bars
            for j, count in enumerate(counts):
                if total > 0:
                    percentage = (count / total) * 100
                    axes[i].text(j, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Set y-axis limits with some headroom
            max_count = max([max(counts_dict[cond]) for cond in self.modulated_classes.keys()])
            axes[i].set_ylim(0, max_count * 1.2)
            
            # Add axis labels and title
            axes[i].set_xlabel('Preferred Distance', fontsize=14)
            axes[i].set_ylabel('Number of Cells', fontsize=14)
            axes[i].set_title(condition.capitalize(), fontsize=18, color=self.modulation_colors[condition], fontweight='bold', pad=10)
            
            # Set x-axis ticks
            axes[i].set_xticks([0, 1, 2, 3])
            axes[i].set_xticklabels(['0', '1', '2', '3'], fontsize=12)
            
            # Remove top and right spines
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        # Add overall title
        fig.suptitle('Preferred Distance Distribution by Modulation Type', fontsize=18, color='black', y=1.05)
        fig.text(0.5, 0.95, self.session_name, fontsize=18, color='grey', ha='center')

        plt.tight_layout()

        return fig, axes
    
    def plot_mean_traces(self, condition = 'both', ax = None):

        # Create subplots for modulation classes excluding 'none'
        conditions_to_plot = [cond for cond in self.modulated_classes.keys() if cond != 'none']
        fig, axes = plt.subplots(1, len(conditions_to_plot), figsize=(5*len(conditions_to_plot), 6))

        # Handle case where there's only one condition (axes won't be an array)
        if len(conditions_to_plot) == 1:
            axes = [axes]

        # Find the maximum number of cells across all conditions (excluding 'none') to normalize y-axis
        max_cells = max([len(self.modulated_classes[condition]) for condition in conditions_to_plot])

        for i, condition in enumerate(conditions_to_plot):
            modulated_cells = self.modulated_classes[condition]
            
            if len(modulated_cells) == 0:
                axes[i].text(0.5, 0.5, 'No cells', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(condition.capitalize(), fontsize=14, color=self.modulation_colors[condition], fontweight='bold')
                continue
            
            RF = self.extract_condition_feature(cell=modulated_cells, condition=condition, feature='RF')
            
            all_mean_traces = []
            for cell in modulated_cells:
                mean_trace = self.R.get_all_traces(cell=cell, mean_trace=True)[0]
                all_mean_traces.append(mean_trace)
            
            all_mean_traces = np.array(all_mean_traces)
            
            # order RF from lowest to highest and apply same order to all_mean_traces
            sorted_indices = np.argsort(RF)
            RF = RF[sorted_indices]
            all_mean_traces = all_mean_traces[sorted_indices]
            
            # plot all_mean_traces - create meshgrid for pcolormesh
            X, Y = np.meshgrid(self.R.angle_range, np.arange(len(all_mean_traces)))
            axes[i].pcolormesh(X, Y, all_mean_traces, shading='auto')
            axes[i].set_title(condition.capitalize(), fontsize=14, color=self.modulation_colors[condition], fontweight='bold')
            axes[i].set_xlabel('Angle (degrees)', fontsize=12)
            axes[i].set_ylabel('Cell (sorted by RF)', fontsize=12)
            
            # Set y-axis limits to match the maximum number of cells
            axes[i].set_ylim(0, max_cells)
            
            # Remove axis spines and ticks
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            
            plt.tight_layout()

        # Create separate figure for 'none' condition
        if 'none' in self.modulated_classes.keys():
            fig_none, ax_none = plt.subplots(1, 1, figsize=(5, 6))
            
            modulated_cells = self.modulated_classes['none']
            
            if len(modulated_cells) == 0:
                ax_none.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax_none.transAxes)
                ax_none.set_title('None', fontsize=14, color=self.modulation_colors['none'], fontweight='bold')
            else:
                RF = self.extract_condition_feature(cell=modulated_cells, condition='none', feature='RF')
                
                all_mean_traces = []
                for cell in modulated_cells:
                    mean_trace = self.R.get_all_traces(cell=cell, mean_trace=True)[0]
                    all_mean_traces.append(mean_trace)
                
                all_mean_traces = np.array(all_mean_traces)
                
                # order RF from lowest to highest and apply same order to all_mean_traces
                sorted_indices = np.argsort(RF)
                RF = RF[sorted_indices]
                all_mean_traces = all_mean_traces[sorted_indices]
                
                # plot all_mean_traces - create meshgrid for pcolormesh
                X, Y = np.meshgrid(self.R.angle_range, np.arange(len(all_mean_traces)))
                ax_none.pcolormesh(X, Y, all_mean_traces, shading='auto')
                ax_none.set_title('None', fontsize=14, color=self.modulation_colors['none'], fontweight='bold')
                ax_none.set_xlabel('Angle (degrees)', fontsize=12)
                ax_none.set_ylabel('Cell (sorted by RF)', fontsize=12)

            
            plt.tight_layout()

    def plot_preferred_distance_polar(self, condition='binocular', ax = None):

        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1, polar=True)
        else:
            fig = ax.get_figure()

        ax.set_theta_zero_location('N')
        # Add gray background for central region (-25 to 25 degrees)
        ax.axvspan(np.radians(-25), np.radians(25), alpha=0.5, color='gray', zorder=0)
        # Add colored y-spans for distance ranges
        distance_colors = self.colors
        for dist in range(1, 5):
            ax.axhspan(dist - 0.30, dist + 0.30, alpha=0.4, color=distance_colors[dist-1], zorder=0)

        if condition is None:
            cells_to_plot = self.responsive_cells
        else:
            cells_to_plot = self.modulated_cells[condition]

        # Plot preferred distance and angle for each condition
        pref_distances = []
        pref_angles = []
        for cell in cells_to_plot:
            if condition is None:
                pref_distance = self.pullout_stat(cell_id=cell, condition='binocular', stat_type='pref_distance')
                pref_angle = self.pullout_stat(cell_id=cell, condition='binocular', stat_type='pref_angle')
            else:
                pref_distance = self.pullout_stat(cell_id=cell, condition=condition, stat_type='pref_distance')
                pref_angle = self.pullout_stat(cell_id=cell, condition=condition, stat_type='pref_angle')
            
            pref_distances.append(pref_distance)
            pref_angles.append(pref_angle)

        theta = np.radians(pref_angles)
        pref_distances = np.array(pref_distances) + np.random.randn(len(pref_distances)) * 0.1 + 1
        sc = ax.scatter(theta, pref_distances, c=pref_angles, cmap='viridis', vmin=-30, vmax=60, s=40, edgecolor='k', alpha=0.8)

        ax.grid(False)
        ax.set_theta_direction(-1)
        ax.set_thetamin(-35)
        ax.set_thetamax(65)
        ax.set_ylabel('Preferred Distance')
        ax.set_yticks([])
        # title will be condition
        if condition is None:
            ax.set_title('Responsive', fontsize=18, color='black', pad=2)
        else:
            ax.set_title(condition.capitalize(), fontsize=18, color='black', pad=2)

        ax.legend()

        plt.tight_layout()

        return fig, ax
    
    def plot_preferred_distance_polar_density(self):
        """
        Plot a density plot of preferred distances (4 bins) vs. preferred angles (binned), 
        as a radial pcolormesh for each condition.
        Rows: distances (4 bins), Columns: angles (binned).
        """
        import matplotlib as mpl

        # Define bins
        distance_bins = np.array([1, 2, 3, 4, 5])  # 4 bins: [1-2), [2-3), [3-4), [4-5)
        angle_min, angle_max = -30, 75
        n_angle_bins = 18
        angle_bins = np.linspace(angle_min, angle_max, n_angle_bins + 1)  # e.g. 6 deg per bin

        # For each condition, build a 2D histogram: (distance_bin, angle_bin)
        condition_colors = {
            'binocular': 'Greens',
            'monocular': 'Reds'
        }

        fig = plt.figure(figsize=(10, 5))
        n_conds = len(condition_colors)
        for i, (cond, cmap_name) in enumerate(condition_colors.items()):
            # Prepare data
            pref_distances = []
            pref_angles = []
            if cond in self.modulated_cells:
                mod_cells = self.modulated_cells[cond]
                # Find the corresponding stat_df for this condition
                stat_df = None
                for df in self.radial_stats:
                    if not df.empty and df['condition'].iloc[0] == cond:
                        stat_df = df
                        break
                if stat_df is not None:
                    for cell in mod_cells:
                        row = stat_df[stat_df['cell_id'] == cell]
                        if not row.empty:
                            # Add small noise to avoid binning artifacts
                            noisy_distance = row['pref_distance'].values[0] + np.random.uniform(-0.2, 0.2)
                            noisy_angle = row['rf_position'].values[0] + np.random.uniform(-1, 1)
                            pref_distances.append(noisy_distance + 1)  # +1 to match original code
                            pref_angles.append(noisy_angle)
            # Bin data
            H, r_edges, theta_edges = np.histogram2d(
                pref_distances, pref_angles,
                bins=[distance_bins, angle_bins]
            )
            # Prepare meshgrid for pcolormesh
            # r: [distance_bins], theta: [angle_bins] in radians
            r, theta = np.meshgrid(distance_bins, np.radians(angle_bins), indexing='ij')
            # H shape: (4, n_angle_bins)
            # For pcolormesh, need (r.shape[0], r.shape[1])
            # Plot
            ax = fig.add_subplot(1, n_conds, i+1, polar=True)
            # pcolormesh expects theta (angle) as 2D array, r as 2D array, and H.T as values
            # But meshgrid above is (4, n_angle_bins+1), H is (4, n_angle_bins)
            # So we need to transpose H for correct orientation
            pcm = ax.pcolormesh(
                np.radians(angle_bins),  # theta edges
                distance_bins,           # r edges
                H,                       # shape (4, n_angle_bins)
                cmap=cmap_name,
                shading='auto'
            )
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_thetamin(angle_min)
            ax.set_thetamax(angle_max)
            ax.set_ylim(distance_bins[0], distance_bins[-1])
            ax.set_yticks(distance_bins[:-1] + 0.5)
            ax.set_yticklabels([f"{int(distance_bins[j])}-{int(distance_bins[j+1])}" for j in range(len(distance_bins)-1)])
            ax.set_xticks(np.radians(np.linspace(angle_min, angle_max, 8)))
            ax.set_title(f"{cond.capitalize()}", fontsize=14)
            cbar = fig.colorbar(pcm, ax=ax, pad=0.1, orientation='vertical', shrink=0.8)
            cbar.set_label('Cell Density', fontsize=10)
            if i == 0:
                ax.set_ylabel('Preferred Distance Bin')
        fig.suptitle('Density of Preferred Distance vs. Angle', fontsize=18, color='black', y=0.98)
        plt.tight_layout()
        return fig

    def plot_reliable_cells_map(self, ax = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        cells_to_plot = self.R.suite2p_id

        # Load Suite2p outputs
        stat = np.load(os.path.join(self.R.suite2p_dir, 'plane0/stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(self.R.suite2p_dir, 'plane0/ops.npy'), allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        ax.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        # Scatter plot each pixel
        for cell_id in cells_to_plot:
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()

            if cell_id not in self.reliable_classes['reliable']:
                sc = ax.scatter(
                    xpix, ypix,
                    c='lightgray',
                    s=120, alpha=0.5,
                    edgecolor='black', linewidth=1
                )
            else:
                rf_location = self.extract_general_feature(cell=[cell_id], feature='RF')
                sc = ax.scatter(
                    xpix, ypix,
                    c=rf_location,
                    cmap='viridis',
                    vmin=-30,
                    vmax=60,
                    s=120, alpha=1,
                    edgecolor='black', linewidth=1
                )

        # Rotate the image by 180 degrees
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        # Remove all axes, spines, ticks, and colorbar for a pure image
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig, ax
    
    def plot_rf_density_map(self, bins = 20):
        """
        Plot a 2D map where each spatial bin is colored by the average RF location
        of the cells whose centers lie in that bin.
        
        Parameters
        ----------
        nx_bins, ny_bins : int
            Number of bins along the x and y dimensions.
        """
        # 1) Load Suite2p outputs
        stat = np.load(self.R.activity_path + '/stat.npy', allow_pickle=True)
        ops = np.load(self.R.activity_path + '/ops.npy', allow_pickle=True).item()
        ref_img = ops['refImg']
        
        # 2) Gather (x, y) centroids and RF values for responsive cells
        xs, ys, rfvals = [], [], []
        for cell_id in self.responsive_cells:
            
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()
            rf = self.pullout_stat(cell_id=cell_id, condition='binocular', stat_type='pref_angle')
            
            xs.append(xpix)
            ys.append(ypix)
            rfvals.append(rf)
        
        xs = np.array(xs)
        ys = np.array(ys)
        rfvals = np.array(rfvals)

        # 3) Compute mean RF in each 2D bin, but only if a bin has more than one sample
        #    We'll use binned_statistic_2d to get both the mean and the count

        # Compute mean RF per bin
        mean_rf, x_edges, y_edges, binnumber = binned_statistic_2d(
            xs, ys, rfvals,
            statistic='mean',
            bins=[bins, bins],
            expand_binnumbers=False
        )
        # Compute count per bin
        count_rf, _, _, _ = binned_statistic_2d(
            xs, ys, rfvals,
            statistic='count',
            bins=[bins, bins],
            expand_binnumbers=False
        )
        # Mask bins with <=1 sample
        mean_rf[count_rf <= 1] = np.nan
        # mean_rf shape is (nx_bins, ny_bins), but imshow wants [rows=y, cols=x],
        # so we'll transpose it.
        
        # 4) Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Show background reference image faintly
        vmin = np.percentile(ref_img, 1)
        vmax = np.percentile(ref_img, 99)
        
        ax.imshow(
            ref_img,
            cmap='gray',
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=[0, ref_img.shape[1], 0, ref_img.shape[0]]
        )
        
        # Overlay the binned RF map
        im = ax.imshow(
            mean_rf.T,
            cmap='viridis',
            alpha=0.8,
            vmin=-30, vmax=75,
            interpolation='nearest',
            origin='upper',
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]]
        )
        cbar = fig.colorbar(im, ax=ax, label='Mean RF Location', shrink=0.8)
        
        ax.set_title(f'{self.session_name} — RF Density Map', fontsize=16, color='black')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
    
    def plot_modulated_cells_map(self, condition='binocular'):

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Load Suite2p outputs
        stat = np.load(os.path.join(self.R.suite2p_dir, 'plane0/stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(self.R.suite2p_dir, 'plane0/ops.npy'), allow_pickle=True).item()
        
        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)
        
        # Show background image
        ax.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Get cells for this condition
        modulated_cells = self.modulated_classes[condition]
        none_cells = self.modulated_classes['none']
        
        # Plot none cells in grey
        for cell_id in none_cells:
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()
            ax.scatter(
                xpix, ypix,
                c='lightgray',
                s=120, alpha=0.5,
                edgecolor='black', linewidth=1
            )
        
        # Plot modulated cells colored by distance
        for cell_id in modulated_cells:
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()
            
            distance = self.extract_condition_feature(cell=[cell_id], condition=condition, feature='distance')
            sc = ax.scatter(
                xpix, ypix,
                c=distance,
                cmap='viridis',
                vmin=0,
                vmax=3,
                s=120, alpha=1,
                edgecolor='black', linewidth=1
            )
        
        # Rotate the image by 180 degrees
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        # Remove axes and spines
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(f'{condition.capitalize()} Modulated Cells', fontsize=16)
        
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)
        plt.show()

        return fig, ax
    
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

        return fig

    def plot_session_summary(self):

        fig_responsive_map, _ = self.plot_modulated_cells_map(condition=None)
        fig_binocular_map, _ = self.plot_modulated_cells_map(condition='binocular')
        fig_monocular_map, _ = self.plot_modulated_cells_map(condition='monocular')

        fig_distance_bars = plt.figure(figsize=(12, 4))
        gs = fig_distance_bars.add_gridspec(1, 3, hspace=0.3, wspace=0.5)
        # Create bar plot axes in second row
        axes_bar = []
        for i in range(3):
            ax = fig_distance_bars.add_subplot(gs[0, i])
            axes_bar.append(ax)

        # Call each plotting function with the appropriate axis
        self.plot_preferred_distance_bars(condition=None, ax=axes_bar[0])
        self.plot_preferred_distance_bars(condition='binocular', ax=axes_bar[1])
        self.plot_preferred_distance_bars(condition='monocular', ax=axes_bar[2])

        fig_distance_polar =plt.figure(figsize=(12, 18))
        gs = fig_distance_polar.add_gridspec(1, 3, hspace=0.3, wspace=0.5)

        # Create polar plots in first row
        axes_polar = []
        for i in range(3):
            ax = fig_distance_polar.add_subplot(gs[0, i], projection='polar')
            axes_polar.append(ax)
        
        self.plot_preferred_distance_polar(condition=None, ax=axes_polar[0])
        self.plot_preferred_distance_polar(condition='binocular', ax=axes_polar[1])
        self.plot_preferred_distance_polar(condition='monocular', ax=axes_polar[2])

        return fig_responsive_map, fig_binocular_map, fig_monocular_map, fig_distance_bars, fig_distance_polar

class MultiSessionPlots:

    def __init__(self, animal_id, experiment_name, base_multi_dir = r'D:/data-distance-coding-project/multi-session-data/', 
        base_processed_dir = r'D:/data-distance-coding-project/processed-data/'):

        self.base_multi_dir = base_multi_dir
        self.base_processed_dir = base_processed_dir
        self.multi_session_dir = os.path.join(self.base_multi_dir, animal_id + '_' + experiment_name)
        
        session_info_file = os.path.join(self.multi_session_dir, animal_id + '_' + experiment_name + '_sessions.txt')
        self.read_config_file(session_info_file)
        self.mean_images = self.load_mean_images()

        self.widefield_fov_pixels = 400
        self.widefield_fov_units = 5200
        self.widefield_img = None
        self.widefield_img_shifted = None
        self.VFS = None
        self.VFS_shifted = None
        self.load_widefield_maps()
        
        self.widefield_dy, self.widefield_dx = self.set_widefield_origin()
        self.shift_widefield()

        self.area_names = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']

        # self.radial_sessions = self.load_radial_sessions()
        # self.areas_verts = self.load_areas_verts()

    def read_config_file(self, session_info_file):

        self.session_names = []
        with open(os.path.join(self.multi_session_dir, session_info_file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                session_name = line.split(',')
                self.session_names.append(session_name)

    def get_session_dir(self, session_number):
        mouse_name, session_date, experiment_name = self.session_names[session_number]
        session_dir = os.path.join(self.base_processed_dir, session_date + '_' + mouse_name + '_' + experiment_name)

        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

        return session_dir

    def load_mean_images(self):

        self.mean_images = []
        for i, session_name in enumerate(self.session_names):
            session_dir = self.get_session_dir(i)
            suite2p_dir = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')
            if not os.path.exists(suite2p_dir):
                raise FileNotFoundError(f"Suite2p directory does not exist: {suite2p_dir}")
            ops = np.load(suite2p_dir, allow_pickle=True).item()
            self.mean_images.append(ops['meanImg'])

        return self.mean_images

    def load_coordinates(self):

        field_coordinates = {}
        try:
            with open(self.animal_dir + r'/' + self.animal_id + '_field_coordinates.pkl', 'rb') as f:
                field_coordinates = pickle.load(f)
        except:
            field_coordinates = {}
            for numeral in self.numeral_names:
                field_coordinates[numeral] = self.align_translation(moving_numeral=numeral)
            
            with open(self.animal_dir + r'/' + self.animal_id + '_field_coordinates.pkl', 'wb') as f:
                pickle.dump(field_coordinates, f)

        return field_coordinates
    
    def load_areas_verts(self):
        try:
            with open(self.animal_dir + r'/' + self.animal_id + '_areas_verts.pkl', 'rb') as f:
                areas_verts = pickle.load(f)
        except:
            areas_verts = self.draw_areas()
            with open(self.animal_dir + r'/' + self.animal_id + '_areas_verts.pkl', 'wb') as f:
                pickle.dump(areas_verts, f)
        return areas_verts
    
    def load_radial_sessions(self):

        try:
            with open(self.animal_dir + r'/' + self.animal_id + '_radial_sessions.pkl', 'rb') as f:
                radial_sessions = pickle.load(f)
            print(f'Radial sessions loaded for {self.animal_id}')
        except:
            print(f'No radial sessions found for {self.animal_id}')
            radial_sessions = self.compute_sessions_stats()
            with open(self.animal_dir + r'/' + self.animal_id + '_radial_sessions.pkl', 'wb') as f:
                pickle.dump(radial_sessions, f)

        return radial_sessions
    
    def compute_sessions_stats(self):

        tagged = []

        for name in self.numeral_names:

            print('Processing session: ', name)

            # load this
            summ_name = self.session_names[name] + '_summ.pkl'
            summ_dir = self.get_animal_dir() + r'/' + summ_name
            with open(summ_dir, 'rb') as f:
                summ = pickle.load(f)
                print('Summ loaded for: ', name)

            session_coords = self.field_coordinates[name]
            suite2p_stat = np.load(self.get_suite2p_dir(self.session_names[name]) + r'\plane0\stat.npy', allow_pickle=True)

            cell_coords = []
            for cell_id in summ.radial_stats['cell_id']:
                x_units, y_units = self.extract_cell_coordinates(suite2p_stat, session_coords, cell_id)
                cell_coords.append([x_units, y_units])

            df = summ.radial_stats.copy()
            df.insert(1, 'session', name)
            df.insert(2, 'coords', cell_coords)

            tagged.append(df)

        radial_sessions = pd.concat(tagged, ignore_index=True)

        return radial_sessions

    def align_to_widefield(self, session_id, slider_limit_px=500):
        """
        Align a single session's mean image to the widefield image with scale, rotation, and translation.
        Previously aligned sessions remain visible in green overlay.
        
        Parameters
        ----------
        session_id : int
            The session index
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.
            
        Returns
        -------
        dict
            Dictionary containing transformation parameters:
            {'scale': float, 'rotation': float, 'dx': float, 'dy': float}
        """

        if hasattr(self, 'field_coordinates_widefield') and len(self.field_coordinates_widefield) > 0:
            # Use the most recent alignment's scale and rotation
            last_params = list(self.field_coordinates_widefield.values())[-1]
            scale0 = last_params['scale']
            rotation0 = last_params['rotation']
        else:
            scale0 = 0.33
            rotation0 = 174

        dx0, dy0 = 0.0, 0.0
        alpha0 = 0.5

        # Helper function to apply transformation to an image
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            
            # Create transformation matrix
            # First scale and rotate around center
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            
            # Then add translation
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            # Apply transformation
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed
        
        # Get the widefield reference image
        widefield = self.widefield_img_shifted.copy()
        
        # Normalize widefield to 0-1 for display
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        
        # Get the mean image to align
        mean_img = self.mean_images[session_id].copy()
        
        # Normalize mean image to 0-1
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
        
        # Resize mean_img to match widefield size
        mean_img_resized = cv2.resize(mean_img_norm, 
                                    (widefield.shape[1], widefield.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        
        # Create green colormap for mean images
        green_cmap = LinearSegmentedColormap.from_list('green', ['black', 'lime'])
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)
        
        # Create initial composite
        # Start with widefield in grayscale
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Overlay previously aligned sessions in green (dimmer, underneath)
        if hasattr(self, 'field_coordinates_widefield'):
            for prev_session_id, params in self.field_coordinates_widefield.items():
                if prev_session_id != session_id:  # Don't show the current one yet
                    prev_img = self.mean_images[prev_session_id].copy()
                    prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                    prev_img_resized = cv2.resize(prev_img_norm,
                                                (widefield.shape[1], widefield.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                    prev_transformed = transform_image(
                        prev_img_resized,
                        params['scale'],
                        params['rotation'],
                        params['dx'],
                        params['dy']
                    )
                    # Blend green on top of grayscale using alpha compositing
                    prev_alpha = 0.3
                    prev_green = np.zeros_like(composite)
                    prev_green[:, :, 1] = prev_transformed
                    composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha
        
        # Add current session on top (brighter)
        current_transformed = transform_image(mean_img_resized, scale0, rotation0, dx0, dy0)
        current_green = np.zeros_like(composite)
        current_green[:, :, 1] = current_transformed
        current_mask = (current_transformed > 0).astype(float)
        composite = composite * (1 - alpha0 * current_mask[:, :, None]) + current_green * alpha0
        
        img_display = ax.imshow(composite)
        ax.set_title(f"Aligning session: {self.session_names[session_id]}")
        ax.axis('off')
        
        # Create sliders
        ax_scale = plt.axes([0.15, 0.25, 0.7, 0.02])
        ax_rotation = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_dx = plt.axes([0.15, 0.15, 0.7, 0.02])
        ax_dy = plt.axes([0.15, 0.10, 0.7, 0.02])
        ax_alpha = plt.axes([0.15, 0.05, 0.7, 0.02])
        
        slider_scale = Slider(ax_scale, 'Scale', 0.1, 1.0, valinit=scale0, valstep=0.01)
        slider_rotation = Slider(ax_rotation, 'Rotation (°)', -180, 180, valinit=rotation0, valstep=1)
        slider_dx = Slider(ax_dx, 'dx (px)', -slider_limit_px, slider_limit_px, valinit=dx0, valstep=1)
        slider_dy = Slider(ax_dy, 'dy (px)', -slider_limit_px, slider_limit_px, valinit=dy0, valstep=1)
        slider_alpha = Slider(ax_alpha, 'Transparency', 0.0, 1.0, valinit=alpha0, valstep=0.05)
        
        def update(val):
            """Update the display when sliders change."""
            scale = slider_scale.val
            rotation = slider_rotation.val
            dx = slider_dx.val
            dy = slider_dy.val
            alpha = slider_alpha.val
            
            # Recreate composite
            composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
            
            # Overlay previously aligned sessions (dimmer, underneath)
            if hasattr(self, 'field_coordinates_widefield'):
                for prev_numeral, params in self.field_coordinates_widefield.items():
                    if prev_session_id != session_id:
                        prev_img = self.mean_images[prev_numeral].copy()
                        prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                        prev_img_resized = cv2.resize(prev_img_norm,
                                                    (widefield.shape[1], widefield.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                        prev_transformed = transform_image(
                            prev_img_resized,
                            params['scale'],
                            params['rotation'],
                            params['dx'],
                            params['dy']
                        )
                        # Add to green channel with reduced intensity
                        prev_alpha = 0.3
                        prev_green = np.zeros_like(composite)
                        prev_green[:, :, 1] = prev_transformed
                        composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha
            
            # Add current session on top (brighter)
            current_transformed = transform_image(mean_img_resized, scale, rotation, dx, dy)
            current_green = np.zeros_like(composite)
            current_green[:, :, 1] = current_transformed
            current_mask = (current_transformed > 0).astype(float)
            composite = composite * (1 - alpha * current_mask[:, :, None]) + current_green * alpha
            
            img_display.set_data(composite)
            ax.set_title(
                f"Aligning session: {self.session_names[session_id]} | "
                f"Scale: {scale:.2f}, Rotation: {rotation:.0f}°, "
                f"dx: {dx:.0f}px, dy: {dy:.0f}px"
            )
            fig.canvas.draw_idle()
        
        # Connect sliders to update function
        slider_scale.on_changed(update)
        slider_rotation.on_changed(update)
        slider_dx.on_changed(update)
        slider_dy.on_changed(update)
        slider_alpha.on_changed(update)
        
        plt.show(block=True)
        
        # Return the transformation parameters
        return {
            'scale': slider_scale.val,
            'rotation': slider_rotation.val,
            'dx': slider_dx.val,
            'dy': slider_dy.val
        }

    def align_all_to_widefield(self, slider_limit_px=500):
        """
        Sequentially align all sessions to the widefield image.
        Each aligned session remains visible while aligning subsequent ones.
        
        Parameters
        ----------
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.
            
        Returns
        -------
        dict
            Dictionary mapping session numeral to transformation parameters
        """
        # Initialize storage for transformations
        
        self.field_coordinates_widefield = {}
        
        print("Starting sequential alignment to widefield...")
        print(f"Sessions to align: {list(self.session_names)}")
        
        for i, session_name in enumerate(self.session_names):
            print(f"\nAligning session {session_name}...")
            params = self.align_to_widefield(i, slider_limit_px=slider_limit_px)
            self.field_coordinates_widefield[i] = params
            print(f"Session {session_name} aligned: {params}")
        
        # # Save to file
        # import pickle
        # save_path = self.multi_session_dir + '/' + self.animal_id + '_field_coordinates_widefield.pkl'
        # with open(save_path, 'wb') as f:
        #     pickle.dump(self.field_coordinates_widefield, f)
        # print(f"\nAll alignments saved to: {save_path}")
        
        return self.field_coordinates_widefield
        
    def plot_aligned_sessions(self, save_path=None):
        """
        Create a result figure showing all aligned sessions overlaid on the widefield image.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        
        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed
        
        # Get widefield reference
        widefield = self.widefield_img_shifted.copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        
        # Create composite
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Overlay all aligned sessions
        if not hasattr(self, 'field_coordinates_widefield') or len(self.field_coordinates_widefield) == 0:
            print("No aligned sessions found. Run align_all_to_widefield() first.")
            return None, None
        
        for idx, (session_id, params) in enumerate(self.field_coordinates_widefield.items()):
            mean_img = self.mean_images[session_id].copy()
            mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
            mean_img_resized = cv2.resize(mean_img_norm,
                                        (widefield.shape[1], widefield.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            transformed = transform_image(
                mean_img_resized,
                params['scale'],
                params['rotation'],
                params['dx'],
                params['dy']
            )
            
            # Blend using alpha compositing in green
            alpha = 0.6
            session_green = np.zeros_like(composite)
            session_green[:, :, 1] = transformed
            mask = (transformed > 0).astype(float)
            composite = composite * (1 - alpha * mask[:, :, None]) + session_green * alpha
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(composite)
        ax.set_title('All Sessions Aligned to Widefield', fontsize=16)
        ax.axis('off')
        
        # # Add legend showing which sessions are included
        # from matplotlib.patches import Patch
        # legend_elements = [Patch(facecolor='gray', label='Widefield')]
        # for session_id in self.field_coordinates_widefield.keys():
        #     legend_elements.append(
        #         Patch(facecolor='lime', alpha=0.6, label=f'Session {self.session_names[session_id]}')
        #     )
        # ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig, ax

    def visualize_cell_transformation(self, session_id, cell_coords_local, save_path=None):
        """
        Visualize how local cell coordinates transform to global widefield coordinates.
        Shows both the local view and the global view with the cells marked.
        
        Parameters
        ----------
        session_id : int or str
            The session identifier
        cell_coords_local : array-like, shape (n_cells, 2)
            Cell coordinates in the session's local pixel space (x, y)
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, axes
            Matplotlib figure and axes
        """
        
        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed
        
        # Convert to numpy array
        cell_coords_local = np.array(cell_coords_local, dtype=float)
        
        # Transform cells to global coordinates
        cell_coords_global = self.transform_cells_to_widefield(session_id, cell_coords_local)
        
        # Get images
        mean_img = self.mean_images[session_id].copy()
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
        
        widefield = self.widefield_img_shifted.copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        
        # Get transformation parameters
        params = self.field_coordinates_widefield[session_id]
        
        # Create transformed version of mean image
        mean_img_resized = cv2.resize(mean_img_norm,
                                    (widefield.shape[1], widefield.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        mean_img_transformed = transform_image(
            mean_img_resized,
            params['scale'],
            params['rotation'],
            params['dx'],
            params['dy']
        )
        
        # Create composite for global view
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        session_green = np.zeros_like(composite)
        session_green[:, :, 1] = mean_img_transformed
        mask = (mean_img_transformed > 0).astype(float)
        composite = composite * (1 - 0.5 * mask[:, :, None]) + session_green * 0.5
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left panel: Local coordinates
        axes[0].imshow(mean_img_norm, cmap='gray')
        axes[0].scatter(cell_coords_local[:, 0], cell_coords_local[:, 1], 
                    c='red', s=100, marker='o', edgecolors='white', linewidths=2)
        for i, coord in enumerate(cell_coords_local):
            axes[0].text(coord[0] + 5, coord[1] + 5, f'Cell {i+1}', 
                        color='red', fontsize=10, fontweight='bold')
        axes[0].set_title(f'Local View: Session {self.session_names[session_id]}', fontsize=14)
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        
        # Right panel: Global coordinates
        axes[1].imshow(composite)
        axes[1].scatter(cell_coords_global[:, 0], cell_coords_global[:, 1],
                    c='red', s=100, marker='o', edgecolors='white', linewidths=2)
        for i, coord in enumerate(cell_coords_global):
            axes[1].text(coord[0] + 5, coord[1] + 5, f'Cell {i+1}',
                        color='red', fontsize=10, fontweight='bold')
        axes[1].set_title('Global View: Widefield Coordinates', fontsize=14)
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        
        # Print transformation info
        print(f"\nTransformation for Session {self.session_names[session_id]}:")
        print(f"  Scale: {params['scale']:.3f}")
        print(f"  Rotation: {params['rotation']:.1f}°")
        print(f"  Translation: dx={params['dx']:.1f}px, dy={params['dy']:.1f}px")
        print(f"\nCell Coordinates:")
        for i in range(len(cell_coords_local)):
            print(f"  Cell {i+1}:")
            print(f"    Local:  ({cell_coords_local[i, 0]:.1f}, {cell_coords_local[i, 1]:.1f})")
            print(f"    Global: ({cell_coords_global[i, 0]:.1f}, {cell_coords_global[i, 1]:.1f})")
        
        plt.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        plt.show()
        
        return fig, axes
    
    def transform_cells_to_widefield(self, session_id, cell_coords_local):
        """
        Transform cell coordinates from a session's local coordinate system to 
        global widefield coordinates.
        
        Parameters
        ----------
        session_id : int or str
            The session identifier
        cell_coords_local : array-like, shape (n_cells, 2)
            Cell coordinates in the session's 760x760 FOV pixel space (x, y).
            
        Returns
        -------
        cell_coords_global : ndarray, shape (n_cells, 2)
            Cell coordinates in widefield pixel space (x, y)
        """
        
        # Check if session has been aligned
        if not hasattr(self, 'field_coordinates_widefield'):
            raise ValueError("No alignments found. Run align_to_widefield() first.")
        
        if session_id not in self.field_coordinates_widefield:
            raise ValueError(f"Session {session_id} has not been aligned yet.")
        
        # Get transformation parameters
        params = self.field_coordinates_widefield[session_id]
        scale = params['scale']
        rotation = params['rotation']
        dx = params['dx']
        dy = params['dy']
        
        # Get image dimensions
        mean_img = self.mean_images[session_id]
        h_orig, w_orig = mean_img.shape 
        h_wf, w_wf = self.widefield_img_shifted.shape 
        
        # Convert to numpy array with float dtype
        cell_coords_local = np.array(cell_coords_local, dtype=float)
        
        # Step 1: Scale coordinates from 760x760 to widefield dimensions (as if resized)
        resize_scale_x = w_wf / w_orig
        resize_scale_y = h_wf / h_orig
        
        coords_resized = cell_coords_local.copy()
        coords_resized[:, 0] *= resize_scale_x
        coords_resized[:, 1] *= resize_scale_y
        
        # Now coords_resized are in widefield pixel space (but not yet transformed)
        
        # Center of widefield image
        center_x = w_wf / 2
        center_y = h_wf / 2
        
        # Convert rotation to radians
        angle_rad = -np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        cell_coords_global = []
        
        for coord in coords_resized:
            x, y = coord
            
            # Step 2: Make relative to widefield center
            x_centered = x - center_x
            y_centered = y - center_y
            
            # # Step 3: Apply scale
            x_scaled = x_centered * scale
            y_scaled = y_centered * scale
            
            # Step 4: Apply rotation
            x_rotated = x_scaled * cos_a - y_scaled * sin_a
            y_rotated = x_scaled * sin_a + y_scaled * cos_a
            
            # Step 5: Translate back to center and add dx, dy offset
            x_final = x_rotated + center_x + dx
            y_final = y_rotated + center_y + dy
            
            cell_coords_global.append([x_final, y_final])
        
        return np.array(cell_coords_global)

    def transform_all_cells_to_widefield(self):
        """
        Transform all cells from all sessions to widefield coordinates.
        Assumes cell coordinates are stored in self.radial_sessions DataFrame
        with columns 'session', 'cell_id', and 'coords'.
        
        Returns
        -------
        radial_sessions_global : pd.DataFrame
            Updated DataFrame with additional 'coords_widefield' column containing
            global coordinates
        """
        
        if not hasattr(self, 'radial_sessions'):
            raise ValueError("No radial_sessions found. Load or compute session stats first.")
        
        # Create a copy of the dataframe
        radial_sessions_global = self.radial_sessions.copy()
        
        # Add new column for global coordinates
        coords_widefield = []
        
        for idx, row in radial_sessions_global.iterrows():
            session_id = row['session']
            cell_coord_local = np.array(row['coords']).reshape(1, 2)  # Single cell coordinate
            
            try:
                coord_global = self.transform_cells_to_widefield(session_id, cell_coord_local)
                coords_widefield.append(coord_global[0])
            except ValueError as e:
                print(f"Warning: Could not transform cell from session {session_id}: {e}")
                coords_widefield.append([np.nan, np.nan])
        
        radial_sessions_global['coords_widefield'] = coords_widefield
        
        return radial_sessions_global

    def align_translation(self, fixed_numeral = 'IV',
                      moving_numeral = 'I',
                      pixels_per_unit=760/2000,
                      canvas_units=4000,
                      slider_limit_px=1000, show_result = True):
        """
        Launch an interactive alignment tool.

        Parameters
        ----------
        fixed_img : ndarray
            The reference image (grayscale or color) as a NumPy array.
        moving_img : ndarray
            The image to be translated to align with fixed_img.
            Must already be the same shape as fixed_img.
        pixels_per_unit : float, optional
            Conversion factor from pixels to your “unit” system.
        canvas_units : int, optional
            Side length (in your units) of the virtual canvas to pad into.
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders.

        Returns
        -------
        dx : int
            Final horizontal translation (in px) you selected.
        dy : int
            Final vertical translation (in px) you selected.
        """
        # Ensure interactive backend for external window
        

        # Utility sub‐functions
        def translate_image(img, dx, dy):
            h, w = img.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

        def pad_to_canvas(img, canvas_size):
            canvas = np.zeros(canvas_size, dtype=img.dtype)
            h, w = img.shape[:2]
            ch, cw = canvas_size
            y_off = (ch - h)//2
            x_off = (cw - w)//2
            canvas[y_off:y_off+h, x_off:x_off+w] = img
            return canvas
        
        fixed_img = self.mean_images[fixed_numeral]
        moving_img = self.mean_images[moving_numeral]

        # Compute canvas in pixels
        canvas_size_px = int(canvas_units / (1/pixels_per_unit))
        canvas_shape = (canvas_size_px, canvas_size_px)

        # Pad both images
        fixed_pad  = pad_to_canvas(fixed_img,  canvas_shape)
        moving_pad = pad_to_canvas(moving_img, canvas_shape)

        # Set up figure + sliders
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # Initial blend
        dx0, dy0 = 0, 0
        blended = cv2.addWeighted(fixed_pad, 0.5,
                                translate_image(moving_pad, dx0, dy0),
                                0.5, 0)
        # define extents in units
        half = canvas_size_px // 2
        extent = [-half*pixels_per_unit, half*pixels_per_unit,
                -half*pixels_per_unit, half*pixels_per_unit]
        img_disp = ax.imshow(blended, cmap='gray', extent=extent)

        # Print the moving image title in the plot title
        ax.set_title(f"Aligning moving image: {moving_numeral}")

        # sliders
        ax_dx = plt.axes([0.25, 0.1, 0.5, 0.03])
        ax_dy = plt.axes([0.25, 0.15, 0.5, 0.03])
        sdx = Slider(ax_dx, 'dx (px)', -slider_limit_px,
                    slider_limit_px, valinit=dx0, valstep=1)
        sdy = Slider(ax_dy, 'dy (px)', -slider_limit_px,
                    slider_limit_px, valinit=dy0, valstep=1)

        def _update(_):
            dx, dy = int(sdx.val), int(sdy.val)
            trans = translate_image(moving_pad, dx, dy)
            blend = cv2.addWeighted(fixed_pad, 0.5, trans, 0.5, 0)
            img_disp.set_data(blend)
            ax.set_title(
                f"Aligning moving image: {moving_numeral} | "
                f"dx = {dx} px ({dx * pixels_per_unit:.2f} units), "
                f"dy = {dy} px ({dy * pixels_per_unit:.2f} units)"
            )
            fig.canvas.draw_idle()

        sdx.on_changed(_update)
        sdy.on_changed(_update)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Show and block until window is closed, in a separate window
        plt.show(block=True)

        if show_result:
            result_img = translate_image(moving_pad, int(sdx.val), int(sdy.val))
            blended_result = cv2.addWeighted(fixed_pad, 0.5, result_img, 0.5, 0)
            plt.figure()
            plt.imshow(blended_result, cmap='gray')
            plt.title("Final alignment: fixed + shifted moving")
            plt.show()

        # Return final values
        return self.pixels_to_units(int(sdx.val), scope='2p'), self.pixels_to_units(int(sdy.val), scope='2p')

    def pixels_to_units(self, pixels, scope='2p'):

        if scope == '2p':
            return pixels * (self.fov_units / self.fov_pixels)
        elif scope == 'widefield':
            return pixels * (self.widefield_fov_units / self.widefield_fov_pixels)
    
    def units_to_pixels(self, units, scope='2p'):

        if scope == '2p':
            return units * (self.fov_pixels / self.fov_units)
        elif scope == 'widefield':
            return units * (self.widefield_fov_pixels / self.widefield_fov_units)

    def plot_window(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 8x8 inches, adjust as needed

        # Plot each image at its field coordinates
        for i, session_name in enumerate(self.session_names):

            img = self.mean_images[i]
            dx_units, dy_units = self.field_coordinates[session_name]
            img_h, img_w = img.shape
            img_w_units = self.pixels_to_units(img_w, scope='2p')
            img_h_units = self.pixels_to_units(img_h, scope='2p')
            # Calculate extent in units for this image, centered at (dx_units, dy_units)
            extent_units = [
                dx_units - img_w_units/2, dx_units + img_w_units/2,
                dy_units + img_h_units/2, dy_units - img_h_units/2  # origin='upper'
            ]
            ax.imshow(img, cmap='gray', alpha=1, extent=extent_units, origin='upper', label=session_name)

        # Set axis limits in units (canvas is 4000x4000 units)
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(2500, -2500)
        ax.set_xlabel("X (units)")
        ax.set_ylabel("Y (units)")
        ax.set_title("All Images on Figure Canvas (units)")

        return fig, ax
     
    def extract_cell_coordinates(self, suite2p_stat, session_coords, cell_id):

        cell_stat = suite2p_stat[cell_id]

        xpix = cell_stat['xpix'][~cell_stat['overlap']].mean() - 380
        ypix = cell_stat['ypix'][~cell_stat['overlap']].mean() - 380

        x_units = self.pixels_to_units(xpix, scope='2p') + session_coords[0]
        y_units = self.pixels_to_units(ypix, scope='2p') + session_coords[1]

        return x_units, y_units
    
    def load_widefield_maps(self):

        # Load the snap.mat file from the animal directory
        snap_mat_path = os.path.join(self.multi_session_dir, 'snap.mat')
        snap_data = loadmat(snap_mat_path)
        widefield_img = snap_data['snap']

        # resize widefield_img to the size of 400x400
        widefield_img = cv2.resize(widefield_img, (400, 400))

        self.widefield_img = widefield_img

        maps_mat_path = os.path.join(self.multi_session_dir, 'maps.mat')
        maps_data = loadmat(maps_mat_path, struct_as_record=False, squeeze_me=True)
        self.VFS = maps_data['maps'].VFS_raw
        self.VFS = gaussian_filter(self.VFS, sigma=1.5)

    def set_widefield_origin(self):

        # 1) show image with origin='lower' so X→cols, Y→rows directly
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(
            self.widefield_img,
            cmap='gray',
            extent=[0, self.widefield_fov_pixels, 0, self.widefield_fov_pixels],
            origin='upper'
        )
        ax.set_title("Click to set the origin (0,0) in widefield pixels")
    
        print("Please click on the image to set the new origin (0,0).")
        x_click, y_click = plt.ginput(1, timeout=-1)[0]
        plt.close(fig)

        # 2) convert to integer array indices
        clicked_col = int(400 - round(y_click))
        clicked_row = int(round(x_click))

        # 3) figure out how many pixels to shift
        n_rows, n_cols = self.widefield_img.shape
        center_col = n_cols // 2
        center_row = n_rows // 2

        dx = center_col - clicked_col   # how far to shift columns (axis=1)
        dy = center_row - clicked_row   # how far to shift rows    (axis=0)

        return dx, dy

    def shift_widefield(self):

        dx = self.widefield_dx
        dy = self.widefield_dy

        # Shift widefield image
        shifted_img = np.roll(self.widefield_img, shift=(dy, dx), axis=(0, 1))
        # Zero out new empty strips for widefield image
        if dy > 0:
            shifted_img[:dy, :] = 0
        elif dy < 0:
            shifted_img[dy:, :] = 0
        if dx > 0:
            shifted_img[:, :dx] = 0
        elif dx < 0:
            shifted_img[:, dx:] = 0
        self.widefield_img_shifted = shifted_img

        # Shift VFS map
        shifted_vfs = np.roll(self.VFS, shift=(dy, dx), axis=(0, 1))
        # Zero out new empty strips for VFS map
        if dy > 0:
            shifted_vfs[:dy, :] = 0
        elif dy < 0:
            shifted_vfs[dy:, :] = 0
        if dx > 0:
            shifted_vfs[:, :dx] = 0
        elif dx < 0:
            shifted_vfs[:, dx:] = 0
        self.VFS_shifted = shifted_vfs

        # Show both shifted images in a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Widefield image subplot
        axs[0].imshow(
            shifted_img,
            cmap='gray',
            extent=[0, self.widefield_fov_units, 0, self.widefield_fov_units],
            origin='upper'
        )
        axs[0].set_title("Shifted Widefield Image")
        axs[0].set_xlabel("Widefield units (um)")
        axs[0].set_ylabel("Widefield units (um)")
        center_units = self.widefield_fov_units / 2
        axs[0].axhline(center_units, color='red', linestyle='--', linewidth=1)
        axs[0].axvline(center_units, color='red', linestyle='--', linewidth=1)

        # VFS map subplot
        im = axs[1].imshow(
            shifted_vfs,
            cmap='jet',
            extent=[0, self.widefield_fov_units, 0, self.widefield_fov_units],
            origin='upper'
        )
        axs[1].set_title("Shifted VFS Map")
        axs[1].set_xlabel("Widefield units (um)")
        axs[1].set_ylabel("Widefield units (um)")
        axs[1].axhline(center_units, color='red', linestyle='--', linewidth=1)
        axs[1].axvline(center_units, color='red', linestyle='--', linewidth=1)
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def draw_areas(self):

        if self.VFS_shifted is None:
            self.shift_widefield()

        # Calculate extent in units for the shifted widefield image and VFS map
        img_h, img_w = self.VFS_shifted.shape
        img_w_units = self.pixels_to_units(img_w, scope='widefield')
        img_h_units = self.pixels_to_units(img_h, scope='widefield')
        extent_units = [
            -img_w_units / 2, img_w_units / 2,
            img_h_units / 2, -img_h_units / 2  # origin='upper'
        ]

        # Create figure with VFS_shifted as the base image
        fig, window_ax = plt.subplots(figsize=(8, 8), dpi=100)
        window_ax.imshow(
            np.rot90(self.VFS_shifted, 2),
            cmap='jet',
            extent=extent_units,
            origin='upper'
        )
        window_ax.set_xlim(extent_units[0], extent_units[1])
        window_ax.set_ylim(extent_units[2], extent_units[3])
        window_ax.set_xlabel("X (units)")
        window_ax.set_ylabel("Y (units)")

        # Overlay VFS_shifted with some transparency
        im = window_ax.imshow(
            np.rot90(self.VFS_shifted, 2),
            cmap='jet',
            extent=extent_units,
            origin='upper',
            alpha=0.4
        )

        # --- Improved lasso/area drawing logic ---
        current_area_idx = [0]
        areas_verts = {}
        patches_dict = {}

        def draw_all_patches():
            for i, (area_name, area_verts) in enumerate(areas_verts.items()):
                color = f"C{i%10}"
                patch = Polygon(area_verts, closed=True, facecolor='none', edgecolor=color, linewidth=3, label=area_name)
                window_ax.add_patch(patch)
                patches_dict[area_name] = patch
            # Remove duplicate legends
            handles, labels = window_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            window_ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            fig.canvas.draw_idle()

        def onselect(verts):
            area = self.area_names[current_area_idx[0]]
            areas_verts[area] = list(verts)
            draw_all_patches()
            # Move to next area if not done
            if current_area_idx[0] < len(self.area_names) - 1:
                current_area_idx[0] += 1
                print(f"Draw lasso for: {self.area_names[current_area_idx[0]]}")
                # Update the title to instruct the user
                window_ax.set_title(self.area_names[current_area_idx[0]])
            else:
                print("All areas drawn. Close the figure window to continue.")
                window_ax.set_title("All areas drawn. Close the figure window to continue.")
            fig.canvas.draw_idle()

        # Set initial title with instruction
        window_ax.set_title(
            f"All Images on Figure Canvas (units)\nDraw lasso for: {self.area_names[0]}"
        )

        print(f"Draw lasso for: {self.area_names[0]}")
        lasso = LassoSelector(window_ax, onselect, props=dict(color='black', linewidth=3))

        # If any areas already have verts (e.g., from a previous run), show them
        if areas_verts:
            draw_all_patches()

        plt.show()

        return areas_verts
    
    def plot_areas(self):

        fig, ax = self.plot_window()

        for i, (area_name, area_verts) in enumerate(self.areas_verts.items()):
            color = f"C{i%10}"
            patch = Polygon(area_verts, closed=True, facecolor='none', edgecolor=color, linewidth=3, label=area_name)
            ax.add_patch(patch)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        return fig, ax
    
    def is_inside_boundary(self, x, y, area_name):

        path = Path(self.areas_verts[area_name])
        return path.contains_point((x, y))
    
    def assign_area_to_cells(self):

        area = []
        for _, row in self.radial_sessions.iterrows():
            x_units, y_units = row['coords']
            for area_name in self.areas_verts.keys():
                if self.is_inside_boundary(x_units, y_units, area_name):
                    area.append(area_name)
                    break
            else:
                area.append('None')
        df = self.radial_sessions.copy()
        df.insert(1, 'area', area)
        self.radial_sessions = df

    def plot_areas_with_cells(self, areas_to_plot=['PM'], modality='binocular', color_by='distance'):

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        # # Calculate extent in units for the shifted widefield image and VFS map
        # img_h, img_w = self.widefield_img_shifted.shape
        # img_w_units = self.pixels_to_units(img_w, scope='widefield')
        # img_h_units = self.pixels_to_units(img_h, scope='widefield')
        # extent_units = [
        #     -img_w_units / 2, img_w_units / 2,
        #     img_h_units / 2, -img_h_units / 2  # origin='upper'
        # ]

        # # Overlay VFS_shifted with some transparency
        # im = ax.imshow(
        #     np.rot90(self.widefield_img_shifted, 2),
        #     cmap='gray',
        #     extent=extent_units,
        #     origin='upper',
        #     alpha=0.7
        # )

        fig, ax = self.plot_window()

        colors = setup_plot_style()
        if modality == 'binocular':
            rows = self.radial_sessions[self.radial_sessions['binocular_modulated'] == True]
        elif modality == 'monocular':
            rows = self.radial_sessions[self.radial_sessions['monocular_modulated'] == True]
        elif modality == 'all':
            rows = self.radial_sessions[self.radial_sessions['is_responsive'] == True]

        for _, row in rows.iterrows():
            x_units, y_units = row['coords']

            for area_name in areas_to_plot:
                inside = self.is_inside_boundary(x_units, y_units, area_name)

                if inside:
                    # Color by preferred distance
                    if color_by == 'distance':
                        distance = row[modality + '_distance']
                        ax.scatter(x_units, y_units, c=colors[distance], s=5)
                    elif color_by == 'rf':
                        ax.scatter(x_units, y_units, c=row['binocular_rf'], cmap='viridis', vmin=-30, vmax=75, s=5)

        for i, area_name in enumerate(areas_to_plot):
            color = f"C{i%10}"
            patch = Polygon(self.areas_verts[area_name], closed=True, facecolor='none', edgecolor=color, linewidth=3, label=area_name)
            ax.add_patch(patch)

        ax.axis('off')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

        return fig, ax
    
    def plot_percetange_of_cells_in_areas(self):
        """
        Plots a bar plot showing the percentage of cells in each area that are:
        - binocular
        - monocular
        - both (binocular and monocular modulated)
        (non-modulated cells are excluded)
        """

        # Get all area names
        area_names = list(self.areas_verts.keys())
        categories = ['binocular', 'monocular', 'both']

        # Prepare a matrix: rows=areas, cols=categories
        percentages = np.zeros((len(area_names), len(categories)))

        # For each area, count cells in each category
        for i, area in enumerate(area_names):
            # Get all cells inside this area
            inside_mask = self.radial_sessions.apply(
                lambda row: self.is_inside_boundary(row['coords'][0], row['coords'][1], area), axis=1
            )
            area_cells = self.radial_sessions[inside_mask]

            # Only consider modulated cells (binocular or monocular)
            modulated_mask = (area_cells['binocular_modulated'] == True) | (area_cells['monocular_modulated'] == True)
            modulated_cells = area_cells[modulated_mask]

            # Boolean masks for each category
            binocular = modulated_cells['binocular_modulated'] == True
            monocular = modulated_cells['monocular_modulated'] == True

            both = binocular & monocular
            only_binocular = binocular & ~monocular
            only_monocular = monocular & ~binocular

            # n_total is now the number of all responsive cells in the area
            responsive_mask = (area_cells['is_responsive'] == True)
            n_total = responsive_mask.sum()
            if n_total == 0:
                percentages[i, 0] = np.nan
            else:
                percentages[i, 0] = only_binocular.sum() / n_total * 100
                percentages[i, 1] = only_monocular.sum() / n_total * 100
                percentages[i, 2] = both.sum() / n_total * 100

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        x = np.arange(len(area_names))

        colors = ['royalblue', 'tomato', 'purple']
        for j, cat in enumerate(categories):
            ax.bar(x + j*bar_width, percentages[:, j], width=bar_width, label=cat, color=colors[j])

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(area_names, rotation=45, ha='right')
        ax.set_ylabel('Percentage of cells (%)')
        ax.set_title('Percentage of modulated cells in each area by modulation type')
        ax.legend(title='Modulation')
        ax.set_ylim(0, 20)
        plt.tight_layout()
        plt.show()

    def plot_preferred_distance_in_areas(self, modality='binocular'):

        # Get all area names
        area_names = list(self.areas_verts.keys())
        # 4 categories for binocular modulated cells by preferred distance
        distance_categories = [0, 1, 2, 3]
        colors = setup_plot_style()
        category_labels = [f'dist {d}' for d in distance_categories]

        # Prepare a matrix: rows=areas, cols=distance categories
        percentages = np.zeros((len(area_names), len(distance_categories)))

        # For each area, count binocular modulated cells in each distance category
        for i, area in enumerate(area_names):
            # Get all cells inside this area
            inside_mask = self.radial_sessions.apply(
                lambda row: self.is_inside_boundary(row['coords'][0], row['coords'][1], area), axis=1
            )
            area_cells = self.radial_sessions[inside_mask]

            # Only consider binocular modulated cells
            if modality == 'binocular':
                binocular_cells = area_cells[(area_cells['is_responsive'] == True) & (area_cells['binocular_modulated'] == True) & (area_cells['monocular_modulated'] == False)]
            elif modality == 'monocular':
                binocular_cells = area_cells[(area_cells['is_responsive'] == True) & (area_cells['monocular_modulated'] == True) & (area_cells['binocular_modulated'] == False)]
            elif modality == 'none':
                binocular_cells = area_cells[(area_cells['is_responsive'] == True) & (area_cells['binocular_modulated'] == False) & (area_cells['monocular_modulated'] == False)]

            # Only consider responsive cells
            responsive_mask = (binocular_cells['is_responsive'] == True)
            responsive_binocular_cells = binocular_cells[responsive_mask]

            n_total = len(responsive_binocular_cells)
            if n_total == 0:
                percentages[i, :] = np.nan
            else:
                # Count for each preferred distance
                for j, d in enumerate(distance_categories):
                    n_in_cat = (responsive_binocular_cells['binocular_distance'] == d).sum()
                    percentages[i, j] = n_in_cat / n_total * 100

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.18
        x = np.arange(len(area_names))

        for j, label in enumerate(category_labels):
            ax.bar(x + j*bar_width, percentages[:, j], width=bar_width, label=label, color=colors[j])

        ax.set_xticks(x + 1.5*bar_width)
        ax.set_xticklabels(area_names, rotation=45, ha='right')
        ax.set_ylabel(f'Percentage of {modality} modulated cells (%)')
        ax.set_title(f'Percentage of {modality} modulated cells in each area by preferred distance')
        ax.legend(title='Preferred Distance')
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.show()

    def plot_cells(self, modality='binocular'):

        if modality == 'binocular':
            rows = self.radial_sessions[(self.radial_sessions['is_responsive'] == True) & (self.radial_sessions['binocular_modulated'] == True)]
        elif modality == 'monocular':
            rows = self.radial_sessions[(self.radial_sessions['is_responsive'] == True) & (self.radial_sessions['monocular_modulated'] == True)]
        elif modality == 'none':
            rows = self.radial_sessions[(self.radial_sessions['is_responsive'] == True) & (self.radial_sessions['monocular_modulated'] == False) & (self.radial_sessions['binocular_modulated'] == False)]

        figure_dir = os.path.join(self.animal_dir, 'figures', modality )
        for name in self.session_names:

            suite2p_dir = self.get_suite2p_dir(self.session_names[name])
            with open(os.path.join(suite2p_dir, f'{self.session_names[name]}_summ.pkl'), 'rb') as f:
                summ = pickle.load(f)

            session_cells = rows[rows['session'] == name]
            for cell in session_cells['cell_id']:
                print(f'Session: {name} - Cell: {cell}', end='\r', flush=True)
                fig = summ.radial_plots.plot_all_traces_colormap_distances(cell, method='regression')
                fig.savefig(os.path.join(figure_dir, f'{name}_{cell}.png'))
                plt.close()

class WidefieldAligner:

    def __init__(self, animal_id, experiment_name):

        self.base_multi_dir = r'D:/data-distance-coding-project/multi-session-data/'
        self.base_processed_dir = r'D:/data-distance-coding-project/processed-data/'
        self.multi_session_dir = os.path.join(self.base_multi_dir, animal_id + '_' + experiment_name)

        self.animal_id = animal_id
        self.experiment_name = experiment_name

        self.session_names = self.read_config_file()
        self.mean_images = self.load_mean_images()
        self.widefield_maps = self.load_widefield_maps()

        self.areas_verts = self.draw_areas(self.widefield_maps['VFS'])
        self.alignment_params = self.align_all_sessions_to_widefield()

    def read_config_file(self, multi_session_info_file = None):

        if multi_session_info_file is None:
            session_info_file = os.path.join(self.multi_session_dir, self.animal_id + '_' + self.experiment_name + '_sessions.txt')
        else:
            session_info_file = multi_session_info_file

        session_names = []
        try:
            with open(session_info_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Multi-session session info file not found: {session_info_file}")
            raise
        except Exception as e:
            print(f"Error reading multi-session session info file {session_info_file}: {e}")
            raise
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                session_name_in_commas = line.split(',')
                mouse_name, record_date, experiment_name = session_name_in_commas
                session_name = record_date + '_' + mouse_name + '_' + experiment_name
                session_names.append(session_name)

        return session_names

    def load_mean_images(self):

        self.mean_images = []
        for session_name in self.session_names:
            session_dir = self.get_session_dir(session_name)
            suite2p_dir = os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy')
            if not os.path.exists(suite2p_dir):
                raise FileNotFoundError(f"Suite2p directory does not exist: {suite2p_dir}")
            ops = np.load(suite2p_dir, allow_pickle=True).item()
            self.mean_images.append(ops['meanImg'])

        return self.mean_images

    def get_session_dir(self, session_name):
        
        session_dir = os.path.join(self.base_processed_dir, session_name)

        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

        return session_dir 

    def load_widefield_maps(self):
        
        widefield_maps = {}

        # Load the snap.mat file from the animal directory
        snap_mat_path = os.path.join(self.multi_session_dir, 'snap.mat')
        snap_data = loadmat(snap_mat_path)
        widefield_img = snap_data['snap']
        # resize widefield_img to the size of 400x400
        widefield_img = cv2.resize(widefield_img, (400, 400))
        widefield_maps['img'] = widefield_img

        maps_mat_path = os.path.join(self.multi_session_dir, 'maps.mat')
        maps_data = loadmat(maps_mat_path, struct_as_record=False, squeeze_me=True)
        VFS_raw = maps_data['maps'].VFS_raw
        VFS_filtered = gaussian_filter(VFS_raw, sigma=1.5)
        widefield_maps['VFS'] = VFS_filtered

        alt_raw = maps_data['maps'].VerticalRetinotopy
        alt_filtered = gaussian_filter(alt_raw, sigma=1.5)
        widefield_maps['alt'] = alt_filtered

        azi_raw = maps_data['maps'].HorizontalRetinotopy
        azi_filtered = gaussian_filter(azi_raw, sigma=1.5)
        widefield_maps['azi'] = azi_filtered

        return widefield_maps

    def draw_areas(self, VFS_map):

        area_names = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']

        fig, window_ax = plt.subplots(figsize=(8, 8), dpi=100)
        window_ax.imshow(
            VFS_map,
            cmap='jet',
            extent=[0, 400, 0, 400],
            origin='upper'
        )
        window_ax.set_xlim(0, 400)
        window_ax.set_ylim(0, 400)
        window_ax.set_xlabel("X (units)")
        window_ax.set_ylabel("Y (units)")

        im = window_ax.imshow(
            VFS_map,
            cmap='jet',
            extent=[0, 400, 0, 400],
            origin='upper',
            alpha=0.4
        )

        # --- Improved lasso/area drawing logic ---
        current_area_idx = [0]
        areas_verts = {}
        patches_dict = {}

        def draw_all_patches():
            for i, (area_name, area_verts) in enumerate(areas_verts.items()):
                color = f"C{i%10}"
                patch = Polygon(area_verts, closed=True, facecolor='none', edgecolor=color, linewidth=3, label=area_name)
                window_ax.add_patch(patch)
                patches_dict[area_name] = patch
            # Remove duplicate legends
            handles, labels = window_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            window_ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            fig.canvas.draw_idle()

        def onselect(verts):
            area = area_names[current_area_idx[0]]
            areas_verts[area] = list(verts)
            draw_all_patches()
            # Move to next area if not done
            if current_area_idx[0] < len(area_names) - 1:
                current_area_idx[0] += 1
                print(f"Draw lasso for: {area_names[current_area_idx[0]]}")
                # Update the title to instruct the user
                window_ax.set_title(area_names[current_area_idx[0]])
            else:
                print("All areas drawn. Close the figure window to continue.")
                window_ax.set_title("All areas drawn. Close the figure window to continue.")
            fig.canvas.draw_idle()

        # Set initial title with instruction
        window_ax.set_title(
            f"All Images on Figure Canvas (units)\nDraw lasso for: {area_names[0]}"
        )

        print(f"Draw lasso for: {area_names[0]}")
        lasso = LassoSelector(window_ax, onselect, props=dict(color='black', linewidth=3))

        # If any areas already have verts (e.g., from a previous run), show them
        if areas_verts:
            draw_all_patches()

        plt.show()

        return areas_verts

    def align_session_to_widefield(self, session_id, slider_limit_px=500):
        """
        Align a single session's mean image to the widefield image with scale, rotation, and translation.
        Previously aligned sessions remain visible in green overlay.
        
        Parameters
        ----------
        session_id : int
            The session index
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.
            
        Returns
        -------
        dict
            Dictionary containing transformation parameters:
            {'scale': float, 'rotation': float, 'dx': float, 'dy': float}
        """

        
        scale0 = 0.33
        rotation0 = 174

        dx0, dy0 = 0.0, 0.0
        alpha0 = 0.5

        # Helper function to apply transformation to an image
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            
            # Create transformation matrix
            # First scale and rotate around center
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            
            # Then add translation
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            # Apply transformation
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed
        
        # Get the widefield reference image
        widefield = self.widefield_maps['img'].copy()
        
        # Normalize widefield to 0-1 for display
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        
        # Get the mean image to align
        mean_img = self.mean_images[session_id].copy()
        
        # Normalize mean image to 0-1
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
        
        # Resize mean_img to match widefield size
        mean_img_resized = cv2.resize(mean_img_norm, 
                                    (widefield.shape[1], widefield.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        
        # Create green colormap for mean images
        green_cmap = LinearSegmentedColormap.from_list('green', ['black', 'lime'])
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)
        
        # Create initial composite
        # Start with widefield in grayscale
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Overlay previously aligned sessions in green (dimmer, underneath)
        if hasattr(self, 'field_coordinates_widefield'):
            for prev_session_id, params in self.field_coordinates_widefield.items():
                if prev_session_id != session_id:  # Don't show the current one yet
                    prev_img = self.mean_images[prev_session_id].copy()
                    prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                    prev_img_resized = cv2.resize(prev_img_norm,
                                                (widefield.shape[1], widefield.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                    prev_transformed = transform_image(
                        prev_img_resized,
                        params['scale'],
                        params['rotation'],
                        params['dx'],
                        params['dy']
                    )
                    # Blend green on top of grayscale using alpha compositing
                    prev_alpha = 0.3
                    prev_green = np.zeros_like(composite)
                    prev_green[:, :, 1] = prev_transformed
                    composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha
        
        # Add current session on top (brighter)
        current_transformed = transform_image(mean_img_resized, scale0, rotation0, dx0, dy0)
        current_green = np.zeros_like(composite)
        current_green[:, :, 1] = current_transformed
        current_mask = (current_transformed > 0).astype(float)
        composite = composite * (1 - alpha0 * current_mask[:, :, None]) + current_green * alpha0
        
        img_display = ax.imshow(composite)
        ax.set_title(f"Aligning session: {self.session_names[session_id]}")
        ax.axis('off')
        
        # Create sliders
        ax_scale = plt.axes([0.15, 0.25, 0.7, 0.02])
        ax_rotation = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_dx = plt.axes([0.15, 0.15, 0.7, 0.02])
        ax_dy = plt.axes([0.15, 0.10, 0.7, 0.02])
        ax_alpha = plt.axes([0.15, 0.05, 0.7, 0.02])
        
        slider_scale = Slider(ax_scale, 'Scale', 0.1, 1.0, valinit=scale0, valstep=0.01)
        slider_rotation = Slider(ax_rotation, 'Rotation (°)', -180, 180, valinit=rotation0, valstep=1)
        slider_dx = Slider(ax_dx, 'dx (px)', -slider_limit_px, slider_limit_px, valinit=dx0, valstep=1)
        slider_dy = Slider(ax_dy, 'dy (px)', -slider_limit_px, slider_limit_px, valinit=dy0, valstep=1)
        slider_alpha = Slider(ax_alpha, 'Transparency', 0.0, 1.0, valinit=alpha0, valstep=0.05)
        
        def update(val):
            """Update the display when sliders change."""
            scale = slider_scale.val
            rotation = slider_rotation.val
            dx = slider_dx.val
            dy = slider_dy.val
            alpha = slider_alpha.val
            
            # Recreate composite
            composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
            
            # Overlay previously aligned sessions (dimmer, underneath)
            if hasattr(self, 'field_coordinates_widefield'):
                for prev_numeral, params in self.field_coordinates_widefield.items():
                    if prev_session_id != session_id:
                        prev_img = self.mean_images[prev_numeral].copy()
                        prev_img_norm = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
                        prev_img_resized = cv2.resize(prev_img_norm,
                                                    (widefield.shape[1], widefield.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                        prev_transformed = transform_image(
                            prev_img_resized,
                            params['scale'],
                            params['rotation'],
                            params['dx'],
                            params['dy']
                        )
                        # Add to green channel with reduced intensity
                        prev_alpha = 0.3
                        prev_green = np.zeros_like(composite)
                        prev_green[:, :, 1] = prev_transformed
                        composite = composite * (1 - prev_alpha * (prev_transformed[:, :, None] > 0)) + prev_green * prev_alpha
            
            # Add current session on top (brighter)
            current_transformed = transform_image(mean_img_resized, scale, rotation, dx, dy)
            current_green = np.zeros_like(composite)
            current_green[:, :, 1] = current_transformed
            current_mask = (current_transformed > 0).astype(float)
            composite = composite * (1 - alpha * current_mask[:, :, None]) + current_green * alpha
            
            img_display.set_data(composite)
            ax.set_title(
                f"Aligning session: {self.session_names[session_id]} | "
                f"Scale: {scale:.2f}, Rotation: {rotation:.0f}°, "
                f"dx: {dx:.0f}px, dy: {dy:.0f}px"
            )
            fig.canvas.draw_idle()
        
        # Connect sliders to update function
        slider_scale.on_changed(update)
        slider_rotation.on_changed(update)
        slider_dx.on_changed(update)
        slider_dy.on_changed(update)
        slider_alpha.on_changed(update)
        
        plt.show(block=True)
        
        # Return the transformation parameters
        return {
            'scale': slider_scale.val,
            'rotation': slider_rotation.val,
            'dx': slider_dx.val,
            'dy': slider_dy.val
        }

    def align_all_sessions_to_widefield(self, slider_limit_px=500):
        """
        Sequentially align all sessions to the widefield image.
        Each aligned session remains visible while aligning subsequent ones.
        
        Parameters
        ----------
        slider_limit_px : int, optional
            Maximum absolute translation (in px) for the sliders. Default is 500.
            
        Returns
        -------
        dict
            Dictionary mapping session numeral to transformation parameters
        """
        # Initialize storage for transformations
        
        alignment_params = {}
        
        print("Starting sequential alignment to widefield...")
        print(f"Sessions to align: {list(self.session_names)}")
        
        for i, session_name in enumerate(self.session_names):
            print(f"\nAligning session {session_name}...")
            params = self.align_session_to_widefield(i, slider_limit_px=slider_limit_px)
            alignment_params[session_name] = params
        
        return alignment_params

    def plot_aligned_sessions_to_widefield(self, save_path=None):
        """
        Create a result figure showing all aligned sessions overlaid on the widefield image.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, just displays the figure.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        
        # Helper function to apply transformation
        def transform_image(img, scale, rotation, dx, dy):
            """Apply scale, rotation, and translation to an image."""
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            
            M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            transformed = cv2.warpAffine(
                img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return transformed
        
        # Get widefield reference
        widefield = self.widefield_maps_shifted['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        
        # Create composite
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Overlay all aligned sessions
        if not hasattr(self, 'field_coordinates_widefield') or len(self.field_coordinates_widefield) == 0:
            print("No aligned sessions found. Run align_all_sessions_to_widefield() first.")
            return None, None
        
        for idx, (session_id, params) in enumerate(self.field_coordinates_widefield.items()):
            mean_img = self.mean_images[session_id].copy()
            mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
            mean_img_resized = cv2.resize(mean_img_norm,
                                        (widefield.shape[1], widefield.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            transformed = transform_image(
                mean_img_resized,
                params['scale'],
                params['rotation'],
                params['dx'],
                params['dy']
            )
            
            # Blend using alpha compositing in green
            alpha = 0.6
            session_green = np.zeros_like(composite)
            session_green[:, :, 1] = transformed
            mask = (transformed > 0).astype(float)
            composite = composite * (1 - alpha * mask[:, :, None]) + session_green * alpha
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(composite)
        ax.set_title('All Sessions Aligned to Widefield', fontsize=16)
        ax.axis('off')
        
        # # Add legend showing which sessions are included
        # from matplotlib.patches import Patch
        # legend_elements = [Patch(facecolor='gray', label='Widefield')]
        # for session_id in self.field_coordinates_widefield.keys():
        #     legend_elements.append(
        #         Patch(facecolor='lime', alpha=0.6, label=f'Session {self.session_names[session_id]}')
        #     )
        # ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig, ax

    def export_alignment_parameters(self):
        
        wf_params = {
            'wf_maps': self.widefield_maps,
            'areas_verts': self.areas_verts,
            'wf_coordinates': self.alignment_params
        }

        with open(self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl', 'wb') as f:
            pickle.dump(wf_params, f)
            print(f"Alignment parameters saved to: {self.multi_session_dir + '/' + self.animal_id + '_' + self.experiment_name + '_alignment_parameters.pkl'}")

        return wf_params

    def plot_widefield_with_areas(self):

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        img = self.widefield_maps['img']
        ax.imshow(img, cmap='gray', alpha=1, extent=[0, 400, 0, 400], origin='upper')

        for i, (area_name, area_verts) in enumerate(self.areas_verts.items()):
            print(np.array(area_verts).shape)
            color = f"C{i%10}"
            patch = Polygon(area_verts, closed=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=3, label=area_name)
            ax.add_patch(patch)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        ax.set_title(f'Widefield Map with Areas\n{self.animal_id} - {self.experiment_name}')
        plt.show()

        return fig, ax

class SessionToWidefield:

    def __init__(self, session_info):

        self.record_date = session_info[0][1]
        self.mouse_name = session_info[0][0]
        self.experiment_name = session_info[0][2]

        session_data_dir = r"D:\data-distance-coding-project\processed-data"
        self.session_data_dir = os.path.join(session_data_dir, self.record_date + '_' + self.mouse_name + '_' + self.experiment_name)

        multi_data_dir = r"D:\data-distance-coding-project\multi-session-data"
        multi_data_dir = os.path.join(multi_data_dir, self.mouse_name + '_' + 'tiling_120um')
        self.multi_data_dir = multi_data_dir

        self.alignment_params, self.field_coordinates = self.load_alignment_parameters()
        self.session_df = self.load_session_df()
        self.suite2p_stat = self.load_suite2p_stat()

    def load_alignment_parameters(self):

        experiment_name = 'tiling_120um'
        alignment_name = f'{self.mouse_name}_{experiment_name}_alignment_parameters.pkl'
        alignment_path = os.path.join(self.multi_data_dir, alignment_name)

        with open(alignment_path, 'rb') as f:
            alignment_params = pickle.load(f)
            print(f"Alignment parameters loaded from {alignment_name}")

        # find the session index in the alignment parameters
        session_index = next((i for i, name in enumerate(alignment_params['session_names']) if name == [self.mouse_name, self.record_date, self.experiment_name]), None)
        self.session_index = session_index
        field_coordinates = alignment_params['field_coordinates_widefield'][self.session_index]

        return alignment_params, field_coordinates

    def load_session_df(self):

        df_name = f'{self.record_date}_{self.mouse_name}_{self.experiment_name}_dataframe.pkl'
        df_path = os.path.join(self.session_data_dir, df_name)

        with open(df_path, 'rb') as f:
            session_df = pickle.load(f)
            print(f"Session dataframe loaded from {df_name}")
        
        return session_df

    def load_suite2p_stat(self):
        
        suite2p_stat_path = os.path.join(self.session_data_dir, 'suite2p', 'plane0', 'stat.npy')
        suite2p_stat = np.load(suite2p_stat_path, allow_pickle=True)

        return suite2p_stat

    def get_original_coords(self, cell_id):

        xpix = self.suite2p_stat[cell_id]['xpix'][~self.suite2p_stat[cell_id]['overlap']].mean()
        ypix = self.suite2p_stat[cell_id]['ypix'][~self.suite2p_stat[cell_id]['overlap']].mean()

        return xpix, ypix

    def transform_coords_to_widefield(self, xpix, ypix):

        scale = self.field_coordinates['scale']
        rotation = self.field_coordinates['rotation']
        dx = self.field_coordinates['dx']
        dy = self.field_coordinates['dy']

        h_orig, w_orig = 760, 760  # Original 2P image dimensions
        h_wf, w_wf = 400, 400  # Widefield image dimensions

        # Step 2: Resize coordinates from 760x760 to 400x400 space
        x_resized = xpix * (w_wf / w_orig)
        y_resized = ypix * (h_wf / h_orig)

        # Step 3: Apply the SAME transformation matrix that cv2 uses
        center = (w_wf / 2, h_wf / 2)
        
        # Create the rotation+scale matrix (same as cv2.getRotationMatrix2D)
        M_rot = cv2.getRotationMatrix2D(center, rotation, scale)
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy
        
        # Apply the transformation matrix to the point
        # M * [x, y, 1]^T = [x', y']^T
        point = np.array([x_resized, y_resized, 1])
        transformed_point = M_rot @ point
        
        xwf, ywf = transformed_point[0], transformed_point[1]

        return xwf, ywf

    def plot_cell_in_widefield(self, cell_id):

        """
        Plot a single cell in both its original session view and transformed to widefield.
        
        Parameters
        ----------
        cell_id : int
            Cell index to plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # LEFT PLOT: Original session
        # Load ops to get mean image
        suite2p_ops_path = os.path.join(self.session_data_dir, 'suite2p', 'plane0', 'ops.npy')
        ops = np.load(suite2p_ops_path, allow_pickle=True).item()
        mean_img = ops['meanImg']
        
        ax1.imshow(mean_img, cmap='gray', extent=[0, 760, 760, 0], origin='upper')
        
        # Get raw pixel coordinates (no transformation needed)
        xpix_raw, ypix_raw = self.get_original_coords(cell_id)
        
        ax1.scatter(xpix_raw, ypix_raw, c='red', s=100, 
                   alpha=0.7, edgecolors='white', linewidths=1.5,
                   label=f'Cell {cell_id}')
        
        ax1.set_title('Original Session View')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.legend(loc='upper right', fontsize=8)
        
        # RIGHT PLOT: Widefield composite
        widefield = self.alignment_params['widefield_maps']['img'].copy()
        widefield_norm = (widefield - widefield.min()) / (widefield.max() - widefield.min())
        composite = np.stack([widefield_norm, widefield_norm, widefield_norm], axis=2)
        
        # Transform and overlay session
        mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min())
        mean_img_resized = cv2.resize(mean_img_norm, (widefield.shape[1], widefield.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        
        h, w = mean_img_resized.shape[:2]
        center = (w / 2, h / 2)
        
        # Get alignment parameters from self.alignment_params
        M_rot = cv2.getRotationMatrix2D(center, self.field_coordinates['rotation'], 
                                         self.field_coordinates['scale'])
        M_rot[0, 2] += self.field_coordinates['dx']
        M_rot[1, 2] += self.field_coordinates['dy']
        
        transformed = cv2.warpAffine(mean_img_resized, M_rot, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Blend session in green
        alpha = 0.5
        session_green = np.zeros_like(composite)
        session_green[:, :, 1] = transformed
        mask = (transformed > 0).astype(float)
        composite = composite * (1 - alpha * mask[:, :, None]) + session_green * alpha
        
        ax2.imshow(composite, extent=[0, 400, 400, 0], origin='upper')
        
        # Transform and plot the cell
        xwf, ywf = self.transform_coords_to_widefield(xpix_raw, ypix_raw)
        
        ax2.scatter(xwf, ywf, c='red', s=100, 
                   alpha=0.7, edgecolors='white', linewidths=1.5,
                   label=f'Cell {cell_id}')
        
        ax2.set_title('Widefield View with Transformed Cell')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)

    def append_widefield_to_session_df(self):

        """
        Append widefield coordinates and corresponding azimuth/altitude values to session_df.
        
        For each cell in session_df, transforms its coordinates to widefield space and 
        retrieves the azimuth and altitude values at those coordinates.
        """

        if not hasattr(self, 'session_df') or self.session_df is None:
            print("No session_df found")
            return
        
        if not hasattr(self, 'alignment_params') or self.alignment_params is None:
            print("No alignment_params found. Run alignment first.")
            return
        
        # Initialize new columns
        widefield_coords = []
        azimuth_values = []
        altitude_values = []
        
        # Get widefield maps
        azi_map = self.alignment_params['widefield_maps']['azi']
        alt_map = self.alignment_params['widefield_maps']['alt']
        
        # Process each cell
        for idx, row in self.session_df.iterrows():

            cell_id = row['cell_id']
            xpix, ypix = self.get_original_coords(cell_id)
            
            # Transform to widefield coordinates
            xwf, ywf = self.transform_coords_to_widefield(xpix, ypix)
        
    
            
            azi_val = azi_map[int(ywf), int(xwf)]
            alt_val = alt_map[int(ywf), int(xwf)]
            
            widefield_coords.append([xwf, ywf])
            azimuth_values.append(azi_val)
            altitude_values.append(alt_val)
        
        # Append to dataframe
        self.session_df['wf_coords'] = widefield_coords
        self.session_df['azi'] = azimuth_values
        self.session_df['alt'] = altitude_values
        
        print(f"Added widefield coordinates and retinotopy values for {len(self.session_df)} cells")

        return self.session_df

    
