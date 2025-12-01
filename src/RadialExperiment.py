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
            print(f"✓ Loading session: {self.session_name}")
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
        rf_position = self.angle_range[rf_idx]

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

        return corr, corr_shifted

    def analyze_cell_reliability(self, cell, n_iterations=100, reliable_threshold = 0.85, distance=None, monocular=None):
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
        visually_responsive = np.mean(corr) > np.percentile(corr_shifted, 99.5)

        corr_control, corr_shifted_control = self.compute_reliability(cell, distance=4, monocular=monocular, n_iterations=n_iterations)
        control_responsive = np.mean(corr_control) > np.percentile(corr_shifted_control, 99)

        rf_values_distances = [np.mean(self.find_rf_values(cell, distance=distance, monocular=monocular)) for distance in [0, 1, 2, 3]]
        rf_values_control = np.mean(self.find_rf_values(cell, distance=4, monocular=monocular))

        # make sure that rf_values_distances are higher than rf_values_control, at least for three distances
        control_comparison = [rf_values_control > rf_values for rf_values in rf_values_distances]
        control_higher = np.sum(control_comparison) > 1

        is_reliable = visually_responsive and not (control_responsive or control_higher) and np.mean(corr) > reliable_threshold

        return is_reliable, visually_responsive, control_responsive, control_higher, np.mean(corr)
    
    def test_linear_trend_rf(self, cell, distances=[0, 1, 2, 3], monocular=None):

        rf_values = [self.find_rf_values(cell, distance=distance, monocular=monocular) for distance in distances]

        # Combine into response vector y and predictor x
        y = np.concatenate(rf_values)
        x = np.repeat(distances, repeats=rf_values[0].size)

        # Simple linear regression via SciPy
        result = linregress(x, y)

        return result.pvalue
    
    def analyze_cell_modulation(self, cell, monocular=None, method='ANOVA'):

        rf_values = [self.find_rf_values(cell, distance=distance, monocular=monocular) for distance in [0, 1, 2, 3, 4]]
        distances_rf = [rf_values[i] for i in range(len(rf_values)-1)]

        if method == 'ANOVA':
            # calculate the f_statistic and p_value
            f_stat, p_value = f_oneway(*distances_rf)
        elif method == 'regression':
            p_value = self.test_linear_trend_rf(cell, distances=[0, 1, 2, 3], monocular=monocular)
        
        # Split trials into two halves for cross-validation
        half_rf_values = []
        for values in distances_rf:
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

        return p_value, preferred_distance, modulation_strength

    def analyze_ocular_dominance(self, cell):

        monocular_mean = self.get_all_traces(cell, distance=None, monocular=True, mean_trace=True)[0]
        binocular_mean = self.get_all_traces(cell, distance=None, monocular=False, mean_trace=True)[0]

        receptive_field_position = self.find_rf_position(cell, distance=None, monocular=None)
        receptive_field_position_index = np.where(self.angle_range == receptive_field_position)[0][0]
    
        monocular_mean_rf = monocular_mean[int(receptive_field_position_index)]
        binocular_mean_rf = binocular_mean[int(receptive_field_position_index)]

        ocular_dominance_index = (binocular_mean_rf - monocular_mean_rf) / (binocular_mean_rf + monocular_mean_rf + 1e-10)

        return ocular_dominance_index
    
    def analyze_condition_stats(self, monocular=None, method='ANOVA'):
        
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

            print(f'Analyzing cell {cell} of {self.suite2p_id[-1]} for {condition_name} condition', end='\r', flush=True)
            
            # Analyze condition
            is_reliable = self.analyze_cell_reliability(cell, monocular=monocular)[0]
            p_value, pref_distance, mod_strength = self.analyze_cell_modulation(cell, monocular=monocular, method=method)
            rf_position = self.find_rf_position(cell, distance=None, monocular=monocular)
            
            # Create a row for this cell
            cell_row = {
                'cell_id': cell,
                'is_reliable': is_reliable,
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
    
    def analyze_session_stats(self, n_iterations=100, method='ANOVA'):

        # Initialize empty lists to collect data for each cell
        cell_data = []
        
        # Analyze each cell
        for cell in self.suite2p_id:

            print(f'Analyzing cell {cell} of {self.suite2p_id[-1]}', end='\r', flush=True)

            rf = self.find_rf_position(cell, distance=None, monocular=None)
            pref_distance = self.analyze_cell_modulation(cell, distance=None, monocular=None, method=method)[1]
            
            # Analyze condition
            is_responsive, visually_responsive, control_responsive, control_higher, mean_corr = self.analyze_cell_reliability(cell, monocular=None, n_iterations=n_iterations)

            p_value_monocular, pref_distance_monocular, mod_strength_monocular = self.analyze_cell_modulation(cell, monocular=True, method=method)
            rf_position_monocular = self.find_rf_position(cell, distance=None, monocular=True) 
            
            p_value_binocular, pref_distance_binocular, mod_strength_binocular = self.analyze_cell_modulation(cell, monocular=False, method=method)
            rf_position_binocular = self.find_rf_position(cell, distance=None, monocular=False)

            ocular_dominance_index = self.analyze_ocular_dominance(cell)
            
            # Create a row for this cell
            cell_row = {
                'cell_id': cell,
                'rf': rf,
                'distance': pref_distance,
                'responsive': is_responsive,
                'responsive_corr': mean_corr,
                'responsive_visually': visually_responsive,
                'responsive_control': control_responsive,
                'responsive_control_higher': control_higher,
                'responsive_mean_corr': mean_corr,
                'ocular_dominance_index': ocular_dominance_index,
                'monocular_modulated': np.bool(p_value_monocular < 0.05),
                'monocular_modulated_p': p_value_monocular,
                'monocular_strength': mod_strength_monocular,
                'monocular_distance': pref_distance_monocular,
                'monocular_rf': rf_position_monocular,
                'binocular_modulated': np.bool(p_value_binocular < 0.05),
                'binocular_modulated_p': p_value_binocular,
                'binocular_strength': mod_strength_binocular,
                'binocular_distance': pref_distance_binocular,
                'binocular_rf': rf_position_binocular,
            }
            
            cell_data.append(cell_row)
        
        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)
        
        # Calculate summary statistics
        n_cells = len(stats_df)
        n_responsive = stats_df['responsive'].sum()
        # n_monocular_modulated = stats_df['monocular_modulated'].sum()
        # n_binocular_modulated = stats_df['binocular_modulated'].sum()

        # Print summary statistics
        print(f"Total cells: {n_cells}")
        print(f"Responsive cells: {n_responsive} ({n_responsive / n_cells * 100:.1f}%)")
        # print(f"Monocular modulated cells: {n_monocular_modulated} ({n_monocular_modulated / n_responsive * 100:.1f}%)")
        # print(f"Binocular modulated cells: {n_binocular_modulated} ({n_binocular_modulated / n_responsive * 100:.1f}%)")
        
        return stats_df

class RadialPlots:

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
    
    def plot_all_traces_colormap_distances(self, cell, method='ANOVA'):
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
            {'monocular': True,  'title': 'Monocular'},
            {'monocular': False, 'title': 'Binocular'},
        ]


       
        all_traces = self.R.get_all_traces(cell, distance=None, monocular=None, mean_trace=False)
        vmin, vmax = np.percentile(all_traces, [1, 99])

        # Process each condition
        for c_idx, condition in enumerate(conditions):
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
            p_value, pref_distance, mod_strength = self.R.analyze_cell_modulation(cell, monocular=condition['monocular'], method=method)
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

            # # Check if p_value is significant and if the cell is responsive for the monocular condition
            responsive, responsive_visually, responsive_control, responsive_control_higher, responsive_corr = self.R.analyze_cell_reliability(cell, monocular=condition['monocular'])
            if p_value < 0.05 and responsive_visually:
                bar_height = vmax * 0.8
                axs[c_idx, 3].plot([1, len(distances_with_control)], [bar_height, bar_height], 'k-', linewidth=1.5)
                axs[c_idx, 3].text((1 + len(distances_with_control))/2, bar_height*1.02, '*', ha='center', va='bottom')

            axs[c_idx, 3].set_title(f'p={p_value:.3f}', fontsize=14)
            axs[c_idx, 3].set_xlabel('Distance')
            axs[c_idx, 3].set_ylabel(self.R.response_mode)
            axs[c_idx, 3].set_ylim(vmin, vmax)

            # --- Stats text column ---
            stats_lines = []
            stats_lines.append(f"Reliable: {responsive}")
            stats_lines.append(f"Visually reliable: {responsive_visually}")
            stats_lines.append(f"Control reliable: {responsive_control}")
            stats_lines.append(f"Control higher: {responsive_control_higher}")
            stats_lines.append(f"Modulated: {np.bool((p_value < 0.05) and responsive_visually)}")
            stats_lines.append(f"RF position: {rf_position[0]:.2f}")
            stats_lines.append(f"Modulation strength: {mod_strength:.3f}")
            stats_lines.append(f"Preferred distance: {pref_distance}")
            stats_lines.append(f"Mean RF values:")
            for i, vals in enumerate(rf_values):
                stats_lines.append(f"  D={distances_with_control[i]}: {np.mean(vals):.2f} ± {np.std(vals):.2f} (n={len(vals)})")

            stats_text = "\n".join(stats_lines)
            axs[c_idx, 4].axis('off')
            axs[c_idx, 4].text(0, 1, stats_text, va='top', ha='left', fontsize=11, family='monospace')

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

    def __init__(self, radial_class, radial_stats, radial_plots):
        self.R = radial_class
        self.radial_stats = radial_stats
        self.session_name = self.R.data_dir.split('/')[-2]
        self.colors = setup_plot_style()
        self.radial_plots = radial_plots

        self.responsive_cells = self.find_responsive_cells()
        self.modulated_cells = self.find_modulated_cells()

    def find_responsive_cells(self, correlation_threshold=0.8):

        responsive_cells = self.radial_stats[
            (self.radial_stats['responsive_corr'] >= correlation_threshold) &
            (self.radial_stats['responsive_visually'] == True) &
            (self.radial_stats['responsive_control'] == False) &
            (self.radial_stats['responsive_control_higher'] == False)
        ]['cell_id'].values

        return responsive_cells

    def find_modulated_cells(self):

        def is_monocular_modulated(cell):
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell, 'monocular_modulated'].values[0]

        def is_binocular_modulated(cell):
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell, 'binocular_modulated'].values[0]

        modulated_cells = {'monocular': [], 'binocular': [], 'both': [], 'none': []}

        for cell in self.responsive_cells:
            if is_monocular_modulated(cell) and is_binocular_modulated(cell):
                modulated_cells['both'].append(cell)
            elif is_monocular_modulated(cell):
                modulated_cells['monocular'].append(cell)
            elif is_binocular_modulated(cell):
                modulated_cells['binocular'].append(cell)
            else:
                modulated_cells['none'].append(cell)
    
        n_cells = len(self.R.suite2p_id)
        n_responsive = len(self.responsive_cells)
        n_monocular_modulated = len(modulated_cells.get('monocular', []))
        n_binocular_modulated = len(modulated_cells.get('binocular', []))
        n_both_modulated = len(modulated_cells.get('both', []))
        n_non_modulated = len(modulated_cells.get('none', []))

        print(f"Total cells: {n_cells}")
        print(f"Responsive cells: {n_responsive} ({n_responsive / n_cells * 100:.1f}%)")
        print(f"Monocular modulated cells: {n_monocular_modulated} ({n_monocular_modulated / n_responsive * 100:.1f}%)")
        print(f"Binocular modulated cells: {n_binocular_modulated} ({n_binocular_modulated / n_responsive * 100:.1f}%)")
        print(f"Both modulated cells: {n_both_modulated} ({n_both_modulated / n_responsive * 100:.1f}%)")
        print(f"Non-modulated cells: {n_non_modulated} ({n_non_modulated / n_responsive * 100:.1f}%)")
        
        return modulated_cells
    
    def plot_single_cells(self):

        results_dir = os.path.join(self.R.data_dir, 'results')

        for condition in self.modulated_cells.keys():
            print(f'Plotting {condition} cells')
            figure_directory = create_new_result_dir(results_dir, f'{condition}')
            for cell in self.modulated_cells[condition]:
                print(f'Plotting cell {cell}', end='\r', flush=True)
                fig = self.radial_plots.plot_all_traces_colormap_distances(cell=cell, method='ANOVA')
                fig.savefig(os.path.join(figure_directory, f'{cell}.png'))
                plt.close(fig)

    def pullout_stat(self, cell_id, condition, stat_type):

        if stat_type == 'pref_angle':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, condition + '_rf'].values[0]
        elif stat_type == 'pref_distance':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, condition + '_distance'].values[0]
        elif stat_type == 'mod_strength':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, condition + '_strength'].values[0]
        elif stat_type == 'modulated':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, condition + '_modulated'].values[0]
        elif stat_type == 'modulated_p':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, condition + '_modulated_p'].values[0]
        elif stat_type == 'responsive':
            return self.radial_stats.loc[self.radial_stats['cell_id'] == cell_id, 'responsive'].values[0]
        else:
            raise ValueError(f"Stat type {stat_type} not found")
        
    def plot_responsive_bars(self):

        # Create and configure bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ['coral', 'lightgreen', 'blue']

        responsive_cells_set = {}
        for i, stat_df in enumerate(self.radial_stats):
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
        
        bar_colors = ['coral', 'lightgreen', 'blue']

        labels = list(self.modulated_cells.keys()) 
        responsive_cells = self.find_responsive_cells()
        modulated_cells = self.find_modulated_cells()
        responsive_counts = [len(responsive_cells)]*len(labels)
        modulated_counts = [len(modulated_cells[label]) for label in labels]
        percentages = [mod/resp*100 for mod, resp in zip(modulated_counts, responsive_counts)]

        # Create stacked bar chart
        bars = ax.bar(labels[:-1], responsive_counts[:-1], color='lightgray', label='Responsive', width=0.75)
        modulated_bars = ax.bar(labels[:-1], modulated_counts[:-1], color=bar_colors, label='Modulated', width=0.75)
        
        # Add value labels on top of bars
        for bar, percentage, count, resp_count in zip(bars, percentages, modulated_counts, responsive_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%\n({count}/{resp_count})',
                   ha='center', va='bottom', fontsize=12,
                   )
            
        # remove y axis
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Customize the plot
        ax.set_ylim(0, max(responsive_counts) * 1.2)  # Add 20% space for labels

        # add title with less padding so they appear closer
        fig.suptitle('Modulated Cells', fontsize=18, color='black', y=0.94)
        ax.set_title(self.session_name, fontsize=14, color='grey', pad=2)
        
        plt.tight_layout()

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

        return fig

    def plot_preferred_distance_bars(self, condition=None):

        fig, ax = plt.subplots(figsize=(4, 5))

        modulated_cells = self.modulated_cells[condition]

        if condition is None:
            all_pref_distances = self.radial_stats[self.radial_stats['cell_id'].isin(modulated_cells)]['distance'].values
        else:
            all_pref_distances = self.radial_stats[self.radial_stats['cell_id'].isin(modulated_cells)][condition + '_distance'].values

        distances = [0, 1, 2, 3]
        counts = [sum(all_pref_distances == d) for d in distances]

        # Calculate percentage of reliable cells
        total_reliable = len(self.responsive_cells)
        percentages = [(count / total_reliable) * 100 for count in counts]

        ax.bar(distances, percentages, width=0.8, color=self.colors, edgecolor='black')

        # Add raw counts on top of bars
        for i, (distance, percentage, count) in enumerate(zip(distances, percentages, counts)):
            ax.text(distance, percentage, str(count), ha='center', va='bottom', fontsize=12)

        ax.set_ylabel('Percentage of Reliable Cells (%)')
        ax.set_xlabel('Preferred Distance')
        ax.set_xticks(distances)
        ax.set_xticklabels(distances)
        ax.set_ylim(0, max(percentages) * 1.2)
        plt.tight_layout()

        return fig
    
    def plot_preferred_distance_polar(self, condition='binocular'):


        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1, polar=True)
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
        sc = ax.scatter(theta, pref_distances, c=pref_angles, cmap='viridis', vmin=-30, vmax=60, s=60, edgecolor='k', alpha=0.8)

        ax.grid(False)
        ax.set_theta_direction(-1)
        ax.set_thetamin(-30)
        ax.set_thetamax(65)
        ax.set_ylabel('Preferred Distance')
        ax.set_yticks([])
        # title will be condition
        if condition is None:
            ax.set_title('Responsive Cells', fontsize=18, color='black', pad=2)
        else:
            ax.set_title(condition.capitalize(), fontsize=18, color='black', pad=2)

        ax.legend()

        plt.tight_layout()
    
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

    def plot_reliable_cells_map(self):

        cells_to_plot = self.R.suite2p_id

        # Load Suite2p outputs
        stat = np.load(os.path.join(self.R.suite2p_dir, 'plane0/stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(self.R.suite2p_dir, 'plane0/ops.npy'), allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        # Scatter plot each pixel
        for cell_id in cells_to_plot:
            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()

            if cell_id not in self.responsive_cells:
                sc = ax.scatter(
                    xpix, ypix,
                    c='lightgray',
                    s=120, alpha=0.8,
                    edgecolor='black', linewidth=1
                )
            else:
                rf_location = self.R.find_rf_position(cell_id, distance=None, monocular=None)
                sc = ax.scatter(
                    xpix, ypix,
                    c=rf_location,
                    cmap='viridis',
                    vmin=-30,
                    vmax=60,
                    s=120, alpha=1,
                    edgecolor='black', linewidth=1
                )

        # Remove all axes, spines, ticks, and colorbar for a pure image
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        return fig
    
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

        cells_to_plot = self.modulated_cells[condition]

        if condition == 'binocular':
            idx_condition = 1
        elif condition == 'monocular':
            idx_condition = 0

        # Load Suite2p outputs
        stat = np.load(os.path.join(self.R.suite2p_dir, 'plane0/stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(self.R.suite2p_dir, 'plane0/ops.npy'), allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        # Scatter plot each pixel
        
        for cell_id in self.responsive_cells:

            if cell_id in cells_to_plot:
                continue

            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()

            ax.scatter(
                xpix, ypix,
                c='lightgray',
                s=150, alpha=0.5,
                edgecolor='black', linewidth=1
            )

        for cell_id in cells_to_plot:

            ypix = stat[cell_id]['ypix'][~stat[cell_id]['overlap']].mean()
            xpix = stat[cell_id]['xpix'][~stat[cell_id]['overlap']].mean()

            rf_location = self.pullout_stat(cell_id=cell_id, condition=condition, stat_type='pref_angle')

            ax.scatter(
                xpix, ypix,
                c=rf_location,
                cmap='viridis',
                vmin=-30,
                vmax=60,
                s=150, alpha=1,
                edgecolor='black', linewidth=1
            )


        ax.axis('off')
        plt.tight_layout(pad=0)

        return fig
    
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

class RadialMultiSession:

    def __init__(self, animal_id, session_names):

        self.animal_id = animal_id
        self.animal_dir = self.get_animal_dir()
        self.fov_pixels = 760
        self.fov_units = 2000

        self.session_names = session_names
        self.numeral_names = session_names.keys()
        self.mean_images = self.load_mean_images()
        self.field_coordinates = self.load_coordinates()

        self.widefield_fov_pixels = 400
        self.widefield_fov_units = 5200
        self.widefield_img = None
        self.widefield_img_shifted = None
        self.VFS = None
        self.VFS_shifted = None
        self.load_widefield_maps()
        self.shift_widefield()
        self.area_names = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']

        self.radial_sessions = self.load_radial_sessions()
        self.areas_verts = self.load_areas_verts()

    def get_animal_dir(self):
        return r'D:/data-distance-coding-project/processed-data/{animal_id}'.format(animal_id=self.animal_id + '_radial')

    def get_suite2p_dir(self, session_name):
        return r'D:/data-distance-coding-project/processed-data/{session_name}/suite2p'.format(session_name=session_name)

    def load_mean_images(self):

        mean_images = {}
        for numeral in self.numeral_names:
            print(self.session_names[numeral])
            ops = np.load(self.get_suite2p_dir(self.session_names[numeral]) + r'\plane0\ops.npy', allow_pickle=True).item()
            mean_images[numeral] = ops['meanImg']
        return mean_images

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
        for name in self.numeral_names:

            img = self.mean_images[name]
            dx_units, dy_units = self.field_coordinates[name]
            img_h, img_w = img.shape
            img_w_units = self.pixels_to_units(img_w, scope='2p')
            img_h_units = self.pixels_to_units(img_h, scope='2p')
            # Calculate extent in units for this image, centered at (dx_units, dy_units)
            extent_units = [
                dx_units - img_w_units/2, dx_units + img_w_units/2,
                dy_units + img_h_units/2, dy_units - img_h_units/2  # origin='upper'
            ]
            ax.imshow(img, cmap='gray', alpha=1, extent=extent_units, origin='upper', label=name)

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
        snap_mat_path = os.path.join(self.animal_dir, 'snap.mat')
        snap_data = loadmat(snap_mat_path)
        widefield_img = snap_data['snap']

        # resize widefield_img to the size of 400x400
        widefield_img = cv2.resize(widefield_img, (400, 400))

        self.widefield_img = widefield_img

        maps_mat_path = os.path.join(self.animal_dir, 'maps.mat')
        maps_data = loadmat(maps_mat_path, struct_as_record=False, squeeze_me=True)
        self.VFS = maps_data['maps'].VFS_raw
        self.VFS = gaussian_filter(self.VFS, sigma=1.5)

        # Allow user to pick a point in the widefield image to set as the origin, then shift the image accordingly

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

        dx, dy = self.set_widefield_origin()

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

        # Plot VFS_shifted on top of the window (output of plot_window)
        fig, window_ax = self.plot_window()

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

