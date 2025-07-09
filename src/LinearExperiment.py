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

class LinearAnalysis:

    def __init__(self, session_path):

        D = ActivityLoader(session_path)
        self.stimulus_path = os.path.join(session_path, 'stimulus')
        self.activity_path = os.path.join(session_path, 'suite2p\plane0')
        self.dFF = D.dFF

        self.frames_time = np.load(self.activity_path + r'/frames_time.npy')

        container = self.load_stimulus()
        self.stepper_steps = container['stim']['stepper_pos'][0][0][0]
        self.stepper_time = container['stim']['time'][0][0][0]
        self.stim_showed = container['stim']['distance_vec'][0][0][0]

        self.distance_traveled = np.interp(self.frames_time, self.stepper_time, self.stepper_steps) / 30
        self.repeat_start_time = np.arange(0, 20 * 25, 25) * 1000
        self.repeat_end_time = self.repeat_start_time + 25000

    def load_stimulus(self):

        filelist = [file for file in os.listdir(self.stimulus_path) if file.endswith(("_stimulus.mat"))]
        if len(filelist) == 0:
            # raise error
            raise Exception('No stimulus file found')
        elif len(filelist) > 1:
            raise Exception('More than one stimulus file found')
        else:
            container = io.loadmat(self.stimulus_path + '/' + str(filelist[0]))

        return container

    def get_repeat_frames(self, repeat):

        repeat_start = self.repeat_start_time[repeat]
        repeat_end = self.repeat_end_time[repeat]

        repeat_frames = (self.frames_time > repeat_start) & (self.frames_time < repeat_end)

        return repeat_frames

    def get_repeat_size(self, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        distance_repeat = self.distance_traveled[repeat_frames]
        stimulus_radius = self.stim_showed[repeat]

        if stimulus_radius == 1:
            theta_repeat = np.degrees(np.arctan2(stimulus_radius, (distance_repeat+2)))
        else:
            theta_repeat = np.degrees(np.arctan2(stimulus_radius, (distance_repeat+10)))

        return theta_repeat

    def get_distancing_frames_within_repeat(self, repeat):

        repeat_size = self.get_repeat_size(repeat=repeat)

        # compute the derivative of repeat_size
        derivative = np.diff(repeat_size)
        derivative = np.insert(derivative, 0, 0)

        return derivative < 0

    def get_distancing_size(self, repeat):

        distancing_frames = self.get_distancing_frames_within_repeat(repeat)

        return self.get_repeat_size(repeat)[distancing_frames]

    def get_distancing_trace(self, cell, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        distancing_frames = self.get_distancing_frames_within_repeat(repeat)

        return self.dFF[cell, repeat_frames][distancing_frames]

    def get_all_distancing_traces(self, cell, stim_shown=None):

        all_traces = []

        for repeat in range(len(self.stim_showed)):

            if stim_shown is not None:
                if self.stim_showed[repeat] != stim_shown:
                    continue

            distance_trace = self.get_distancing_trace(cell, repeat)
            all_traces.append(distance_trace)

        # check all traces and make them the same size by cutting by the end
        min_size = min([len(trace) for trace in all_traces])
        all_traces = [trace[:min_size] for trace in all_traces]

        return np.array(all_traces)


    def get_stopped_frames_within_repeat(self, repeat):

        stim = self.stim_showed[repeat]

        if stim == 1:
            stopped_steps = 100
        elif stim == 2:
            stopped_steps = 20
        else:
            raise Exception('Unknown stimulus')

        repeat_frames = self.get_repeat_frames(repeat)
        repeat_steps = self.distance_traveled[repeat_frames] * 30

        stopped_frames = repeat_steps == stopped_steps

        return stopped_frames

    def get_stopped_trace(self, cell, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        stopped_frames = self.get_stopped_frames_within_repeat(repeat)

        return self.dFF[cell, repeat_frames][stopped_frames]

    def get_all_stopped_traces(self, cell, stim_shown=None):

        all_traces = []

        for repeat in range(len(self.stim_showed)):

            if stim_shown is not None:
                if self.stim_showed[repeat] != stim_shown:
                    continue

            stopped_trace = self.get_stopped_trace(cell, repeat)
            all_traces.append(stopped_trace)

        # check all traces and make them the same size by cutting by the end
        min_size = min([len(trace) for trace in all_traces])
        all_traces = [trace[:min_size] for trace in all_traces]

        return np.array(all_traces)

class LinearDevice:

    def __init__(self, session_path, response_mode='dFF', tau=None):

        self.session_path = session_path
        self.results_path = create_new_result_dir(self.session_path, r'results')
        self.stimulus_path = os.path.join(self.session_path, r'stimulus')
        self.activity_path = os.path.join(self.session_path, r'suite2p\plane0')

        self._load_activity(tau)
        self._load_stimulus_variables()
        self._synchronize_stimulus_and_activity()

        self.response_mode = response_mode

        if self.response_mode == 'dFF':
            self.response = self.dFF
            self.response = gaussian_filter1d(self.response, 1.5, axis=1)
        elif self.response_mode == 'dFF-zscore':
            # do gaussian smoothing
            self.response = gaussian_filter1d(self.dFF, 2, axis=1)
            # normalize by baseline response
            baseline_mean, baseline_std = self.get_baseline_response()
            # Calculate z-score using baseline mean and std
            self.response = (self.response - baseline_mean[:, np.newaxis]) / (baseline_std[:, np.newaxis] + 1e-10)  # Add small constant to avoid division by zero
        elif self.response_mode == 'spikes':
            self.response = self.spikes
        elif self.response_mode == 'spikes-zscore':
            # do gaussian smoothing
            self.response = gaussian_filter1d(self.spikes, 2, axis=1)
            # normalize by baseline response
            baseline_mean, baseline_std = self.get_baseline_response()
            # Calculate z-score using baseline mean and std
            self.response = (self.response - baseline_mean[:, np.newaxis]) / (baseline_std[:, np.newaxis] + 1e-10)  # Add small constant to avoid division by zero
        else:
            raise Exception('Unknown activity mode')
        
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

        self.time = locals_dict['time']
        self.stepper_pos = locals_dict['stepper_pos']
        self.servo_id = locals_dict['servo_id']
        self.mini_servo_active = locals_dict['mini_servo_active']
        self.trial_sequence = locals_dict['trial_sequence']
        self.history_sequence = locals_dict['history_sequence']
        self.trial_start_time = locals_dict['trial_start_time']
        self.trial_end_time = locals_dict['trial_end_time']

        try:
            self.monocular_sequence = locals_dict['monocular_sequence']
        except:
            self.monocular_sequence = np.zeros_like(self.trial_sequence)

        if len(self.monocular_sequence) == 0:
            self.monocular_sequence = np.ones_like(self.trial_sequence)

        try:
            self.servo_movement_time = locals_dict['SERVO_MOVEMENT_TIME'][0]
        except:
            print('No servo movement time found, using 100ms')
            self.servo_movement_time = 100

        # Try to get miniservo_to_active_time
        try:
            self.miniservo_to_active_time = locals_dict['MOVEMENT_TO_ACTIVE_TIME'][0]
        except:
            print('No miniservo to active time found, using 2500ms')
            self.miniservo_to_active_time = 2500
            
        # Try to get miniservo_hold
        try:
            self.miniservo_hold = locals_dict['HOLD_TIME'][0]
        except:
            print('No miniservo hold time found, using 100ms')
            self.miniservo_hold = 100
            
        # Try to get miniservo_to_inactive_time
        try:
            self.miniservo_to_inactive_time = locals_dict['MOVEMENT_TO_REST_TIME'][0]
        except:
            print('No miniservo to inactive time found, using 2500ms')
            self.miniservo_to_inactive_time = 2500
            
        # Try to get miniservo_pre_time
        try:
            self.miniservo_pre_time = locals_dict['PRE_ACTIVATION_WAIT_TIME'][0]
        except:
            print('No miniservo pre time found, using 1000ms')
            self.miniservo_pre_time = 1000

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

        self.spikes = self.spikes[:,self.frames_time <= 1000*self.trial_end_time[-1]]
        # self.dFF = self.dFF[:,self.frames_time <= 1000*self.trial_end_time[-1]]
        self.time = np.linspace(0, self.trial_end_time[-1]*1000, len(self.stepper_pos))
        self.frames_time = self.frames_time[self.frames_time <= 1000*self.trial_end_time[-1]]
        self.stepper_pos_interp = np.interp(self.frames_time, self.time, self.stepper_pos)
        self.on_time_interp = np.round(np.interp(self.frames_time, self.time, self.mini_servo_active)).astype(bool)
        self.identity_interp = np.round(np.interp(self.frames_time, self.time, self.servo_id)).astype(int)

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

        trial_start = 1000*self.trial_start_time[repeat]
        trial_end = 1000*self.trial_end_time[repeat]

        # addendum to the repeat where it goes back to on time
        on_frames = (self.frames_time > trial_start) & (self.frames_time < trial_end) & self.on_time_interp

        # find the last index that is true
        on_start_idx = np.where(on_frames == True)[0][0]
        on_end_idx = np.where(on_frames == True)[0][-1]

        # find the time of the last true index
        on_start = self.frames_time[on_start_idx]

        if self.miniservo_to_inactive_time > 500:
            on_end = self.frames_time[on_end_idx] + self.miniservo_to_inactive_time + self.miniservo_hold
        else:
            on_end = self.frames_time[on_end_idx]

        return on_start, on_end
    
    def get_baseline_response(self, pre_time = 1000):

        baseline_mean = np.zeros(self.response.shape[0])
        baseline_std = np.zeros(self.response.shape[0])

        for idx in range(self.response.shape[0]):
            
            off_responses = []
            for repeat in range(len(self.trial_sequence)):
                on_start, _ = self.get_on_times(repeat)

                if self.miniservo_pre_time > pre_time:
                    off_start, off_end = on_start-pre_time, on_start
                else:
                    off_start, off_end = on_start-self.miniservo_pre_time, on_start

                off_frames = (self.frames_time > off_start) & (self.frames_time < off_end)

                off_responses.append(np.mean(self.response[idx, off_frames]))

            off_responses = np.array(off_responses)
            baseline_mean[idx] = np.mean(off_responses)
            baseline_std[idx] = np.std(off_responses)

        return baseline_mean, baseline_std


    def get_on_trace(self, idx, repeat):

        on_start, on_end = self.get_on_times(repeat)
        on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)

        if self.on_length is not None:
            return self.response[idx, on_frames][:self.on_length]
        else:
            return self.response[idx, on_frames]
        
    def get_condition_trials(self, monocular=None, history=None, distance=None):

        # find the indices of repeats that match the criteria
        if monocular is not None:
            monocular_indices = np.where(self.monocular_sequence == monocular)[0]
        else:
            monocular_indices = np.arange(len(self.trial_sequence))

        if history is not None:
            history_indices = np.where(self.history_sequence == history)[0]
        else:
            history_indices = np.arange(len(self.trial_sequence))

        if distance is not None:
            distance_indices = np.where(self.trial_sequence == distance)[0]
        else:
            distance_indices = np.arange(len(self.trial_sequence))

        # do intersection
        condition_indices = np.intersect1d(monocular_indices, history_indices)
        repeat_indices = np.intersect1d(condition_indices, distance_indices)

        # get the intersection of the indices
        return repeat_indices
    
    def get_all_traces(self, cell, distance=None, monocular=None, history=None, mean_trace=False):

        idx = np.where(self.suite2p_id == cell)[0][0]

        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance, history=history)

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        if mean_trace:
            return np.mean(all_traces, axis=0), np.std(all_traces, axis=0)/np.sqrt(all_traces.shape[0])
        else:
            return all_traces

    def analyze_cell_reliability(self, cell, n_iterations=100, distance=None, monocular=None, history=None):
        """
        Analyzes the reliability of a cell's response by comparing split-half correlations.
        
        Args:
            cell: Cell ID to analyze
            n_iterations: Number of iterations for reliability calculation
            distance: Optional distance condition to filter trials
            monocular: Optional monocular condition to filter trials
            history: Optional history condition to filter trials
            additional_reliability: Whether to perform additional reliability checks
            
        Returns:
            tuple: (is_reliable, reliability, corr_shifted, corr) where is_reliable indicates if the cell
                   is considered reliable based on correlation analysis
        """
        
        # Get traces for the specified conditions
        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance, history=history)
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
        is_reliable = reliability > np.percentile(corr_shifted, 95)

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
        
        repeat_indices = self.get_condition_trials(monocular=monocular, distance=distance, history=history)
        
        # Calculate medians for this condition
        on_max = []
        off_max = []

        idx = np.where(self.suite2p_id == cell)[0][0]

        # get the mean trace for this cell
        mean_trace = self.get_all_traces(cell, distance=distance, mean_trace=True, monocular=monocular)[0]
        max_pos = np.argmax(mean_trace)

        for repeat in repeat_indices:
            on_start, on_end = self.get_on_times(repeat)

            if self.miniservo_pre_time > 1000:
                off_start, off_end = on_start-1000, on_start
            else:
                off_start, off_end = on_start-self.miniservo_pre_time, on_start

            on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)   
            off_frames = (self.frames_time > off_start) & (self.frames_time < off_end)

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
    
    def analyze_cell_modulation(self, cell, monocular=None, history=None):

        if self.response_mode == 'spikes-zscore':
            window_size = 3
        else:
            window_size = 5

        distances = np.unique(self.trial_sequence)
        # get the mean trace for this cell
        mean_trace = self.get_all_traces(cell, mean_trace=True, monocular=monocular, history=history)[0]

        # find the receptive fied for this cell
        rf_idx = np.argmax(mean_trace)

        rf_values = []
        for distance in distances:
            on_traces = self.get_all_traces(cell,  
                                     distance=distance,
                                     monocular=monocular,
                                     history=history)
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

        return p_value, preferred_distance, modulation_strength, rf_values
    
    def analyze_condition_stats(self, monocular=None, history=None):
        
        # Define condition name based on monocular and history options
        if monocular is None:
            condition_name = ''
        elif monocular:
            condition_name = 'monocular'
        else:
            condition_name = 'binocular'
            
        # Add history information to condition name if specified
        if history is not None and monocular is not None:
            condition_name += '_history' if history else '_no_history'
        elif monocular is None and history is not None:
            condition_name += 'history' if history else 'no_history'

        if monocular is None and history is None:
            condition_name = 'all_conditions'
        
        # Initialize empty lists to collect data for each cell
        cell_data = []
        
        # Analyze each cell
        for cell in self.suite2p_id:
            
            # Analyze condition
            is_responsive, reliability, _, _ = self.analyze_cell_responsiveness(cell, monocular=monocular, history=history)
            p_value, pref_distance, mod_strength, _ = self.analyze_cell_modulation(cell, monocular=monocular, history=history)
            
            # Create a row for this cell
            cell_row = {
                'cell_id': cell,
                'responsive': np.bool(is_responsive),
                'reliable': reliability,
                'modulated': np.bool(p_value < 0.05 and is_responsive),
                'modulated_p': p_value,
                'mod_strength': mod_strength,
                'pref_distance': pref_distance,
                'condition': condition_name
            }
            
            cell_data.append(cell_row)
        
        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)
        
        # Calculate summary statistics
        n_cells = len(stats_df)
        n_responsive = stats_df['responsive'].sum()
        n_modulated = stats_df['modulated'].sum()

        # Print summary statistics
        print(f'Condition: {condition_name}')
        print(f"Total cells: {n_cells}")
        print(f"Responsive cells: {n_responsive} ({n_responsive / n_cells * 100:.1f}%)")
        print(f"Modulated cells: {n_modulated} ({n_modulated / n_responsive * 100:.1f}%)")
        
        return stats_df

    

class LinearPlots:

    def __init__(self, linear_class):
        self.L = linear_class
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
            idx = np.where(self.L.suite2p_id == cell)[0][0]
    
        # Handle both single number and range inputs for n_repeats
        if isinstance(repeats, (list, tuple)) and len(repeats) == 2:
            repeats = np.arange(repeats[0], repeats[1])
        else:
            repeats = np.arange(repeats)

        repeat_start = self.L.trial_start_time[repeats[0]]*1000
        repeat_end = self.L.trial_end_time[repeats[-1]]*1000

        repeats_frames = (self.L.frames_time > repeat_start) & (self.L.frames_time < repeat_end)

        ax = plt.gca()

        if cell is None:
            ax.plot(self.L.frames_time[repeats_frames]/1000, self.L.stepper_pos_interp[repeats_frames]/30, lw=2, color='black')
        else:
            ax.plot(self.L.frames_time[repeats_frames]/1000, self.L.response[idx, repeats_frames], lw=1.5, color='black')

        for rep in repeats:
            on_start, on_end = self.L.get_on_times(rep)
            if self.L.monocular_sequence[rep] == 1:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.L.trial_sequence[rep]], alpha=0.5)
            else:
                plt.axvspan(on_start/1000, on_end/1000, color=colors[self.L.trial_sequence[rep]], alpha=0.5, hatch='///')

            # path for off_time    
            plt.axvspan(on_start/1000-self.L.miniservo_pre_time/1000, on_start/1000, color='gray', alpha=0.4)

            # patch for servo movement time
            plt.axvspan(on_start/1000-self.L.miniservo_pre_time/1000-self.L.servo_movement_time/1000, on_start/1000-self.L.miniservo_pre_time/1000, color='red', alpha=0.4)

        #if cell is not None, set the ylim to the 99 and 1 percentiles of the response
        if cell is not None:
            ax.set_ylim(np.min(self.L.response[idx, :]), np.max(self.L.response[idx, :]))

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
            plt.ylabel('Stepper Position (cm)')
            plt.title('Stimulus Response')
        else:
            plt.ylabel(self.L.response_mode)
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

        monocular_condition_name = 'monocular' if monocular else 'binocular'
        history_condition_name = 'history' if history else 'no_history'

        # function will be two columns, one row.
        # first column pcolormesh of all traces
        # second column plot of mean traces
        idx = np.where(self.L.suite2p_id == cell)[0][0]
        # Handle the case of a single distance (1D array of axes)
        if distance == None:
            fig, axs = plt.subplots(1, 3, figsize=(14, 4), 
                               gridspec_kw={'width_ratios': [2, 1, 1]})
            axs = np.array([axs])
        else:
            fig, axs = plt.subplots(len(distance), 3, figsize=(14, len(distance)*3.5), 
                               gridspec_kw={'width_ratios': [2, 1, 1]})
       
        vmin, vmax = np.percentile(self.L.response[idx], [1, 99])

        if distance is None:
            distance = [None]
        
        for i, dist in enumerate(distance):

            repeat_indices = self.L.get_condition_trials(monocular=monocular, history=history, distance=dist)

            repeat_traces = []
            for repeat in repeat_indices:
                on_start, on_end = self.L.get_on_times(repeat)
                frames = (self.L.frames_time > (on_start-pre_time)) & (self.L.frames_time < (on_end+post_time))
                trace_length = self.L.on_length + int((pre_time + post_time)/100)
                
                # Make sure we don't exceed the available frames
                trace = self.L.response[idx, frames]
                if len(trace) >= trace_length:
                    repeat_traces.append(trace[:trace_length].T)
            
            if not repeat_traces:
                continue
                
            repeat_traces = np.array(repeat_traces)
            repeat_traces_mean = np.mean(repeat_traces, axis=0)
            repeat_traces_sem = np.std(repeat_traces, axis=0)/np.sqrt(len(repeat_traces))
            
            # Plot heatmap
            if self.L.response_mode == 'spikes-zscore' or self.L.response_mode == 'spikes':
                cmap = 'viridis'
            elif self.L.response_mode == 'dFF' or self.L.response_mode == 'dFF-zscore':
                cmap = 'viridis'

            axs[i, 0].pcolormesh(repeat_traces, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[i, 0].axvline(x=pre_time/100-self.L.miniservo_pre_time/100-self.L.servo_movement_time/100, color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].axvline(x=(pre_time/100+self.L.on_length), color='lightgray', linestyle='--', linewidth=1.5)
            axs[i, 0].set_xlabel('Time (frames)')
            axs[i, 0].set_ylabel('Trials')
            axs[i, 0].set_title(f'All Traces')
            
            # Plot mean trace
            if dist is not None:
                plot_color = self.colors[dist]
            else:
                plot_color = 'black'

            axs[i, 1].axvline(x=pre_time/100-self.L.miniservo_pre_time/100-self.L.servo_movement_time/100, color='lightgray', linestyle='--', linewidth=2)
            axs[i, 1].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=2)
            axs[i, 1].axvline(x=(pre_time/100+self.L.on_length), color='lightgray', linestyle='--', linewidth=2)
            axs[i, 1].plot(repeat_traces_mean, color=plot_color, label=f'Distance {dist}', linewidth=3)
            axs[i, 1].set_xlabel('Time (frames)')
            axs[i, 1].set_ylabel(self.L.response_mode)
            axs[i, 1].set_title(f'Mean Traces ')
            axs[i, 1].fill_between(np.arange(len(repeat_traces_mean)), 
                                    repeat_traces_mean - repeat_traces_sem, 
                                    repeat_traces_mean + repeat_traces_sem, 
                                    color=plot_color, alpha=0.2)
            axs[i, 1].set_ylim(vmin, vmax)
            axs[i, 1].legend()

            # plot boxplot of on and off
            
            # Analyze cell responsiveness and plot boxplot as third column
            if distance[0] is None:
                is_responsive, p_val, on_max, off_max = self.L.analyze_cell_responsiveness(cell, distance=None, monocular=monocular, history=history)
            else:
                is_responsive, p_val, on_max, off_max = self.L.analyze_cell_responsiveness(cell, distance=dist, monocular=monocular, history=history)
            
            # Add scatter points to each column - on with plot_color, off with grey
            # ON data with plot_color
            x_on = np.random.normal(1, 0.05, size=len(on_max))
            axs[i, 2].scatter(x_on, on_max, alpha=0.3, s=30, color=plot_color, zorder=3)
            
            # OFF data with grey
            x_off = np.random.normal(2, 0.05, size=len(off_max))
            axs[i, 2].scatter(x_off, off_max, alpha=0.3, s=30, color='grey', zorder=3)

            axs[i, 2].axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

            # Plot boxplot of on and off
            axs[i, 2].boxplot([on_max, off_max], patch_artist=False, showfliers=False)
            
            axs[i, 2].set_xticks([1, 2])
            axs[i, 2].set_xticklabels(['ON', 'OFF'])
            axs[i, 2].set_ylabel(self.L.response_mode)
            axs[i, 2].set_ylim(vmin, vmax)
            
            # Add a horizontal line between the boxplots
            axs[i, 2].plot([1, 2], [vmax * 0.95, vmax * 0.95], color='black', linewidth=1.5)
            
            # Add significance indicators
            if not is_responsive:
                axs[i, 2].text(1.5, vmax * 0.95, 'n.s.', color='black', ha='center', va='bottom', fontsize=12)
            elif p_val < 0.001:
                axs[i, 2].text(1.5, vmax * 0.95, '***', color='black', ha='center', va='bottom', fontsize=14)
            elif p_val < 0.01:
                axs[i, 2].text(1.5, vmax * 0.95, '**', color='black', ha='center', va='bottom', fontsize=14)
            elif p_val < 0.05:
                axs[i, 2].text(1.5, vmax * 0.95, '*', color='black', ha='center', va='bottom', fontsize=14)
            else:
                axs[i, 2].text(1.5, vmax * 0.95, 'n.s.', color='black', ha='center', va='bottom', fontsize=12)

        # add suptitle to the figure
        fig.suptitle(f'Cell {cell} - {monocular_condition_name} - {history_condition_name}', fontsize=20)
        plt.tight_layout()

        return fig
    
    def plot_all_traces_colormap_distances(self, cell, pre_time=2500, post_time=1000, conditions=None):
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
        idx = np.where(self.L.suite2p_id == cell)[0][0]
        
        # Define distances to plot
        distance = [0, 1, 2, 3]
        
        if conditions is None:
            conditions = [
                {'monocular': None, 'history': None, 'title': 'All Conditions'},
            ]
        
        # Create figure with a row for each condition, 2 columns per row
        fig, axs = plt.subplots(len(conditions), 3, figsize=(15, 4*len(conditions)), 
                               gridspec_kw={'width_ratios': [2, 1, 1]})
        
        # Handle single condition case
        if len(conditions) == 1:
            axs = np.array([axs])
       
        vmin, vmax = np.percentile(self.L.response[idx], [0, 99])

        # Process each condition
        for c_idx, condition in enumerate(conditions):
            repeat_traces = []
            repeats_per_distance = []
            
            for i, dist in enumerate(distance):
                repeat_traces_distance = []
                repeat_indices = self.L.get_condition_trials(
                    monocular=condition['monocular'], 
                    history=condition['history'], 
                    distance=dist
                )

                for repeat in repeat_indices:
                    on_start, on_end = self.L.get_on_times(repeat)
                    frames = (self.L.frames_time > (on_start-pre_time)) & (self.L.frames_time < (on_end+post_time))
                    trace_length = self.L.on_length + int((pre_time + post_time)/100)
                    
                    # Make sure we don't exceed the available frames
                    trace = self.L.response[idx, frames]
                    if len(trace) >= trace_length:
                        repeat_traces.append(trace[:trace_length].T)
                        repeat_traces_distance.append(trace[:trace_length].T)
                
                if not repeat_traces_distance:
                    continue

                repeats_per_distance.append(len(repeat_indices))

                axs[c_idx, 1].plot(np.mean(repeat_traces_distance, axis=0), 
                                  color=self.colors[dist], 
                                  label=f'Distance {dist}', 
                                  linewidth=3)
                axs[c_idx, 1].fill_between(
                    np.arange(len(np.mean(repeat_traces_distance, axis=0))), 
                    np.mean(repeat_traces_distance, axis=0) - np.std(repeat_traces_distance, axis=0)/np.sqrt(len(repeat_traces_distance)), 
                    np.mean(repeat_traces_distance, axis=0) + np.std(repeat_traces_distance, axis=0)/np.sqrt(len(repeat_traces_distance)), 
                    color=self.colors[dist], alpha=0.2
                )

            if not repeat_traces:
                axs[c_idx, 0].text(0.5, 0.5, 'No data for this condition', 
                                  ha='center', va='center', transform=axs[c_idx, 0].transAxes)
                axs[c_idx, 1].text(0.5, 0.5, 'No data for this condition', 
                                  ha='center', va='center', transform=axs[c_idx, 1].transAxes)
                continue
                
            repeat_traces = np.array(repeat_traces)

            # Add horizontal lines to mark different distances
            for i, n_repeats in enumerate(repeats_per_distance):
                if i > 0:
                    y_pos = np.sum(repeats_per_distance[:i])
                    axs[c_idx, 0].axhline(y=y_pos, color=self.colors[i-1], linestyle='--', linewidth=2)
            
            # Plot heatmap
            if self.L.response_mode == 'spikes-zscore' or self.L.response_mode == 'spikes':
                cmap = 'viridis'
            elif self.L.response_mode == 'dFF' or self.L.response_mode == 'dFF-zscore':
                cmap = 'viridis'

            axs[c_idx, 0].pcolormesh(repeat_traces, cmap=cmap, vmin=vmin, vmax=vmax)

            # Add vertical lines to mark stimulus timing
            axs[c_idx, 0].axvline(x=pre_time/100-self.L.miniservo_pre_time/100-self.L.servo_movement_time/100, 
                                color='lightgray', linestyle='--', linewidth=1.5)
            axs[c_idx, 0].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=1.5)
            axs[c_idx, 0].axvline(x=(pre_time/100+self.L.on_length), color='lightgray', linestyle='--', linewidth=1.5)
            
            axs[c_idx, 0].set_xlabel('Time (frames)')
            axs[c_idx, 0].set_ylabel('Trials')
            axs[c_idx, 0].set_title(f'{condition["title"]} - All Traces', fontsize=14)

            # Add vertical lines to mean traces plot
            axs[c_idx, 1].axvline(x=pre_time/100-self.L.miniservo_pre_time/100-self.L.servo_movement_time/100, 
                                color='lightgray', linestyle='--', linewidth=2)
            axs[c_idx, 1].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=2)
            axs[c_idx, 1].axvline(x=(pre_time/100+self.L.on_length), color='lightgray', linestyle='--', linewidth=2)
            
            axs[c_idx, 1].set_xlabel('Time (frames)')
            axs[c_idx, 1].set_ylabel(self.L.response_mode)
            axs[c_idx, 1].set_title(f'{condition["title"]} - Mean Traces', fontsize=14)
            axs[c_idx, 1].set_ylim(vmin, vmax)

            p_value, _, _, rf_values = self.L.analyze_cell_modulation(cell, monocular=condition['monocular'], history=condition['history'])

            bp = axs[c_idx, 2].boxplot(rf_values, patch_artist=True, labels=distance, showfliers=False)
            
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
            for box in bp['boxes']:
                box.set_facecolor('none')

            for i, rf_vals in enumerate(rf_values):
                x = np.random.normal(i+1, 0.04, size=len(rf_vals))
                axs[c_idx, 2].scatter(x, rf_vals, alpha=0.8, color=colors[i], s=40)

            # # Check if p_value is significant and if the cell is responsive for the monocular condition
            if p_value < 0.05 and self.L.analyze_cell_responsiveness(cell, monocular=condition['monocular'], history=condition['history'])[0]:
                bar_height = vmax * 0.8
                axs[c_idx, 2].plot([1, len(distance)], [bar_height, bar_height], 'k-', linewidth=1.5)
                axs[c_idx, 2].text((1 + len(distance))/2, bar_height*1.02, '*', ha='center', va='bottom')

            axs[c_idx, 2].set_title(f'p={p_value:.3f}', fontsize=14)
            axs[c_idx, 2].set_xlabel('Distance')
            axs[c_idx, 2].set_ylabel(self.L.response_mode)
            axs[c_idx, 2].set_ylim(vmin, vmax)


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
    

class LinearSummaries():

    def __init__(self, linear_class, linear_stats):
        self.L = linear_class
        self.linear_stats = linear_stats
        self.session_name = self.L.session_path.split('/')[-1]
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
        for i, stat_df in enumerate(self.linear_stats):
            condition_name = stat_df['condition'].iloc[0] if not stat_df.empty else f"condition_{i}"
            # Get cells that are responsive but not modulated
            non_modulated_cells[condition_name] = stat_df[(stat_df['responsive']==True) & (stat_df['modulated']==False)]['cell_id'].values

        # Find cells that are non-modulated in all conditions
        all_conditions = set.intersection(*[set(cells) for cells in non_modulated_cells.values()]) if non_modulated_cells else set()
        non_modulated_cells = np.array(list(all_conditions))

        return non_modulated_cells
    
    def find_modulated_cells(self):

        # find the cells that are modulated in any condition
        modulated_cells = {}
        responsive_cells = {}
        for i, stat_df in enumerate(self.linear_stats):
            condition_name = stat_df['condition'].iloc[0] if not stat_df.empty else f"condition_{i}"
            modulated_cells[condition_name] = stat_df[stat_df['modulated']==True]['cell_id'].values
            responsive_cells[condition_name] = stat_df[stat_df['responsive']==True]['cell_id'].values

        # Find cells that are modulated in all conditions
        all_conditions = set.intersection(*[set(cells) for cells in modulated_cells.values()]) if modulated_cells else set()
        modulated_cells['all_conditions'] = np.array(list(all_conditions))

        # Find cells that are responsive in all conditions
        all_conditions = set.intersection(*[set(cells) for cells in responsive_cells.values()]) if responsive_cells else set()
        responsive_cells['all_conditions'] = np.array(list(all_conditions))

        # # Remove cells from single conditions if they are in all conditions
        # for condition_name in modulated_cells:
        #     if condition_name != 'all_conditions':
        #         modulated_cells[condition_name] = np.setdiff1d(modulated_cells[condition_name], 
        #                                                      modulated_cells['all_conditions'])
        #         responsive_cells[condition_name] = np.setdiff1d(responsive_cells[condition_name], 
        #                                                      responsive_cells['all_conditions'])
        
        # # find the non-modulated, that is the responsive cells that are not in any categorymodulated cells
        # non_modulated = np.setdiff1d(responsive_cells['all_conditions'], 
        #                            np.concatenate([cells for cells in modulated_cells.values()]))
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
        ax.set_ylabel('Number of Responsive Cells', fontsize=14)
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

        labels = list(self.modulated_cells.keys())[:-1]
        responsive_counts = [len(self.responsive_cells[label]) for label in labels]
        total_counts = [len(self.L.suite2p_id) for label in labels]

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
        fig, axes = plt.subplots(1, 1+len(self.linear_stats), figsize=(12, 2*len(self.linear_stats)))

        # first axes will be the total responsive pie chart
        # Extract unique responsive cells across conditions
        responsive_cells = []
        for i, stat_df in enumerate(self.linear_stats):
            responsive = set(stat_df.loc[stat_df['responsive'] == True, 'cell_id'].values)
            responsive_cells.append(np.array(list(responsive)))

        # concatenate the arrays
        responsive_cells = np.concatenate(responsive_cells)
        # find the unique cells
        responsive_cells = np.unique(responsive_cells)

        axes[0].pie([len(responsive_cells), len(self.L.suite2p_id) - len(responsive_cells)], 
                        labels=[f'Responsive ({len(responsive_cells)}/{len(self.L.suite2p_id)})', f'Non-responsive ({len(self.L.suite2p_id) - len(responsive_cells)}/{len(self.L.suite2p_id)})'],
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        startangle=90,
                        shadow=True,
                        textprops={'fontsize': 16})
        axes[0].set_title('Either condition', fontsize=20)

        # Data for pie charts from stats DataFrame
        responsive_counts = []
        non_responsive_counts = []

        for i, df in enumerate(self.linear_stats):

            condition_name = df['condition'][0]
            responsive_count = df['responsive'].sum()
            total_count = len(df)
            non_responsive_count = total_count - responsive_count
            
            responsive_counts.append(responsive_count)
            non_responsive_counts.append(non_responsive_count)
            
            axes[i+1].pie([responsive_count, non_responsive_count], 
                        labels=[f'Responsive ({responsive_count}/{total_count})', f'Non-responsive ({non_responsive_count}/{total_count})'],
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
                    strength = self.linear_stats[-1].loc[self.linear_stats[-1]['cell_id'] == cell, 'mod_strength'].values[0]
                else:
                    strength = self.linear_stats[i].loc[self.linear_stats[i]['cell_id'] == cell, 'mod_strength'].values[0]
                
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
                    pref_distance = self.linear_stats[-1].loc[self.linear_stats[-1]['cell_id'] == cell, 'pref_distance'].values[0]
                else:
                    pref_distance = self.linear_stats[i].loc[self.linear_stats[i]['cell_id'] == cell, 'pref_distance'].values[0]
                
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
    
    def plot_location_map(self, condition):

        # Plot all responsive cells using scatter at pixel locations
        cells_to_plot = self.responsive_cells[condition]

        # Filter stats for given condition
        stats_df = self.linear_stats['condition'==condition]

        # Load Suite2p outputs
        stat = np.load(self.L.activity_path + r'\stat.npy', allow_pickle=True)
        ops = np.load(self.L.activity_path + r'\ops.npy', allow_pickle=True).item()

        # Background image setup
        mean_img = ops['meanImg']
        vmin = np.percentile(mean_img, 1)
        vmax = np.percentile(mean_img, 99)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(ops['refImg'], alpha=0.5, cmap='gray', vmin=vmin, vmax=vmax)

        # Scatter plot each pixel
        for cell_id in cells_to_plot:
            idx = np.where(self.L.suite2p_id == cell_id)[0][0]
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




