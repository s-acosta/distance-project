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
        elif self.response_mode == 'dFF-zscore':
            self.response = (self.dFF - np.mean(self.dFF)) / np.std(self.dFF)
        elif self.response_mode == 'spikes':
            self.response = self.spikes
        elif self.response_mode == 'spikes-zscore':
            # do gaussian smoothing
            self.response = gaussian_filter1d(self.spikes, 3, axis=1)
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
        self.dFF = self.dFF[:,self.frames_time <= 1000*self.trial_end_time[-1]]
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
        on_end = self.frames_time[on_end_idx] + self.miniservo_to_inactive_time + self.miniservo_hold

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
    
    def get_all_traces(self, cell, distance=None, monocular=None, mean_trace=False):

        idx = np.where(self.suite2p_id == cell)[0][0]

        # find the indices of repeats that match the criteria
        if monocular is not None:
            monocular_indices = np.where(self.monocular_sequence == monocular)[0]
        else:
            monocular_indices = np.arange(len(self.trial_sequence))

        if distance is not None:
            distance_indices = np.where(self.trial_sequence == distance)[0]
        else:
            distance_indices = np.arange(len(self.trial_sequence))

        repeat_indices = np.intersect1d(monocular_indices, distance_indices)

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        if mean_trace:
            return np.mean(all_traces, axis=0), np.std(all_traces, axis=0)/np.sqrt(all_traces.shape[0])
        else:
            return all_traces

    def analyze_cell_responsiveness(self, cell, distance=None, monocular=None, threshold=1):
        """
        Analyzes if a cell is responsive to stimuli under a specific condition.
        
        Args:
            cell: Cell index to analyze
            distance: Distance condition to analyze (None for all distances)
            
        Returns:
            tuple: (is_responsive, p_value, on_max, off_max) where is_responsive is True if the cell is 
                   significantly more active during ON periods (p < 0.05 and ON median > OFF median)
        """
        
        # Determine which trials to use based on condition
        if distance is None and monocular is None:
            repeat_indices = np.arange(len(self.trial_sequence))
        else:
            # Filter by distance if specified
            if distance is None:
                distance_indices = np.arange(len(self.trial_sequence))
            else:
                distance_indices = np.where(self.trial_sequence == distance)[0]
            
            # Filter by monocular if specified
            if monocular is None:
                monocular_indices = np.arange(len(self.trial_sequence))
            else:
                monocular_indices = np.where(self.monocular_sequence == monocular)[0]
            
            # Get intersection of both conditions
            repeat_indices = np.intersect1d(distance_indices, monocular_indices)
        
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
            window_end = min(len(on_trace), max_pos + 5)  # +3 because end is exclusive
            on_max.append(np.mean(on_trace[window_start:window_end]))
            off_max.append(np.mean(off_trace))
        
        # Perform t-test
        t_stat, p_val = ttest_ind(on_max, off_max)
        
        # Determine if cell is responsive (p < 0.05 and median ON > median OFF)
        # Ensure max_pos isn't the last index to avoid window issues
        is_responsive = (p_val < 0.05) and (np.median(on_max)>np.median(off_max)) and (np.percentile(on_max, 75) > threshold) and (max_pos < len(on_trace) - 1) and (max_pos > 0)
        
        
        return is_responsive, p_val, on_max, off_max
    
    def analyze_population_stats(self):

        # Initialize empty lists to collect data for each cell
        cell_data = []
        
        # Analyze each cell
        for cell in self.suite2p_id:
            # Analyze monocular condition
            is_responsive_monocular = self.analyze_cell_responsiveness(cell, monocular=True)[0]
            
            # Analyze binocular condition
            is_responsive_binocular = self.analyze_cell_responsiveness(cell, monocular=False)[0]
            
            # Get distance modulation analysis results
            p_value_monocular, pref_monocular, mod_strength_monocular, _ = self.analyze_cell_modulation(cell, monocular=True)
            p_value_binocular, pref_binocular, mod_strength_binocular, _ = self.analyze_cell_modulation(cell, monocular=False)

            # Create a row for this cell
            cell_row = {
                'cell_id': cell,
                'mono_responsive': np.bool(is_responsive_monocular),
                'mono_modulated': np.bool(p_value_monocular < 0.05 and is_responsive_monocular),
                'mono_modulated_p': p_value_monocular,
                'mono_pref': pref_monocular if p_value_monocular < 0.05 and is_responsive_monocular else np.nan,
                'mono_strength': mod_strength_monocular if p_value_monocular < 0.05 and is_responsive_monocular else np.nan,
                'bino_responsive': np.bool(is_responsive_binocular),
                'bino_modulated': np.bool(p_value_binocular < 0.05 and is_responsive_binocular),
                'bino_modulated_p': p_value_binocular,
                'bino_pref': pref_binocular if p_value_binocular < 0.05 and is_responsive_binocular else np.nan,
                'bino_strength': mod_strength_binocular if p_value_binocular < 0.05 and is_responsive_binocular else np.nan,
            }
            
            cell_data.append(cell_row)
        
        # Create DataFrame from collected data
        stats_df = pd.DataFrame(cell_data)
        
        # Calculate summary statistics
        n_cells = len(stats_df)
        n_responsive_either = stats_df['mono_responsive'].sum() + stats_df['bino_responsive'].sum() - \
                             (stats_df['mono_responsive'] & stats_df['bino_responsive']).sum()
        
        n_both_modulated = (stats_df['mono_modulated'] & stats_df['bino_modulated']).sum()
        n_mono_modulated = (stats_df['mono_modulated'] & ~stats_df['bino_modulated']).sum()
        n_bino_modulated = (~stats_df['mono_modulated'] & stats_df['bino_modulated']).sum()
        
        # Print summary statistics
        print(f"Total cells: {n_cells}")
        print(f"Responsive cells (either condition): {n_responsive_either} ({n_responsive_either / n_cells * 100:.1f}%)")
        
        if n_responsive_either > 0:
            print(f"Monocular modulated cells: {n_mono_modulated} ({n_mono_modulated / n_responsive_either * 100:.1f}% of responsive, {n_mono_modulated / n_cells * 100:.1f}% of total)")
            print(f"Binocular modulated cells: {n_bino_modulated} ({n_bino_modulated / n_responsive_either * 100:.1f}% of responsive, {n_bino_modulated / n_cells * 100:.1f}% of total)")
            print(f"Both modulated cells: {n_both_modulated} ({n_both_modulated / n_responsive_either * 100:.1f}% of responsive, {n_both_modulated / n_cells * 100:.1f}% of total)")
        else:
            print("Monocular modulated cells: 0 (0%)")
            print("Binocular modulated cells: 0 (0%)")
            print("Both modulated cells: 0 (0%)")
        
        
        return stats_df

    def analyze_cell_modulation(self, cell, monocular=None):

        window_size = 5
        distances = np.unique(self.trial_sequence)

        # get the mean trace for this cell
        mean_trace = self.get_all_traces(cell, mean_trace=True, monocular=None)[0]

        # find the receptive fied for this cell
        rf_idx = np.argmax(mean_trace)

        rf_values = []
        for distance in distances:
            on_traces = self.get_all_traces(cell,  
                                     distance=distance,
                                     monocular=monocular)
            # Find the max value within window +5 -5 frames from rf_index
            window_start = max(0, rf_idx - window_size)
            window_end = min(on_traces.shape[1], rf_idx + window_size + 1)
            on_traces_max = np.max(on_traces[:, window_start:window_end], axis=1)
            rf_values.append(on_traces_max)

        f_stat, p_value = f_oneway(*rf_values)

        # preffered distance is the one with the highest mean response
        preferred_distance = np.argmax(np.mean(rf_values, axis=1))
        least_preferred_distance = np.argmin(np.mean(rf_values, axis=1))

        # modulation strength is how much higher the response is at the preferred distance compared to the least preferred distance
        preferred_mean = np.mean(rf_values[preferred_distance])
        least_preferred_mean = np.mean(rf_values[least_preferred_distance])
        modulation_strength = (preferred_mean - least_preferred_mean) / (preferred_mean + least_preferred_mean)  # Avoid division by zero

        return p_value, preferred_distance, modulation_strength, rf_values

class LinearPlots:

    def __init__(self, linear_class):
        self.L = linear_class
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
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc='orange', label='Servo Up'))
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            
        
        plt.xlabel('Time (s)')
        if cell is None:
            plt.ylabel('Stepper Position (cm)')
            plt.title('Stimulus Response')
        else:
            plt.ylabel(self.L.response_mode)
            plt.title(f'Stimulus Response for Cell {cell}')

        plt.tight_layout()

    def plot_all_traces_colormap(self, cell, pre_time=2500, post_time=1000, distance=None, monocular=None):
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

            if dist is None:
                distance_indices = np.arange(len(self.L.trial_sequence))
            else:
                distance_indices = np.where(self.L.trial_sequence == dist)[0]

            # find the indices of repeats that match the criteria
            if monocular is not None:
                monocular_indices = np.where(self.L.monocular_sequence == monocular)[0]
            else:
                monocular_indices = np.arange(len(self.L.trial_sequence))

            repeat_indices = np.intersect1d(distance_indices, monocular_indices)

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
            axs[i, 0].set_title(f'All Traces - Distance {dist}')
            
            # Plot mean trace
            axs[i, 1].axvline(x=(pre_time-self.L.miniservo_pre_time)/100, color='lightgray', linestyle='--', linewidth=3)
            axs[i, 1].axvline(x=(pre_time/100), color='lightgray', linestyle='--', linewidth=3)
            axs[i, 1].axvline(x=(pre_time/100+self.L.on_length), color='lightgray', linestyle='--', linewidth=3)
            axs[i, 1].plot(repeat_traces_mean, color='black')
            axs[i, 1].set_xlabel('Time (frames)')
            axs[i, 1].set_ylabel(self.L.response_mode)
            axs[i, 1].set_title(f'Mean Traces - Distance {dist}')
            axs[i, 1].fill_between(np.arange(len(repeat_traces_mean)), 
                                    repeat_traces_mean - repeat_traces_sem, 
                                    repeat_traces_mean + repeat_traces_sem, 
                                    color='black', alpha=0.2)
            axs[i, 1].set_ylim(vmin, vmax)

            # plot boxplot of on and off
            
            # Analyze cell responsiveness and plot boxplot as third column
            if distance[0] is None:
                is_responsive, p_val, on_max, off_max = self.L.analyze_cell_responsiveness(cell, distance=None, monocular=monocular)
            else:
                is_responsive, p_val, on_max, off_max = self.L.analyze_cell_responsiveness(cell, distance=dist, monocular=monocular)
            
            # Add scatter points to each column
            for j, data in enumerate([on_max, off_max], 1):
                # Add jitter to x positions for better visibility
                x = np.random.normal(j, 0.05, size=len(data))
                axs[i, 2].scatter(x, data, alpha=0.2, s=30, color='black', zorder=3)

            axs[i, 2].axhline(y=np.mean(on_max), color='black', linestyle='--', linewidth=1.5, alpha=0.5)

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
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(2)]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(2)]
        
        plt.suptitle(f'Cell {cell}', fontsize=24)
        
        colors = setup_plot_style()
        distances = [0, 1, 2, 3]
        
        # Calculate global min/max for top row
        vmin_top = np.percentile(self.L.response[idx], 1)
        vmax_top = np.percentile(self.L.response[idx], 99)

        vmin_bottom = np.min(self.L.response[idx])
        vmax_bottom = np.max(self.L.response[idx])

        # Plot both rows
        for j, condition in enumerate(conditions):
            # Top row - distance traces
            ax_top = axes_top[j]
            for i, distance in enumerate(distances):
                mean_trace, sem_trace = self.L.get_all_traces(cell, distance=distance,
                                                    monocular=condition['monocular'],
                                                    mean_trace=True)
                
                ax_top.plot(mean_trace.T, color=colors[i], linewidth=2, label=f'Distance {distance}')
                ax_top.fill_between(range(len(mean_trace)), 
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color=colors[i], alpha=0.2)
            
            ax_top.set_xlabel('Time (frames)')
            ax_top.set_ylabel(self.L.response_mode)
            ax_top.set_title(condition['title'])
            ax_top.legend()
            ax_top.set_ylim(vmin_top, vmax_top)

            # Bottom row - RF values
            ax_bottom = axes_bottom[j]
            ax_bottom.set_ylim(vmin_bottom, vmax_bottom)

            p_value, _, _, rf_values = self.L.analyze_cell_modulation(cell, monocular=condition['monocular'])

            bp = ax_bottom.boxplot(rf_values, patch_artist=True, labels=distances)
            
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
            if p_value < 0.05 and self.L.analyze_cell_responsiveness(cell, monocular=condition['monocular'])[0]:
                bar_height = vmax_bottom * 0.95
                ax_bottom.plot([1, len(distances)], [bar_height, bar_height], 'k-', linewidth=1.5)
                ax_bottom.text((1 + len(distances))/2, bar_height*1.02, '*', ha='center', va='bottom')

            ax_bottom.set_title(f'p={p_value:.3f}')
            ax_bottom.set_xlabel('Distance')
            ax_bottom.set_ylabel(self.L.response_mode)
            

        plt.tight_layout()
        return fig