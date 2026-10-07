import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import copy
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from src.utils import setup_plot_style, import_vars_from_matlab, create_new_result_dir
from src.ActivityPipelines import ActivityLoader
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# one value per trial; select_trials subsets these together
TRIAL_FIELDS = ('distance_sequence', 'monocular_sequence', 'eye_sequence', 'direction_sequence',
                'block_sequence', 'sweep_from', 'sweep_to', 'off_start', 'on_start', 'on_end')


def _subset_trials(device, trials):
    """keep only these trials (int indices) in every per-trial field, in place"""
    for field in TRIAL_FIELDS:
        if hasattr(device, field):
            setattr(device, field, np.asarray(getattr(device, field))[trials])

class RadialDevice:

    def __init__(self, data_dir=None, response_mode='spikes', tau=None,
                 mouse_name=None, record_date=None, experiment_name=None,
                 base_data_dir=r'D:/data-distance-coding-project/processed-data',
                 verbose=False):
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

        self.verbose = verbose

        # Build data_dir from session info if provided
        if data_dir is None and all([mouse_name, record_date, experiment_name]):
            self.mouse_name = mouse_name
            self.record_date = record_date
            self.experiment_name = experiment_name
            self.session_name = f'{record_date}_{mouse_name}_{experiment_name}'
            data_dir = os.path.join(base_data_dir, self.session_name)
            if self.verbose:
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
        self._drop_incomplete_trials()

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
            if self.verbose:
                print('Stimulus file found')
            return io.loadmat(self.stimulus_dir + '/' + str(filelist[0]))

    def _load_stimulus_variables(self):

        container = self._load_stimulus_container()

        struct_keys = [k for k in container if not k.startswith('__')]
        if len(struct_keys) != 1:
            raise Exception(f'Expected exactly one stimulus struct in .mat file, found: {struct_keys}')
        structure_name = struct_keys[0]

        locals_dict = import_vars_from_matlab(container, structure_name)

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

        # radialDeviceControl_v3: sweeps alternate direction (+1 start_angle -> end_angle,
        # -1 back) and the eye condition comes in blocks (eye_sequence 1 left, 2 right, 3 both).
        # Older files sweep start_angle -> end_angle only.
        self.direction_sequence = locals_dict.get('direction_sequence', np.ones(len(self.distance_sequence), dtype=int))
        for field in ('eye_sequence', 'block_sequence', 'block_order', 'sweep_from', 'sweep_to',
                      'block_start', 'switch_start', 'switch_end'):
            if field in locals_dict:
                setattr(self, field, locals_dict[field])

    def _drop_incomplete_trials(self):
        # a trial still running when imaging stopped would set on_length (the shortest trial)
        # for every trial, so keep only trials whose on-period lies inside the imaging
        complete = (1000 * self.on_start >= self.frames_time[0]) & (1000 * self.on_end <= self.frames_time[-1])
        if not complete.all():
            print(f'Dropping trial(s) {np.flatnonzero(~complete).tolist()}: on-period outside the imaging')
            _subset_trials(self, np.flatnonzero(complete))

    def _load_activity(self, tau=None):

        try:
            D = ActivityLoader(self.data_dir, tau)
            self.frames_time = D.frames_time
            self.dFF = D.dFF
            self.spikes = D.spikes
            self.suite2p_id = D.suite2p_id
            if self.verbose:
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
        if self.verbose:
            print("Repeat stimulus on length distribution:")
            for length, count in sorted(frame_counts.items()):
                print(f"    {count} repeats have length {length}")

        # set the on length to the shortest length
        self.on_length = min(frame_counts.keys())
        if self.verbose:
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

    def get_on_frames(self, repeat):
        """
        Frame indices of a trial's on-period, on_length long and ordered along angle_range
        (start_angle -> end_angle). Reverse sweeps (v3, end_angle -> start_angle) are flipped
        and keep the frames nearest start_angle, as forward sweeps do, so trimming to on_length
        drops the same end of the sweep in both directions.
        """
        on_start, on_end = self.get_on_times(repeat)
        frames = np.flatnonzero((self.frames_time > on_start) & (self.frames_time < on_end))
        if self.direction_sequence[repeat] < 0:
            frames = frames[::-1]
        return frames[:self.on_length]

    def get_on_trace(self, idx, repeat):

        return self.response[idx, self.get_on_frames(repeat)]

    def select_trials(self, trials, label=None):
        """
        Shallow copy that keeps only some trials, e.g. one eye condition of an interleaved
        session: R.select_trials(R.eye_sequence == 1, 'left'). Activity arrays are shared, not
        copied; on_length and angle_range stay those of the full session.

        Parameters
        ----------
        trials : bool mask or int indices over the session's trials
        label : str, optional
            appended to session_name (so cached dataframes don't collide)
        """
        trials = np.asarray(trials)
        if trials.dtype == bool:
            trials = np.flatnonzero(trials)
        sub = copy.copy(self)
        _subset_trials(sub, trials)
        if label is not None and hasattr(self, 'session_name'):
            sub.session_name = f'{self.session_name}_{label}'
        return sub

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
