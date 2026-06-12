import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")                       # repo root: path_setup, utils
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project\disparity-analysis")    # so `src.*` = disparity's own
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils.suite2p import ActivityLoader
from utils.helpers import setup_plot_style, create_new_result_dir
from utils.matlab_io import load_mat_struct


class DisparityDevice:
    """
    Data access for a goggles (left/right eye) disparity session.

    Mirrors ``RadialDevice`` but conditions trials on **eye** instead of
    distance/monocular. A session may show one eye or both; ``present_eyes`` /
    ``single_eye`` / ``eye`` describe what was actually presented.

    Can be constructed two ways:
      1. Direct path:  DisparityDevice(data_dir='path/to/session')
      2. Session info: DisparityDevice(mouse_name='SAM045', record_date='260609',
                                       experiment_name='goggles_left')

    Parameters
    ----------
    data_dir : str, optional
        Direct path to the session directory.
    response_mode : str
        'spikes' (z-scored + smoothed) or 'dFF' (smoothed).
    tau : float, optional
        Indicator timescale; if set, spikes are recomputed via deconvolution.
    mouse_name, record_date, experiment_name : str, optional
        Used to build ``data_dir`` from ``base_data_dir`` when ``data_dir`` is None.
    base_data_dir : str
        Base directory for processed data (only used with session info).
    on_length : int
        Number of stimulus-on frames to keep per trial (default 60).
    verbose : bool
        Print loading progress.
    """

    # stim['eye_order'] codes
    EYE_CODES = {'right': 1, 'left': 2}

    def __init__(self, data_dir=None, response_mode='spikes', tau=None,
                 mouse_name=None, record_date=None, experiment_name=None,
                 base_data_dir=r'D:/data-distance-coding-project/processed-data',
                 on_length=60, verbose=False):

        self.verbose = verbose
        self.mouse_name = mouse_name
        self.record_date = record_date
        self.experiment_name = experiment_name

        # Build data_dir from session info if provided
        if data_dir is None and all([mouse_name, record_date, experiment_name]):
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
        else:
            # direct path: derive a session name from the folder for downstream use
            self.session_name = os.path.basename(os.path.normpath(data_dir))

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
            self.response = (self.response - np.mean(self.response, axis=1, keepdims=True)) / np.std(self.response, axis=1, keepdims=True)
            self.response = gaussian_filter1d(self.response, 1.5, axis=1)
        else:
            raise Exception('Unknown activity mode')

        self._detect_eyes()
        self._set_on_length(on_length)

    # ------------------------------------------------------------------ loading

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

    def _load_stimulus_container(self):
        """Return the single stimulus .mat in the session as a cleaned dict."""
        filelist = [f for f in os.listdir(self.stimulus_dir) if f.endswith(".mat")]
        if len(filelist) == 0:
            raise FileNotFoundError(f"No .mat stimulus file found in {self.stimulus_dir}")
        elif len(filelist) > 1:
            raise RuntimeError(
                f"Multiple .mat files found in {self.stimulus_dir}: {sorted(filelist)}. "
                "Expected exactly one."
            )
        if self.verbose:
            print('Stimulus file found')
        return load_mat_struct(os.path.join(self.stimulus_dir, sorted(filelist)[0]))

    def _load_stimulus_variables(self):

        self.stim = self._load_stimulus_container()

        self.stim_start = np.asarray(self.stim['stim_start'])
        self.stim_end = np.asarray(self.stim['stim_end'])
        self.pre_start = np.asarray(self.stim['pre_start'])
        self.eye_order = np.asarray(self.stim['eye_order']).astype(int)
        self.parameters = self.stim['parameters']
        self.n_repeats = len(self.eye_order)

    def _detect_eyes(self):
        """Determine which eye(s) were stimulated (1=right, 2=left)."""
        code_to_eye = {v: k for k, v in self.EYE_CODES.items()}
        present_codes = np.unique(self.eye_order)
        self.present_eyes = [code_to_eye[int(c)] for c in present_codes if int(c) in code_to_eye]
        self.single_eye = len(self.present_eyes) == 1
        self.eye = self.present_eyes[0] if self.single_eye else None
        if self.verbose:
            print(f"Eyes present: {self.present_eyes}"
                  + (f" (single-eye: {self.eye})" if self.single_eye else " (both eyes)"))

    def _set_on_length(self, on_length=60):
        self.on_length = on_length
        if self.verbose:
            print(f'Set stimulus on length: {self.on_length}')

    # -------------------------------------------------------------------- timing

    def get_stim_times(self, repeat):
        """(on_start, on_end) in ms for the stimulus-on window of a repeat."""
        return 1000 * self.stim_start[repeat], 1000 * self.stim_end[repeat]

    def get_pre_times(self, repeat):
        """(pre_start, stim_start) in ms — the pre-stimulus baseline window."""
        return 1000 * self.pre_start[repeat], 1000 * self.stim_start[repeat]

    # -------------------------------------------------------------------- traces

    def get_on_trace(self, idx, repeat):
        """Stimulus-on response for response-row ``idx`` (truncated to on_length)."""
        on_start, on_end = self.get_stim_times(repeat)
        on_frames = (self.frames_time > on_start) & (self.frames_time < on_end)

        trace = self.response[idx, on_frames]
        if self.on_length is not None and len(trace) > self.on_length:
            trace = trace[:self.on_length]
        return trace

    def get_repeat_trace(self, cell, repeat):
        """Stimulus-on response for a ``suite2p_id`` cell on one repeat."""
        idx = np.where(self.suite2p_id == cell)[0][0]
        return self.get_on_trace(idx, repeat)

    def get_eye_repeats(self, eye=None):
        """Trial indices for an eye ('left'/'right'), or all trials if None."""
        if eye in self.EYE_CODES:
            return np.where(self.eye_order == self.EYE_CODES[eye])[0]
        return np.arange(self.n_repeats)

    # --------------------------------------------------------- screen geometry

    def get_screen_positions(self, eye):
        """(start_pos, end_pos) screen pixel coordinates for an eye's sweep."""
        start_pos = self.parameters[f'{eye}_start']
        end_pos = self.parameters[f'{eye}_end']
        return start_pos, end_pos

    def get_screen_range(self, eye, signed=False):
        """
        Per-frame pixel positions across the sweep (length == on_length).

        When ``signed`` is True the left eye is negated, so left maps to the
        negative visual field and right to the positive (as the notebook plots).
        """
        start_pos, end_pos = self.get_screen_positions(eye)
        rng = np.linspace(start_pos, end_pos, self.on_length)
        if signed and eye == 'left':
            rng = -rng
        return rng

    def get_all_traces(self, cell, eye=None, mean_trace=False):
        """
        Stimulus-on traces across the repeats of an eye condition.

        Returns an array of shape (n_repeats, on_length), or, when
        ``mean_trace`` is True, a (mean, sem) tuple over repeats.
        """
        idx = np.where(self.suite2p_id == cell)[0][0]
        repeat_indices = self.get_eye_repeats(eye)

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        if mean_trace:
            return np.mean(all_traces, axis=0), np.std(all_traces, axis=0) / np.sqrt(all_traces.shape[0])
        return all_traces

    # ------------------------------------------------------------ receptive field

    def find_rf_position(self, cell, eye=None):
        """
        Peak (receptive-field) frame of the mean trace for an eye.

        Returns
        -------
        (rf_idx, rf_value) : (int, float)
            Index of the peak frame and the mean response there.
        """
        mean_trace = self.get_all_traces(cell, eye=eye, mean_trace=True)[0]
        rf_idx = int(np.argmax(mean_trace))
        rf_value = float(mean_trace[rf_idx])
        return rf_idx, rf_value

    def find_rf_values(self, cell, eye=None, window_size=5):
        """Per-trial mean response in a window around the RF peak frame."""
        rf_idx, _ = self.find_rf_position(cell, eye=eye)
        on_traces = self.get_all_traces(cell, eye=eye)

        window_start = max(0, rf_idx - window_size)
        window_end = min(on_traces.shape[1], rf_idx + window_size + 1)
        return np.mean(on_traces[:, window_start:window_end], axis=1)
