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


class GogglesDevice:
    """
    Paradigm-independent data access for a goggles session.

    Handles everything shared by the goggles stimuli: activity loading, stimulus
    timing, stimulus-on traces, screen/angle geometry, and receptive-field peak.
    Trial *conditioning* is left to subclasses through two hooks:

      - ``_detect_conditions()`` — set the condition attributes (which eyes /
        which disparities were shown) after the stimulus is loaded.
      - ``_get_repeats(**cond)`` — map a condition (e.g. ``eye='left'`` or
        ``disparity=0``) to the trial indices it selects.

    Because :meth:`get_all_traces`, :meth:`find_rf_position`, and
    :meth:`find_rf_values` accept ``**cond`` and delegate to ``_get_repeats``, a
    subclass only has to define those two hooks to get the full trace/RF API.

    Can be constructed two ways:
      1. Direct path:  GogglesDevice(data_dir='path/to/session')
      2. Session info: GogglesDevice(mouse_name='SAM045', record_date='260609',
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

        self._detect_conditions()
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
        """Load the stimulus fields common to every goggles paradigm."""
        self.stim = self._load_stimulus_container()

        self.stim_start = np.asarray(self.stim['stim_start'])
        self.stim_end = np.asarray(self.stim['stim_end'])
        self.pre_start = np.asarray(self.stim['pre_start'])
        self.parameters = self.stim['parameters']
        self.n_repeats = len(self.stim_start)

        # Per-eye deg->pixel fits [slope, intercept] (teensyStimulusAngle), i.e. the
        # mapping this session was actually run with. The legacy goggles_left format
        # predates them, so they are absent there -> None.
        self.left_fit = self._stim_fit('left')
        self.right_fit = self._stim_fit('right')

        # Elevation sweep (teensyStimulusMonocular's opts.elevations), if used. Empty
        # (has_elevation False) for every session recorded before that option existed,
        # or recorded with it left at its default -- both look identical here since
        # the field is simply absent/empty in the stim log either way.
        self.elevations = np.atleast_1d(
            np.asarray(self.stim.get('elevations', []), dtype=float)).ravel()
        if 'trial_elevation_px' in self.stim:
            self.trial_elevation_px = np.asarray(self.stim['trial_elevation_px'], dtype=float).ravel()
        else:
            self.trial_elevation_px = np.full(self.n_repeats, np.nan)

    def _stim_fit(self, eye):
        """That eye's [slope, intercept] deg->pixel fit from the stim log, or None."""
        key = f'{eye}_fit'
        if key not in self.stim:
            return None
        return np.asarray(self.stim[key], float).ravel()

    def _set_on_length(self, on_length=60):
        self.on_length = on_length
        if self.verbose:
            print(f'Set stimulus on length: {self.on_length}')

    # ------------------------------------------------------- conditioning hooks

    def _detect_conditions(self):
        """Set condition attributes from the stimulus. Overridden by subclasses."""
        raise NotImplementedError

    def _get_repeats(self, **cond):
        """Map a condition (e.g. ``eye=`` or ``disparity=``) to trial indices.

        With no condition, returns all trials. Overridden by subclasses.
        """
        return np.arange(self.n_repeats)

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

    def get_all_traces(self, cell, mean_trace=False, **cond):
        """
        Stimulus-on traces across the repeats selected by ``**cond``.

        Returns an array of shape (n_repeats, on_length), or, when
        ``mean_trace`` is True, a (mean, sem) tuple over repeats. The condition
        keyword (``eye=`` / ``disparity=`` / none) is resolved by the subclass's
        :meth:`_get_repeats`.
        """
        idx = np.where(self.suite2p_id == cell)[0][0]
        repeat_indices = self._get_repeats(**cond)

        all_traces = []
        for repeat in repeat_indices:
            on_trace = self.get_on_trace(idx, repeat)
            all_traces.append(on_trace.T)

        all_traces = np.array(all_traces)

        if mean_trace:
            return np.mean(all_traces, axis=0), np.std(all_traces, axis=0) / np.sqrt(all_traces.shape[0])
        return all_traces

    # --------------------------------------------------------- screen geometry

    def get_screen_positions(self, eye):
        """(start_pos, end_pos) screen pixel coordinates for an eye's sweep."""
        # new teensyStimulusBoth format: per-trial px arrays (NaN on the off-eye trials)
        if f'{eye}_eye_start_px' in self.stim:
            start = np.asarray(self.stim[f'{eye}_eye_start_px'], float)
            end = np.asarray(self.stim[f'{eye}_eye_end_px'], float)
            start, end = start[~np.isnan(start)], end[~np.isnan(end)]
            return float(start[0]), float(end[0])   # constant across shown trials
        # legacy goggles_left format
        return self.parameters[f'{eye}_start'], self.parameters[f'{eye}_end']

    def get_screen_range(self, eye, signed=False):
        """
        Per-frame pixel positions across the sweep (length == on_length).

        When ``signed`` is True the left eye is negated unconditionally.

        .. warning::
           That negation is only correct for the legacy ``goggles_left`` format,
           whose left pixels are drawn-panel columns (non-negative). The
           ``teensyStimulusBoth`` format already stores signed pixels, so
           ``signed=True`` flips those the wrong way. Use
           :meth:`get_signed_screen_range` to plot. This method keeps its
           behaviour because ``SessionAnalyzer`` writes the ``{eye}_px`` columns
           through it, so the saved dataframes follow this convention and
           changing it would silently invalidate them.
        """
        start_pos, end_pos = self.get_screen_positions(eye)
        rng = np.linspace(start_pos, end_pos, self.on_length)
        if signed and eye == 'left':
            rng = -rng
        return rng

    def get_signed_screen_range(self, eye):
        """
        Per-frame pixels on the stimulus' signed joint axis, for either format.

        Both panels form one axis: left spans [-W, 0], right [0, +W], meeting at
        a nasal edge of 0. Left pixels above 0 are off its panel — the sweep has
        run past the nasal edge and the left eye is showing gray.

        The two stimulus formats disagree on the left eye's sign:
        ``teensyStimulusBoth`` stores it already signed (``left_eye_start_px``
        runs negative), while legacy ``goggles_left`` stores the drawn panel
        column (non-negative). Negate only the latter — an unconditional
        ``-abs()`` folds an already-signed range back on itself at the nasal
        edge, and an unconditional negation puts it on the wrong side.
        """
        rng = self.get_screen_range(eye)
        if eye == 'left' and rng.min() >= 0:      # legacy drawn panel -> signed axis
            rng = -rng
        return rng

    @property
    def has_elevation(self):
        """Whether this session swept multiple fixed elevations (opts.elevations
        in teensyStimulusMonocular). False for every session that predates that
        option, and for one recorded with it left at its default (single sweep)."""
        return self.elevations.size > 0

    @property
    def has_angle_range(self):
        """Whether this session's stimulus log has start_deg/end_deg (teensyStimulusAngle only)."""
        return 'start_deg' in self.parameters and 'end_deg' in self.parameters

    def get_angle_range(self):
        """
        Per-frame visual angle (deg) across the sweep (length == on_length).

        Shared by both eyes: they sweep the same ``start_deg``..``end_deg`` range,
        just mapped onto each eye's screen by its own pixel fit.

        Only available for sessions recorded with ``teensyStimulusAngle`` — legacy
        pixel-format sessions (``teensyStimulusMonocular``/``...Both``/plain
        ``goggles_left``) never logged a degree sweep at acquisition time, and it
        can't be recovered after the fact. Check :attr:`has_angle_range` first, or
        use :meth:`get_signed_screen_range`/:meth:`get_screen_range` (pixels), which
        work for every format.
        """
        if not self.has_angle_range:
            raise ValueError(
                "This session has no start_deg/end_deg in its stimulus parameters "
                "-- it predates the angle-calibrated stimulus format "
                "(teensyStimulusAngle) and never logged a degree sweep. Use "
                "get_signed_screen_range()/get_screen_range() (pixels) instead, or "
                "check `device.has_angle_range` before calling this."
            )
        return np.linspace(self.parameters['start_deg'], self.parameters['end_deg'],
                           self.on_length)

    # ------------------------------------------------------------ receptive field

    def find_rf_position(self, cell, **cond):
        """
        Peak (receptive-field) frame of the mean trace for a condition.

        Returns
        -------
        (rf_idx, rf_value) : (int, float)
            Index of the peak frame and the mean response there.
        """
        mean_trace = self.get_all_traces(cell, mean_trace=True, **cond)[0]
        rf_idx = int(np.argmax(mean_trace))
        rf_value = float(mean_trace[rf_idx])
        return rf_idx, rf_value

    def find_rf_values(self, cell, window_size=5, **cond):
        """Per-trial mean response in a window around the RF peak frame."""
        rf_idx, _ = self.find_rf_position(cell, **cond)
        on_traces = self.get_all_traces(cell, **cond)

        window_start = max(0, rf_idx - window_size)
        window_end = min(on_traces.shape[1], rf_idx + window_size + 1)
        return np.mean(on_traces[:, window_start:window_end], axis=1)
