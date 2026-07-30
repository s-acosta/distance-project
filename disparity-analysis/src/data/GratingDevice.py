import numpy as np

from src.data.GogglesDevice import GogglesDevice


class GratingDevice(GogglesDevice):
    """
    Data access for a binocular **phase-disparity drifting grating** session
    (``teensyStimulusGrating``).

    Both eyes drift the same full-field vertical sinusoidal grating; the left eye
    carries a static per-trial **phase** offset set by a **disparity** (deg).
    Trials are conditioned on disparity, exactly like :class:`DisparityDevice`,
    and all loading / timing / trace logic is inherited from
    :class:`GogglesDevice`.

    Unlike the bar paradigms, the grating is **full-field and drifts in time**:
    there is no spatial sweep and therefore no receptive-field *position*. The
    stimulus log has no ``start_deg``/``end_deg`` and no ``left_fit``/``right_fit``
    (only a scalar ``left_slope``/``right_slope``). The sweep-geometry methods
    that assume a spatial sweep are overridden to raise ``NotImplementedError`` so
    they cannot be called by mistake; condition on disparity and read the temporal
    response instead (``get_all_traces(cell, disparity=...)``).

    Attributes
    ----------
    disparities : list of int
        Sorted unique disparities present (e.g. [-10, 0, 10]).
    n_disparities : int
    trial_disparity_deg : ndarray
        Per-trial left-eye disparity in degrees.
    trial_phase_offset_deg : ndarray or None
        Per-trial left-eye phase offset in degrees, if logged.
    stim_size, temp_freq
        Grating pixel period and drift frequency, carried for reference / plots.
    """

    _NO_GEOMETRY = (
        "GratingDevice presents a full-field drifting grating: it has no spatial "
        "sweep and no receptive-field position, so '{name}' is undefined. Condition "
        "on disparity instead, e.g. get_all_traces(cell, disparity=...)."
    )

    # ------------------------------------------------------- conditioning hooks

    def _detect_conditions(self):
        """Determine which disparities were shown, from ``trial_disparity_deg``."""
        if self.stim.get('mode') == 'monocular' or 'eye_order' in self.stim:
            print("[warning] this looks like a MONOCULAR grating session "
                  "(mode='monocular' / eye_order present). GratingDevice handles "
                  "binocular phase-disparity gratings only; monocular grating "
                  "support is not implemented. Proceeding as a single disparity "
                  "condition.")

        self.trial_disparity_deg = np.asarray(self.stim['trial_disparity_deg']).astype(int)
        self.disparities = sorted(int(d) for d in np.unique(self.trial_disparity_deg))
        self.n_disparities = len(self.disparities)

        # grating parameters carried for reference / tuning plots (not used for loading)
        self.trial_phase_offset_deg = (
            np.asarray(self.stim['trial_phase_offset_deg']).astype(float)
            if 'trial_phase_offset_deg' in self.stim else None
        )
        self.stim_size = self.stim.get('stim_size')
        self.temp_freq = self.stim.get('temp_freq')

        if self.verbose:
            print(f"Disparities present (deg): {self.disparities}")

    def get_disparity_repeats(self, disparity=None):
        """Trial indices for a disparity (deg), or all trials if None."""
        if disparity is None:
            return np.arange(self.n_repeats)
        return np.where(self.trial_disparity_deg == disparity)[0]

    def _get_repeats(self, disparity=None):
        return self.get_disparity_repeats(disparity)

    # --------------------------------------------- spatial geometry: not defined

    def get_angle_range(self, *args, **kwargs):
        raise NotImplementedError(self._NO_GEOMETRY.format(name='get_angle_range'))

    def get_screen_positions(self, *args, **kwargs):
        raise NotImplementedError(self._NO_GEOMETRY.format(name='get_screen_positions'))

    def get_screen_range(self, *args, **kwargs):
        raise NotImplementedError(self._NO_GEOMETRY.format(name='get_screen_range'))

    def get_signed_screen_range(self, *args, **kwargs):
        raise NotImplementedError(self._NO_GEOMETRY.format(name='get_signed_screen_range'))
