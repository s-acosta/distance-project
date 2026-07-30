import numpy as np

from src.data.GogglesDevice import GogglesDevice


class DisparityDevice(GogglesDevice):
    """
    Data access for a binocular disparity goggles session (teensyStimulusBoth).

    Unlike :class:`MonocularDevice`, the bar is shown to **both eyes at once**,
    offset by a per-trial disparity, so there is one response per trial and
    trials are conditioned on **disparity** (deg), not eye. The stimulus has no
    ``eye_order``; it carries ``trial_disparity_deg`` and a ``disparities`` list.

    All loading, timing, trace, and geometry logic lives in
    :class:`GogglesDevice`; this subclass only adds the disparity conditioning.

    Attributes
    ----------
    disparities : list of int
        Sorted unique disparities present (e.g. [-20, -10, 0, 10, 20]).
    n_disparities : int
    trial_disparity_deg : ndarray
        Per-trial disparity in degrees.
    """

    # ------------------------------------------------------- conditioning hooks

    def _detect_conditions(self):
        """Determine which disparities were shown, from ``trial_disparity_deg``."""
        self.trial_disparity_deg = np.asarray(self.stim['trial_disparity_deg']).astype(int)
        self.disparities = sorted(int(d) for d in np.unique(self.trial_disparity_deg))
        self.n_disparities = len(self.disparities)
        # left_fit / right_fit are loaded by GogglesDevice._load_stimulus_variables
        if self.verbose:
            print(f"Disparities present (deg): {self.disparities}")

    def get_disparity_repeats(self, disparity=None):
        """Trial indices for a disparity (deg), or all trials if None."""
        if disparity is None:
            return np.arange(self.n_repeats)
        return np.where(self.trial_disparity_deg == disparity)[0]

    def _get_repeats(self, disparity=None):
        return self.get_disparity_repeats(disparity)

    # --------------------------------------------------------- pixel geometry

    def angle_to_px(self, angle, eye, disparity=0):
        """
        Screen pixel for a visual angle in one eye, via that eye's deg->pixel fit.

        Disparity shifts the **left** eye's angle before conversion (matching the
        stimulus: ``left_px = left_slope*(deg - disparity) + left_intercept``); the
        right eye is unaffected by disparity.
        """
        angle = np.asarray(angle, float)
        if eye == 'left':
            slope, intercept = self.left_fit
            return slope * (angle - disparity) + intercept
        elif eye == 'right':
            slope, intercept = self.right_fit
            return slope * angle + intercept
        raise ValueError("eye must be 'left' or 'right'")

    def get_px_range(self, eye, disparity=0):
        """Per-frame screen pixels for an eye across the sweep at a given disparity."""
        return self.angle_to_px(self.get_angle_range(), eye, disparity)

    def find_rf_px(self, cell, disparity):
        """
        (left_px, right_px) screen pixels where the cell's RF peaks at a disparity.

        The RF frame is found on the (disparity-conditioned) mean trace, then that
        frame's visual angle is mapped to each eye's screen pixel.
        """
        rf_idx, _ = self.find_rf_position(cell, disparity=disparity)
        angle = self.get_angle_range()[rf_idx]
        return (float(self.angle_to_px(angle, 'left', disparity)),
                float(self.angle_to_px(angle, 'right', disparity)))
