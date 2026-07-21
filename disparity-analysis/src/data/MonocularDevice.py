import numpy as np

from src.data.GogglesDevice import GogglesDevice


class MonocularDevice(GogglesDevice):
    """
    Data access for a monocular goggles session: one eye stimulated at a time,
    the other showing gray.

    Conditions trials on **eye**. A session may show one eye or both (in
    alternating trials — never simultaneously; that is :class:`DisparityDevice`);
    ``present_eyes`` / ``single_eye`` / ``eye`` describe what was actually
    presented. All loading, timing, trace, and geometry logic lives in
    :class:`GogglesDevice`; this subclass only adds the eye conditioning.

    Can be constructed two ways:
      1. Direct path:  MonocularDevice(data_dir='path/to/session')
      2. Session info: MonocularDevice(mouse_name='SAM045', record_date='260609',
                                       experiment_name='goggles_left')
    """

    # stim['eye_order'] codes
    EYE_CODES = {'right': 1, 'left': 2}

    # ------------------------------------------------------- conditioning hooks

    def _detect_conditions(self):
        """Determine which eye(s) were stimulated (1=right, 2=left)."""
        self.eye_order = np.asarray(self.stim['eye_order']).astype(int)

        code_to_eye = {v: k for k, v in self.EYE_CODES.items()}
        present_codes = np.unique(self.eye_order)
        self.present_eyes = [code_to_eye[int(c)] for c in present_codes if int(c) in code_to_eye]
        self.single_eye = len(self.present_eyes) == 1
        self.eye = self.present_eyes[0] if self.single_eye else None
        if self.verbose:
            print(f"Eyes present: {self.present_eyes}"
                  + (f" (single-eye: {self.eye})" if self.single_eye else " (both eyes)"))

    def get_eye_repeats(self, eye=None):
        """Trial indices for an eye ('left'/'right'), or all trials if None."""
        if eye in self.EYE_CODES:
            return np.where(self.eye_order == self.EYE_CODES[eye])[0]
        return np.arange(self.n_repeats)

    def _get_repeats(self, eye=None):
        return self.get_eye_repeats(eye)
