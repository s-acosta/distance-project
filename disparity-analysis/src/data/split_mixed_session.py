"""Split a mixed-mode goggles session into per-paradigm sessions.

``teensyStimulusAngle`` with ``opts.mode = "mixed"`` intermingles binocular
disparity trials and monocular (one-eye) trials in a single recording, and logs
both into one stimulus ``.mat``. The analysis pipeline predates that: it expects
one paradigm per session directory, because :class:`DisparityDevice` conditions
on ``trial_disparity_deg`` and :class:`MonocularDevice` conditions on
``eye_order``, and neither knows the other's trials exist.

Reading a mixed session with either device directly is **wrong**, not merely
noisy: monocular trials are logged with ``trial_disparity_deg = 0``, so
``DisparityDevice`` silently folds them into the D=0 condition (for a 7-disparity
x 15-repeat session with 15 monocular repeats per eye, D=0 would get 45 trials, of
which 30 are monocular). ``MonocularDevice`` has the mirror problem: binocular
trials carry ``eye_order = 0``, which is not a valid eye code.

:func:`split_mixed_session` writes two derived session directories that look
exactly like separately-recorded sessions, so the existing notebooks and devices
work unchanged:

    260916_SAM046_goggles/          (source, untouched)
    260916_SAM046_disparity/        stimulus/*.mat = binocular trials only
    260916_SAM046_monocular/        stimulus/*.mat = monocular trials only

Both derived sessions reference the *same* suite2p output, since the two
paradigms share one recording: trials are selected by time window, not by frame
slice, so no activity data is split or duplicated. By default ``suite2p`` is a
directory junction (no copy, no admin rights needed, and ``iscell.npy`` curation
stays shared); pass ``link_suite2p=False`` to copy it instead.
"""

import os
import shutil
import subprocess

import numpy as np
from scipy.io import loadmat, savemat

from utils.matlab_io import load_mat_struct

# stim_log.trial_kind codes written by teensyStimulusAngle
TRIAL_KIND = {'binocular': 1, 'monocular': 2}

# Per-trial fields in the stim log. Anything else whose leading dimension happens
# to equal the trial count is detected too (and reported), so fields added to the
# stimulus later are carried through without editing this list.
PER_TRIAL_FIELDS = (
    'trial_start', 'pre_start', 'stim_start', 'stim_end', 'post_end',
    'frame_times', 'lead_eye', 'eye_order', 'trial_eye', 'trial_kind',
    'trial_cond', 'trial_disparity_deg', 'trial_elevation_px',
    'left_eye_start_px', 'left_eye_end_px',
    'right_eye_start_px', 'right_eye_end_px',
)

# Session-level fields that must never be subset, even if their length collides
# with the trial count (per-condition tables, the angle sweep, the fits, ...).
SESSION_FIELDS = (
    'angles', 'disparities', 'cond_kind', 'cond_eye', 'cond_disparity_deg',
    'left_fit', 'right_fit', 'start_deg', 'end_deg', 'stim_size', 'mode',
    'parameters', 'finish_time', 'elevations',
)


def _scrub(value):
    """Make a loaded stimulus field writable by ``savemat``.

    ``clean_matlab_struct`` maps MATLAB string objects (MatlabOpaque) to None --
    here that is ``parameters.mode``, ``PORT``, ``PORT_left`` and ``scope`` --
    and ``savemat`` cannot write None. Replace with an empty string.
    """
    if value is None:
        return ''
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def _is_per_trial(name, value, n_trials):
    """Whether a stimulus field carries one entry per trial."""
    if name in SESSION_FIELDS:
        return False
    if name in PER_TRIAL_FIELDS:
        return True
    if isinstance(value, list):
        return len(value) == n_trials
    return isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_trials


def _subset(value, keep):
    """Take the trials in ``keep`` from a per-trial field (list or ndarray).

    Cell arrays (``frame_times``) come back from ``clean_matlab_struct`` as
    Python lists, not arrays, so indexing has to handle both.
    """
    if isinstance(value, list):
        return [value[i] for i in keep]
    return value[keep]


def _share_suite2p(src_suite2p, dst_suite2p, link=True, verbose=True):
    """Point a derived session at the source suite2p output.

    Returns 'junction', 'copy' or 'exists'. A junction is tried first: the two
    paradigms come from one recording, so sharing keeps a single source of truth
    for ``iscell.npy`` and avoids duplicating hundreds of MB per session.
    """
    if os.path.exists(dst_suite2p):
        return 'exists'

    if link:
        result = subprocess.run(
            ['cmd', '/c', 'mklink', '/J', dst_suite2p, src_suite2p],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return 'junction'
        if verbose:
            print(f"  ! junction failed ({result.stderr.strip() or result.stdout.strip()}); copying instead")

    shutil.copytree(src_suite2p, dst_suite2p)
    return 'copy'


def split_mixed_session(session_dir, kinds=('disparity', 'monocular'),
                        out_base=None, link_suite2p=True, overwrite=False,
                        verbose=True):
    """Write per-paradigm session directories from a mixed-mode goggles session.

    Parameters
    ----------
    session_dir : str
        The mixed session directory (contains ``stimulus/`` and ``suite2p/``).
    kinds : tuple of str
        Which paradigms to emit: 'disparity' (binocular trials) and/or
        'monocular'. The derived session is named by replacing the source
        folder's experiment suffix, e.g. ``260916_SAM046_goggles`` ->
        ``260916_SAM046_disparity``.
    out_base : str, optional
        Where to write the derived sessions (default: alongside ``session_dir``).
    link_suite2p : bool
        Share suite2p via a directory junction (default) instead of copying it.
    overwrite : bool
        Rewrite the stimulus ``.mat`` if the derived session already exists. The
        derived directory itself is never deleted -- removing a tree that
        contains a junction risks deleting the *source* suite2p through it.

    Returns
    -------
    dict
        ``{kind: derived_session_dir}`` for each kind written.
    """
    session_dir = os.path.abspath(session_dir)
    session_name = os.path.basename(os.path.normpath(session_dir))
    if out_base is None:
        out_base = os.path.dirname(session_dir)

    # ---------------------------------------------------------------- load
    stim_dir = os.path.join(session_dir, 'stimulus')
    mats = [f for f in os.listdir(stim_dir) if f.endswith('.mat')]
    if len(mats) != 1:
        raise RuntimeError(f"Expected exactly one .mat in {stim_dir}, found {sorted(mats)}")
    src_mat = os.path.join(stim_dir, mats[0])

    raw = loadmat(src_mat, squeeze_me=True)
    top_vars = [k for k in raw if not k.startswith('__')]
    if len(top_vars) != 1:
        raise RuntimeError(f"Expected one top-level variable in {src_mat}, found {top_vars}")
    top_var = top_vars[0]

    stim = load_mat_struct(src_mat)
    if 'trial_kind' not in stim:
        raise ValueError(
            f"{session_name} has no 'trial_kind' field -- it was not recorded with "
            "teensyStimulusAngle in mixed mode, so there is nothing to split. "
            "Point the device at the session directly."
        )

    trial_kind = np.asarray(stim['trial_kind']).astype(int).ravel()
    n_trials = trial_kind.size

    per_trial = [k for k, v in stim.items() if _is_per_trial(k, v, n_trials)]
    if verbose:
        mode = stim.get('mode', '?')
        print(f"Source: {session_name}  (mode={mode}, {n_trials} trials)")
        print(f"  binocular={int((trial_kind == TRIAL_KIND['binocular']).sum())} "
              f"monocular={int((trial_kind == TRIAL_KIND['monocular']).sum())}")
        print(f"  per-trial fields subset: {sorted(per_trial)}")
        extra = sorted(set(per_trial) - set(PER_TRIAL_FIELDS))
        if extra:
            print(f"  (detected by length, not in PER_TRIAL_FIELDS: {extra})")

    # ------------------------------------------------------------- split
    written = {}
    for kind in kinds:
        if kind == 'disparity':
            keep = np.flatnonzero(trial_kind == TRIAL_KIND['binocular'])
            new_mode = 'binocular'
        elif kind == 'monocular':
            keep = np.flatnonzero(trial_kind == TRIAL_KIND['monocular'])
            new_mode = 'monocular'
        else:
            raise ValueError(f"Unknown kind {kind!r}; expected 'disparity' or 'monocular'")

        if keep.size == 0:
            if verbose:
                print(f"  ! no {kind} trials in {session_name} -- skipping")
            continue

        out = {}
        for key, value in stim.items():
            out[key] = _subset(value, keep) if key in per_trial else value
            out[key] = _scrub(out[key])

        # The subset is a single-paradigm session now; say so in the log, and
        # keep only the disparities actually present in it.
        out['mode'] = new_mode
        present = np.unique(np.asarray(out['trial_disparity_deg']).ravel())
        out['disparities'] = present.astype(int)
        out['split_from'] = session_name
        out['split_trial_index'] = keep.astype(np.int32) + 1   # 1-based, for MATLAB

        new_name = f"{session_name.rsplit('_', 1)[0]}_{kind}"
        new_dir = os.path.join(out_base, new_name)
        new_stim_dir = os.path.join(new_dir, 'stimulus')
        new_mat = os.path.join(new_stim_dir, f"{new_name}_stimulus.mat")

        if os.path.exists(new_mat) and not overwrite:
            if verbose:
                print(f"  = {new_name}: stimulus exists, leaving it (overwrite=True to rewrite)")
        else:
            os.makedirs(new_stim_dir, exist_ok=True)
            savemat(new_mat, {top_var: out}, do_compression=True)
            if verbose:
                print(f"  + {new_name}: {keep.size} trials -> {os.path.basename(new_mat)}")

        how = _share_suite2p(os.path.join(session_dir, 'suite2p'),
                             os.path.join(new_dir, 'suite2p'),
                             link=link_suite2p, verbose=verbose)
        if verbose:
            print(f"    suite2p: {how}")

        written[kind] = new_dir

    return written
