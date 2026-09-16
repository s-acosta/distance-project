"""
Run OMSI spike inference on a suite2p session and write the result as sibling
session folders the analysis pipelines can load unmodified.

OMSI (Optimized MCMC Spike Inference, https://github.com/dylanmmartins/OMSI) is a
continuous-time MCMC spike sampler that replaces suite2p's OASIS deconvolution.
It returns spike *times*, not a continuous trace, so two representations are
written:

    <session>_newspikes       spks.npy = spike counts per frame
    <session>_newspikes_prob  spks.npy = per-frame posterior spike probability

Both folders are complete session folders (suite2p/plane0 + stimulus), copied
from the source, so the only thing that differs between them and the original is
spks.npy.

OMSI lives in its own conda env (`spikeinf`) because it needs ray, which the
notebook env does not have. Run this module from that interpreter.
"""

import glob
import os
import shutil
import time

# The spikeinf env ships conda-forge MKL 2026.1, whose default (OpenMP)
# threading layer fails to resolve at runtime and takes the process down with a
# delay-load fault (0xc06d007f) on the first BLAS call. Sequential MKL loads
# fine and is what we want anyway: OMSI pins every Ray worker to one thread, so
# threaded BLAS would only oversubscribe the cores. Set before numpy is
# imported so MKL sees it when it initialises.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np

PLANE = os.path.join('suite2p', 'plane0')

# Files copied verbatim from the source session's plane0 directory. spks.npy is
# deliberately absent -- that is the one array we replace.
PLANE_FILES = ['F.npy', 'Fneu.npy', 'iscell.npy', 'ops.npy', 'stat.npy',
               'frames_time.npy']
REQUIRED_PLANE_FILES = ['F.npy', 'Fneu.npy', 'iscell.npy', 'ops.npy']

DEFAULT_RAY_DIR = r'C:\ray_tmp'

# Indicator rise time constant, seconds. Same value the OMSI paper figures use
# when pinning kinetics (figures/figure4.py).
TAU_RISE = 0.045


def configure_ray_dir(ray_dir=None):
    """Point OMSI's Ray scratch directory at a known path.

    OMSI asks for this with a tkinter dialog the first time it runs and caches
    the answer in internals.yaml. That dialog blocks a headless or scripted run,
    so write the config ourselves if the key is not set yet.

    Note the config file lives next to the package (``<clone>/src``), not at the
    repo root, because OMSI derives it from the module path.

    Parameters
    ----------
    ray_dir : str, optional
        Directory for Ray temporary files. A short path on a local SSD works
        best on Windows. Defaults to DEFAULT_RAY_DIR.

    Returns
    -------
    str
        The ray_dir now configured.
    """

    from OMSI import _config

    ray_dir = ray_dir or DEFAULT_RAY_DIR
    existing = _config._load()

    if existing.get('ray_dir', '').strip():
        return existing['ray_dir'].strip()

    os.makedirs(ray_dir, exist_ok=True)
    existing['ray_dir'] = ray_dir
    _config._save(existing)
    print(f'[omsi] ray_dir = {ray_dir}  (written to {_config._CONFIG_FILE})')

    return ray_dir


def build_params(fs, tau=None, spike_method='map'):
    """Build the OMSI sampler parameter dict.

    Parameters
    ----------
    fs : float
        Frame rate in Hz.
    tau : float, optional
        Indicator decay time constant in seconds (suite2p's ops['tau']). When
        given, the AR(2) kernel is pinned to it instead of being estimated per
        cell.
    spike_method : str
        'map', 'last' or 'prob'. OMSI's code defaults to 'map' while its README
        claims 'last', so always pass this explicitly.

    Returns
    -------
    dict
        Parameters for OMSI.deconv. 'f' is filled in by deconv_from_array.
    """

    params = {'spike_method': spike_method, 'p': 2, 'auto_stop': True}

    if tau is None:
        return params

    g_rise = float(np.exp(-1.0 / (TAU_RISE * fs)))
    g_decay = float(np.exp(-1.0 / (tau * fs)))

    params.update({
        'g': [g_rise + g_decay, -g_rise * g_decay],
        'defg': [g_rise, g_decay],
        'TauStd': [TAU_RISE * fs, tau * fs],
        'upd_gam': 1,
        'marg': 1,
        'con_lam': False,
    })

    return params


def spikes_to_counts(spike_times, n_frames, fs):
    """Bin per-cell spike times (seconds) into a counts-per-frame array.

    OMSI has its own version of this but ``deconv`` calls it with a leaked loop
    variable, so the 'spike_train' it returns holds only the last cell's spikes.
    Build the array here from the per-cell 'spikes' object array, which is
    correct.

    Parameters
    ----------
    spike_times : sequence of np.ndarray
        Per-cell spike times in seconds, one array per cell.
    n_frames : int
        Number of imaging frames.
    fs : float
        Frame rate in Hz.

    Returns
    -------
    np.ndarray
        (n_cells, n_frames) float32 spike counts.
    """

    counts = np.zeros((len(spike_times), n_frames), dtype=np.float32)

    for i, spikes in enumerate(spike_times):
        spikes = np.asarray(spikes, dtype=float).ravel()
        spikes = spikes[np.isfinite(spikes)]
        if spikes.size == 0:
            continue
        frames = np.clip(np.round(spikes * fs).astype(int), 0, n_frames - 1)
        np.add.at(counts[i], frames, 1)

    return counts


def _scatter_to_rois(cell_array, mask, n_rois):
    """Place a cells-only array back into full ROI order.

    ActivityLoader loads the whole spks.npy and *then* subsets it by iscell, so
    a file with only the accepted cells would be silently misaligned. Rejected
    ROIs get zero rows; they are dropped before any analysis sees them.

    Parameters
    ----------
    cell_array : np.ndarray
        (n_cells, n_frames) array covering only ROIs where iscell is 1.
    mask : np.ndarray
        Boolean iscell[:, 0] mask, length n_rois.
    n_rois : int
        Total number of suite2p ROIs.

    Returns
    -------
    np.ndarray
        (n_rois, n_frames) float32 array.
    """

    full = np.zeros((n_rois, cell_array.shape[1]), dtype=np.float32)
    full[mask] = cell_array.astype(np.float32)

    return full


def _copy_session_skeleton(src_dir, dst_dir, overwrite=False):
    """Copy everything a session folder needs except spks.npy.

    dFF.npy is deliberately not written: with it absent ActivityLoader falls
    back to ``dFF = spks``, which is what the source session already does.

    Parameters
    ----------
    src_dir : str
        Source session directory.
    dst_dir : str
        Destination session directory, created if needed.
    overwrite : bool
        Allow writing into a destination that already exists.
    """

    if os.path.isdir(dst_dir) and not overwrite:
        raise FileExistsError(
            f'{dst_dir} already exists. Pass overwrite=True to rebuild it.'
        )

    dst_plane = os.path.join(dst_dir, PLANE)
    os.makedirs(dst_plane, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'stimulus'), exist_ok=True)

    src_plane = os.path.join(src_dir, PLANE)
    for fname in PLANE_FILES:
        src_file = os.path.join(src_plane, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(dst_plane, fname))

    # Timing XML sits one level up, beside plane0. ActivityLoader parses it only
    # when frames_time.npy is missing, but copy it so the folder is standalone.
    for xml in glob.glob(os.path.join(src_dir, 'suite2p', '*.xml')):
        shutil.copy2(xml, os.path.join(dst_dir, 'suite2p', os.path.basename(xml)))

    stim_files = glob.glob(os.path.join(src_dir, 'stimulus', '*_stimulus.mat'))
    if len(stim_files) != 1:
        raise FileNotFoundError(
            f'Expected exactly one *_stimulus.mat in {src_dir}/stimulus, '
            f'found {len(stim_files)}.'
        )
    shutil.copy2(stim_files[0],
                 os.path.join(dst_dir, 'stimulus', os.path.basename(stim_files[0])))


def build_omsi_sessions(src_session_dir, counts_dir=None, prob_dir=None,
                        hz=None, f_corr=0.7, tau=None, spike_method='map',
                        ray_dir=None, overwrite=False, max_cells=None):
    """Run OMSI on a suite2p session and write two new session folders.

    The source session is only read, never modified.

    Parameters
    ----------
    src_session_dir : str
        Session directory holding suite2p/plane0/ and stimulus/.
    counts_dir : str, optional
        Destination whose spks.npy holds spike counts per frame. Defaults to
        ``<src_session_dir>_newspikes``.
    prob_dir : str, optional
        Destination whose spks.npy holds the posterior probability trace.
        Defaults to ``<src_session_dir>_newspikes_prob``. Pass None explicitly
        via skip_prob to omit it.
    hz : float, optional
        Frame rate. Read from ops.npy when omitted.
    f_corr : float
        Neuropil correction coefficient used when OMSI computes dF/F.
    tau : float, optional
        Indicator decay constant in seconds. Pins the kernel when given.
    spike_method : str
        OMSI spike extraction method: 'map', 'last' or 'prob'.
    ray_dir : str, optional
        Ray scratch directory.
    overwrite : bool
        Rebuild destinations that already exist.
    max_cells : int, optional
        Debug aid: run only the first N accepted cells. The remaining cells get
        zero rows, so the output is not usable for analysis.

    Returns
    -------
    dict
        Summary with the destination paths and per-cell spike statistics.
    """

    import OMSI

    src_session_dir = os.path.abspath(src_session_dir)
    if counts_dir is None:
        counts_dir = src_session_dir + '_newspikes'
    if prob_dir is None:
        prob_dir = src_session_dir + '_newspikes_prob'

    src_plane = os.path.join(src_session_dir, PLANE)
    for fname in REQUIRED_PLANE_FILES:
        path = os.path.join(src_plane, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Required suite2p file not found: {path}')

    dst_dirs = [d for d in (counts_dir, prob_dir) if d]
    for dst in dst_dirs:
        if os.path.isdir(dst) and not overwrite:
            raise FileExistsError(
                f'{dst} already exists. Pass --overwrite to rebuild it.'
            )

    iscell = np.load(os.path.join(src_plane, 'iscell.npy'))
    mask = iscell[:, 0].astype(bool)
    cell_ids = np.arange(len(mask))[mask]

    raw_f = np.load(os.path.join(src_plane, 'F.npy'), mmap_mode='r')
    n_rois, n_frames = raw_f.shape

    ops = np.load(os.path.join(src_plane, 'ops.npy'), allow_pickle=True).item()
    fs = float(hz) if hz else float(ops.get('fs', ops.get('fs2', 0)))
    if fs <= 0:
        raise ValueError(f'Bad frame rate {fs}. Pass hz explicitly.')

    print(f'[omsi] source      : {src_session_dir}')
    print(f'[omsi] rois        : {n_rois}  cells: {mask.sum()}  frames: {n_frames}')
    print(f'[omsi] frame rate  : {fs} Hz')
    print(f'[omsi] spike method: {spike_method}   tau: {tau if tau else "estimated per cell"}')

    configure_ray_dir(ray_dir)

    # Ray spawns the raylet, and the raylet spawns workers, both inheriting this
    # process's environment -- so the MKL threading layer set at import time
    # reaches the workers that actually run the sampler.
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

    for dst in dst_dirs:
        _copy_session_skeleton(src_session_dir, dst, overwrite=overwrite)
        print(f'[omsi] session skeleton copied to {dst}')

    params = build_params(fs, tau=tau, spike_method=spike_method)

    # OMSI writes spike_inference.npz into outdir; without it the file would land
    # in the *source* plane0 directory.
    omsi_outdir = os.path.join(dst_dirs[0], PLANE)

    start = time.time()
    if max_cells is None:
        results = OMSI.deconv_from_suite2p(
            src_session_dir, hz=fs, f_corr=f_corr, cells_only=True,
            outdir=omsi_outdir, params=params,
        )['plane0']
    else:
        # Debug path: deconv_from_suite2p has no cell limit, so slice F/Fneu here.
        print(f'[omsi] DEBUG: running only the first {max_cells} cells')
        f_cells = np.asarray(raw_f)[mask][:max_cells]
        fneu_cells = np.load(os.path.join(src_plane, 'Fneu.npy'),
                             mmap_mode='r')[mask][:max_cells]
        results = OMSI.deconv_from_array(
            f=np.asarray(f_cells, dtype=np.float32),
            fneu=np.asarray(fneu_cells, dtype=np.float32),
            hz=fs, f_corr=f_corr, outdir=omsi_outdir, params=params,
        )
    elapsed = time.time() - start

    spike_times = list(results['spikes'])
    n_spikes = np.array([np.isfinite(np.asarray(s, dtype=float)).sum()
                         for s in spike_times], dtype=np.int32)

    counts_cells = spikes_to_counts(spike_times, n_frames, fs)
    prob_cells = np.asarray(results['prob_trace'], dtype=np.float32)

    if max_cells is not None:
        # Pad the debug subset back out so the scatter step still lines up.
        pad = mask.sum() - counts_cells.shape[0]
        counts_cells = np.vstack([counts_cells, np.zeros((pad, n_frames), np.float32)])
        prob_cells = np.vstack([prob_cells, np.zeros((pad, n_frames), np.float32)])
        n_spikes = np.concatenate([n_spikes, np.zeros(pad, dtype=np.int32)])
        spike_times = spike_times + [np.array([])] * pad

    counts_full = _scatter_to_rois(counts_cells, mask, n_rois)
    prob_full = _scatter_to_rois(prob_cells, mask, n_rois)

    zero_spike_ids = cell_ids[n_spikes == 0]

    spike_times_obj = np.empty(len(spike_times), dtype=object)
    for i, spikes in enumerate(spike_times):
        spike_times_obj[i] = np.asarray(spikes, dtype=float)

    npz_src = os.path.join(omsi_outdir, 'spike_inference.npz')

    for dst, primary, secondary, secondary_name in [
        (counts_dir, counts_full, prob_full, 'spks_prob.npy'),
        (prob_dir, prob_full, counts_full, 'spks_counts.npy'),
    ]:
        if not dst:
            continue
        plane = os.path.join(dst, PLANE)
        np.save(os.path.join(plane, 'spks.npy'), primary)
        np.save(os.path.join(plane, secondary_name), secondary)
        np.save(os.path.join(plane, 'omsi_spike_times.npy'), spike_times_obj,
                allow_pickle=True)
        np.save(os.path.join(plane, 'omsi_cell_ids.npy'), cell_ids)
        np.save(os.path.join(plane, 'omsi_zero_spike_cell_ids.npy'), zero_spike_ids)
        if os.path.isfile(npz_src) and os.path.abspath(plane) != os.path.abspath(omsi_outdir):
            shutil.copy2(npz_src, os.path.join(plane, 'spike_inference.npz'))

    duration_min = n_frames / fs / 60.0
    rate = n_spikes / (n_frames / fs)

    print()
    print(f'[omsi] done in {elapsed / 60:.1f} min for {mask.sum()} cells '
          f'({duration_min:.1f} min of recording)')
    print(f'[omsi] spikes per cell: mean {n_spikes.mean():.1f}  '
          f'median {np.median(n_spikes):.0f}  min {n_spikes.min()}  max {n_spikes.max()}')
    print(f'[omsi] firing rate    : mean {rate.mean():.2f} Hz  '
          f'median {np.median(rate):.2f} Hz')
    print(f'[omsi] zero-spike cells: {len(zero_spike_ids)} / {mask.sum()}')
    if len(zero_spike_ids):
        preview = ', '.join(str(c) for c in zero_spike_ids[:25])
        more = ' ...' if len(zero_spike_ids) > 25 else ''
        print(f'[omsi]   cell_ids: {preview}{more}')
        print('[omsi]   these rows are all zero, so RadialDevice z-scoring makes '
              'them NaN. Filter them in the notebook.')
    for dst in dst_dirs:
        print(f'[omsi] wrote {dst}')

    return {
        'counts_dir': counts_dir,
        'prob_dir': prob_dir,
        'cell_ids': cell_ids,
        'n_spikes': n_spikes,
        'zero_spike_cell_ids': zero_spike_ids,
        'fs': fs,
        'n_rois': n_rois,
        'n_frames': n_frames,
        'elapsed_s': elapsed,
    }
