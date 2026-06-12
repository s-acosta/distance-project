import os
import shutil
import numpy as np
import pandas as pd
import scipy.stats as stats

def split_suite2p_recordings(joint_plane0_dir, output_dirs, frame_counts):
    """
    Split a combined suite2p plane0 directory into N separate directories,
    one per recording, based on explicit frame counts.

    Args:
        joint_plane0_dir: Path to the combined suite2p/plane0 directory
        output_dirs:      List of output directory paths, one per recording
        frame_counts:     List of ints, number of frames per recording
                          (must sum to the total number of frames in F.npy)
    """
    if len(output_dirs) != len(frame_counts):
        raise ValueError("output_dirs and frame_counts must have the same length")

    raw_f = np.load(os.path.join(joint_plane0_dir, 'F.npy'))
    total_frames = raw_f.shape[1]

    if sum(frame_counts) != total_frames:
        raise ValueError(
            f"frame_counts sum to {sum(frame_counts)} but F.npy has {total_frames} frames"
        )

    # Preload all frame-based arrays
    f_neu = np.load(os.path.join(joint_plane0_dir, 'Fneu.npy'))
    spks  = np.load(os.path.join(joint_plane0_dir, 'spks.npy'))

    frames_time_path = os.path.join(joint_plane0_dir, 'frames_time.npy')
    frames_time = np.load(frames_time_path) if os.path.isfile(frames_time_path) else None

    dff_path = os.path.join(joint_plane0_dir, 'dFF.npy')
    dff = np.load(dff_path) if os.path.isfile(dff_path) else None

    # Compute slice boundaries from cumulative frame counts
    boundaries = np.concatenate([[0], np.cumsum(frame_counts)])

    for i, out_dir in enumerate(output_dirs):
        os.makedirs(out_dir, exist_ok=True)
        start, end = boundaries[i], boundaries[i + 1]

        np.save(os.path.join(out_dir, 'F.npy'),    raw_f[:, start:end])
        np.save(os.path.join(out_dir, 'Fneu.npy'), f_neu[:, start:end])
        np.save(os.path.join(out_dir, 'spks.npy'), spks[:, start:end])

        if frames_time is not None:
            np.save(os.path.join(out_dir, 'frames_time.npy'), frames_time[start:end])

        if dff is not None:
            np.save(os.path.join(out_dir, 'dFF.npy'), dff[:, start:end])

        # Cell metadata is the same for all recordings — copy unchanged
        for fname in ['iscell.npy', 'stat.npy', 'ops.npy', 'redcell.npy']:
            src = os.path.join(joint_plane0_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_dir, fname))

        print(f"Recording {i + 1}: {frame_counts[i]} frames → {out_dir}")

    print(f"Done. {raw_f.shape[0]} cells split into {len(output_dirs)} recordings.")


class ActivityLoader:
    """
    Load suite2p activity for a single session directory.

    Reads spikes / raw F / neuropil, restricts to accepted cells (iscell), and
    pulls frame timing from a cached frames_time.npy or the Bruker T-series XML.
    Generic across experiments (radial, disparity, ...).

    Parameters
    ----------
    activity_path : str
        Session directory containing a `suite2p/plane0` folder.
    tau : float, optional
        Indicator timescale. If provided, spikes are recomputed via OASIS
        deconvolution (`compute_spike_deconvolution`).
    skew_filter : bool or float, optional
        If truthy, drop cells whose stat['skew'] is below this threshold
        (True uses the default 0.75) before loading.
    """

    def __init__(self, activity_path, tau=None, skew_filter=False):

        self.path = activity_path
        self.dFF = []
        self.spikes = []
        self.raw_f = []
        self.f_neu = []

        self.frames_time = []
        self.is_cell = []
        self.suite2p_id = []

        if skew_filter:
            self.filter_by_skewness(skew_filter)
        else:
            self.is_cell = np.load(self.path + '/suite2p/plane0/iscell.npy')

        self.import_timing_data()
        self.import_activity_data()

        if tau is not None:
            print('Computing spike deconvolution with tau = ' + str(tau))
            self.spikes = self.compute_spike_deconvolution(tau)

    def import_activity_data(self):

        self.suite2p_id = np.arange(len(self.is_cell[:, 0]))[self.is_cell[:, 0].astype(bool)]
        self.spikes = np.load(self.path + '/suite2p/plane0/spks.npy')
        self.f_neu = np.load(self.path + '/suite2p/plane0/Fneu.npy')
        self.raw_f = np.load(self.path + '/suite2p/plane0/F.npy')

        self.spikes = self.spikes[self.is_cell[:, 0].astype(bool), :]
        self.raw_f = self.raw_f[self.is_cell[:, 0].astype(bool), :]
        self.f_neu = self.f_neu[self.is_cell[:, 0].astype(bool), :]

        if os.path.isfile(self.path + '/suite2p/plane0/dFF.npy'):
            self.dFF = np.load(self.path + '/suite2p/plane0/dFF.npy')
        else:
            # self.compute_dFF()
            self.dFF = self.spikes

    def import_timing_data(self):

        # Find any XML file in the suite2p directory
        xml_files = [f for f in os.listdir(self.path + '/suite2p') if f.endswith('.xml')]
        tseries_data_filename = self.path + '/suite2p/' + xml_files[0] if xml_files else None
        frames_time_filename = self.path + '/suite2p/plane0/frames_time.npy'

        if os.path.isfile(frames_time_filename):
            self.frames_time = np.load(frames_time_filename)
        elif tseries_data_filename and os.path.isfile(tseries_data_filename):
            data_frame = pd.read_xml(tseries_data_filename, xpath=".//Frame")
            self.frames_time = data_frame['relativeTime'].values * 1000
            np.save(self.path + '/suite2p/plane0/frames_time.npy', self.frames_time)
        else:
            print('No timing data imported')

    def filter_by_skewness(self, skewness_threshold=0.75):
        """
        Filter cells by skewness of the dFF trace
        """
        stat = np.load(self.path + '/suite2p/plane0/stat.npy', allow_pickle=True)
        self.is_cell = np.load(self.path + '/suite2p/plane0/iscell.npy')

        # save the original is_cell
        original_is_cell = self.is_cell.copy()
        np.save(self.path + '/suite2p/plane0/original_iscell.npy', original_is_cell)
        initial_cell_count = np.sum(self.is_cell[:, 0].astype(bool))

        for cell in range(len(stat)):
            skew = stat[cell]['skew']
            if skew < skewness_threshold:
                self.is_cell[cell] = False

        final_cell_count = np.sum(self.is_cell[:, 0].astype(bool))
        print(f'Skewness filtering: {final_cell_count} cells accepted out of {initial_cell_count} total cells')
        np.save(self.path + '/suite2p/plane0/iscell.npy', self.is_cell)

    def compute_dFF(self):
        print('Computing dFF')

        f = self.raw_f - 0.7 * self.f_neu
        dFF = []

        for cell in range(f.shape[0]):
            print('\r Cell ' + str(cell) + ' of ' + str(f.shape[0]), end="")
            kde = stats.gaussian_kde(f[cell, :])
            x = np.linspace(np.min(f[cell, :]), np.max(f[cell, :]), 1000)
            f0 = x[np.argmax(kde(x))]
            dFF.append((f[cell, :] - f0) / f0)

        self.dFF = np.array(dFF)
        np.save(self.path + '/suite2p/plane0/dFF.npy', self.dFF)

    def compute_spike_deconvolution(self, tau=1.5):

        # Lazy import: only require the (heavy) suite2p package when deconvolution
        # is actually requested, so `import utils.suite2p` stays lightweight.
        from suite2p.extraction import dcnv

        if tau is None:
            tau = 1.0  # timescale of indicator

        fs = 10.0  # sampling rate in Hz
        neucoeff = 0.7  # neuropil coefficient
        batch_size = 500

        # load traces and subtract neuropil
        F = self.raw_f
        Fneu = self.f_neu
        Fc = F - neucoeff * Fneu

        # get spikes
        spks = dcnv.oasis(F=Fc, tau=tau, batch_size=batch_size, fs=fs)

        return spks
