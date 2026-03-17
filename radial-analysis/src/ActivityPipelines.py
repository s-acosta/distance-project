import numpy as np
import pandas as pd
import scipy.stats as stats
import os
from suite2p.extraction import dcnv

def split_sessions(joint_plane0_dir, session1_plane0_dir, session2_plane0_dir):
    """
    Split a combined suite2p plane0 directory (two sessions recorded back-to-back)
    into two separate plane0 directories at the temporal midpoint.

    Cell IDs (iscell, stat, ops) are identical in both outputs.

    Parameters
    ----------
    joint_plane0_dir : str
        Path to the combined suite2p/plane0/ directory.
    session1_plane0_dir : str
        Output path for the first session's plane0/ directory (will be created).
    session2_plane0_dir : str
        Output path for the second session's plane0/ directory (will be created).
    """
    import shutil

    os.makedirs(session1_plane0_dir, exist_ok=True)
    os.makedirs(session2_plane0_dir, exist_ok=True)

    raw_f = np.load(joint_plane0_dir + '/F.npy')
    f_neu = np.load(joint_plane0_dir + '/Fneu.npy')
    spks  = np.load(joint_plane0_dir + '/spks.npy')

    split = raw_f.shape[1] // 2

    for arr, fname in [(raw_f, 'F.npy'), (f_neu, 'Fneu.npy'), (spks, 'spks.npy')]:
        np.save(session1_plane0_dir + '/' + fname, arr[:, :split])
        np.save(session2_plane0_dir + '/' + fname, arr[:, split:])

    frames_time_path = joint_plane0_dir + '/frames_time.npy'
    if os.path.isfile(frames_time_path):
        ft = np.load(frames_time_path)
        np.save(session1_plane0_dir + '/frames_time.npy', ft[:split])
        np.save(session2_plane0_dir + '/frames_time.npy', ft[split:])

    dff_path = joint_plane0_dir + '/dFF.npy'
    if os.path.isfile(dff_path):
        dff = np.load(dff_path)
        np.save(session1_plane0_dir + '/dFF.npy', dff[:, :split])
        np.save(session2_plane0_dir + '/dFF.npy', dff[:, split:])

    for fname in ['iscell.npy', 'stat.npy', 'ops.npy', 'redcell.npy']:
        src = joint_plane0_dir + '/' + fname
        if os.path.isfile(src):
            shutil.copy2(src, session1_plane0_dir + '/' + fname)
            shutil.copy2(src, session2_plane0_dir + '/' + fname)

    n_cells = raw_f.shape[0]
    print(f'Split {n_cells} cells: {split} frames → session 1, {raw_f.shape[1] - split} frames → session 2')

class ActivityLoader:

    def __init__(self, activity_path, tau=None, skew_filter=False):
        """
        activity_path: path to the activity data
        tau: timescale of indicator
        """

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

        if tau is None:
            tau = 1.0 # timescale of indicator
        
        fs = 10.0 # sampling rate in Hz
        neucoeff = 0.7 # neuropil coefficient
        # for computing and subtracting baseline
        sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
        win_baseline = 60.0 # in seconds, window in which to compute max/min filters
        batch_size = 500

        # load traces and subtract neuropil
        F = self.raw_f
        Fneu = self.f_neu
        Fc = F - neucoeff * Fneu

        # get spikes
        spks = dcnv.oasis(F=Fc, tau=tau, batch_size=batch_size, fs=fs)

        return spks




