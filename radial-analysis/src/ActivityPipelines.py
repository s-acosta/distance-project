import numpy as np
import pandas as pd
import scipy.stats as stats
import os
from suite2p.extraction import dcnv

# def split_activity_data(joint_data_dir, split_number, new_session_data_dir):
#     f_neu = np.load(joint_data_dir + '/Fneu.npy')
#     raw_f = np.load(joint_data_dir + '/F.npy')
#     is_cell = np.load(joint_data_dir + '/iscell.npy')
#     spikes = compute_spike_deconvolution(raw_f, f_neu, tau=1.5, normalize=True)

#     split = int(spikes.shape[1] / 2)

#     # see if file exists
#     try:
#         frames_times = np.load(joint_data_dir + '/frames_time.npy')
#         frames_times = [frames_times[:split], frames_times[split:]]
#         np.save(new_session_data_dir + '/frames_time.npy', frames_times[split_number])
#     except:
#         print('No frames time file found, computing it from xml file')

#     split_spikes = [spikes[:, :split], spikes[:, split:]]
#     split_f_neu = [f_neu[:, :split], f_neu[:, split:]]
#     split_raw_f = [raw_f[:, :split], raw_f[:, split:]]

#     np.save(new_session_data_dir + '/spks.npy', split_spikes[split_number])
#     np.save(new_session_data_dir + '/Fneu.npy', split_f_neu[split_number])
#     np.save(new_session_data_dir + '/F.npy', split_raw_f[split_number])
#     np.save(new_session_data_dir + '/iscell.npy', is_cell)

#     return spikes, split_spikes[split_number], split_f_neu[split_number], split_raw_f[split_number]

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




