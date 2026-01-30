import numpy as np
from scipy import io
import scipy.stats as stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from src.ActivityPipelines import ActivityLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random
import os
from scipy.stats import friedmanchisquare, f_oneway
from pathlib import Path
import pickle


def create_new_result_dir(results_working_directory, new_result_dir_name):
    """
    Creates a new directory for results if it does not already exist
    :param results_working_directory:
    :param new_result_dir_name:
    :return: new_result_dir
    """
    new_result_dir = os.path.join(results_working_directory, new_result_dir_name)
    if not os.path.exists(new_result_dir):
        print('Creating directory: ' + new_result_dir_name)
        os.makedirs(new_result_dir)
    else:
        print('Directory ' + new_result_dir_name + ' already exists')

    return new_result_dir


class DistanceStimulusLoader:

    def __init__(self, distance_path):

        self.stimulus_path = distance_path + r'/stimulus'
        self.activity_path = distance_path + r'/suite2p/plane0'
        self.frames_time = []
        self.n_repeats = []
        self.distance_mat = []
        self.leg_1_start = []
        self.leg_1_end = []
        self.leg_2_start = []
        self.leg_2_end = []
        self.off_time = []
        self.encoder_time = []
        self.encoder_angle = []

        self.import_stimulus_data()
        self.interpolate_angle()

    def import_stimulus_data(self, frames_time=None, container=None):

        if frames_time is None:
            self.frames_time = np.load(self.activity_path + '/frames_time.npy')
        else:
            self.frames_time = frames_time

        if container is None:
            filelist = [file for file in os.listdir(self.stimulus_path) if file.endswith(("_stimulus.mat"))]
            if len(filelist) == 0:
                # raise error
                raise Exception('No stimulus file found')
            elif len(filelist) > 1:
                raise Exception('More than one stimulus file found')
            else:
                container = io.loadmat(self.stimulus_path + '/' + str(filelist[0]))

        # New version
        self.n_repeats = container['stim_data']['n_repeats'][0][0][0][0]
        self.distance_mat = container['stim_data']['distance'][0][0][0]
        self.leg_1_start = container['stim_data']['leg_1_start'][0][0][0] * 1000
        self.leg_1_end = np.sort(container['stim_data']['leg_1_end'][0][0][0]) * 1000
        self.off_time = np.sort(container['stim_data']['off_time'][0][0][0]) * 1000

        self.encoder_time = container['stim_data']['encoder_time'][0][0][0]
        self.encoder_angle = container['stim_data']['encoder_angle'][0][0][0]

    def interpolate_angle(self):
        # check that the length of encoder time is the same as the frames time and fix it if it is not

        self.encoder_time = self.encoder_time[1:]
        self.encoder_time = self.encoder_time - self.encoder_time[0]

        if len(self.encoder_time) != len(self.encoder_angle):
            if len(self.encoder_time) > len(self.encoder_angle):
                self.encoder_time = self.encoder_time[:len(self.encoder_angle)]
            else:
                self.encoder_angle = self.encoder_angle[:len(self.encoder_time)]

        self.encoder_angle = np.interp(self.frames_time, self.encoder_time, self.encoder_angle)


class DistanceAnalysis:

    def __init__(self, distance_path, angle_range, activity_mode='dFF', run_session=False, n_iterations=500):

        self.path = distance_path

        # Session name will be the last part of distance_path after the slash
        self.session_name = distance_path.split('\\')[-1]

        Activity = ActivityLoader(distance_path)
        self.dFF = Activity.dFF
        self.suite2p_id = Activity.suite2p_id

        self.spks = Activity.spikes
        # spks_norm = np.zeros_like(self.spks)
        # for cell, spks in enumerate(self.spks):
        #     spks_norm[cell] = spks / np.max(spks)
        # self.spks = spks_norm

        Stimulus = DistanceStimulusLoader(self.path)
        self.frames_time = Stimulus.frames_time
        self.distance_mat = Stimulus.distance_mat
        self.leg_1_start = Stimulus.leg_1_start
        self.leg_1_end = Stimulus.leg_1_end
        self.leg_2_start = Stimulus.leg_2_start
        self.leg_2_end = Stimulus.leg_2_end
        self.off_time = Stimulus.off_time
        self.angle = Stimulus.encoder_angle

        self.one_eye = []
        self.two_eye = []

        if angle_range is None:
            self.angle_range = np.arange(0, 90, 1)
        else:
            self.angle_range = angle_range

        if np.shape(self.distance_mat)[0] == 144:
            self.one_eye = np.array([np.arange(0, 36, 1), np.arange(72, 108, 1)])
            self.two_eye = np.array([np.arange(36, 72, 1), np.arange(108, 144, 1)])
        elif np.shape(self.distance_mat)[0] == 180:
            self.one_eye = np.array([np.arange(0, 45, 1), np.arange(90, 135, 1)])
            self.two_eye = np.array([np.arange(45, 90, 1), np.arange(135, 180, 1)])

        self.missed_repeats = []

        self.stats = {}
        self.activity_mode = activity_mode

        if run_session:
            self.run_session(n_iterations=n_iterations)

    def save_session(self, path=None):

        if path is None:
            path = self.path

        with open(path + '/' + self.session_name + '_distance.pkl', 'wb') as f:
            pickle.dump(self, f)

    def run_session(self, n_iterations):

        self.identify_missed_repeats()
        self.compute_reliability_all_cells(block='one eye', n_iterations=n_iterations)
        self.compute_reliability_all_cells(block='two eye', n_iterations=n_iterations)

        self.compute_distance_modulation_all_cells(block='one eye')
        self.compute_distance_modulation_all_cells(block='two eye')

        self.extract_characteristics_all()

    def get_repeat_frames(self, repeat, time_until_movement=2500):

        repeat_start = self.leg_1_start[repeat] + time_until_movement
        repeat_end = self.leg_1_end[repeat]

        off_start = self.leg_1_start[repeat]
        off_end = self.leg_1_start[repeat] + 1000

        off_frames = (self.frames_time > off_start) & (self.frames_time < off_end)
        repeat_frames = (self.frames_time > repeat_start) & (self.frames_time < repeat_end)

        return repeat_frames, off_frames

    def get_repeat_angles(self, repeat):

        repeat_frames, _ = self.get_repeat_frames(repeat)
        repeat_angles = self.angle[repeat_frames]

        return repeat_angles

    def identify_missed_repeats(self):

        self.missed_repeats = []
        for repeat in range(len(self.distance_mat)):
            if repeat in self.missed_repeats:
                continue

            repeat_angles = self.get_repeat_angles(repeat)
            # if repeat angles is empty
            if len(repeat_angles) == 0:
                self.missed_repeats.append(repeat)
                print('Repeat ' + str(repeat) + ' is missing. Repeat angle for this repeat is empty')
                continue
            if np.sum(repeat_angles - repeat_angles[0]) == 0:
                self.missed_repeats.append(repeat)
                print('Repeat ' + str(repeat) + ' is missing. Arm has not moved or has not detected movement')
            if self.leg_1_end[repeat] > self.frames_time[-1]:
                self.missed_repeats.append(repeat)
                print('2p finished earlier than expected, last one missed')

        self.distance_mat = np.delete(self.distance_mat, self.missed_repeats)
        self.leg_1_start = np.delete(self.leg_1_start, self.missed_repeats)
        self.leg_1_end = np.delete(self.leg_1_end, self.missed_repeats)

    def get_repeat_trace(self, cell, repeat, spikes_mode=True):

        repeat_frames, off_frames = self.get_repeat_frames(repeat)

        if len(repeat_frames) > np.shape(self.dFF)[1]:
            repeat_frames = repeat_frames[:np.shape(self.dFF)[1]]

        if self.activity_mode == 'spikes':
            repeat_trace = self.spks[cell, repeat_frames]
            window = 3
            repeat_trace = np.convolve(repeat_trace, np.ones(window) / window, mode='same')
        elif self.activity_mode == 'dFF':
            repeat_trace = self.dFF[cell, repeat_frames]
        else:
            print('Activity mode not recognized')
            return

        return repeat_trace

    def interpolate_repeat_trace(self, cell, repeat):

        repeat_trace = []
        repeat_angles = []

        leg_trace = self.get_repeat_trace(cell, repeat)
        leg_angles = self.get_repeat_angles(repeat)

        repeat_trace.append(np.interp(self.angle_range, leg_angles, leg_trace))
        repeat_angles.append(self.angle_range)

        repeat_trace = np.array(repeat_trace)
        repeat_angles = np.array(repeat_angles)

        return repeat_trace, repeat_angles

    def get_all_traces(self, cell, distance=None, mean_trace=False, block=None):

        all_traces = []

        if block is None:
            trace_vector = range(len(self.leg_1_start))
        elif block == 'one eye':
            trace_vector = self.one_eye.flatten()
        elif block == 'two eye':
            trace_vector = self.two_eye.flatten()
        else:
            print('Block not recognized')
            return

        for i in trace_vector:
            # if i is contained in missed_repeats then skip iteration
            if i in self.missed_repeats:
                continue
            if distance is not None:
                for dist in distance:
                    if self.distance_mat[i] != dist:
                        continue

                    repeat_trace, _ = self.interpolate_repeat_trace(cell, repeat=i)
                    all_traces.append(repeat_trace.T)

            else:

                repeat_trace, _ = self.interpolate_repeat_trace(cell, repeat=i)
                all_traces.append(repeat_trace.T)

        if mean_trace:
            all_traces_std = np.std(all_traces, axis=0)
            mean_traces = np.mean(all_traces, axis=0)
            return np.squeeze(mean_traces), np.squeeze(all_traces_std)
        else:
            return np.squeeze(np.array(all_traces))

    def compute_reliability(self, cell, n_iterations, distance=None, block=None,
                            additional_reliability=True):

        if distance is not None:
            all_traces = self.get_all_traces(cell, distance=distance, block=block)
        else:
            all_traces = self.get_all_traces(cell, distance=[1, 2, 3, 4], block=block)
            all_traces = np.squeeze(np.array(all_traces))

        n_repeats = all_traces.shape[0]
        n_halves = int(np.ceil(n_repeats / 2))
        halves_vector = np.concatenate((np.full(n_halves, True), np.full(n_halves, False)))

        corr = []
        corr_shifted = []
        for i in range(n_iterations):

            shuffled_halves = sorted(halves_vector, key=lambda k: random.random())[:n_repeats]

            first_half_traces = all_traces[shuffled_halves, :]
            second_half_traces = all_traces[np.invert(shuffled_halves), :]

            if len(first_half_traces[0]) != len(second_half_traces[0]):
                if len(first_half_traces) > len(second_half_traces):
                    first_half_traces = first_half_traces[:len(second_half_traces[0]), :]
                else:
                    second_half_traces = second_half_traces[:len(first_half_traces[0]), :]

            first_half_mean = np.mean(first_half_traces, axis=0)
            second_half_mean = np.mean(second_half_traces, axis=0)

            corr.append(stats.pearsonr(first_half_mean, second_half_mean)[0])

            first_half_shifted = []
            second_half_shifted = []
            for trace in first_half_traces:
                circular_shift = np.random.randint(-180, 180)
                first_half_shifted.append(np.roll(trace, circular_shift))
            for trace in second_half_traces:
                circular_shift = np.random.randint(-180, 180)
                second_half_shifted.append(np.roll(trace, circular_shift))

            first_half_shifted_mean = np.mean(first_half_shifted, axis=0)
            second_half_shifted_mean = np.mean(second_half_shifted, axis=0)
            corr_shifted.append(stats.pearsonr(first_half_shifted_mean, second_half_shifted_mean)[0])

        reliability = np.mean(corr)
        is_reliable = reliability > np.percentile(corr_shifted, 99)

        if is_reliable and additional_reliability:
            is_active, distance = self.is_active(cell, block=block)
            is_noise_reliable = self.is_noise_reliable(cell, n_iterations=n_iterations, block=block)

            if is_noise_reliable:
                is_reliable = False
            if not is_active:
                is_reliable = False

        return is_reliable, reliability, corr_shifted, corr

    def is_active(self, cell, threshold=12.5, block=None):

        all_traces = self.get_all_traces(cell, distance=[1, 2, 3, 4], mean_trace=False, block=block)
        all_traces = gaussian_filter1d(all_traces, sigma=5, axis=1)

        max_location = []
        for trace in all_traces:
            max_location.append(self.angle_range[np.argmax(trace)])

        receptive_field = self.compute_receptive_field(cell, block=block)
        max_location = np.array(max_location)
        max_displacement = np.abs(max_location - self.angle_range[receptive_field])

        if np.median(max_displacement) < threshold:
            return True, np.median(max_displacement)
        else:
            return False, np.median(max_displacement)

    def is_noise_reliable(self, cell, n_iterations=50, block=None):

        is_noise_reliable, _, _, _ = self.compute_reliability(cell, n_iterations=n_iterations,
                                                              block=block, distance=[5], additional_reliability=False)
        return is_noise_reliable

    def compute_reliability_all_cells(self, n_iterations, block=None):

        is_reliable = []
        reliability = []
        for cell in range(self.dFF.shape[0]):
            print('\r Cell ' + str(cell), end="")
            is_reliable_temp, reliability_temp, _, _ = self.compute_reliability(cell, n_iterations,
                                                                                block=block)
            is_reliable.append(is_reliable_temp)
            reliability.append(reliability_temp)

        if block == 'one eye':
            self.stats['is_reliable_one_eye'] = is_reliable
            self.stats['reliability_one_eye'] = reliability

        elif block == 'two eye':
            self.stats['is_reliable_two_eye'] = is_reliable
            self.stats['reliability_two_eye'] = reliability

        return is_reliable, reliability

    def compute_receptive_field(self, cell, distance=None, block=None, return_amplitude=False):

        if distance is None:
            response = self.get_all_traces(cell, distance=[1, 2, 3, 4], block=block)
        else:
            response = self.get_all_traces(cell, distance=[distance], block=block)

        response_mean = np.mean(response, axis=0)
        response_mean = gaussian_filter1d(response_mean, 3)
        receptive_field = np.argmax(response_mean)

        if return_amplitude:
            return receptive_field, np.max(response_mean)
        else:
            return receptive_field

    def compute_values_at_RF(self, cell, block=None, general_receptive_field=None):

        distance_values = []

        for distance in range(5):

            if general_receptive_field is None:
                receptive_field = self.compute_receptive_field(cell, distance=[distance + 1], block=block)
            else:
                receptive_field = general_receptive_field

            response = self.get_all_traces(cell, distance=[distance + 1], block=block)
            response = gaussian_filter1d(response, 3, axis=1)

            # see if the range of receptive_field +5 or -5 is within the angle_range
            lower_bound = receptive_field - 5
            upper_bound = receptive_field + 5
            if receptive_field + 5 > len(self.angle_range):
                upper_bound = len(self.angle_range)
            elif receptive_field - 5 < 0:
                lower_bound = 0

            # and get the max within that range for every response
            response = np.array([np.max(response[i, lower_bound:upper_bound]) for i in range(len(response))])
            distance_values.append(response)

        return distance_values

    def compute_distance_modulation(self, cell, block=None):

        if block == 'one eye':
            is_reliable = self.stats['is_reliable_one_eye'][cell]
        elif block == 'two eye':
            is_reliable = self.stats['is_reliable_two_eye'][cell]
        else:
            is_reliable = self.compute_reliability(cell)[0]

        if not is_reliable:
            return False, 2, np.nan

        receptive_field = self.compute_receptive_field(cell, block=block)
        vals = self.compute_values_at_RF(cell, block=block, general_receptive_field=receptive_field)

        F, p = f_oneway(vals[0], vals[1], vals[2], vals[3])

        if p < 0.05:
            return True, p, F
        else:
            return False, p, F

    def compute_distance_modulation_all_cells(self, block):

        is_modulated = []
        pvalues = []
        Fvalues = []

        if block == 'one eye':
            is_reliable = self.stats['is_reliable_one_eye']
        elif block == 'two eye':
            is_reliable = self.stats['is_reliable_two_eye']
        else:
            print('Block not recognized')
            return

        for cell in range(self.dFF.shape[0]):

            if not is_reliable[cell]:
                is_modulated.append(False)
                pvalues.append(None)
                Fvalues.append(None)
                continue

            print('\r Cell ' + str(cell), end="")
            is_modulated_temp, pvalue, F = self.compute_distance_modulation(cell, block)
            is_modulated.append(is_modulated_temp)
            pvalues.append(pvalue)
            Fvalues.append(F)

        if block == 'one eye':
            self.stats['is_modulated_one_eye'] = is_modulated
            self.stats['pvalues_one_eye'] = pvalues
            self.stats['Fvalues_one_eye'] = Fvalues
        elif block == 'two eye':
            self.stats['is_modulated_two_eye'] = is_modulated
            self.stats['pvalues_two_eye'] = pvalues
            self.stats['Fvalues_two_eye'] = Fvalues

    def extract_modulation_cell(self, cell):

        is_reliable_one = self.stats['is_reliable_one_eye'][cell]
        is_reliable_two = self.stats['is_reliable_two_eye'][cell]

        if is_reliable_one or is_reliable_two:

            is_monocular_modulated = self.stats['is_modulated_one_eye'][cell]
            is_binocular_modulated = self.stats['is_modulated_two_eye'][cell]

            if is_monocular_modulated and is_binocular_modulated:
                return 'both'
            elif is_monocular_modulated:
                return 'monocular'
            elif is_binocular_modulated:
                return 'binocular'
            else:
                return 'un-modulated'

        else:
            return 'un-reliable'

    def extract_characteristics_all(self):

        type = []
        preferred_distance = []
        preferred_angle = []

        for cell in range(self.dFF.shape[0]):
            print('\r Cell ' + str(cell), end="")
            type_temp, distance_temp, angle_temp = self.extract_characteristics(cell)
            type.append(type_temp)
            preferred_distance.append(distance_temp)
            preferred_angle.append(angle_temp)

        self.stats['type'] = type
        self.stats['preferred_distance'] = preferred_distance
        self.stats['preferred_angle'] = preferred_angle

        return type, preferred_distance, preferred_angle

    def extract_characteristics(self, cell):

        type = self.extract_modulation_cell(cell)

        if type == 'un-reliable':
            receptive_field = self.compute_receptive_field(cell, block=None)
            return type, 0, self.angle_range[receptive_field]

        else:

            if type == 'monocular':
                block = 'one eye'
            elif type == 'binocular':
                block = 'two eye'
            else:
                block = None

            amplitude = []
            for d in range(1, 5):
                _, a = self.compute_receptive_field(cell=cell, distance=d, block=block, return_amplitude=True)
                amplitude.append(a)

            receptive_field = self.compute_receptive_field(cell, block=block)
            preferred_distance = np.argmax(amplitude) + 1


            return type, preferred_distance, self.angle_range[receptive_field]
