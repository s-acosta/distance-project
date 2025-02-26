import numpy as np
from scipy import io
from ActivityPipelines import ActivityLoader
import os

class LinearAnalysis:

    def __init__(self, session_path):

        D = ActivityLoader(session_path)
        self.stimulus_path = os.path.join(session_path, 'stimulus')
        self.activity_path = os.path.join(session_path, 'suite2p\plane0')
        self.dFF = D.dFF

        self.frames_time = np.load(self.activity_path + r'/frames_time.npy')

        container = self.load_stimulus()
        self.stepper_steps = container['stim']['stepper_pos'][0][0][0]
        self.stepper_time = container['stim']['time'][0][0][0]
        self.stim_showed = container['stim']['distance_vec'][0][0][0]

        self.distance_traveled = np.interp(self.frames_time, self.stepper_time, self.stepper_steps) / 30
        self.repeat_start_time = np.arange(0, 20 * 25, 25) * 1000
        self.repeat_end_time = self.repeat_start_time + 25000

    def load_stimulus(self):

        filelist = [file for file in os.listdir(self.stimulus_path) if file.endswith(("_stimulus.mat"))]
        if len(filelist) == 0:
            # raise error
            raise Exception('No stimulus file found')
        elif len(filelist) > 1:
            raise Exception('More than one stimulus file found')
        else:
            container = io.loadmat(self.stimulus_path + '/' + str(filelist[0]))

        return container

    def get_repeat_frames(self, repeat):

        repeat_start = self.repeat_start_time[repeat]
        repeat_end = self.repeat_end_time[repeat]

        repeat_frames = (self.frames_time > repeat_start) & (self.frames_time < repeat_end)

        return repeat_frames

    def get_repeat_size(self, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        distance_repeat = self.distance_traveled[repeat_frames]
        stimulus_radius = self.stim_showed[repeat]

        if stimulus_radius == 1:
            theta_repeat = np.degrees(np.arctan2(stimulus_radius, (distance_repeat+2)))
        else:
            theta_repeat = np.degrees(np.arctan2(stimulus_radius, (distance_repeat+10)))

        return theta_repeat

    def get_distancing_frames_within_repeat(self, repeat):

        repeat_size = self.get_repeat_size(repeat=repeat)

        # compute the derivative of repeat_size
        derivative = np.diff(repeat_size)
        derivative = np.insert(derivative, 0, 0)

        return derivative < 0

    def get_distancing_size(self, repeat):

        distancing_frames = self.get_distancing_frames_within_repeat(repeat)

        return self.get_repeat_size(repeat)[distancing_frames]

    def get_distancing_trace(self, cell, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        distancing_frames = self.get_distancing_frames_within_repeat(repeat)

        return self.dFF[cell, repeat_frames][distancing_frames]

    def get_all_distancing_traces(self, cell, stim_shown=None):

        all_traces = []

        for repeat in range(len(self.stim_showed)):

            if stim_shown is not None:
                if self.stim_showed[repeat] != stim_shown:
                    continue

            distance_trace = self.get_distancing_trace(cell, repeat)
            all_traces.append(distance_trace)

        # check all traces and make them the same size by cutting by the end
        min_size = min([len(trace) for trace in all_traces])
        all_traces = [trace[:min_size] for trace in all_traces]

        return np.array(all_traces)


    def get_stopped_frames_within_repeat(self, repeat):

        stim = self.stim_showed[repeat]

        if stim == 1:
            stopped_steps = 100
        elif stim == 2:
            stopped_steps = 20
        else:
            raise Exception('Unknown stimulus')

        repeat_frames = self.get_repeat_frames(repeat)
        repeat_steps = self.distance_traveled[repeat_frames] * 30

        stopped_frames = repeat_steps == stopped_steps

        return stopped_frames

    def get_stopped_trace(self, cell, repeat):

        repeat_frames = self.get_repeat_frames(repeat)
        stopped_frames = self.get_stopped_frames_within_repeat(repeat)

        return self.dFF[cell, repeat_frames][stopped_frames]

    def get_all_stopped_traces(self, cell, stim_shown=None):

        all_traces = []

        for repeat in range(len(self.stim_showed)):

            if stim_shown is not None:
                if self.stim_showed[repeat] != stim_shown:
                    continue

            stopped_trace = self.get_stopped_trace(cell, repeat)
            all_traces.append(stopped_trace)

        # check all traces and make them the same size by cutting by the end
        min_size = min([len(trace) for trace in all_traces])
        all_traces = [trace[:min_size] for trace in all_traces]

        return np.array(all_traces)

