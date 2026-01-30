"""
Stimulus-related functions for orientation tuning experiments.
"""
import os
from scipy import io
import numpy as np
from src.utils import import_vars_from_matlab
import matplotlib.pyplot as plt
from src.ActivityPipelines import ActivityLoader

def load_stimulus_from_matlab(stimulus_path):
    """
    Load stimulus file from matlab file.
    """

    # check files inside stimulus_path
    filelist = [file for file in os.listdir(stimulus_path) if file.endswith(("_stimulus.mat"))]
    if len(filelist) == 0:
        # raise error
        raise Exception('No stimulus file found')
    elif len(filelist) > 1:
        raise Exception('More than one stimulus file found')
    else:
        print('Stimulus file found')

    container = io.loadmat(stimulus_path + '/' + str(filelist[0]))
    stim_params = import_vars_from_matlab(container, 'stim_data')
    stim_params['list_orientations'] = np.unique(stim_params['orientation_sequence'])
    
    return stim_params

def load_responses(session_path):

    D = ActivityLoader(session_path)
    frames_time = D.frames_time

    return D, frames_time


# def get_on_frames(stim_params, repeat):
    
#     trial_start = 1000*stim_params['on_time_start'][repeat]
#     trial_end = 1000*stim_params['on_time_end'][repeat]

#     return (stim_params['frames_time'] > trial_start) & (stim_params['frames_time'] < trial_end)


# def get_off_times(stim_params, repeat):

#     trial_start = 1000*stim_params['pre_time_start'][repeat]
#     trial_end = 1000*stim_params['on_time_start'][repeat]

#     return (stim_params['frames_time'] > trial_start) & (stim_params['frames_time'] < trial_end)

# def get_trace(D, stim_params, cell, repeat, on_frames=True):

#     if on_frames:
#         frames = get_on_frames(stim_params, repeat)
#     else:
#         frames = get_off_times(stim_params, repeat)

#     trace = D.dFF[cell, frames]

#     return trace[:20]
    