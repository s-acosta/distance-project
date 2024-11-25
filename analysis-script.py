# Here goes the sessions that you want to analyze
data_dir = r'D:/data-distance-coding-project/processed-data'
imaging_filenames = [r'240802_SAM025']

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ActivityPipelines import split_activity_data, ActivityLoader
from DistanceProject import DistanceStimulusLoader, DistanceAnalysis, create_new_result_dir, \
    plot_monocular_vs_binocular, plot_monocular_vs_binocular_modulation
from DistancePlots import plot_repeats, plot_trials, plot_raw_traces, plot_trials_ordered, \
    plot_reliability_distribution, plot_trials_ordered_2, plot_boxplot
import os
from matplotlib.backends.backend_pdf import PdfPages
import random
from pathlib import Path

# Load the data
Data = []
results_directories = []
for imaging_filename in imaging_filenames:
    print('Loading ' + imaging_filename)
    imaging_dir = os.path.join(data_dir, imaging_filename)

    # create folder
    results_dir = create_new_result_dir(imaging_dir, 'results')

    D = DistanceAnalysis(imaging_dir, angle_range=np.linspace(-40, 60, 100), activity_mode='dFF')

    if np.shape(D.distance_mat)[0] == 144:
        D.one_eye = np.array([np.arange(0, 36, 1), np.arange(72, 108, 1)])
        D.two_eye = np.array([np.arange(36, 72, 1), np.arange(108, 144, 1)])
    elif np.shape(D.distance_mat)[0] == 180:
        D.one_eye = np.array([np.arange(0, 45, 1), np.arange(90, 135, 1)])
        D.two_eye = np.array([np.arange(45, 90, 1), np.arange(135, 180, 1)])

    D.identify_missed_repeats()
    Data.append(D)

    print('Plotting raw traces of ' + imaging_filename)
    plot_raw_traces(D, title=imaging_filename)
    plt.savefig(results_dir + '/raw_traces.png')
    plt.close()

    results_directories.append(results_dir)

# Computing reliability
is_reliable_one_all = []
is_reliable_two_all = []

reliability_one_all = []
reliability_two_all = []

for i, D in enumerate(Data):
    print('Computing Reliability of ', imaging_filenames[i])

    is_reliable_one, reliability_one = D.compute_reliability_all_cells(block='one eye', n_iterations=200)

    is_reliable_two, reliability_two = D.compute_reliability_all_cells(block='two eye', n_iterations=200)

    is_reliable_one_all.append(is_reliable_one)
    is_reliable_two_all.append(is_reliable_two)

    reliability_one_all.append(reliability_one)
    reliability_two_all.append(reliability_two)

    results_dir = results_directories[i]

    plot_reliability_distribution(np.concatenate(is_reliable_one_all), np.concatenate(reliability_one_all),
                                  title='All sessions one eye')
    plt.savefig(results_dir + '/reliability_one_eye.png')
    plt.close()

    plot_reliability_distribution(np.concatenate(is_reliable_two_all), np.concatenate(reliability_two_all),
                                  title='All sessions two eye')
    plt.savefig(results_dir + '/reliability_two_eye.png')
    plt.close()

# Computing distance modulation
for D in Data:
    D.compute_distance_modulation_all_cells(block='one eye')
    D.compute_distance_modulation_all_cells(block='two eye')

for i, D in enumerate(Data):

    monocular_count = 0
    binocular_count = 0
    monocular_binocular_count = 0
    not_modulated_count = 0

    reliable_cells = np.where(np.logical_or(D.stats['is_reliable_one_eye'], D.stats['is_reliable_two_eye']))[0]

    monocular_dir = create_new_result_dir(results_directories[i], 'monocular_modulated')
    binocular_dir = create_new_result_dir(results_directories[i], 'binocular_modulated')
    monocular_and_binocular_dir = create_new_result_dir(results_directories[i], 'monocular_binocular_modulated')
    not_modulated_dir = create_new_result_dir(results_directories[i], 'not_modulated')

    for cell in reliable_cells:

        is_monocular_modulated, _, _ = D.compute_distance_modulation(cell, block='one eye')
        is_binocular_modulated, _, _ = D.compute_distance_modulation(cell, block='two eye')

        if is_binocular_modulated and is_monocular_modulated:
            monocular_binocular_count += 1
            plot_trials_ordered_2(D, cell=cell, title=imaging_filenames[i])
            plt.savefig(os.path.join(monocular_and_binocular_dir, imaging_filenames[i] + f'_{cell}.png'))
            plt.close()

        elif is_monocular_modulated:
            monocular_count += 1
            plot_trials_ordered_2(D, cell=cell, title=imaging_filenames[i])
            plt.savefig(os.path.join(monocular_dir, imaging_filenames[i] + f'_{cell}.png'))
            plt.close()

        elif is_binocular_modulated:
            binocular_count += 1
            plot_trials_ordered_2(D, cell=cell, title=imaging_filenames[i])
            plt.savefig(os.path.join(binocular_dir, imaging_filenames[i] + f'_{cell}.png'))
            plt.close()

        else:
            not_modulated_count += 1
            plot_trials_ordered_2(D, cell=cell, title=imaging_filenames[i])
            plt.savefig(os.path.join(not_modulated_dir, imaging_filenames[i] + f'_{cell}.png'))
            plt.close()

    print('Monocular modulated: ', monocular_count)
    print('Binocular modulated: ', binocular_count)
    print('Monocular and Binocular modulated: ', monocular_binocular_count)
    print('Not modulated: ', not_modulated_count)

    # make pie plot of the counts
    plt.figure()
    labels = ['Monocular', 'Binocular', 'Monocular and Binocular', 'Not modulated']
    sizes = [monocular_count, binocular_count, monocular_binocular_count, not_modulated_count]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    # set title
    plt.title(D.path)
    plt.savefig(os.path.join(results_directories[i], 'modulation_pie.png'))
    plt.close()

# Plotting modulation vs RF
for i,D in enumerate(Data):

    RF = []
    monocular_rho = []
    binocular_rho = []
    monocular_F = []
    binocular_F = []

    reliable_cells = np.where(np.logical_or(D.stats['is_reliable_one_eye'], D.stats['is_reliable_two_eye']))[0]

    for cell in reliable_cells:

        if D.stats['is_reliable_one_eye'][cell] and D.stats['is_reliable_two_eye'][cell]:
            RF.append(D.angle_range[D.compute_receptive_field(cell)])
        elif D.stats['is_modulated_one_eye'][cell]:
            RF.append(D.angle_range[D.compute_receptive_field(cell, block='one eye')])
        else:
            RF.append(D.angle_range[D.compute_receptive_field(cell, block='two eye')])

        is_monocular_modulated, one_eye_rho, one_eye_F = D.compute_distance_modulation(cell, block='one eye')
        is_binocular_modulated, two_eye_rho, two_eye_F = D.compute_distance_modulation(cell, block='two eye')

        monocular_rho = np.append(monocular_rho, one_eye_rho)
        binocular_rho = np.append(binocular_rho, two_eye_rho)
        monocular_F = np.append(monocular_F, one_eye_F)
        binocular_F = np.append(binocular_F, two_eye_F)

        # get the RF of the non_modulated_cells
    non_modulated_cells = np.logical_and(monocular_rho > 0.05, binocular_rho > 0.05)
    monocular_modulated = np.logical_and(monocular_rho < 0.05, binocular_rho > 0.05)
    binocular_modulated = np.logical_and(monocular_rho > 0.05, binocular_rho < 0.05)
    monocular_binocular_modulated = np.logical_and(monocular_rho < 0.05, binocular_rho < 0.05)

    # Select the F indices where non_modulated_cells is true
    F_non_modulated = monocular_F[non_modulated_cells]
    RF_non_modulated = np.array(RF)[non_modulated_cells]

    # monocular modulated
    F_monocular = monocular_F[monocular_modulated]
    RF_monocular = np.array(RF)[monocular_modulated]

    # binocular modulated
    F_binocular = binocular_F[binocular_modulated]
    RF_binocular = np.array(RF)[binocular_modulated]

    # monocular and binocular modulated
    F_monocular_binocular = (binocular_F[monocular_binocular_modulated] + monocular_F[
        monocular_binocular_modulated]) / 2
    RF_monocular_binocular = np.array(RF)[monocular_binocular_modulated]

    plt.figure(figsize=(10, 5))

    plt.scatter(RF_non_modulated, F_non_modulated, color='gray', label='Non modulated', alpha=0.8)
    plt.scatter(RF_monocular, F_monocular, color='red', label='Monocular modulated', alpha=0.8)
    plt.scatter(RF_binocular, F_binocular, color='blue', label='Binocular modulated', alpha=0.8)
    plt.scatter(RF_monocular_binocular, F_monocular_binocular, color='purple',
                label='Monocular and Binocular modulated', alpha=0.8)
    plt.xlabel('RF position (degrees)', fontsize=16)
    plt.ylabel('F value', fontsize=16)
    plt.title(D.path)
    plt.legend()
    plt.savefig(os.path.join(results_directories[i], 'RF_vs_modulation.png'))



