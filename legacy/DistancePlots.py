import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random
from scipy.stats import f_oneway
from scipy.ndimage import gaussian_filter1d

# set rcPArams to default
plt.rcdefaults()
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams.update({'legend.fontsize': 12})
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'figure.titlesize': 22})
colors = ['orange', '#35b779', '#31688e', '#440154', 'gray']
names = ['1', '2', '3', '4', 'control']


def plot_repeats(D, repeats):
    # First trial
    plt.figure(figsize=(14, 4))
    plt.title('Repeats')

    for n_trial in repeats:
        start = int(D.leg_1_start[n_trial])
        end = int(D.leg_1_end[n_trial])

        distance = D.distance_mat[n_trial]

        plt.axvspan(D.leg_1_start[n_trial] + 2500, D.leg_1_end[n_trial], color=colors[distance - 1], alpha=0.4,
                    label='Distance ' + str(distance))

    start = int(D.leg_1_start[repeats[0]])
    end = int(D.leg_1_end[repeats[-1]])

    all_trial_time = D.frames_time[(D.frames_time > start) & (D.frames_time < end)]
    all_trial_angle = D.angle[(D.frames_time > start) & (D.frames_time < end)]

    plt.plot(all_trial_time, all_trial_angle, color='black')

    legend_elements = [Patch(facecolor=colors[0], alpha=0.4, label='Distance 1'),
                       Patch(facecolor=colors[1], alpha=0.4, label='Distance 2'),
                       Patch(facecolor=colors[2], alpha=0.4, label='Distance 3'),
                       Patch(facecolor=colors[3], alpha=0.4, label='Distance 4'),
                       Patch(facecolor=colors[4], alpha=0.4, label='Control')
                       ]
    # add labels
    plt.legend(handles=legend_elements)
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (degrees)')


def plot_raw_traces(D, title=None):
    session_traces = []
    session_mean_traces = []

    for cell in range(len(D.spks)):
        session_traces.append(D.spks[cell])
        session_mean_traces.append(D.get_all_traces(cell, mean_trace=True)[0])

    session_traces = np.array(session_traces)
    session_mean_traces = np.array(session_mean_traces)

    limits = np.linspace(D.angle_range[0], D.angle_range[-1], 101)

    # # make a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.pcolor(session_traces, cmap='Greys', vmin=0, vmax=np.percentile(session_traces.flatten(), 99))
    ax2.pcolor(session_mean_traces, cmap='Greys', vmin=0,
               vmax=np.percentile(session_mean_traces.flatten(), 99))
    # add y axis label
    ax1.set_ylabel('Cell #', fontsize=16)
    ax1.set_xlabel('Frame #', fontsize=16)

    ax2.set_xlabel('Angle (degrees)', fontsize=16)
    ax2.set_xticklabels(['-40', '-20', '0', '20', '40', '60'])

    # Set figure title
    if title is not None:
        fig.suptitle(title, fontsize=22)
    else:
        fig.suptitle('Raw Traces', fontsize=22)


def plot_trials(D, cell, title=None):
    angles = D.angle_range

    all_traces = D.get_all_traces(cell, distance=[1, 2, 3, 4])
    vmin = np.percentile(all_traces, 1)
    vmax = np.percentile(all_traces, 99)

    spks = np.squeeze(D.get_all_traces(cell, distance=[1, 2, 3, 4]))
    spks = gaussian_filter1d(spks, sigma=3, axis=1)
    # trials_to_be_deleted = D.distance_mat == 5
    # spks = np.delete(spks, np.where(trials_to_be_deleted), axis=0)
    spks_mean = np.mean(spks, axis=0)
    print(spks.shape[0])
    spks_sem = np.std(spks, axis=0)/np.sqrt(spks.shape[0])

    fig = plt.figure(figsize=(12, 10))
    if title is not None:
        title = 'Cell ' + str(cell) + ', ' + title
    else:
        title = 'Cell ' + str(cell)

    fig.suptitle(title, fontsize=22)

    fig.subplots_adjust(top=0.90)
    fig.supxlabel('Angle (degrees)', fontsize=16)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(wspace=0.01, hspace=0.1)  # set the spacing between axes.

    # First ax
    ax0 = plt.subplot(gs[0])
    ax0.pcolor(spks, vmin=vmin, vmax=vmax)
    ax0.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelsize=14)
    ax0.set_ylabel('Repeat #', fontsize=16)


    ax1 = ax0.twinx()
    ax1.set_ylim([0, 160])
    ax1.axhline(D.one_eye[0][-1] - 5, color='white', linewidth=2, linestyle='--')
    ax1.axhline(D.one_eye[1][-1] - 15, color='white', linewidth=2, linestyle='--')
    ax1.axhline(D.two_eye[0][-1] - 10, color='white', linewidth=2, linestyle='--')
    ax1.set_yticks([20, 60, 100, 140])
    ax1.set_yticklabels(['one eye', 'two eyes', 'one eye', 'two eyes'], fontsize=14)

    # Third ax
    ax2 = plt.subplot(gs[1])
    ax2.tick_params(left=True, right=False, labelleft=True, labelbottom=True, bottom=True, labelsize=14)

    # if D.stats['is_reliable_one_eye'][cell] and D.stats['is_reliable_two_eye'][cell] exists and are non empty

    # check if D.stats is empty
    if len(D.stats) == 0:
        color = 'gray'
    else:
        if D.stats['is_reliable_one_eye'][cell] or D.stats['is_reliable_two_eye'][cell]:
            color = 'blue'
        else:
            color = 'red'

    ax2.plot(angles, spks_mean, linewidth=3, color=color)
    ax2.fill_between(angles, spks_mean - spks_sem, spks_mean + spks_sem, alpha=0.2, color=color)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_visible(True)
    if D.activity_mode == 'spikes':
        ax2.set_ylabel('inf. spks', fontsize=16)
    else:
        ax2.set_ylabel('dF/F', fontsize=16)
    ax2.set_ylim([vmin, vmax])


def plot_trials_ordered(D, cell, block=None):
    angles = D.angle_range

    all_traces = D.get_all_traces(cell)
    vmin = np.percentile(all_traces, 0.5)
    vmax = np.percentile(all_traces, 99.5)

    fig = plt.figure(figsize=(8, 11))
    if block is None:
        fig.suptitle('Cell ' + str(cell))
    elif block == 'one eye':
        fig.suptitle('Cell ' + str(cell) + ' - One eye')
    elif block == 'two eye':
        fig.suptitle('Cell ' + str(cell) + ' - Two eyes')

    fig.subplots_adjust(top=0.90)
    fig.supxlabel('Angle (degrees)', fontsize=16)
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 0.5, 1.25])
    gs.update(wspace=0.01, hspace=0.4)  # set the spacing between axes.

    for distance in range(5):

        spks = D.get_all_traces(cell, distance=distance + 1, block=block)

        ax = plt.subplot(gs[distance, 0])
        ax.pcolor(spks, vmin=vmin, vmax=vmax)
        ax.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelsize=12, left=False,
                       labelleft=False)

        # add horizontal line as divider
        if block is False:
            ax2 = ax.twinx()
            ax2.set_yticks([18, 54, 90, 126])
            # set ax2 to be from 0 to 144
            ax2.set_ylim([0, 144])
            ax2.set_yticklabels(['one eye', 'two eyes', 'one eye', 'two eyes'], fontsize=10)
            ax2.axhline(D.one_eye[0][-1], color='gray', linewidth=1, linestyle='--')
            ax2.axhline(D.one_eye[1][-1], color='gray', linewidth=1, linestyle='--')
            ax2.axhline(D.two_eye[0][-1], color='gray', linewidth=1, linestyle='--')

        if (distance + 1) == 5:
            ax.set_title('Control', fontsize=16)
        else:
            ax.set_title('Distance ' + str(distance + 1), fontsize=14, color=colors[distance])

    ax1 = plt.subplot(gs[5, 0])

    for distance in range(5):
        mean_distance, std_distance = D.get_all_traces(cell, distance=distance + 1, mean_trace=True,
                                                       block=block)

        sem = std_distance / np.sqrt(spks.shape[0])

        if (distance + 1) == 5:
            ax1.plot(angles, mean_distance, '--', linewidth=2, color=colors[distance], label='Control')
        else:
            ax1.plot(angles, mean_distance, linewidth=2, color=colors[distance],
                     label='Distance ' + str(distance + 1))

        ax1.fill_between(angles, mean_distance - sem, mean_distance + sem, color=colors[distance], alpha=0.3)

        ax1.tick_params(left=True, right=False, labelleft=True, labelbottom=True, bottom=True, labelsize=12)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.spines[['left', 'bottom']].set_visible(True)
        ax1.set_ylim(np.percentile(all_traces, 0.5), np.percentile(all_traces, 99.5))
        ax1.set_ylabel('inf. spks')
        # add legend
        ax1.legend(loc='upper right', fontsize=10)


def plot_trials_ordered_2(D, cell, title=None):
    angles = D.angle_range

    names = ['10cm', '16cm', '22cm', '28cm', 'control']

    all_traces = D.get_all_traces(cell)
    vmin = np.percentile(all_traces, 0.5)
    vmax = np.percentile(all_traces, 99.5)

    fig = plt.figure(figsize=(16, 11))

    fig.subplots_adjust(top=0.90)
    if title is None:
        fig.suptitle('Cell ' + str(cell), fontsize=22)
        # add text below title
    else:
        fig.suptitle(' Cell ' + str(cell) + ', ' + title, fontsize=22)

    fig.text(0.3, 0.94, 'One Eye', ha='center', va='center', fontsize=18, color='grey')
    fig.text(0.725, 0.94, 'Two Eyes', ha='center', va='center', fontsize=18, color='grey')
    # add subtitle
    gs = gridspec.GridSpec(7, 2, height_ratios=[1, 1, 1, 1, 0.5, 1.25, 2.5])
    gs.update(wspace=0.2, hspace=0.8)  # set the spacing between axes.

    blocks = ['one eye', 'two eye']

    for i, block in enumerate(blocks):

        for distance in range(5):

            spks = D.get_all_traces(cell, distance=[distance + 1], block=block)

            if D.activity_mode == 'spikes':
                # run a rolling window
                spks = np.squeeze(spks)
                spks = gaussian_filter1d(spks, sigma=3, axis=1)

            ax = plt.subplot(gs[distance, i])
            ax.pcolor(spks, vmin=vmin, vmax=vmax)
            ax.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelsize=12, left=False,
                           labelleft=False)

            if (distance + 1) == 5:
                ax.set_title('Control', fontsize=16)
            else:
                ax.set_title('Distance ' + str(distance + 1) + ' (' + names[distance] + ')', fontsize=14, color=colors[distance])

        ax1 = plt.subplot(gs[5, i])

        for distance in range(5):
            mean_distance, std_distance = D.get_all_traces(cell, distance=[distance + 1], mean_trace=True,
                                                           block=block)
            mean_distance = gaussian_filter1d(mean_distance, sigma=3)

            sem = std_distance / np.sqrt(spks.shape[0])

            if (distance + 1) == 5:
                ax1.plot(angles, mean_distance, '--', linewidth=2, color=colors[distance], label='Control')
            else:
                ax1.plot(angles, mean_distance, linewidth=2, color=colors[distance],
                         label='Distance ' + str(distance + 1) + ' (' + names[distance] + ')')

            ax1.fill_between(angles, mean_distance - sem, mean_distance + sem, color=colors[distance], alpha=0.3)

        if block == 'one eye':
            if D.stats['is_reliable_one_eye'][cell]:
                receptive_field = D.compute_receptive_field(cell, block=block)
                ax1.axvline(angles[receptive_field], color='blue', linestyle='--', linewidth=2, alpha=0.6)
            else:
                receptive_field = D.compute_receptive_field(cell, block='two eye')
                ax1.axvline(angles[receptive_field], color='red', linestyle='--', linewidth=2, alpha=0.6)
        elif block == 'two eye':
            if D.stats['is_reliable_two_eye'][cell]:
                receptive_field = D.compute_receptive_field(cell, block=block)
                ax1.axvline(angles[receptive_field], color='blue', linestyle='--', linewidth=2, alpha=0.6)
            else:
                receptive_field = D.compute_receptive_field(cell, block='one eye')
                ax1.axvline(angles[receptive_field], color='red', linestyle='--', linewidth=2, alpha=0.6)

        ax1.tick_params(left=True, right=False, labelleft=True, labelbottom=True, bottom=True, labelsize=12)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.spines[['left', 'bottom']].set_visible(True)
        ax1.set_ylim(np.percentile(all_traces, 0.5), np.percentile(all_traces, 99.5))
        if D.activity_mode == 'spikes':
            ax1.set_ylabel('inf. spks', fontsize=16)
        else:
            ax1.set_ylabel('dF/F')
        ax1.set_xlabel('Angle (degrees)', fontsize=16)

        # set more distance between ax1 and ax2
        ax2 = plt.subplot(gs[6, i])
        # set ax2 to start lower in the overall figure
        receptive_field = D.compute_receptive_field(cell, block=block)
        vals = D.compute_values_at_RF(cell=cell, block=block, general_receptive_field=receptive_field)

        F, p = f_oneway(vals[0], vals[1], vals[2], vals[3])

        # min_len = min([len(val) for val in vals])
        # vals = [val[:min_len] for val in vals]
        # res = friedmanchisquare(vals[0], vals[1], vals[2], vals[3])

        box = ax2.boxplot(vals, labels=names, medianprops=dict(color="black", linewidth=1.5))

        for distance, box in enumerate(box['boxes']):
            box.set(color=colors[distance], linewidth=2)

        for distance in range(5):
            y = vals[distance]
            x = np.random.normal(distance + 1, 0.04, y.shape[0])
            ax2.scatter(x, y, alpha=0.6, color=colors[distance])

        # set ylimits
        ax2.set_ylim(vmin + 0.5 * vmin, vmax + 0.5 * vmax)
        # set text below xlabel iwht p value and F statistic
        ax2.set_xlabel("F = " + str(round(F, 2)) + " , p = " + str(round(p, 4)),
                       fontsize=16)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_ylabel('inf. spks', fontsize=16)
        # set names for the xticks
        ax2.set_xticklabels(names, fontsize=12)
        # set ticks to fontsize 12
        ax2.tick_params(axis='both', which='major', labelsize=12)



    import matplotlib as mpl
    cmap = 'viridis'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create a ScalarMappable and set the colormap
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # # Create a figure and axis for the colorbar
    # fig, ax = plt.subplots()  # Adjust the figsize to make it long and thin
    #
    # # Create the colorbar
    # cbar = fig.colorbar(sm, cax=ax, orientation='horizontal', ticks=np.linspace(vmin, vmax, 500))
    #
    # # set only 5 ticks
    # cbar.set_ticks(np.linspace(vmin, vmax, 2))
    # # set fontsize of the ticks to be 14
    # cbar.ax.tick_params(labelsize=30)
    #
    # #set title of the colorbar
    # cbar.set_label('inf. spks', fontsize=30, labelpad=10)
    # # set label above the colorbar
    # cbar.ax.xaxis.set_label_position('top')
    #
    # # Set the aspect ratio
    # cbar.ax.set_aspect(70)  # Adjust the aspect ratio as needed
    # plt.show()


def plot_reliability_distribution(is_reliable, reliability, title=None):
    percentage_reliable = 100 * np.sum(is_reliable) / len(is_reliable)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for cell, corr in enumerate(reliability):
        if not is_reliable[cell]:
            color = 'red'
        else:
            color = 'blue'

        ax.scatter(corr, random.uniform(-.1, .1), color=color, alpha=0.6)

    ax.set_ylim(-1, 1)
    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Reliable' + r' (' + str(np.sum(is_reliable)) + r')'),
                       Patch(facecolor='red', alpha=0.6,
                             label='Not reliable' + r' (' + str(np.sum(np.invert(is_reliable))) + r')'),
                       Line2D([0], [0], color='gray', lw=3, ls='--', alpha=0.6, label='Population Mean')]
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.axvline(np.mean(reliability), ls='--', alpha=0.6, color='gray', linewidth=3)
    ax.set_yticks([])
    plt.xticks(fontsize=12)

    ax.legend(handles=legend_elements, loc='upper left')

    if title is None:
        ax.set_title('Reliability: ' + str(round(percentage_reliable, 2)) + '% of cells are reliable', fontsize=16)
    else:
        ax.set_title(title + ' Reliability: ' + str(round(percentage_reliable, 2)), fontsize=16)

    ax.set_xlabel('Reliability', fontsize=14)


def plot_boxplot(D, cell, vals=None):
    all_traces = D.get_all_traces(cell, spikes_mode=True)
    vmin = np.percentile(all_traces, 0.1) - 5
    vmax = np.percentile(all_traces, 99.9) + 5

    from scipy.stats import friedmanchisquare
    colors = ['orange', '#35b779', '#31688e', '#440154', 'gray']
    names = ['1', '2', '3', '4', 'control']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(f'Cell {cell}', fontsize=18)

    ax = axes[0]
    if vals is None:
        vals = D.compute_values_at_RF(cell=cell, spikes_mode=True, block='one eye')

    res = friedmanchisquare(vals[0], vals[1], vals[2], vals[3])
    box = ax.boxplot(vals, labels=names, medianprops=dict(color="black", linewidth=1.5))

    for distance, box in enumerate(box['boxes']):
        box.set(color=colors[distance], linewidth=2)

    for distance in range(5):
        y = vals[distance]
        x = np.random.normal(distance + 1, 0.04, y.shape[0])
        ax.scatter(x, y, alpha=0.6, color=colors[distance])

    # set ylimits
    ax.set_ylim(vmin, vmax)

    ax.set_title("F = " + str(round(res.statistic, 2)) + " , p = " + str(round(res.pvalue, 4)),
                 fontsize=16)

    ax = axes[1]
    vals = D.compute_values_at_RF(cell=cell, spikes_mode=True, block='two eye')
    res = friedmanchisquare(vals[0], vals[1], vals[2], vals[3])
    box = ax.boxplot(vals, labels=names, medianprops=dict(color="black", linewidth=1.5))

    for distance, box in enumerate(box['boxes']):
        box.set(color=colors[distance], linewidth=2)

    for distance in range(5):
        y = vals[distance]
        x = np.random.normal(distance + 1, 0.04, y.shape[0])
        ax.scatter(x, y, alpha=0.6, color=colors[distance])

    ax.set_ylim(vmin, vmax)

    ax.set_title("F = " + str(round(res.statistic, 2)) + " , p = " + str(round(res.pvalue, 4)),
                 fontsize=16)


def plot_RF_vs_modulation(D):
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
    plt.axvline(-20, color='gray', linestyle='--')
    plt.axvline(20, color='gray', linestyle='--')
    plt.title(D.session_name)
    plt.legend()


def plot_cell_reliability(D, cell, block, n_iterations=200):
    """ Plot the reliability of a cell and the distance of the max response from the RF"""

    is_reliable, reliability, corr_shifted, corr_real = D.compute_reliability(cell=cell, n_iterations=n_iterations, block=block,
                                                                   additional_reliability=False)

    receptive_field = D.compute_receptive_field(cell, block=block)

    all_trials = D.get_all_traces(cell=cell, distance=[1, 2, 3, 4], block=block)
    all_trials = np.squeeze(np.array(all_trials))
    all_trials = gaussian_filter1d(all_trials, sigma=5, axis=1)

    plt.subplots(1, 3, figsize=(20, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # add more space between subplots
    plt.subplots_adjust(wspace=0.3)

    ax1.pcolor(all_trials)
    ax1.set_title('Cell ' + str(cell) + ' ' + block)
    # remove ticks and tick labels FOR BOTH AXES
    ax1.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelsize=12)


    ax2.hist(corr_shifted, bins=np.linspace(-1, 1, 50), color='coral', alpha=0.8, label='Circulary Shuffled')
    ax2.hist(corr_real, bins=np.linspace(-1, 1, 50), color='skyblue', alpha=0.8, label='Unshuffled')

    ax2.axvline(np.mean(corr_shifted), color='red', linestyle='--')
    ax2.axvline(np.mean(corr_real), color='blue', linestyle='--')

    # add axis labels
    ax2.set_xlabel('Correlation', fontsize=16)
    ax2.set_ylabel('Frequency', fontsize=16)

    # compute cohen d
    # compute cohen d for these two distributions
    mean_diff = np.mean(corr_real) - np.mean(corr_shifted)
    std_diff = np.sqrt((np.std(corr_real) ** 2 + np.std(corr_shifted) ** 2) / 2)
    cohen_d = mean_diff / std_diff

    ax2.set_title(r'Between Trials CC')

    distance = []
    for trial in all_trials:
        # find the index where there is the max
        max_pos = D.angle_range[np.argmax(trial)]
        dist = abs(max_pos - D.angle_range[receptive_field])
        distance.append(dist)

        if abs(max_pos - D.angle_range[receptive_field]) < 10:
            ax3.plot(D.angle_range, trial, color='skyblue')
        else:
            ax3.plot(D.angle_range, trial, color='orange')

    # add manual legend with skyblue being consistent and orange unconsistent
    legend_elements = [Line2D([0], [0], color='skyblue', lw=2, label='RF Consistent'),
                       Line2D([0], [0], color='orange', lw=2, label='RF Inconsistent')]
    ax3.legend(handles=legend_elements, loc='upper left')

    median_distance = np.median(distance)
    if median_distance < 10:
        ax3.axvline(D.angle_range[receptive_field], color='blue',)
    else:
        ax3.axvline(D.angle_range[receptive_field], color='red')

    ax3.axvline(D.angle_range[receptive_field] - 10, color='grey', linestyle='--')
    ax3.axvline(D.angle_range[receptive_field] + 10, color='grey', linestyle='--')

    ax3.set_xlabel('Angle (degrees)', fontsize=16)
    ax3.set_ylabel('inf. spks', fontsize=16)

    # set title for ax3
    ax3.set_title('Receptive Field consistency')

    # set legend
    ax2.legend()





