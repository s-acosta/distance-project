"""Small project-wide helpers shared across experiments (radial, disparity, ...)."""
import os
import matplotlib.pyplot as plt


def create_new_result_dir(results_working_directory, new_result_dir_name):
    """
    Create a results subdirectory if it does not already exist.

    Parameters
    ----------
    results_working_directory : str
        Parent directory in which to create the results folder.
    new_result_dir_name : str
        Name of the results subdirectory.

    Returns
    -------
    str
        Full path to the (possibly pre-existing) results directory.
    """
    new_result_dir = os.path.join(results_working_directory, new_result_dir_name)
    if not os.path.exists(new_result_dir):
        print('Creating directory: ' + new_result_dir_name)
        os.makedirs(new_result_dir)
    else:
        print('Directory ' + new_result_dir_name + ' already exists')

    return new_result_dir


def setup_plot_style():
    """Apply the project matplotlib style and return the shared color palette."""
    plt.rcdefaults()
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 22,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    colors = ['orange', '#35b779', '#31688e', '#440154', 'gray']
    return colors
