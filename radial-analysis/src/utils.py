## add some documentation: i am going to put here all the useful functions for the whole project
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

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

def setup_plot_style():
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

    # Define and return your project color palette
    colors = ['orange', '#35b779', '#31688e', '#440154', 'gray']
    return colors

def import_vars_from_matlab(container, structure_name):
    """Extract all fields from a MATLAB structure and create variables.

    Args:
        container: The loaded .mat file
        structure_name: Name of the structure (e.g., 'stim')
        locals_dict: Dictionary to store the extracted variables

    Returns:
        Dictionary containing field names and their values

    Example:
        stim_params = import_vars_from_matlab(container, 'stim', {})
    """

    locals_dict = {}

    if structure_name not in container:
        print(f"Structure {structure_name} not found")
        return locals_dict

    structure = container[structure_name]
    field_names = structure.dtype.names

    for field in field_names:
        try:
            # Standard MATLAB format from scipy.io.loadmat
            locals_dict[field] = structure[field][0][0][0]
        except (IndexError, ValueError):
            try:
                locals_dict[field] = structure[field][0][0]
            except:
                print(f"Warning: Could not extract {field}")
                locals_dict[field] = None

    return locals_dict
