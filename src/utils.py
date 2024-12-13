## add some documentation: i am going to put here all the useful functions for the whole project
import os
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