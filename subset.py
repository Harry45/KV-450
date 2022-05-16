# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""
This code is used to create a subset of the data.
"""

import sys
import os
import numpy as np


def store_arrays(array: str, folder_name: str, file_name: str):
    """Given an array, folder name and file name, we will store the array in the same format for KV-450.

    Args:
        array (str): array which we want to store
        folder_name (str): the name of the folder
        file_name (str): name of the file
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savetxt(folder_name + '/' + file_name + '.asc', array)


def main(number: int):
    """This function is used to create a subset of the data.

    Args:
        number (int): The number of objects to be selected.
    """

    # interval in redshifts
    z_interval = ['0.1t0.3', '0.3t0.5', '0.5t0.7', '0.7t0.9', '0.9t1.2']

    # file name
    fname = 'Nz_DIR' + str(number) + '/Nz_DIR'+str(number) + '_Bootstrap'

    for i in range(5):

        file = fname + '/Nz_z' + z_interval[i]
        file += '_FULL_DIR' + str(number) + '.asc'

        # load the file
        samples = np.loadtxt(file)

        # samples of n(z
        sub_samples = samples[:, -1000:]

        # we need the redshift values
        redshift = samples[:, 0].reshape(samples.shape[0], 1)

        # concatenate redshift and samples
        file_save = np.concatenate([redshift, sub_samples], axis=1)

        # folder name
        folder = 'Nz_Bayes/Nz_Bayes_Bootstrap_' +str(number)

        # file name
        name_out = 'Nz_Bayes_z'+z_interval[i]

        store_arrays(file_save, folder,name_out)


if __name__ == '__main__':
    num = sys.argv[1]
    main(num)
