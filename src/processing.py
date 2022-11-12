"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Process the catalogue and save the outputs
"""
import os
from ml_collections import ConfigDict
from astropy.io import fits
import numpy as np


def extract_column(col: str, config: ConfigDict, save: bool = False, **kwargs) -> np.ndarray:
    """Extract the different columns and save them to a file in the folder data/processed/. Important columns are:

    9 columns (9 bands)
    -------------------
    - MAG_GAAP
    - MAGERR_GAAP
    - FLUX_GAAP
    - FLUXERR_GAAP
    - EXTINCTION
    - MAG_LIM

    Only one column
    ---------------
    - Z_B
    - THELI_NAME
    - MAG_AUTO
    - recal_weight

    Args:
        col (str): name of the column to combine
        config (ConfigDict): a configuration file containing the configurations
        save (bool, optional): _description_. Defaults to False.
    """
    array_complete = list()
    for cat in range(config.catnames):
        fits_file = fits.open(cat, memmap=True)
        data = fits_file[1].data

        # identify the flags in the data
        flag = data['GAAP_Flag_ugriZYJHKs']

        # find the names of the columns
        names = np.array(data.names)
        columns = names[[names[i].startswith(col) for i in range(len(names))]]

        if len(columns) > 1:
            colname = [f'{col}_{b}' for b in config.bands]
            arr = np.asarray([data[c] for c in colname]).T
        else:
            arr = data[col]
        array_complete.append(arr[flag == 0])
    if len(columns) > 1:
        array_complete = np.concatenate(array_complete, axis=0)
    else:
        array_complete = np.array(array_complete)

    if save:
        os.makedirs(config.path.processed, exist_ok=True)
        fname = kwargs.pop('fname')
        np.save(config.path.processed + fname + '.npy', array_complete)

    return array_complete


def simple_cleaning(config: ConfigDict):
    """Extract the important columns to a folder.

    Args:
        config (ConfigDict): the main configuration file.
    """
    # multiple columns (9 bands) in the catalogue
    extract_column('MAG_GAAP', config, True, fname='mag')
    extract_column('MAGERR_GAAP', config, True, fname='mag_err')
    extract_column('FLUX_GAAP', config, True, fname='flux')
    extract_column('FLUXERR_GAAP', config, True, fname='flux_err')
    extract_column('EXTINCTION', config, True, fname='ex')
    extract_column('MAG_LIM', config, True, fname='lim')

    # for these ones, we have a single column in the catalogue
    extract_column('Z_B', config, True, fname='bpz')
    extract_column('THELI_NAME', config, True, fname='name')
    extract_column('MAG_AUTO', config, True, fname='mag_0')
    extract_column('recal_weight', config, True, fname='weight')
