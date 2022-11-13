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


def extract_column_bands(col: str, config: ConfigDict, save: bool = False, **kwargs) -> np.ndarray:
    """Extract the different columns and save them to a file in the folder data/processed/. Important columns are:

    9 columns (9 bands)
    -------------------
    - MAG_GAAP
    - MAGERR_GAAP
    - FLUX_GAAP
    - FLUXERR_GAAP
    - EXTINCTION
    - MAG_LIM

    Args:
        col (str): name of the column to combine
        config (ConfigDict): a configuration file containing the configurations
        save (bool, optional): save the file to output. Defaults to False.
    Returns:
        np.ndarray: the array of values
    """
    colname = [f'{col}_{b}' for b in config.bands]
    array_complete = list()
    for cat in config.catnames:
        fits_file = fits.open(config.path.catalogue + cat, memmap=True)
        data = fits_file[1].data

        flag = data['GAAP_Flag_ugriZYJHKs']
        arr = np.asarray([data[c] for c in colname]).T
        array_complete.append(arr[flag == 0])
    array_complete = np.concatenate(array_complete, axis=0)

    if save:
        os.makedirs(config.path.processed, exist_ok=True)
        fname = kwargs.pop('fname')
        np.save(config.path.processed + fname + '.npy', array_complete)

    return array_complete


def extract_column_single(col: str, config: ConfigDict, save: bool = False, **kwargs) -> np.ndarray:
    """_summary_

    Only one column
    ---------------
    - Z_B
    - THELI_NAME
    - MAG_AUTO
    - recal_weight

    Args:
        col (str): name of the column to combine
        config (ConfigDict): a configuration file containing the configurations
        save (bool, optional): save the file to output. Defaults to False.
    Returns:
        np.ndarray: the array of values
    """

    array_complete = list()
    for cat in config.catnames:
        fits_file = fits.open(config.path.catalogue + cat, memmap=True)
        data = fits_file[1].data

        flag = data['GAAP_Flag_ugriZYJHKs']
        arr = data[col]
        array_complete.append(arr[flag == 0])
    array_complete = np.concatenate(array_complete)

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
    # _ = extract_column_bands('MAG_GAAP', config, True, fname='mag')
    # _ = extract_column_bands('MAGERR_GAAP', config, True, fname='mag_err')
    # _ = extract_column_bands('FLUX_GAAP', config, True, fname='flux')
    # _ = extract_column_bands('FLUXERR_GAAP', config, True, fname='flux_err')
    # _ = extract_column_bands('EXTINCTION', config, True, fname='ex')
    # _ = extract_column_bands('MAG_LIM', config, True, fname='lim')

    # for these ones, we have a single column in the catalogue
    # _ = extract_column_single('Z_B', config, True, fname='bpz')
    # _ = extract_column_single('THELI_NAME', config, True, fname='name')
    # _ = extract_column_single('MAG_AUTO', config, True, fname='mag_0')
    # _ = extract_column_single('recal_weight', config, True, fname='weight')
