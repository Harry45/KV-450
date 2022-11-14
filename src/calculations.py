"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Perform some calculations on the values found in the catalogue.
"""
import os
import random
import numpy as np
from ml_collections import ConfigDict


def calculate_offset(config: ConfigDict):
    """Calculate the offset and save it to data/calculation/

    Args:
        config (ConfigDict): the main configuration file.
    """

    # load the relevant processed files
    name = np.load(config.path.processed + 'name.npy')
    magnitude = np.load(config.path.processed + 'mag.npy')
    flux = np.load(config.path.processed + 'flux.npy')

    offset = np.ones_like(flux)
    unique_name = np.unique(name)

    for n in unique_name:

        idx_name = name == n
        mag_sel = magnitude[idx_name]
        flu_sel = flux[idx_name]

        for j in range(config.nband):
            idx = mag_sel[:, j] != 99
            mag_sel = mag_sel[idx]
            flu_sel = flu_sel[idx]

        # this has been added by me
        # -------------------------------
        flu_sel[flu_sel <= 0] = 1E-10
        # -------------------------------

        offset_individual = mag_sel + 2.5 * np.log10(flu_sel)

        if offset_individual.shape[0] >= 1:
            offset_median = np.mean(offset_individual, axis=0)
            offset[idx_name] *= offset_median

    # save the calculations
    np.save(config.path.processed + 'offset.npy', offset)


def correct_flux(config: ConfigDict):
    """Calculates the corrected flux and flux error.

    Args:
        config (ConfigDict): the configuration file for the KV-450 catalogue
    """

    # these are from the catalogue
    flux = np.load(config.path.processed + 'flux.npy')
    flux_err = np.load(config.path.processed + 'flux_err.npy')
    ext = np.load(config.path.processed + 'ex.npy')

    print(np.amin(flux))
    print(np.amax(flux))

    # these are calculated in the previous function
    offset = np.load(config.path.processed + 'offset.npy')

    # # apply extinction
    # flux = flux * 10**(0.4 * ext)
    # flux_err = flux_err * 10**(0.4 * ext)

    # # correct for offset
    # flux = flux * 10**(-0.4 * offset)
    # flux_err = flux_err * 10**(-0.4 * offset)

    print(np.amin(ext))
    print(np.amax(ext))
    print(np.amin(offset))
    print(np.amax(offset))
    # # save the calculations
    # np.save(config.path.processed + 'fluxcorr.npy', flux)
    # np.save(config.path.processed + 'fluxcorr_err.npy', flux_err)


def correct_magnitude(config: ConfigDict):
    """Apply corrections to the magnitude as well.

    Args:
        config (ConfigDict): the main configuration file for KV-450
    """

    # these are from the catalogue
    mag = np.load(config.path.processed + 'mag.npy')
    mag_err = np.load(config.path.processed + 'mag_err.npy')
    lim = np.load(config.path.processed + 'lim.npy')

    # apply correction
    for i in range(config.nband):
        mag[mag[:, i] > lim[:, i], i] = 99.
        mag[mag_err[:, i] > 1, i] = 99.
        mag_err[mag[:, i] == 99, i] = lim[mag[:, i] == 99, i]

    # save the corrections
    np.save(config.path.processed + 'magcorr.npy', mag)
    np.save(config.path.processed + 'magcorr_err.npy', mag_err)


def split_file(config: ConfigDict, filelist: list):
    """Split the main file into five parts and randomly select 100,000 objects from it.

    Args:
        config (ConfigDict): the main configuration file for KV-450.
        filelist (list): names of the files to be split

    files
    -----
    - bpz
    - fluxcorr
    - fluxcorr_err
    - weight
    """
    os.makedirs(config.path.split, exist_ok=True)
    bpz = np.load(config.path.processed + 'bpz.npy')
    nsources = len(bpz)
    index = random.sample(range(0, nsources), nsources)

    for fname in filelist:
        # load file and shuffle it
        file = np.load(config.path.processed + f'{fname}.npy')[index]
        filesplit = np.array_split(file, 5, axis=0)
        for s in range(5):
            splitname = f'{config.path.split}split_{s}/{fname}'
            np.save(splitname + '.npy', filesplit[s][0:100000])