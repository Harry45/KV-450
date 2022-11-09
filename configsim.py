"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Contains all the configurations required for running the code (Simulations)
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Contains a series of configurations required for the BHM method.

    Returns:
        ml_collections.ConfigDict: the set of configurations
    """

    conf = ml_collections.ConfigDict()

    # general settings
    conf.filters = ['u', 'g', 'r', 'i', 'Z2', 'Y', 'J', 'H', 'Ks']

    # paths
    conf.path = path = ml_collections.ConfigDict()
    path.data = 'data/sim/'
    path.output = 'output/sim/'
    path.filter = 'filters/'
    path.sed = 'sed/'

    # filenames (for the outputs)
    conf.fnames = fnames = ml_collections.ConfigDict()
    fnames.fluxcorr = 'fluxcorr'
    fnames.fluxcorr_err = 'fluxcorr_err'

    # redshift configurations
    conf.redshift = redshift = ml_collections.ConfigDict()
    redshift.zmin = 0.0
    redshift.zmax = 3.0
    redshift.zstep = 0.05
    redshift.zlist = [0.05] * 6

    # fixed values
    conf.values = fixed = ml_collections.ConfigDict()
    fixed.nsources = 100000
    fixed.chunk = 1000
    fixed.nlikechunk = 1
    fixed.nchain = 1

    # numbers related to sampling procedure
    conf.sampling = sampling = ml_collections.ConfigDict()
    sampling.nsamples = 1000000
    sampling.resume = 55000
    sampling.minsamples = 1000
    sampling.maxsamples = 100000
    sampling.ncore = 4

    # boolean configurations
    conf.boolean = boolean = ml_collections.ConfigDict()
    boolean.likelihood = True
    boolean.samples = True
    boolean.resume = False
    boolean.weights = True
    boolean.random = False
    return conf