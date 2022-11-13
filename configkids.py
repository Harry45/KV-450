"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Contains all the configurations required for running the code (KV-450)
"""


from ml_collections import ConfigDict
import numpy as np


def get_config() -> ConfigDict:
    """Contains a series of configurations required for the BHM method.

    Returns:
        ConfigDict: the set of configurations
    """

    conf = ConfigDict()

    # filters
    conf.filters = ['u', 'g', 'r', 'i', 'Z2', 'Y', 'J', 'H', 'Ks']
    conf.filterlist = [f'KiDSVIKING_{f}.res' for f in conf.filters]

    # bands
    conf.bands = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']
    conf.nband = len(conf.filters)

    # catalogue
    conf.cat = ['9', 'S', '23', '15', '12']
    conf.catnames = [f'KV450_G{c}_reweight_3x4x4_v2_good.cat' for c in conf.cat]

    # fixed numbers
    conf.ntomo = 1
    conf.eps = 1E-300
    conf.logeps = np.log(conf.eps)

    # paths
    conf.path = path = ConfigDict()
    path.catalogue = 'data/catalogue/'
    path.processed = 'data/processed/kv/'
    path.split = 'data/splits/kv/'
    path.output = 'output/kv/'
    path.filter = 'filters/'
    path.sed = 'sed/'

    # redshift configurations
    conf.redshift = redshift = ConfigDict()
    redshift.zmin = 0.0
    redshift.zmax = 3.0
    redshift.zstep = 0.05
    redshift.zlist = [0.05] * 6

    # magnitude setup
    conf.mag = mag = ConfigDict()
    mag.min = 19.0
    mag.max = 25.0
    mag.finedelta = 0.1
    mag.delta = 1.0

    # fixed values
    conf.values = fixed = ConfigDict()
    fixed.nsources = 100000
    fixed.chunk = 1000
    fixed.nlikechunk = 1
    fixed.nchain = 1
    fixed.referencefilter = 2

    # numbers related to sampling procedure
    conf.sampling = sampling = ConfigDict()
    sampling.nsamples = 1000000
    sampling.resume = 55000
    sampling.minsamples = 1000
    sampling.maxsamples = 100000
    sampling.ncore = 4

    # boolean configurations
    conf.boolean = boolean = ConfigDict()
    boolean.likelihood = True
    boolean.samples = True
    boolean.resume = False
    boolean.weights = True
    boolean.random = False
    return conf
