# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""Calculates the Gelman Rubin convergence test for two set of samples."""

import numpy as np


def gelmanRubin(sample1, sample2):

    # the number of chains
    M = 2

    # concatenate the samples in a list
    samples = [sample1, sample2]

    # the number of parameters in the chain
    Npar = samples[0].shape[1]

    # the number of samples in each chain
    N = np.array([samples[i].shape[0] for i in range(M)])

    # combine the two set of samples
    samples_all = np.concatenate(samples)

    # empty array to store the mean
    mu = np.zeros((Npar, M))

    # empty array to store the variance
    var = np.zeros((Npar, M))

    for i in range(M):
        mu[:, i] = np.mean(samples[i], axis=0)
        var[:, i] = np.var(samples[i], axis=0, ddof=1)

    # mean across all samples
    mua = np.mean(samples_all, axis=0)

    # difference between the two means
    diff = mu - mua.reshape(Npar, 1)

    # the B-factor (see Gelman and Rubin 1992)
    B = np.var(diff, axis=1, ddof=1)

    # the W factor
    W = np.mean(var, axis=1)

    # the R-hat statistics
    Rs = (W + B*(1.0 + (1.0/M)))/W

    return Rs
