"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Contains the main code for doing the BHM
"""

from ml_collections import ConfigDict
import numpy as np


def luminosity_distance(redshift: np.ndarray) -> np.ndarray:
    """Calculates the luminosity_distance

    Args:
        redshift (np.ndarray): the redshift of the source

    Returns:
        np.ndarray: the luminosity distance
    """
    return np.exp(30.5 * redshift**0.04 - 21.7)


def likelihood_ztm(data: np.ndarray, var: np.ndarray, mod: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Calculates the likelihood.

    Args:
        data (np.ndarray): the input data
        var (np.ndarray): the variance
        mod (np.ndarray): the model (the mean)
        config (ConfigDict): a set of configurations

    Returns:
        np.ndarray: _description_
    """

    var = np.diag(1. / var)
    chi2 = (np.dot((data - mod), var**2) * (data - mod)).sum(-1)
    like = np.exp(-0.5 * np.clip((chi2 - np.min(chi2)), 0., -2 * config.logeps))
    if np.isnan(like[0, 0, 0]):
        like = np.ones_like(like)
    like /= like.sum()
    return like


def likelihood_integration(like: np.ndarray, trange: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Performs an integration of the likelihood

    Args:
        like (np.ndarray): the likelihood values
        trange (np.ndarray): an array of integers for the template
        config (ConfigDict): the configuration file

    Returns:
        np.ndarray: the numerical integration
    """

    # setting up the range for the magnitude and redshift
    magmax1 = config.mag.max + 3 * config.mag.finedelta
    magmax2 = config.mag.max + 3 * config.mag.delta
    redmax = config.redshift.max + 3 * config.redshift.zstep

    mag1 = np.arange(config.mag.min, magmax1, config.mag.finedelta)
    mag2 = np.arange(config.mag.min, magmax2, config.mag.delta)
    redshifts = np.arange(config.redshift.min, redmax, config.redshift.zstep)

    like_reduced1 = np.zeros((len(mag2) - 2, len(trange), len(redshifts)))

    # iterate over to perform the numerical integration
    for i in range(len(mag2) - 2):
        idx = (mag1 >= mag2[i]) & (mag1 <= mag2[i + 1])
        like_reduced1[i, :, :] = np.mean(like[idx, :, :], axis=0)

    for j in range(len(mag2) - 2):
        zmax = config.redshift.max + 2 * config.reshift.zlist[j]
        red = np.arange(config.redshift.min, zmax, config.reshift.zlist[j])
        like_reduced3 = np.zeros((len(trange), len(red) - 2))

        for i in range(len(red) - 2):
            idx = (redshifts >= red[i]) & (redshifts <= red[i + 1])
            like_reduced3[:, i] = np.mean(like_reduced1[j, :, idx], axis=0)
        if j == 0:
            total_like = like_reduced3.flatten()
        else:
            total_like = np.concatenate((total_like, like_reduced3.flatten()))
    return total_like / total_like.sum()


def model_zt_to_ztm(model: np.ndarray, mgrid: np.ndarray) -> np.ndarray:
    """Calculate the model related to redshift and template only.

    Args:
        model (np.ndarray): the whole model
        mgrid (np.ndarray): a grid of values for the magnitudes

    Returns:
        np.ndarray: the model related to redshift and template only.
    """
    fgrid = 10.**(-.4 * mgrid)
    model /= model[:, :, 2][:, :, None]
    model = np.outer(model, fgrid).reshape((model.shape[0], model.shape[1], model.shape[2], len(fgrid)))
    model = np.swapaxes(model, 2, 3)
    return model


def big_sampler(likelihood: np.ndarray, dirichletsamples: np.ndarray, nbins: int) -> list:
    """The big sampler

    Args:
        likelihood (np.ndarray): an array of the likelihood values
        dirichletsamples (np.ndarray): samples generated from the Dirichlet distribution
        nbins (int): number of bins

    Returns:
        list: a list of random bins
    """
    prod = likelihood * dirichletsamples
    prod /= np.sum(prod, axis=1)[:, None]
    x_point = [np.random.choice(np.arange(nbins), p=i) for i in prod]
    return x_point


def prior_maker(nbins: int, trange: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Calculates the prior for specific range.

    Args:
        nbins (int): number of bins
        trange (np.ndarray): number of templates
        config (ConfigDict): a collection of the configurations

    Returns:
        np.ndarray: the prior
    """
    prior = np.ones(nbins)
    zmax = config.redshift.max + 2 * config.redshift.zlist[0]
    magmax = config.mag.max + 2 * config.mag.delta

    redshifts = np.arange(config.redshift.min, zmax, config.redshift.zlist[0])[:-2]
    mag = np.arange(config.mag.min, magmax, config.mag.delta)[:-2]

    total_z = 0
    for i in range(len(mag)):
        maxz = config.redshift.max + 2 * config.redshift.zlist[i]
        zrange = np.arange(config.redshift.min, maxz, config.redshift.zlist[i])
        total_z2 = total_z + len(zrange) - 2
        prior[(len(trange) * total_z): (total_z2 * len(trange))] = zrange[1] / redshifts[1]
        total_z = total_z2
    return prior


def nbins_caculator(zlist, mlist, t1):
    m1 = np.round(np.arange(mlist[0], mlist[1]+(2*mlist[3]), mlist[3]), 5)[:-2]
    total_z = 0
    for i in range(len(m1)):
        z = np.round(np.arange(zlist[0], zlist[1]+(2*zlist[3+i]), zlist[3+i]), 5)
        total_z += len(z)-2
    return total_z * len(t1)


def running_mean_general(x, N):
    mid = int(np.floor(N/2))
    y = np.zeros(len(x)+(2*mid))
    y[mid:-mid] = x

    y[:mid] = x[0]
    y[-mid:] = x[-1]

    y[:mid] = 0
    y[-mid:] = 0

    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def running_mean_general_convol(x, ratio):
    N = len(ratio)
    mid = int(np.floor(N/2))
    y = np.zeros(len(x)+(2*mid))
    y[mid:-mid] = x

    y[:mid] = x[0]
    y[-mid:] = x[-1]

    y[:mid] = 0
    y[-mid:] = 0

    ratio = np.array(ratio)
    ratio /= np.sum(ratio)

    y = np.convolve(y, ratio, mode='valid')

    return y


def distribution_average(like, zlist, mlist, t1, equation, N):
    like_2 = np.ones_like(like)
    z1 = np.round(np.arange(zlist[0], zlist[1]+(2*zlist[3]), zlist[3]), 5)[:-2]
    m = np.round(np.arange(mlist[0], mlist[1]+(2*mlist[3]), mlist[3]), 5)[:-2]
    total_z = 0
    for i in range(len(m)):
        z = np.round(np.arange(zlist[0], zlist[1] +
                               (2*zlist[3+i]), zlist[3+i]), 5)
        total_z2 = total_z + len(z)-2
        trial = like[(len(t1)*total_z):(total_z2*len(t1))].reshape(
            len(t1), (len(z)-2))
        trial_2 = np.ones_like(trial)
        for k in range(len(t1)):
            trial_2[k] = equation(trial[k], N)
            #trial_2[k] = idt.gaussian_filter(trial[k],1)
            trial_2[k][trial_2[k] == np.min(trial_2[k])] = np.min(trial[k])
        like_2[(len(t1)*total_z):(total_z2*len(t1))] = trial_2.flatten()
        total_z = total_z2

    return like_2
