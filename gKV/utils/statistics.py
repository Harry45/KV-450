# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KV-450 likelihood)
# Description: Calculation of the basic statistics from MCMC samples

import numpy as np
import scipy.stats as ss


def summary(samples: np.ndarray, labels: list, p: int = 1) -> None:
    """Calculate the mean and p-sigma credible intervals of each parameter given the MCMC samples

    Args:
        samples (np.ndarray): the MCMC samples
        labels (list): the names of the different parameters
        p (int): p-sigma credible interval
    """

    assert isinstance(samples, np.ndarray), 'samples must be a numpy array'
    assert samples.shape[0] > samples.shape[1], 'samples must be a 2D array, with shape (n_samples, n_parameters)'

    normal = ss.norm(0, 1)

    # calculate the mean
    mean = np.mean(samples, axis=0)

    # calculate the median
    median = np.median(samples, axis=0)

    # calculate the p-sigma credible interval
    intervals = np.array([normal.cdf(-p), normal.cdf(p)]) * 100

    # calculate the lower and upper bounds
    cf_mean = np.percentile(samples, intervals, axis=0) - mean
    cf_median = np.percentile(samples, intervals, axis=0) - median

    print('Using the median')
    print('-' * 80)
    for i in range(len(labels)):
        print('{0:<40s} : {1:6.3f} ± ({2:.3f}, {3:.3f})'.format(labels[i],
                                                                median[i], cf_median[:, i][1], -cf_median[:, i][0]))

    print('')
    print('Using the mean')
    print('-' * 80)
    for i in range(len(labels)):
        print('{0:<40s} : {1:6.3f} ± ({2:.3f}, {3:.3f})'.format(labels[i],
                                                                mean[i], cf_mean[:, i][1], -cf_mean[:, i][0]))
