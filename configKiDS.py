"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Contains all the configurations required for running the code (KV-450)
"""

import ml_collections


def config() -> ml_collections.ConfigDict:
    """Contains a series of configurations required for the BHM method.

    Returns:
        ml_collections.ConfigDict: the set of configurations
    """

    conf = ml_collections.ConfigDict()

    return conf
