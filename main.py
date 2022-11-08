"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Main script for running the code
"""

import ml_collections
from configkids import configkids
# from configsim import configsim


def run(configurations: ml_collections.ConfigDict) -> None:
    """Run the main script and output the results.

    Args:
        configurations (ml_collections.ConfigDict): An ML configurations. See configKiDS.py for an example.
    """

    print(configurations.filters)


if __name__ == "__main__":
    CONFIGS = configkids()
    run(CONFIGS)
