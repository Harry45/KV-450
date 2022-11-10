"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Main script for running the code
"""

from absl import flags, app
from ml_collections.config_flags import config_flags
# import argparse
# from configkids import configkids
# from configsim import configsim

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)


def main(argv):
    """
    Run the main script.
    """

    print(FLAGS.config.filters)
    print(FLAGS.config.redshift.zmin)


if __name__ == "__main__":
    app.run(main)
