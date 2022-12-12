"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Main script for running the code
"""

from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts
from src.processing import simple_cleaning
from src.calculations import calculate_offset, correct_flux, split_file

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)


def main(argv):
    """
    Run the main script.
    """

    # simple_cleaning(FLAGS.config)
    # calculate_offset(FLAGS.config)
    # correct_flux(FLAGS.config)
    split_file(FLAGS.config, ['fluxcorr', 'fluxcorr_err', 'weight'])


if __name__ == "__main__":
    app.run(main)
