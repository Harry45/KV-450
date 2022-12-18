
from absl import flags, app
from ml_collections.config_flags import config_flags
from ml_collections import ConfigDict
import os
import numpy as np
from scipy.interpolate import interp1d
from mpi4py import MPI

from src.bhm import luminosity_distance, model_zt_to_ztm
from src.bhm import nbins_calculator, likelihood_ztm, likelihood_integration

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)

# parallel processing
# COMM = MPI.COMM_WORLD
# SIZE = COMM.Get_size()
# RANK = COMM.Get_rank()


def load_filters(cfg: ConfigDict):
    record = dict()
    cfg = cfg.filters
    for j in range(cfg.nfilter):
        path = os.path.join(cfg.path, cfg.names[j])
        data = np.genfromtxt(path)
        wavelength, target = data[:, 0], data[:, 1]
        target /= wavelength
        record[cfg.types[j]] = (wavelength, target)
    return record


def load_templates(config: ConfigDict):

    record = dict()
    cfg = config.template
    temp_path = os.path.join(cfg.list, cfg.choice)
    template_names = np.loadtxt(temp_path, dtype=str)
    ntemplate = len(template_names)
    for j in range(ntemplate):
        path = os.path.join(cfg.path, template_names[j])
        temp_data = np.genfromtxt(path)
        temp_data[:, 1] *= temp_data[:, 0]**2 / config.parsec
        reference = np.interp(config.lambdaref, temp_data[:, 0], temp_data[:, 1])
        temp_data[:, 1] /= reference
        interpolator = interp1d(temp_data[:, 0], temp_data[:, 1], fill_value="extrapolate")
        record[j] = (temp_data, interpolator)
    return record


def load_observations(cfg: ConfigDict):
    split_path = cfg.path.split + 'split_' + str(cfg.split) + '/'
    f_obs = np.load(split_path + 'fluxcorr.npy')
    f_obs_err = np.load(split_path + 'fluxcorr_err.npy')
    return f_obs, f_obs_err


def process_filter(wavelength, target):

    ind = np.where(target > 0.01 * np.max(target))[0]
    lambda_min, lambda_max = wavelength[ind[0]], wavelength[ind[-1]]
    norm = np.trapz(target, x=wavelength)

    return norm, lambda_min, lambda_max


def calculate_fmod(cfg: ConfigDict):

    record = dict()

    filters = load_filters(cfg)
    templates = load_templates(cfg)

    zmax = cfg.redshift.zmax + (3 * cfg.redshift.zfine)
    magmax = cfg.mag.max + (3 * cfg.mag.finedelta)

    redshifts = np.arange(cfg.redshift.zmin, zmax, cfg.redshift.zfine)
    mag = np.arange(cfg.mag.min, magmax, cfg.mag.finedelta)

    ntemplate = len(templates)
    nredshift = len(redshifts)

    f_mod = np.zeros((nredshift, ntemplate, cfg.filters.nfilter))
    record['temp_index'] = np.arange(0, ntemplate)

    for j_tem in range(ntemplate):
        _, sed_interp = templates[j_tem]
        for j_fil, jft in enumerate(cfg.filters.types):
            wavelength, target = filters[jft]
            norm, lambda_min, lambda_max = process_filter(wavelength, target)
            for j_red in range(nredshift):
                opz = (redshifts[j_red] + 1)
                xf_z = np.linspace(lambda_min / opz, lambda_max / opz, num=5000)
                yf_z = interp1d(wavelength / opz, target)(xf_z)
                ysed = sed_interp(xf_z)
                f_mod[j_red, j_tem, j_fil] = np.trapz(ysed * yf_z, x=xf_z) / norm
                f_mod[j_red, j_tem, j_fil] *= opz**2
                f_mod[j_red, j_tem, j_fil] /= (4.0 * np.pi * luminosity_distance(redshifts[j_red])**2)

    model_ztm = model_zt_to_ztm(f_mod, mag)
    return f_mod, model_ztm, record


def calculate_likelihood(cfg: ConfigDict):

    f_obs, f_obs_err = load_observations(cfg)
    f_mod, model_ztm, record = calculate_fmod(cfg)

    nobj = f_obs.shape[0]
    nbins = nbins_calculator(record['temp_index'], cfg)
    likelihood = np.zeros((nobj, nbins))
    for i in range(10):

        like_i = likelihood_ztm(f_obs[i], f_obs_err[i], model_ztm, cfg)
        like_i = np.swapaxes(like_i, 2, 0)
        likelihood[i] = likelihood_integration(like_i, record['temp_index'], cfg)


def main(argv):
    # f_obs, f_obs_err = load_observations(FLAGS.config)
    # filters = load_filters(FLAGS.config)
    # templates = load_templates(FLAGS.config)
    calculate_likelihood(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
