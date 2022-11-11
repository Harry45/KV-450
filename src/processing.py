"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Process the catalogue and save the outputs
"""

from ml_collections import ConfigDict
from astropy.io import fits
from astropy.table import Table
import numpy as np


def cleaning(config: ConfigDict, save: bool) -> None:
    """Process the data and store them. We have 5 catalogues - we combine them into a single file.

    Args:
        config (ConfigDict): a configuration files.
        save (bool): option to save the files.
    """

    for cat in range(config.catnames):
        fits_file = fits.open(cat, memmap=True)
        data = fits_file[1].data

        # important columns
        fobs = np.asarray([data[config.cols.mag[i]] for i in range(config.nband)]).T
        fobserr = np.asarray([data[config.cols.mag_err[i]] for i in range(config.nband)]).T
        flux = np.asarray([data[config.cols.flux[i]] for i in range(config.nband)]).T
        fluxerr = np.asarray([data[config.cols.flux_err[i]] for i in range(config.nband)]).T
        ext = np.asarray([data[config.cols.ext[i]] for i in range(config.nband)]).T
        mag_lim = np.asarray([data[config.cols.mag_lim[i]] for i in range(config.nband)]).T

        # flags
        flag = data['GAAP_Flag_ugriZYJHKs']
        bpz = data['Z_B']
        seq = data['SeqNr_field']
        seq_n = data['SeqNr']
        name = data['THELI_NAME']
        mag_0 = data['MAG_AUTO']
        weight = data['recal_weight']

        # Filter data by flags
        bpz = bpz[flag == 0]
        fobserr = fobserr[flag == 0, :]
        fobs = fobs[flag == 0, :]
        ext = ext[flag == 0, :]
        mag_lim = mag_lim[flag == 0, :]
        seq = seq[flag == 0]
        name = name[flag == 0]
        mag_0 = mag_0[flag == 0]
        seq_n = seq_n[flag == 0]
        flux = flux[flag == 0]
        fluxerr = fluxerr[flag == 0]
        weight = weight[flag == 0]

    if save:
        # to create *_complete
        np.save('mag.npy', fobs_complete)
        np.save('mag_err.npy', fobserr_complete)
        np.save('bpz.npy', bpz_complete)
        np.save('ex.npy', ex_complete)
        np.save('lim.npy', lim_complete)
        np.save('name.npy', name_complete)
        np.save('mag_0.npy', mag_0_complete)
        np.save('flux.npy', flux_complete)
        np.save('flux_err.npy', flux_e_complete)
        np.save('weight.npy', weight_complete)


# for i in range(5):
#     astro_file = fits.open('KV450_G{}_reweight_3x4x4_v2_good.cat'.format(labels[i]), memmap=True)

#     fobs1 = astro_file[1].data['MAG_GAAP_u']
#     fobs2 = astro_file[1].data['MAG_GAAP_g']
#     fobs3 = astro_file[1].data['MAG_GAAP_r']
#     fobs4 = astro_file[1].data['MAG_GAAP_i']
#     fobs5 = astro_file[1].data['MAG_GAAP_Z']
#     fobs6 = astro_file[1].data['MAG_GAAP_Y']
#     fobs7 = astro_file[1].data['MAG_GAAP_J']
#     fobs8 = astro_file[1].data['MAG_GAAP_H']
#     fobs9 = astro_file[1].data['MAG_GAAP_Ks']

#     fobs = (np.asarray([fobs1, fobs2, fobs3, fobs4, fobs5, fobs6, fobs7, fobs8, fobs9])).T

#     fobserr1 = astro_file[1].data['MAGERR_GAAP_u']
#     fobserr2 = astro_file[1].data['MAGERR_GAAP_g']
#     fobserr3 = astro_file[1].data['MAGERR_GAAP_r']
#     fobserr4 = astro_file[1].data['MAGERR_GAAP_i']
#     fobserr5 = astro_file[1].data['MAGERR_GAAP_Z']
#     fobserr6 = astro_file[1].data['MAGERR_GAAP_Y']
#     fobserr7 = astro_file[1].data['MAGERR_GAAP_J']
#     fobserr8 = astro_file[1].data['MAGERR_GAAP_H']
#     fobserr9 = astro_file[1].data['MAGERR_GAAP_Ks']

#     fobserr = (np.array([fobserr1, fobserr2, fobserr3, fobserr4, fobserr5, fobserr6, fobserr7, fobserr8, fobserr9])).T

#     flux1 = astro_file[1].data['FLUX_GAAP_u']
#     flux2 = astro_file[1].data['FLUX_GAAP_g']
#     flux3 = astro_file[1].data['FLUX_GAAP_r']
#     flux4 = astro_file[1].data['FLUX_GAAP_i']
#     flux5 = astro_file[1].data['FLUX_GAAP_Z']
#     flux6 = astro_file[1].data['FLUX_GAAP_Y']
#     flux7 = astro_file[1].data['FLUX_GAAP_J']
#     flux8 = astro_file[1].data['FLUX_GAAP_H']
#     flux9 = astro_file[1].data['FLUX_GAAP_Ks']

#     flux = (np.asarray([flux1, flux2, flux3, flux4, flux5, flux6, flux7, flux8, flux9])).T

#     flux_e1 = astro_file[1].data['FLUXERR_GAAP_u']
#     flux_e2 = astro_file[1].data['FLUXERR_GAAP_g']
#     flux_e3 = astro_file[1].data['FLUXERR_GAAP_r']
#     flux_e4 = astro_file[1].data['FLUXERR_GAAP_i']
#     flux_e5 = astro_file[1].data['FLUXERR_GAAP_Z']
#     flux_e6 = astro_file[1].data['FLUXERR_GAAP_Y']
#     flux_e7 = astro_file[1].data['FLUXERR_GAAP_J']
#     flux_e8 = astro_file[1].data['FLUXERR_GAAP_H']
#     flux_e9 = astro_file[1].data['FLUXERR_GAAP_Ks']

#     flux_e = (np.asarray([flux_e1, flux_e2, flux_e3, flux_e4, flux_e5, flux_e6, flux_e7, flux_e8, flux_e9])).T

#     ex1 = astro_file[1].data['EXTINCTION_u']
#     ex2 = astro_file[1].data['EXTINCTION_g']
#     ex3 = astro_file[1].data['EXTINCTION_r']
#     ex4 = astro_file[1].data['EXTINCTION_i']
#     ex5 = astro_file[1].data['EXTINCTION_Z']
#     ex6 = astro_file[1].data['EXTINCTION_Y']
#     ex7 = astro_file[1].data['EXTINCTION_J']
#     ex8 = astro_file[1].data['EXTINCTION_H']
#     ex9 = astro_file[1].data['EXTINCTION_Ks']

#     ex = (np.asarray([ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9])).T

#     lim1 = astro_file[1].data['MAG_LIM_u']
#     lim2 = astro_file[1].data['MAG_LIM_g']
#     lim3 = astro_file[1].data['MAG_LIM_r']
#     lim4 = astro_file[1].data['MAG_LIM_i']
#     lim5 = astro_file[1].data['MAG_LIM_Z']
#     lim6 = astro_file[1].data['MAG_LIM_Y']
#     lim7 = astro_file[1].data['MAG_LIM_J']
#     lim8 = astro_file[1].data['MAG_LIM_H']
#     lim9 = astro_file[1].data['MAG_LIM_Ks']

#     lim = (np.asarray([lim1, lim2, lim3, lim4, lim5, lim6, lim7, lim8, lim9])).T

#     # Flagged Data
#     flag = astro_file[1].data['GAAP_Flag_ugriZYJHKs']
#     #flag = flag.astype(int) & 87
#     # BPZ Value
#     BPZ = astro_file[1].data['Z_B']
#     seq = astro_file[1].data['SeqNr_field']
#     seq_n = astro_file[1].data['SeqNr']
#     name = astro_file[1].data['THELI_NAME']
#     M_0 = astro_file[1].data['MAG_AUTO']
#     weight = astro_file[1].data['recal_weight']

#     # Filter data by flags
#     BPZ = BPZ[flag == 0]
#     fobserr = fobserr[flag == 0, :]
#     fobs = fobs[flag == 0, :]
#     ex = ex[flag == 0, :]
#     lim = lim[flag == 0, :]
#     seq = seq[flag == 0]
#     name = name[flag == 0]
#     M_0 = M_0[flag == 0]
#     seq_n = seq_n[flag == 0]
#     flux = flux[flag == 0]
#     flux_e = flux_e[flag == 0]
#     weight = weight[flag == 0]

#     if i == 0:
#         fobs_complete = fobs
#         fobserr_complete = fobserr
#         BPZ_complete = BPZ
#         ex_complete = ex
#         lim_complete = lim
#         seq_complete = seq
#         name_complete = name
#         M_0_complete = M_0
#         flux_complete = flux
#         flux_e_complete = flux_e
#         weight_complete = weight
#     else:
#         fobs_complete = np.append(fobs_complete, fobs, axis=0)
#         fobserr_complete = np.append(fobserr_complete, fobserr, axis=0)
#         BPZ_complete = np.append(BPZ_complete, BPZ, axis=0)
#         ex_complete = np.append(ex_complete, ex, axis=0)
#         lim_complete = np.append(lim_complete, lim, axis=0)
#         seq_complete = np.append(seq_complete, seq, axis=0)
#         name_complete = np.append(name_complete, name, axis=0)
#         M_0_complete = np.append(M_0_complete, M_0, axis=0)
#         flux_complete = np.append(flux_complete, flux, axis=0)
#         flux_e_complete = np.append(flux_e_complete, flux_e, axis=0)
#         weight_complete = np.append(weight_complete, weight, axis=0)

# print(fobs_complete.shape)
# np.save('Mag.npy', fobs_complete)
# np.save('Mag_e.npy', fobserr_complete)
# np.save('BPZ.npy', BPZ_complete)
# np.save('ex.npy', ex_complete)
# np.save('lim.npy', lim_complete)
# # np.save('../data/BPZ_TEST_seq.npy',seq_complete)
# np.save('name.npy', name_complete)
# np.save('M_0.npy', M_0_complete)
# np.save('Flux.npy', flux_complete)
# np.save('Flux_e.npy', flux_e_complete)
# np.save('weight.npy', weight_complete)
# stop

# np.save('data/{0}'.format('Full_Survey'),complete_data_set)


# for i in range(9):
#    plt.scatter(complete_data_set[:,i],complete_data_set[:,9+i])
#    plt.show()
# plt.hist(complete_data_set[:,18])
# plt.show()
# z_max = [0.3, 0.5, 0.7, 0.9, 1.2]
# z_min = [0.1, 0.3, 0.5, 0.7, 0.9]

# #'''Name Output Files'''
# mag_names = ['mag_bin_1', 'mag_bin_2', 'mag_bin_3', 'mag_bin_4', 'mag_bin_5']
# error_names = ['error_bin_1', 'error_bin_2', 'error_bin_3', 'error_bin_4', 'error_bin_5']

# for i in range(len(z_max)):
#     '''Bin Data by Redshift'''
#     new_data_set = complete_data_set[complete_data_set[:, -1] < z_max[i]]
#     new_data_set = new_data_set[new_data_set[:, -1] > z_min[i]]

#     fobs = new_data_set[:, :9]
#     fobserr = new_data_set[:, 9:18]
#     print(z_max[i], fobs.shape[0])
#     np.save('../data/{0}'.format(mag_names[i]), fobs)
#     np.save('../data/{0}'.format(error_names[i]), fobserr)
