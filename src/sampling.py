"""
Authors: Arrykrishna Mootoovaloo and George Kyriacou
Supervisors: Alan Heavens, Andrew Jaffe, Florent Leclercq
Date: November 2022
Project: KiDS+VIKING-450 cosmology with Bayesian hierarchical model redshift distributions
Script: Contains the main code for sampling the posterior using MPI
"""
import time
import gc
from tqdm import tqdm

import os
import numpy as np
from mpi4py import MPI
from ml_collections import ConfigDict


def generate_samples(config: ConfigDict):

    ntemplate = 0
    colour = 1
    template_names = None

    if RANK == 0:
        temp_path = os.path.join(config.template.list, config.template.choice)
        template_names = np.loadtxt(temp_path, dtype=str)
        ntemplate = len(template_names)
        if ntemplate > SIZE:
            template_names = np.array_split(template_names, SIZE, axis=0)
        else:
            template_names = np.array_split(template_names, ntemplate, axis=0)

    ntemplate = COMM.bcast(ntemplate, root=0)

    if RANK < ntemplate:
        colour = 0

    sub_comm = COMM.Split(colour, RANK)
    template_names = sub_comm.scatter(template_names, root=0)

    return template_names, ntemplate, colour


# load filters
# filters = np.loadtxt('{}/{}'.format(filter_list_folder, filter_list), dtype=str)

# load templates
nt = 0
template_names_full = None
template_names = None
if rank == 0:
    template_names_full = np.loadtxt('{}/{}'.format(template_list_folder, template_list), dtype=str)

    nt = len(template_names_full)
    if nt > size:
        template_names = np.array_split(template_names_full, size, axis=0)
    else:
        template_names = np.array_split(template_names_full, nt, axis=0)


nt = comm.bcast(nt, root=0)
if rank < nt:
    colour = 0
else:
    colour = 1
key = rank
sub_comm = comm.Split(colour, key)
sub_size = sub_comm.Get_size()
sub_rank = sub_comm.Get_rank()

template_names = sub_comm.scatter(template_names, root=0)


# removes a few functions that dont work on HPC
HPC = False

if not HPC and rank == 0:
    total_time_start = time.time()

"""
# MAIN CODE

for files_number in range(number_of_tomograthic_bins):

    z1 = np.round(np.arange(zlist[0], zlist[1]+(3*zlist[2]), zlist[2]), 5)
    m1 = np.round(np.arange(mlist[0], mlist[1]+(3*mlist[2]), mlist[2]), 5)
    t1 = np.arange(0, nt)

    m1 = np.round(np.arange(mlist[0], mlist[1]+(3*mlist[2]), mlist[2]), 5)
    t1 = np.arange(0, nt)

    if not HPC:
        if rank == 0:
            t_start = time.time()

    if likelihood_generator:

        if rank == 0:
            print('Inputing data....', flush=True)

            # input data
            f_obs = np.load('{}/{}'.format(flux_folder, flux_name))[:, :len(filters)]  # the fluxes for each filter
            f_obs_err = np.load('{}/{}'.format(flux_folder, flux_error_name)
                                )[:, :len(filters)]  # the error of fluxes at each filet
            random = np.arange(len(f_obs))
            if RANDOM:
                np.random.shuffle(random)
            f_obs = f_obs[random]
            f_obs_err = f_obs_err[random]
            f_obs = f_obs[:n_objects, :len(filters)]
            f_obs_err = f_obs_err[:n_objects, :len(filters)]
            np.save('{}/random.npy'.format(output_folder), random)
            print('Total File Size : ', f_obs.shape, flush=True)

            f_obs_chunk = np.array_split(f_obs, size, axis=0)
            f_obs_err_chunk = np.array_split(f_obs_err, size, axis=0)

            print('Chunk File Size : ', f_obs_chunk[0].shape, flush=True)
            if not HPC:
                gc.collect()
        else:
            f_obs_chunk = None
            f_obs_err_chunk = None

        f_obs_chunk = comm.scatter(f_obs_chunk, root=0)
        f_obs_err_chunk = comm.scatter(f_obs_err_chunk, root=0)

        if rank == 0:
            print('[DONE]', flush=True)

        if rank < nt:
            nt_chunk = len(template_names)
            f_mod = np.zeros((len(z1), nt_chunk, len(filters)))
            model_ztm = np.zeros((len(z1), nt_chunk, len(m1), len(filters)))
        else:
            f_mod = None
            model_ztm = None
        # Approximate luminosity distance for flat LCDM
        # We will now use the BPZ routines to load the flux-redshift model for each template.
        # We'll have interpolated and discretized versions

        lambdaRef = 4.5e3
        if rank == 0:
            print('Inputing Templates and Filters....', flush=True)
            if not HPC:
                pbar = tqdm(total=nt_chunk)
        if rank < nt:
            for it in range(nt_chunk):
                if not HPC:
                    if rank == 0:
                        pbar.update(1)
                seddata = np.genfromtxt('{}/{}'.format(template_folder, template_names[it]))
                seddata[:, 1] *= seddata[:, 0]**2. / 3e18
                ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
                seddata[:, 1] /= ref
                sed_interp = interp1d(seddata[:, 0], seddata[:, 1])
                for jf in range(len(filters)):
                    data = np.genfromtxt('{}/{}'.format(filter_folder, filters[jf]))
                    xf, yf = data[:, 0], data[:, 1]
                    yf /= xf  # divide by lambda
                    # Only consider range where >1% max
                    ind = np.where(yf > 0.01*np.max(yf))[0]
                    lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
                    norm = np.trapz(yf, x=xf)
                    for iz in range(len(z1)):
                        opz = (z1[iz] + 1)
                        xf_z = np.linspace(lambdaMin / opz, lambdaMax / opz, num=5000)
                        yf_z = interp1d(xf / opz, yf)(xf_z)
                        ysed = sed_interp(xf_z)
                        f_mod[iz, it, jf] = np.trapz(ysed * yf_z, x=xf_z) / norm
                        f_mod[iz, it, jf] *= opz**2. / DL(z1[iz])**2. / (4*np.pi)

            f_mod = sub_comm.gather(f_mod, root=0)

        if rank == 0:
            f_mod = np.concatenate(f_mod, axis=1)
            model_ztm = model_zt_to_ztm(f_mod, m1)
            # np.save('{}/model.npy'.format(output_folder),model_ztm)

        comm.Barrier()
        f_mod = comm.bcast(f_mod, root=0)
        model_ztm = comm.bcast(model_ztm, root=0)
        shape = model_ztm.shape
        comm.Barrier()
        if rank == 0:
            print('[DONE]', flush=True)

        if rank == 0:
            print('Generate z,t,m Likelihoods....', flush=True)

        # convert magnitudes to fluxes
        f_obs_chunk = np.array_split(f_obs_chunk, n_split, axis=0)
        f_obs_err_chunk = np.array_split(f_obs_err_chunk, n_split, axis=0)
        ff_grid_edges = 10.**(-.4*m1)

        # surces are split to save RAM
        for split in range(n_split):
            nobj = f_obs_chunk[split].shape[0]
            nbins = nbins_caculator(zlist, mlist, t1)
            like = np.zeros((nobj, nbins), dtype=float)
            if not HPC:
                if rank == 0:
                    pbar = tqdm(total=nobj)
            for i in range(nobj):
                if not HPC:
                    if rank == 0:
                        pbar.update(1)
                like[i] = likelihood_integration(
                    np.swapaxes(likelihood_ztm(f_obs_chunk[split][i, :],
                                               f_obs_err_chunk[split][i, :], model_ztm), 2, 0), zlist, mlist, t1)

            np.save('{}/likelihoods/likelihood_file_{}_{}.npy'.format(output_folder, rank, split), like)

            like = None

        if not HPC:
            gc.collect()
            if rank == 0:
                pbar.close()
        comm.Barrier()

        if rank == 0:
            print('[DONE]', flush=True)

    if sample_generator:
        if rank == 0:

            print('N(z) Gibbs Sampling....', flush=True)
            # input likelihoods
            nbins = np.load(
                '{}/likelihoods/likelihood_file_0_0.npy'.format(output_folder)).shape[1]
        else:
            nbins = None

        nbins = comm.bcast(nbins, root=0)
        prior = prior_maker(nbins, zlist, mlist, t1)

        for GRT in range(number_of_chains):

            if rank == 0:
                if sample_resume:
                    # continue from previus session
                    nbs = np.load('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                        output_folder, files_number+1, GRT, resume_number))[-1]
                    nbs = np.histogram(
                        nbs, bins=np.arange(nbins+1))[0]
                    nbs = nbs.astype(float) + prior
                    hbs = np.random.dirichlet(nbs, 1)
                else:
                    # print(nbins)
                    # starting point assumes random histogram
                    nbs = np.random.rand(nbins)*prior
                    # create starting distribution
                    hbs = np.random.dirichlet(nbs, 1)
            else:
                hbs = None

            comm.Barrier()

            hbs = comm.bcast(hbs, root=0)

            if rank == 0:
                fbs = np.zeros((nsamples_split, n_objects))
                nsamples_split_counter = 0
                if not HPC:
                    pbar = tqdm(total=nsamples+1)
            # run per sample
            for kk in range(1, nsamples+1):
                for split in range(n_split):
                    # load likelihood files and flatten
                    bpz_like_ztm_chunk = np.load(
                        '{}/likelihoods/likelihood_file_{}_{}.npy'.format(output_folder, rank, split))
                    # count bincounts of {z,t,m}_survey
                    if split == 0:
                        nbs = big_sampler(bpz_like_ztm_chunk, hbs)
                    else:
                        nbs = np.concatenate(
                            [nbs, big_sampler(bpz_like_ztm_chunk, hbs)])
                        # nbs += big_sampler(bpz_like_ztm_chunk,hbs)

                # sum bincounts from all cores
                # nbs_all = comm.reduce(nbs, op=MPI.SUM, root=0)
                nbs_all = comm.gather(nbs, root=0)
                nbs = None
                if rank == 0:
                    # save samples
                    nbs_all = np.hstack(nbs_all)
                    if kk >= 0:
                        # BIG CHANGE save the number counts NOT,the probabilities
                        fbs[nsamples_split_counter, :] = nbs_all
                        nsamples_split_counter += 1
                        if nsamples_split_counter == nsamples_split:
                            if sample_resume:
                                np.save('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                                    output_folder, files_number+1, GRT, kk+resume_number), fbs)
                            else:
                                np.save('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                                    output_folder, files_number+1, GRT, kk), fbs)
                            nsamples_split_counter = 0

                    # use bincounts to draw distribution sample p(distibution|flux,{z,t,m}_survey)
                    nbs_hist = np.histogram(
                        nbs_all, bins=np.arange(nbins+1))[0]
                    nbs_all = None
                    nbs_hist = nbs_hist.astype(float) + prior
                    # nbs_all = nbs_all.astype(float)
                    # nbs_all += prior

                    hbs = np.random.dirichlet(nbs_hist, 1)
                    # hbs = distribution_average(hbs[0],zlist,mlist,t1,running_mean_general_convol,[1.,20.,1.])
                nbs_all = None
                hbs = comm.bcast(hbs, root=0)
                if not HPC:
                    if rank == 0:
                        pbar.update(1)
                '''
                    gc.collect()
                '''
            if not HPC:
                if rank == 0:
                    pbar.close()

        bpz_like_ztm_chunk = None
        fbs = None
        fbs_samples_ztm_chunk = None

        if not HPC:
            gc.collect()

        if rank == 0:
            print('[DONE]', flush=True)

if not HPC:
    if rank == 0:
        total_time_end = time.time()

        print('Total Time =', float(total_time_end-total_time_start), flush=True)
"""
