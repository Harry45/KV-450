 
import numpy as np
from scipy.special import erf
from scipy.special import erfc
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from matplotlib.patches import Rectangle
from mpi4py import MPI
import argparse


# input param file
parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()

p_args = args.p
p = np.genfromtxt('{}'.format(p_args),dtype=None,delimiter=',',encoding='ascii')
p_float = np.genfromtxt('{}'.format(p_args),dtype=float,delimiter=',',encoding='ascii')
p_int = np.genfromtxt('{}'.format(p_args),dtype=int,delimiter=',',encoding='ascii')
p_bool = np.genfromtxt('{}'.format(p_args),dtype=bool,delimiter=',',encoding='ascii')


# avoid zeros
eps = 1e-300
eeps = np.log(eps)

# functions

# Approximate luminosity distance for flat LCDM
def DL(z):
    return np.exp(30.5 * z**0.04 - 21.7)

def likelihood_ztm(data, var, mod):
    
    var = np.diag(1/var)
    #var = np.where(data/var < 1e-6, 0.0, var**-1.0)
    #var - np.diag(var)
    chi2 = (np.dot((data-mod), var**2)*(data-mod)).sum(-1)
    #chi2 *= -1/2
    #chi2 -= np.max(chi2)
    like = np.exp(-0.5 *
           np.clip((chi2-np.min(chi2)), 0., -2*eeps))
    #like = np.exp(chi2)
    if np.isnan(like[0,0,0]):
        like = np.ones_like(like)
    like /= like.sum()
    return like

#selection = np.load('/rds/general/user/gk1513/home/SCRATCH/cx2-scratch2/Photo_Z_3D/SELECTION_FUNCTION/model_function.npy')

def likelihood_integration(like,zlist,mlist,t1):
    m1 = np.round(np.arange(mlist[0], mlist[1]+(3*mlist[2]), mlist[2]),5)
    m2 = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)
    z1 = np.round(np.arange(zlist[0],zlist[1]+(3*zlist[2]),zlist[2]),5)
    like_reduced1 = np.zeros((len(m2)-2, len(t1), len(z1)))
    for i in range(len(m2)-2):
        like_reduced1[i, :, :] = np.mean(
            like[(m1 >= m2[i]) & (m1 <= m2[i+1]), :, :], axis=0)
        
    for j in range(len(m2)-2):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+j]),zlist[3+j]), 5) 
        like_reduced3 = np.zeros((len(t1), len(z)-2))
        for i in range(len(z)-2):
            like_reduced3[:, i] = np.mean(
                like_reduced1[j,:, (z1 >= z[i]) & (z1 <= z[i+1])], axis=0)
        if j == 0 :
            total_like = like_reduced3.flatten()
        else : 
            total_like = np.concatenate((total_like,like_reduced3.flatten()))
    #total_like /= selection
    return total_like/total_like.sum()


def model_zt_to_ztm(model, mgrid):
    fgrid = 10.**(-.4*mgrid)
    model /= model[:, :, 2][:, :, None]
    model = np.outer(model, fgrid).reshape(
        (model.shape[0], model.shape[1], model.shape[2], len(fgrid)))
    model = np.swapaxes(model, 2, 3)
    return model

def big_sampler(likelihood, hbs):
    prod = likelihood * hbs
    prod /= np.sum(prod, axis=1)[:, None]
    x_point = [np.random.choice(np.arange(nbins), p=i) for i in prod]
    #x_point = np.histogram(x_point, bins=np.arange(nbins+1))[0]
    return x_point

def prior_maker(nbins,zlist,mlist,t1):
    prior = np.ones(nbins)
    z1 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3]),zlist[3]), 5)[:-2] 
    m1 = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
    total_z = 0 
    for i in range(len(m1)):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+i]),zlist[3+i]), 5) 
        total_z2 = total_z + len(z)-2
        prior[(len(t1)*total_z):(total_z2*len(t1))] = z[1]/z1[1]
        total_z = total_z2
    return prior

def nbins_caculator(zlist,mlist,t1):
    m1 = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
    total_z = 0
    for i in range(len(m1)):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+i]),zlist[3+i]), 5) 
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


def running_mean_general_convol(x,ratio) : 
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

def distribution_average(like, zlist, mlist, t1,equation,N):
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

# ----------------------------------------------------------------------------------------------------------
# paralell processing
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# tomograthic bins set to 1 as were looking at whole survey
number_of_tomograthic_bins = 1

# assign parameter file to varibles 

flux_folder, output_folder, filter_folder, template_folder, template_list_folder, filter_list_folder = p[:6]
flux_name, flux_error_name, filter_list, template_list = p[6:10]
zlist = np.loadtxt('{}'.format(p_args),skiprows=32,max_rows=1)
mlist = np.loadtxt('{}'.format(p_args),skiprows=33,max_rows=1)
reffilter, n_objects, nsamples, nsamples_split, n_split, number_of_chains = p_int[12:18]
likelihood_generator, sample_generator, sample_resume, WEIGHTS, RANDOM = p_bool[18:23]
resume_number = p_int[23]

# load filters
filters = np.loadtxt('{}/{}'.format(filter_list_folder,filter_list),dtype=str)

# load templates
nt = 0
template_names_full = None
template_names = None
if rank == 0 :
    template_names_full = np.loadtxt('{}/{}'.format(template_list_folder,template_list),dtype=str)
    
    nt = len(template_names_full)
    if nt > size:
        template_names = np.array_split(template_names_full,size,axis=0)
    else : 
        template_names = np.array_split(template_names_full,nt,axis=0)

nt = comm.bcast(nt,root=0)
if rank < nt :
    colour = 0
else :
    colour = 1
key = rank
sub_comm = comm.Split(colour,key)
sub_size = sub_comm.Get_size()
sub_rank = sub_comm.Get_rank()

template_names = sub_comm.scatter(template_names, root=0)


# removes a few functions that dont work on HPC
HPC = False
if not HPC :
    import time
    import gc
    from tqdm import tqdm
    #import getdist
    #import getdist.plots as pt
    if rank == 0:
        total_time_start = time.time()

# MAIN CODE
# ----------------------------------------------------------------------------------------------------------
for files_number in range(number_of_tomograthic_bins):
    
    z1 = np.round(np.arange(zlist[0],zlist[1]+(3*zlist[2]),zlist[2]),5)
    m1 = np.round(np.arange(mlist[0], mlist[1]+(3*mlist[2]), mlist[2]),5)
    t1 = np.arange(0,nt)

    m1 = np.round(np.arange(mlist[0], mlist[1]+(3*mlist[2]), mlist[2]),5)
    t1 = np.arange(0,nt)


    if not HPC : 
        if rank == 0 :
            t_start = time.time()
    
    if likelihood_generator :
        
        if rank == 0 :
            print('Inputing data....',flush=True)
            
            # input data
            f_obs = np.load('{}/{}'.format(flux_folder,flux_name))[:,:len(filters)]  #the fluxes for each filter
            f_obs_err = np.load('{}/{}'.format(flux_folder,flux_error_name))[:,:len(filters)] # the error of fluxes at each filet
            random = np.arange(len(f_obs))
            if RANDOM :
                np.random.shuffle(random)
            f_obs = f_obs[random]
            f_obs_err = f_obs_err[random]
            f_obs = f_obs[:n_objects,:len(filters)]
            f_obs_err = f_obs_err[:n_objects,:len(filters)]
            np.save('{}/random.npy'.format(output_folder),random)
            print('Total File Size : ',f_obs.shape,flush=True)             

            f_obs_chunk = np.array_split(f_obs,size,axis=0)
            f_obs_err_chunk = np.array_split(f_obs_err,size,axis=0)
            


            
            print('Chunk File Size : ',f_obs_chunk[0].shape,flush=True)
            if not HPC :
                gc.collect()
        else:
            f_obs_chunk = None
            f_obs_err_chunk = None
        
        f_obs_chunk = comm.scatter(f_obs_chunk, root=0)
        f_obs_err_chunk = comm.scatter(f_obs_err_chunk, root=0)

        
        if rank == 0 :
            print('[DONE]',flush=True)
        

        if rank < nt :
            nt_chunk = len(template_names)
            f_mod = np.zeros((len(z1),nt_chunk,len(filters)))
            model_ztm = np.zeros((len(z1),nt_chunk,len(m1),len(filters)))
        else :
            f_mod = None
            model_ztm = None
        # Approximate luminosity distance for flat LCDM
        # We will now use the BPZ routines to load the flux-redshift model for each template.
        # We'll have interpolated and discretized versions
        
        lambdaRef = 4.5e3
        if rank == 0 :
            print('Inputing Templates and Filters....',flush=True)
            if not HPC :
                pbar = tqdm(total = nt_chunk)
        if rank < nt :    
            for it in range(nt_chunk):
                if not HPC :
                    if rank == 0 :
                        pbar.update(1)
                seddata = np.genfromtxt('{}/{}'.format(template_folder,template_names[it]))
                seddata[:, 1] *= seddata[:, 0]**2. / 3e18
                ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
                seddata[:, 1] /= ref
                sed_interp = interp1d(seddata[:, 0], seddata[:, 1])
                for jf in range(len(filters)):
                    data = np.genfromtxt('{}/{}'.format(filter_folder,filters[jf]))
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
                        
            f_mod = sub_comm.gather(f_mod,root=0)
        
        if rank == 0 :
            f_mod = np.concatenate(f_mod,axis=1)
            model_ztm = model_zt_to_ztm(f_mod,m1)
            #np.save('{}/model.npy'.format(output_folder),model_ztm)
       
        comm.Barrier()
        f_mod = comm.bcast(f_mod,root=0)
        model_ztm = comm.bcast(model_ztm, root=0)
        shape = model_ztm.shape
        comm.Barrier()
        if rank == 0 :
            print('[DONE]',flush=True)

        if rank == 0 :
            print('Generate z,t,m Likelihoods....',flush=True)

        
        # convert magnitudes to fluxes
        f_obs_chunk = np.array_split(f_obs_chunk, n_split, axis=0)
        f_obs_err_chunk = np.array_split(f_obs_err_chunk, n_split, axis=0)
        ff_grid_edges = 10.**(-.4*m1)
        
        # surces are split to save RAM
        for split in range(n_split):
            nobj = f_obs_chunk[split].shape[0]
            nbins = nbins_caculator(zlist,mlist,t1)
            like = np.zeros((nobj,nbins),dtype=float)
            if not HPC :
                if rank == 0 :
                    pbar = tqdm(total = nobj)
            for i in range(nobj):  
                if not HPC:
                    if rank == 0 :
                        pbar.update(1)
                like[i] = likelihood_integration(
                           np.swapaxes(likelihood_ztm(f_obs_chunk[split][i, :], 
                                        f_obs_err_chunk[split][i, :],model_ztm),2,0)
                            ,zlist,mlist,t1)

            np.save('{}/likelihoods/likelihood_file_{}_{}.npy'.format(output_folder,rank,split),like)
            
            like = None

        if not HPC :
            gc.collect( )
            if rank == 0 :
                pbar.close()
        comm.Barrier()

        if rank == 0 :
            print('[DONE]',flush=True)

    if sample_generator:
        if rank == 0 :

            print('N(z) Gibbs Sampling....',flush=True)
            # input likelihoods
            nbins = np.load(
                '{}/likelihoods/likelihood_file_0_0.npy'.format(output_folder)).shape[1]
        else :
            nbins = None

        nbins = comm.bcast(nbins, root=0)
        prior = prior_maker(nbins,zlist,mlist,t1)

        for GRT in range(number_of_chains):

            if rank == 0:
                if sample_resume :
                    # continue from previus session
                    nbs = np.load('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                    output_folder, files_number+1, GRT, resume_number))[-1]
                    nbs = np.histogram(
                        nbs, bins=np.arange(nbins+1))[0]
                    nbs = nbs.astype(float) + prior
                    hbs  = np.random.dirichlet(nbs, 1)
                else :
                    #print(nbins)
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
                if not HPC :
                    pbar = tqdm(total=nsamples+1)
            # run per sample
            for kk in range(1, nsamples+1):
                for split in range(n_split):
                    # load likelihood files and flatten 
                    bpz_like_ztm_chunk = np.load('{}/likelihoods/likelihood_file_{}_{}.npy'.format(output_folder,rank,split))                       
                    # count bincounts of {z,t,m}_survey
                    if split == 0 :
                        nbs = big_sampler(bpz_like_ztm_chunk,hbs)
                    else : 
                        nbs = np.concatenate(
                            [nbs, big_sampler(bpz_like_ztm_chunk, hbs)])
                        #nbs += big_sampler(bpz_like_ztm_chunk,hbs)
                    

                # sum bincounts from all cores
                #nbs_all = comm.reduce(nbs, op=MPI.SUM, root=0)
                nbs_all = comm.gather(nbs,root=0)
                nbs = None
                if rank == 0:
                    # save samples 
                    nbs_all = np.hstack(nbs_all)
                    if kk >= 0:
                        fbs[nsamples_split_counter,:] = nbs_all      #BIG CHANGE save the number counts NOT,the probabilities
                        nsamples_split_counter += 1
                        if nsamples_split_counter == nsamples_split :
                            if sample_resume :
                                np.save('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                                    output_folder, files_number+1, GRT, kk+resume_number),fbs)
                            else :
                                np.save('{}/sample_chains_counts/sample_chain_noS_{}_{}_{}.npy'.format(
                                    output_folder, files_number+1, GRT, kk),fbs)
                            nsamples_split_counter = 0

                    # use bincounts to draw distribution sample p(distibution|flux,{z,t,m}_survey)
                    nbs_hist = np.histogram(
                        nbs_all, bins=np.arange(nbins+1))[0]
                    nbs_all = None
                    nbs_hist = nbs_hist.astype(float) + prior
                    #nbs_all = nbs_all.astype(float)
                    #nbs_all += prior
                    
                    hbs  = np.random.dirichlet(nbs_hist, 1)
                    #hbs = distribution_average(hbs[0],zlist,mlist,t1,running_mean_general_convol,[1.,20.,1.])
                nbs_all = None
                hbs = comm.bcast(hbs, root=0)
                if not HPC :
                    if rank == 0:    
                        pbar.update(1)
                '''    
                    gc.collect()
                '''
            if not HPC :
                if rank == 0 :
                    pbar.close()



        bpz_like_ztm_chunk = None
        fbs = None
        fbs_samples_ztm_chunk = None
        
        if not HPC :
            gc.collect()

        if rank == 0 :
            print('[DONE]',flush=True)

if not HPC :
    if rank ==0:
        total_time_end = time.time()
        
        print('Total Time =',float(total_time_end-total_time_start),flush=True)
