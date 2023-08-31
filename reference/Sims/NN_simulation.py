import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from mpi4py import MPI
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()
p_args = args.p
p = np.genfromtxt('{}'.format(p_args),dtype=None,delimiter=',',encoding='ascii')
p_float = np.genfromtxt('{}'.format(p_args),dtype=float,delimiter=',',encoding='ascii')
p_int = np.genfromtxt('{}'.format(p_args),dtype=int,delimiter=',',encoding='ascii')
p_bool = np.genfromtxt('{}'.format(p_args),dtype=bool,delimiter=',',encoding='ascii')

output_folder = '{}/blind_test'.format(p[1])
filter_folder = p[2]
template_folder = p[3]
template_list_folder = p[4]
error_folder = p[0]
filter_list_folder = p[5]
filter_list = p[8]


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def scatter(dataList, root=0, tag=0):

    """Scatter data across the nodes
    The default version apparently pickles the entire 'dataList',
    which can cause errors if the pickle size grows over 2^31 bytes
    due to fundamental problems with pickle in python 2. Instead,
    we send the data to each slave node in turn; this reduces the
    pickle size.

    @param dataList  List of data to distribute; one per node
                    (including root)
    @param root  Index of root node
    @param tag  Message tag (integer)
    @return  Data for this node
    """

    if comm.Get_rank() == root:
        for rank, data in enumerate(dataList):
            if rank == root:
                continue
            comm.send(data, rank, tag=tag)
        return dataList[root]
    else:
        return comm.recv(source=root, tag=tag)


true_e = np.load(
    '{}/error_function_flux.npy'.format(error_folder), allow_pickle=True)
bad_e = np.load(
    '{}/large_error_function.npy'.format(error_folder), allow_pickle=True)


def flux_to_mag_conversion(f, fe, bad_e):
    m = np.zeros_like(f)
    me = np.zeros_like(fe)

    for i in range(9):
        if i < 4:
            m[np.where(f[:, i] < fe[:, i]), i] = 99.
            me[np.where(f[:, i] < fe[:, i]), i] = bad_e[i][np.random.randint(len(bad_e[i]),
                                                                             size=len(np.where(f[:, i] < fe[:, i])))]
            m[np.where(f[:, i] > fe[:, i]), i] = -2.5 * (
                np.log10(f[np.where(f[:, i] > fe[:, i]), i]))
            me[np.where(f[:, i] > fe[:, i]), i] = 1.086 / \
                (np.abs(f[np.where(f[:, i] > fe[:, i]), i]) /
                 fe[np.where(f[:, i] > fe[:, i]), i])
        else:

            m[np.where(f[:, i] < 0), i] = 99.

            #m[np.where(f[:, i] > 0), i] = 30 - 2.5 * (
            #    np.log10(f[np.where(f[:, i] > 0), i]))
            m[np.where(f[:, i] > 0), i] = - 2.5 * (
                np.log10(f[np.where(f[:, i] > 0), i]))
            me[:, i] = 1.086 / (np.abs(f[:, i])/fe[:, i])

        random = np.arange(len(m))
        np.random.shuffle(random)
        lim = np.load('{}/BPZ_TEST_lim.npy'.format(error_folder))[random]

        for i in range(9):
            me[m[:,i]>lim[:,i],i] = lim[m[:,i]>lim[:,i],i]
            m[m[:,i]>lim[:,i],i] = 99.
            m[me[:,i]>1.,i] = 99.
            me[me[:, i] > 1., i] = lim[me[:, i] > 1., i]
        
    return m, me


def DL(z):
                return np.exp(30.5 * z**0.04 - 21.7)

#ex = np.load('{}/extraction.npy'.format(error_folder))
flux = np.load('{}/flux_corrected.npy'.format(error_folder))
flux_e = np.load('{}/flux_e_corrected.npy'.format(error_folder))

#flux *= 10**(0.4*ex)
#flux_e *= 10**(0.4*ex)
#flux[:, 4:] *= 10**(-12)
#flux_e[:, 4:] *= 10**(-12):

flux_mean = np.mean(flux, axis=0)
flux_e_mean = np.mean(flux_e, axis=0)
flux /= flux_mean
flux_e /= flux_e_mean

train_objects = 6000000
random = np.arange(flux.shape[0])
np.random.shuffle(random)
flux = flux[random]
flux_e = flux_e[random]
flux = flux[:train_objects]
flux_e = flux_e[:train_objects]


model = MLPRegressor(random_state=1, max_iter=500, activation='logistic')
model.fit(flux.tolist(), flux_e.tolist())

''' File Names '''

filters = np.loadtxt(
    '{}/{}'.format(filter_list_folder, filter_list), dtype=str)
nf = len(filters)
reffilter = p_int[17]

z_max, z_min, z_binsize = p_float[10:13]
m_max, m_min = p_int[15:17]

nobj = 1000000

template_list = p[9]

nt = 0
template_names_full = None
template_names = None
if rank == 0:
    template_names_full = np.loadtxt(
        '{}/{}'.format(template_list_folder, template_list), dtype=str)
    
    nt = len(template_names_full)
    if nt > size:
        template_names = np.array_split(template_names_full, size, axis=0)
    else:
        template_names = np.array_split(template_names_full, nt, axis=0)
#'''
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

z_grid_binsize = p_float[13]
z_grid_edges = np.arange(z_min, z_max, z_grid_binsize)
z_grid = (z_grid_edges[1:] + z_grid_edges[:-1])/2.
nz = len(z_grid)

if rank < nt:
    nt_chunk = len(template_names)
    f_mod = np.zeros((nz, nt_chunk, nf))*0.
else:
    f_mod = None

lambdaRef = 4.5e3
if rank == 0:
    print('Inputing Templates and Filters....', flush=True)
    pbar = tqdm(total=nt_chunk)
if rank < nt:
    for it in range(nt_chunk):
        if rank == 0:
            pbar.update(1)
        seddata = np.genfromtxt(
            '{}/{}'.format(template_folder, template_names[it]))
        seddata[:, 1] *= seddata[:, 0]**2. / 3e18
        ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
        seddata[:, 1] /= ref
        sed_interp = interp1d(seddata[:, 0], seddata[:, 1])
        for jf in range(nf):
            data = np.genfromtxt(
                '{}/{}'.format(filter_folder, filters[jf]))
            xf, yf = data[:, 0], data[:, 1]
            yf /= xf  # divide by lambda
            # Only consider range where >1% max
            ind = np.where(yf > 0.01*np.max(yf))[0]
            lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
            norm = np.trapz(yf, x=xf)
            for iz in range(z_grid.size):
                opz = (z_grid[iz] + 1)
                xf_z = np.linspace(
                    lambdaMin / opz, lambdaMax / opz, num=5000)
                yf_z = interp1d(xf / opz, yf)(xf_z)
                ysed = sed_interp(xf_z)
                f_mod[iz, it, jf] = np.trapz(
                    ysed * yf_z, x=xf_z) / norm
                f_mod[iz, it, jf] *= opz**2. / \
                    DL(z_grid[iz])**2. / (4*np.pi)

    #f_mod[:, :,:] = np.clip(f_mod[:,:,:],0.,1e300) #new
    f_mod = sub_comm.gather(f_mod, root=0)

if rank == 0:
    f_mod = np.concatenate(f_mod, axis=1)
comm.Barrier()
f_mod = comm.bcast(f_mod, root=0)
f_mod_interps = np.zeros((nt, nf), dtype=interp1d)

for jt, jf in np.ndindex(f_mod_interps.shape):
    f_mod_interps[jt, jf] = InterpolatedUnivariateSpline(
        z_grid, f_mod[:, jt, jf])

comm.Barrier()

if rank == 0:
    print('[DONE]', flush=True)

# input distribution parameters 
if rank == 0:
    print('Inputing Distibution Parameters....', flush=True)

    
    '''
    redshifts, imags, types = np.random.uniform(z_min, z_max, size=nobj), np.random.uniform(
        m_min, m_max, size=nobj), np.random.choice(nt, size=nobj)
    data = np.vstack((redshifts, types, imags)).T
    np.save('{}/true_distribution.npy'.format(output_folder), data)
    '''

    #'''
    dist = np.load('{}/true_distribution.npy'.format(output_folder))
    redshifts, imags, types = dist[:, 0], dist[:, 2], dist[:, 1]
    types = types.astype(int)
    #'''
    
    nobj = len(redshifts)
    redshifts = np.array_split(redshifts, size, axis=0)
    imags = np.array_split(imags, size, axis=0)
    types = np.array_split(types, size, axis=0)
else:
    redshifts = None
    imags = None
    types = None
redshifts = comm.scatter(redshifts, root=0)
imags = comm.scatter(imags, root=0)
types = comm.scatter(types, root=0)


if rank == 0 :
    f_obs = np.zeros((nobj,nf))
    f_obs_err = np.zeros((nobj,nf))
    f_obs = np.array_split(f_obs, size, axis=0)
    f_obs_err = np.array_split(f_obs_err, size, axis=0)
else :
    f_obs = None
    f_obs_err = None
f_obs = comm.scatter(f_obs, root=0)
f_obs_err = comm.scatter(f_obs_err, root=0)

for jf in range(nf):
    if jf == reffilter:
        f_obs[:,reffilter] = imags
    else:
        cmod = np.array([-2.5*np.log10(np.clip(
                        f_mod_interps[types[i],jf](redshifts[i])/f_mod_interps[types[i],reffilter](redshifts[i])
                        ,1e-5,1e5)) for i in range(f_obs.shape[0])])       
        f_obs[:,jf] = imags + cmod

# convert mags to fluxs
'''
flux = np.zeros_like(f_obs)
for jf in range(nf):
    if jf < 4 :
        flux[:,jf] = 10**(-.4*f_obs[:,jf])
    else:
        flux[:,jf] = 10**(-.4*(f_obs[:,jf]-30))
'''

flux = 10**(-.4*f_obs)



# fit with flux errors with NN
flux /= flux_mean
flux_error = np.abs(np.array(model.predict(flux.tolist())))*flux_e_mean
flux *= flux_mean

#flux += flux_error * np.random.randn(flux.shape)
for jf in range(nf):
    flux[:,jf] = flux[:,jf] + flux_error[:,jf] * np.random.randn(flux.shape[0])

np.save('{}/sim_flux.npy'.format(output_folder), flux)
np.save('{}/sim_fluxe.npy'.format(output_folder), flux_error)
f_obs,f_obs_err = flux_to_mag_conversion(flux,flux_error,bad_e)

f_obs = comm.gather(f_obs, root=0)
f_obs_err = comm.gather(f_obs_err, root=0)

if rank == 0:
    f_obs = np.concatenate(f_obs, axis=0)
    f_obs_err = np.concatenate(f_obs_err, axis=0)
    print('[DONE]', flush=True)


    np.save('{}/sim_f.npy'.format(output_folder), f_obs)
    np.save('{}/sim_fe.npy'.format(output_folder), f_obs_err)

if rank == 0 :
    print('Simulation Compleat')
    print('Data Chunk Size : ',f_obs.shape)
    print('Gathering Data...')

    bpz_file = np.concatenate((f_obs,f_obs_err),axis=1)
    bpz_file = bpz_file[:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17]]
    print('Data Gathered',bpz_file.shape)

    print('Add Unobserved Criteria')
    ff = bpz_file

    print('convert undetected (m=99) to unobserved (m=-99)')

    for i in range(9):
        ff[ff[:,(2*i)]==99.,(2*i)+1] = 0.
        ff[ff[:,(2*i)]==99.,(2*i)] = -99.

    print('Estimate M_0 values')

    def sim_to_M_0(m,m_r,true_e, true_e_r):
        
        bin_size = 0.05
        bins = np.arange(17, 28, bin_size)
        me = np.zeros_like(m)
        
        for j in range(len(bins)):
            if j == 0:
                n_sim = len(me[m <= bins[j]+bin_size])
                if n_sim == 0:
                    continue
                if true_e[j][0] == 'None':
                    used_value = true_e[j][1]
                    me[m <= bins[j]+bin_size] = true_e[used_value][np.random.randint(len(true_e[used_value]),
                                                                                                            size=n_sim)]
                else:
                    me[m <= bins[j]+bin_size] = true_e[j][np.random.randint(len(true_e[j]),size=n_sim)]
                selected = np.where((m_r <= bins[j]+bin_size)&(m == -99.),True,False)
                n_sim = np.count_nonzero(selected)
                if n_sim == 0:
                    continue
                if true_e_r[j][0] == 'None':
                    used_value = true_e_r[j][1]
                    
                    me[selected] = true_e_r[used_value][np.random.randint(len(true_e_r[used_value]),size=n_sim)]
                else:
                    me[selected] = true_e_r[j][np.random.randint(len(true_e_r[j]),size=n_sim)]
            elif j == len(bins)-1:
                n_sim = len(me[m > bins[j]])
                if n_sim == 0:
                    continue
                if true_e[j][0] == 'None':
                    used_value = true_e[j][1]
                    me[m > bins[j]] = true_e[used_value][np.random.randint(len(true_e[used_value]),size=n_sim)]
                else:
                    me[m > bins[j]] = true_e[j][np.random.randint(len(true_e[j]),size=n_sim)]
                selected = np.where((m_r > bins[j])&(m == -99.),True,False)
                n_sim = np.count_nonzero(selected)
                if n_sim == 0:
                    continue
                if true_e_r[j][0] == 'None':
                    used_value = true_e_r[j][1]
                    me[selected] = true_e_r[used_value][np.random.randint(len(true_e_r[used_value]),size=n_sim)]
                else:
                    me[selected] = true_e[j][np.random.randint(len(true_e[j]),size=n_sim)]
            else:
                n_sim = len(me[np.where((m > bins[j]) & (m <= bins[j]+bin_size))])
                if n_sim == 0:
                    continue
                if true_e[j][0] == 'None':
                    used_value = true_e[j][1]
                    me[np.where((m > bins[j]) & (m <= bins[j]+bin_size))] = true_e[used_value][np.random.randint(len(true_e[used_value]),size=n_sim)]
                else:
                    me[np.where((m > bins[j]) & (m <= bins[j]+bin_size))] = true_e[j][np.random.randint(len(true_e[j]),size=n_sim)]
                selected = np.where((m_r > bins[j]) & (m_r <= bins[j]+bin_size) & (m == -99.),True,False)
                n_sim = np.count_nonzero(selected)
                if n_sim == 0:
                    continue
                if true_e_r[j][0] == 'None':
                    used_value = true_e_r[j][1]
                    me[selected] = true_e_r[used_value][np.random.randint(len(true_e_r[used_value]),size=n_sim)]
                else:
                    me[selected] = true_e_r[j][np.random.randint(len(true_e_r[j]),size=n_sim)]
        return me


    e_i = np.load('{}/M_0_i_function.npy'.format(error_folder),
                    allow_pickle=True,encoding = 'latin1')
    e_r = np.load('{}/M_0_r_function.npy'.format(error_folder),
                    allow_pickle=True,encoding = 'latin1')


    ff = np.array_split(ff,4,axis=0)

    for i in range(4):
        M_0 = sim_to_M_0(ff[i][:,6],ff[i][:,4],e_i,e_r)
        print('M_0',M_0.shape)
        print('old ff',ff[i].shape)
        ff[i] = np.concatenate([ff[i],np.array([M_0]).T],axis=1)
        print('new ff',ff[i].shape)
        np.savetxt('{}/Final_model_{}.cat'.format(output_folder,i),ff[i])
