import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
#from getdist import plots, MCSamples

from matplotlib.offsetbox import AnchoredText
import argparse


print('Input Parameters...', flush=True)

# inputs param file
parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()
p_args = args.p
p = np.genfromtxt('{}'.format(p_args), dtype=None,
                  delimiter=',', encoding='ascii')
p_float = np.genfromtxt('{}'.format(p_args), dtype=float,
                        delimiter=',', encoding='ascii')
p_int = np.genfromtxt('{}'.format(p_args), dtype=int,
                      delimiter=',', encoding='ascii')
p_bool = np.genfromtxt('{}'.format(p_args), dtype=bool,
                       delimiter=',', encoding='ascii')

# assign varibles from param file
flux_folder, output_folder, filter_folder, template_folder, template_list_folder, filter_list_folder = p[
    :6]
flux_name, flux_error_name, filter_list, template_list = p[6:10]
zlist = np.loadtxt('{}'.format(p_args), skiprows=32, max_rows=1)
mlist = np.loadtxt('{}'.format(p_args), skiprows=33, max_rows=1)
reffilter, n_objects, nsamples, nsamples_split, n_split, number_of_chains = p_int[12:18]
likelihood_generator, sample_generator, sample_resume, WEIGHTS, RANDOM = p_bool[18:23]

smin, smax, n_paralell = p_int[24:27]


t = np.loadtxt('{}/{}'.format(template_list_folder, template_list), dtype=str)
nt = len(t)

z2 = np.round(np.arange(zlist[0], zlist[1]+(2*zlist[3]), zlist[3]), 5)
m2 = np.round(np.arange(mlist[0], mlist[1]+(2*mlist[3])), 5)
t1 = np.arange(0, nt)

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out.T

vectorised_mn = np.vectorize(multinomial_rvs, excluded = ['p'] ,signature='(i),(i,j)->(j,i)')
vectorised_mn_test = np.vectorize(multinomial_rvs, signature='(i),(i,j)->(j,i)')


def remake(like,zlist,mlist,t1):
    z1 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3]),zlist[3]), 5)[:-2] 
    m = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
    final = np.zeros((like.shape[0],len(m), len(t1), len(z1)))
    total_z = 0   
    for i in range(len(m)):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+i]),zlist[3+i]), 5) 
        total_z2 = total_z + len(z)-2
        trial = like[:,(len(t1)*total_z):(total_z2*len(t1))].reshape(
                    like.shape[0],len(t1),(len(z)-2))
        for j in range(len(z)-2):
            final[:,i,:,(z1 >= z[j]) & (z1 < z[j+1])] = np.moveaxis(trial[:,:,j][:,:,None],2,0)/ \
                            np.sum((z1 >= z[j]) & (z1 < z[j+1]))
        
        
        total_z = total_z2
        trial = None
    return np.swapaxes(final,1,3)

def remake_tomo(like,zlist,mlist,t1):
    z1 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3]),zlist[3]), 5)[:-2] 
    m = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
    final = np.zeros((len(m), len(t1), len(z1)))
    total_z = 0   
    for i in range(len(m)):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+i]),zlist[3+i]), 5) 
        total_z2 = total_z + len(z)-2
        trial = like[(len(t1)*total_z):(total_z2*len(t1))].reshape(
                    len(t1),(len(z)-2))
        for j in range(len(z)-2):
            final[i,:,(z1 >= z[j]) & (z1 < z[j+1])] = np.moveaxis(trial[:,j][:,None],1,0)/ \
                            np.sum((z1 >= z[j]) & (z1 < z[j+1]))
        
        
        total_z = total_z2
    return np.swapaxes(final,0,2)

def remake3(like,zlist,mlist,t1):
    #print(like.shape,zlist,mlist,t1)
    z1 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3]),zlist[3]), 5)[:-2]
    m = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
    final = np.zeros((like.shape[0],len(m), len(t1), len(z1)))
    total_z = 0
    for i in range(len(m)):
        z = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3+i]),zlist[3+i]), 5)
        total_z2 = total_z + len(z)-2
        trial = like[:,(len(t1)*total_z):(total_z2*len(t1))].reshape(
                    like.shape[0],len(t1),(len(z)-2))
        for j in range(len(z)-2):
            p = np.random.uniform(size=[like.shape[0],len(t1),np.sum((z1 >= z[j]) & (z1 < z[j+1]))])
            p /= p.sum(axis=(2))[:,:,None]
            final[:,i,:,(z1 >= z[j]) & (z1 < z[j+1])] = np.moveaxis(vectorised_mn_test(trial[:,:,j].astype(int),p),0,1)
        total_z = total_z2
    trial = None
    p = None
    return np.swapaxes(final,1,3)


def nbins_caculator(zlist, mlist, t1):
    m1 = np.round(np.arange(mlist[0], mlist[1]+(2*mlist[3]), mlist[3]), 5)[:-2]
    total_z = 0
    for i in range(len(m1)):
        z = np.round(np.arange(zlist[0], zlist[1] +
                               (2*zlist[3+i]), zlist[3+i]), 5)
        total_z += len(z)-2
    return total_z * len(t1)

def rebin(like,zlist,mlist,t1):
   m = np.round(np.arange(mlist[0],mlist[1]+(2*mlist[3]),mlist[3]), 5)[:-2]
   z1 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[3]),zlist[3]), 5)
   z2 = np.round(np.arange(zlist[0],zlist[1]+(2*zlist[2+len(m)]),zlist[2+len(m)]), 5)
   final = np.zeros((like.shape[0],len(z2)-2, len(t1), len(m)))
   for i in range(len(z2)-2):
      final[:,i,:,:] = np.sum(like[:,(z1[:-2] >= z2[i]) & (z1[:-2] < z2[i+1]),:,:],axis=(1))
   return final


print('Loading Weights and BPZ files', flush=True)

random = np.load('{}/random.npy'.format(output_folder))

if WEIGHTS:
   weights = np.load('{}/weight.npy'.format(flux_folder))[random]
   weights = weights[:n_objects]
else:
   weights = np.ones(n_objects)

weights_split = np.array_split(weights, n_paralell)
bpz_results = np.load('{}/BPZ.npy'.format(flux_folder))[random]
bpz_results = bpz_results[:n_objects]
bpz_results_split = np.array_split(bpz_results, n_paralell)


data_folder = '/rds/general/user/gk1513/home/SCRATCH/cx2-scratch2/Photo_Z_3D/SIMULATION_DATA'
data = np.load('{}/true_distribution.npy'.format(data_folder))
data_weights = np.load('{}/weight.npy'.format(data_folder))
data_bpz = np.load('{}/BPZ.npy'.format(data_folder))

print('Loading Samples... : ',flush=True)

# load sample chains 
n_bins = nbins_caculator(zlist, mlist, t1)
mod = 0
#print(int(smax/3))
for i in np.arange(smin, smax+nsamples_split, nsamples_split):
    if i <= int(smax/3):
        continue
    else : 
        mod +=1
    if mod == 1 :
        sample_chunck = np.load(
            '{}/sample_chains_counts/sample_chain_noS_1_0_{}.npy'.format(output_folder, i))[(int(smax/3)%nsamples_split):]
    else : 
        sample_chunck = np.load(
            '{}/sample_chains_counts/sample_chain_noS_1_0_{}.npy'.format(output_folder, i))
    sample_hist = np.zeros([sample_chunck.shape[0], n_bins])
    for j in range(sample_hist.shape[0]):
        sample_hist[j] = np.histogram(sample_chunck[j], bins=np.arange(n_bins+1),weights=weights,density=False)[0]
    sample_chunck = None
    if mod == 1 :
        sample_full = sample_hist
    else :
        sample_full = np.concatenate((sample_full,sample_hist),axis=0)
    sample_hist = None

print('Full Sample Chains : ',sample_full.shape,flush=True)
sample_full = remake3(sample_full,zlist,mlist,t1)
sample_full /= np.sum(sample_full,axis=(1,2,3))[:,None,None,None]
print('After Remake : ',sample_full.shape,flush=True)


#'''
print('Loading Likelihood...', flush=True)

for j in range(n_paralell):

    for i in range(n_split):

        like_chunck = np.load(
            '{}/likelihoods/likelihood_file_{}_{}.npy'.format(output_folder, j, i))
        like_chunck /= np.sum(like_chunck, axis=(1))[:, None]
        like_chunck *= np.array_split(weights, n_paralell)[j][:, None]
        like_chunck = remake(like_chunck, zlist, mlist, t1)
        like_chunck = np.sum(like_chunck, axis=0)
        if j == 0 and i == 0:
            like_stacked = like_chunck
        else:
            like_stacked += like_chunck
        like_chunck = None
#n_bins = np.prod(like_stacked.shape)
like_stacked /= np.sum(like_stacked)
print('Stacked Likelihood : ', like_stacked.shape, flush=True)
#'''




t1 = np.arange(0, nt)


x_z_plot = np.round(
    np.arange(zlist[0], zlist[1]+(2*zlist[3]), zlist[3]), 5)[:-2] + (zlist[3]/2)
x_m_plot = np.round(
    np.arange(mlist[0], mlist[1]+(2*mlist[3]), mlist[3]), 5)[:-2] + (mlist[3]/2)
x_t = np.arange(0,nt)+0.5

x_z_data = np.round(
    np.arange(zlist[0], zlist[1]+(3*zlist[3]), zlist[3])[:-2], 5)
x_m_data = np.round(
    np.arange(mlist[0], mlist[1]+(2*mlist[3]), mlist[3]), 5)[:-1]
x_t_data = np.arange(0, nt+1)





def function_plot(samples,data,axis_label,file_name):
   if axis_label == 'z' :
      a1,a2,a3 = 2,3,0
      x = x_z_plot
      x_data = x_z_data

   elif axis_label == 'm' :
      a1,a2,a3 = 1,2,2
      x = x_m_plot
      x_data = x_m_data
   else :
      a1,a2,a3 = 1,3,1
      x = x_t
      x_data = x_t_data

   sigma_lowlow, sigma_low, sigma_high, sigma_highhigh,mean = np.percentile(
       np.sum(samples, axis=(a1, a2)), [2.3, 15.9, 84.1, 97.7, 50.0], axis=0)
      
   fig, ax = plt.subplots()
   plt.fill_between(x,sigma_low,sigma_high,alpha=0.4,color='C3',zorder=1)
   plt.fill_between(x, sigma_lowlow,sigma_highhigh, alpha=0.15, color='C3', zorder=0)
   plt.plot(x,np.sum(like_stacked, axis=(a1-1, a2-1)), color='C0', zorder=10,label='Likelihood')
   plt.plot(x,mean, color='C3', zorder=5,label='Posterior')

   truth = np.histogram(data[:,a3],bins=x_data
            ,range=[x[0],x[-1]],weights=data_weights)[0]
   plt.plot(x,truth/np.sum(truth), color='C2', zorder=20,label='Truth')
   
   
   if axis_label == 'z':
      values = np.ones([np.sum(samples, axis=(a1, a2)).shape[0],len(x)])*x
      mean_r = np.average(values,weights=np.sum(samples, axis=(a1, a2)),axis=1)
      print('truth: ',np.average(x,weights=truth/np.sum(truth)))
      print('mean: ',np.mean(mean_r),'sigma:',np.std(mean_r))
   plt.xlabel('{}'.format(axis_label))
   plt.ylabel('N({})'.format(axis_label))
   plt.legend()

   plt.savefig('{}/plots/{}'.format(output_folder,file_name))   
   '''
   true = np.histogram(data[:,a3],bins=x
            ,range=[x[0],x[-1]])[0]
   labels = ['{}'.format(i) for i in x[:sample_plot.shape[1]]]
   markers = {'{}'.format(x[:sample_plot.shape[1]][i]):true[i] for i in np.arange(sample_plot.shape[1])}
   samples = MCSamples(samples=sample_plot,labels=labels,names=labels)
   g = plots.get_subplot_plotter()
   g.triangle_plot(samples, filled=True,markers=markers)
   g.export('{}/{}/tri_{}'.format(output_folder,plot_folder,file_name))
   '''


#print('True Distribution : ',data.shape,flush=True)
#print(weights.shape)
function_plot(sample_full,data,'z','Full_Survey_z.png')
print('z',flush=True)
function_plot(sample_full,data,'m','Full_Survey_m.png')
print('m',flush=True)
function_plot(sample_full,data,'t','Full_Survey_t.png')

print('Full Sky Ploted',flush=True)

n_rows = np.int(np.ceil(sample_full.shape[3]/3)) 

fig_z, ax_z = plt.subplots(n_rows, 3, figsize=(15, n_rows*5), sharex=True, sharey=False)
fig_t, ax_t = plt.subplots(n_rows, 3, figsize=(15, n_rows*5), sharex=True, sharey=False)


for i in range(sample_full.shape[3]):
    
    
    row = i // 3
    col = i % 3

    sample  = sample_full[:,:,:,i]
    sample /= np.sum(sample,axis=(1,2))[:,None,None]
    sigma_lowlow, sigma_low, sigma_high, sigma_highhigh = np.percentile(
        np.sum(sample,axis=2), [2.3, 15.9, 84.1, 97.7, ], axis=0)

    ax_z[row,col].fill_between(x_z_plot,sigma_low,sigma_high,alpha=0.4,color='C3',zorder=1)
    ax_z[row,col].fill_between(x_z_plot,sigma_lowlow,sigma_highhigh,alpha=0.15,color='C3',zorder=0)
    ax_z[row, col].plot(x_z_plot, like_stacked[:, :, i].sum(axis=(1))/np.sum(like_stacked[:,:,i]), color='C0', zorder=10,label='Likelihood')
    ax_z[row, col].plot(x_z_plot, np.mean(
        sample_full[:, :, :, i].sum(axis=(2)), axis=0), color='C3', zorder=5,label='Posterior')
    anchored_text = AnchoredText('m={}-{}'.format(np.round(x_m_plot[i], 2), np.round(x_m_plot[i]+1, 2)), loc='upper center',frameon='True')
    ax_z[row,col].add_artist(anchored_text)
    truth  = np.histogram(data[((data[:, 2] > x_m_data[i]) & (data[:, 2] < x_m_data[i+1])), 0], bins=x_z_data, density=False,weights=data_weights[((data[:, 2] > x_m_data[i]) & (data[:, 2] < x_m_data[i+1]))])[0]
    ax_z[row,col].plot(x_z_plot,truth/np.sum(truth), color='C2', zorder=20,label='Truth')
    
    if i == 2 :
      ax_z[row, col].legend()
    if i > 1 :
      ax_z[row,col].set_xlabel('z')
    if i == 0 or i == 3 :
      ax_z[row,col].set_ylabel("N(z)")
           

    sigma_lowlow, sigma_low, sigma_high, sigma_highhigh = np.percentile(
        np.sum(sample,axis=1), [2.3, 15.9, 84.1, 97.7, ], axis=0)
    ax_t[row,col].fill_between(x_t,sigma_low,sigma_high,alpha=0.4,color='C3',zorder=1)
    ax_t[row,col].fill_between(x_t,sigma_lowlow,sigma_highhigh,alpha=0.15,color='C3',zorder=0)
    ax_t[row, col].plot(x_t, like_stacked[:,:, i].sum(axis=(0))/np.sum(like_stacked[:,:,i]), color='C0',zorder=10,label='Likelihood')
    ax_t[row, col].plot(x_t, np.mean(sample_full[:,:, :, i].sum(axis=(1)), axis=0), color='C3',zorder=5,label='Posterior')
    anchored_text = AnchoredText('m={}-{}'.format(np.round(x_m_plot[i], 2), np.round(x_m_plot[i]+1, 2)), loc='upper center',frameon='True')
    ax_t[row,col].add_artist(anchored_text)

    truth = np.histogram(data[((data[:, 2] > x_m_data[i]) & (data[:, 2] < x_m_data[i+1])), 1], bins=x_t_data,weights=data_weights[((data[:, 2] > x_m_data[i]) & (data[:, 2] < x_m_data[i+1]))])[0]
    ax_t[row,col].plot(x_t,truth/np.sum(truth),color='C2',zorder=20,label='Truth')


    if i == 2 :
      ax_t[row, col].legend()
    if i > 1 :
      ax_t[row,col].set_xlabel('t')
    if i == 0 or i == 3 :
      ax_t[row,col].set_ylabel("N(t)")





n_rows = np.int(np.ceil(len(x_t[:-1])/3))
fig_tz, ax_tz = plt.subplots(n_rows, 3, figsize=(15, n_rows*5), sharex=True, sharey=False)


for i in range(sample_full.shape[2]):

    row = i // 3
    col = i % 3
    
    sample  = sample_full[:,:,i,:]
    sample /= np.sum(sample,axis=(1,2))[:,None,None]    

    sigma_lowlow, sigma_low, sigma_high, sigma_highhigh = np.percentile(
        np.sum(sample,axis=2), [2.3, 15.9, 84.1, 97.7, ], axis=0)
    

    ax_tz[row,col].fill_between(x_z_plot,sigma_low,sigma_high,alpha=0.4,color='C3',zorder=1)
    ax_tz[row,col].fill_between(x_z_plot,sigma_lowlow,sigma_highhigh,alpha=0.15,color='C3',zorder=0)
    ax_tz[row, col].plot(x_z_plot, like_stacked[:, i, :].sum(axis=(1))/np.sum(like_stacked[:,i,:]), color='C0',zorder=10,label='Likelihood')
    ax_tz[row, col].plot(x_z_plot, np.mean(sample_full[:, :, i, :].sum(axis=(2)), axis=0), 'C3',zorder=5,label='Posterior')
    #ax_tz[row, col].hist(data[(data[:, 1] == x_t[i]), 0], bins=x_z, density=False,
    #                     histtype='step', label='t={}'.format(x_t[i]), color='C2', 
    #                     weights=weights[data[:, 1] == x_t[i]],zorder=20)
    truth = np.histogram(data[(data[:, 1] == x_t[i]), 0], bins=x_z_data, density=False,weights=data_weights[data[:, 1] == x_t[i]])[0]
    ax_tz[row,col].plot(x_z_plot,truth/np.sum(truth),color='C2',zorder=20,label='Truth')
    ax_tz[row, col].legend()

sample_full = None


fig_tz.tight_layout()
fig_tz.subplots_adjust(hspace=.0)
fig_tz.subplots_adjust(wspace=.0)
fig_tz.savefig('{}/plots/Full_Survey_zt.png'.format(output_folder))
print('z/t Distribution Printed',flush=True)
fig_z.tight_layout()
fig_z.subplots_adjust(hspace=.0)
fig_z.subplots_adjust(wspace=.0)
fig_z.savefig('{}/plots/Full_Survey_zm.png'.format(output_folder))
print('z/m Distribution Printed',flush=True)
fig_t.tight_layout()
fig_t.subplots_adjust(hspace=.0)
fig_t.subplots_adjust(wspace=.0)
fig_t.savefig('{}/plots/Full_Survey_tm.png'.format(output_folder))
print('t/m Distribution Printed',flush=True)


bpz_min = [0.1,0.3,0.5,0.7,0.9]
bpz_max = [0.3,0.5,0.7,0.9,1.2]

def function_plot_tomo(samples,data,weight,ax,row,col,axis_label,i,like_stacked):
   if axis_label == 'z' :
      a1,a2,a3 = 2,3,0
      x = x_z_plot
      x_data = x_z_data

   elif axis_label == 'm' :
      a1,a2,a3 = 1,2,2
      x = x_m_plot
      x_data = x_m_data
   else :
      a1,a2,a3 = 1,3,1
      x = x_t
      x_data = x_t_data

   
   sigma_lowlow, sigma_low, sigma_high, sigma_highhigh, mean = np.percentile(
       np.sum(samples, axis=(a1, a2)), [2.3, 15.9, 84.1, 97.7,50.0 ], axis=0)
   
   ax[row,col].fill_between(x,sigma_low,sigma_high,alpha=0.4,color='C3',zorder=1)
   ax[row,col].fill_between(x, sigma_lowlow,sigma_highhigh, alpha=0.15, color='C3', zorder=0)
   ax[row,col].plot(x,np.sum(like_stacked, axis=(a1-1, a2-1)), color='C0', zorder=10,label='Likelihood')
   ax[row,col].plot(x,mean, color='C3', zorder=5,label='Posterior')
   anchored_text = AnchoredText(r'${} < z_B < {}$'.format(bpz_min[i],bpz_max[i]), loc='upper center',frameon='True')
   ax[row,col].add_artist(anchored_text)
   truth = np.histogram(data[:,a3],bins=x_data
            ,range=[x[0],x[-1]],weights=weight)[0]
   ax[row,col].plot(x,truth/np.sum(truth), color='C2', zorder=20,label='Truth')


   if axis_label == 'z':
      values = np.ones([np.sum(samples, axis=(a1, a2)).shape[0],len(x)])*x
      mean_r = np.average(values,weights=np.sum(samples, axis=(a1, a2)),axis=1)
      print('mean: ',np.mean(mean_r),'sigma:',np.std(mean_r))
      print('truth: ',np.average(x,weights=truth/np.sum(truth)))
   if i == 2 :
      ax[row,col].legend()
   if i > 1 :
      ax[row,col].set_xlabel("{}".format(axis_label))
   if i == 0 or i == 3 :
      ax[row,col].set_ylabel("N({})".format(axis_label))

figz, axz = plt.subplots(2, 3, figsize=(15, 2*5), sharex=True, sharey=False)
figt, axt = plt.subplots(2, 3, figsize=(15, 2*5),sharex=True, sharey=False)
figm, axm = plt.subplots(2, 3, figsize=(15, 2*5),sharex=True, sharey=False)

for i in reversed(range(5)):
    print('Tomographic Bin : ',i)
    
    
    row = i // 3
    col = i % 3

    weight_tomo = weights[((bpz_results > bpz_min[i]) &
                           (bpz_results < bpz_max[i]))]
    mod = 0
    for j in np.arange(smin, smax+nsamples_split, nsamples_split):
        if j <= int(smax/3):
            continue
        else : 
            mod +=1
        if mod == 1 :
            sample_chunck = np.load('{}/sample_chains_counts/sample_chain_noS_1_0_{}.npy'
                                    .format(output_folder, j))[(int(smax/3) % nsamples_split):, ((bpz_results > bpz_min[i])
                                                               & (bpz_results < bpz_max[i]))]
        else:
           sample_chunck = np.load('{}/sample_chains_counts/sample_chain_noS_1_0_{}.npy'
                                .format(output_folder, j))[:, ((bpz_results > bpz_min[i])
                                                                                             & (bpz_results < bpz_max[i]))]
        sample_hist = np.zeros([sample_chunck.shape[0], n_bins])
        for k in range(sample_hist.shape[0]):
            sample_hist[k] = np.histogram(
                sample_chunck[k], bins=np.arange(n_bins+1), weights=weight_tomo)[0]
            sample_chunck[k] = None
        if mod == 1:
            sample_tomo = sample_hist
        else:
            sample_tomo = np.concatenate((sample_tomo, sample_hist), axis=0)
        sample_chunck = None
        sample_hist = None
    

    
    for j in range(n_paralell):

        for k in range(n_split):

            like_chunck = np.load(
                '{}/likelihoods/likelihood_file_{}_{}.npy'
                .format(output_folder, j, k))[((np.array_split(bpz_results, n_paralell)[j] > bpz_min[i]) & (
                    np.array_split(bpz_results, n_paralell)[j] < bpz_max[i]))]
            

            weight_tomo = np.array_split(weights, n_paralell)[j][((np.array_split(bpz_results, n_paralell)[j] > bpz_min[i]) & (
                np.array_split(bpz_results, n_paralell)[j] < bpz_max[i]))]
            like_chunck /= np.sum(like_chunck, axis=(1))[:, None]
            like_chunck *= weight_tomo[:,None]
            like_chunck = remake(like_chunck, zlist, mlist, t1)
            like_chunck = np.sum(like_chunck, axis=0)
            if j == 0 and k == 0:
                like_stacked = like_chunck
            else:
                like_stacked += like_chunck
            like_chunck = None
    
    like_stacked /= np.sum(like_stacked)
    data_tomo = data[((data_bpz > bpz_min[i]) & (data_bpz < bpz_max[i]))]
    weight_tomo = data_weights[((data_bpz > bpz_min[i]) &
                           (data_bpz < bpz_max[i]))]
    
    sample_tomo = remake3(sample_tomo, zlist, mlist, t1)
    sample_tomo /= np.sum(sample_tomo,axis=(1,2,3))[:,None,None,None]
    function_plot_tomo(sample_tomo,data_tomo,weight_tomo,axz,row,col,'z',i,like_stacked)
    function_plot_tomo(sample_tomo,data_tomo,weight_tomo,axt,row,col,'t',i,like_stacked)
    function_plot_tomo(sample_tomo,data_tomo,weight_tomo,axm,row,col,'m',i,like_stacked)
    sample_tomo = None
    like_stacked = None
    data_tomo = None
    weight_tomo = None


figz.tight_layout()
figz.subplots_adjust(hspace=.0)
figz.subplots_adjust(wspace=.0)
figt.tight_layout()
figt.subplots_adjust(hspace=.0)
figt.subplots_adjust(wspace=.0)
figm.tight_layout()
figm.subplots_adjust(hspace=.0)
figm.subplots_adjust(wspace=.0)
figz.savefig('{}/plots/tomo_z.png'.format(output_folder))
figm.savefig('{}/plots/tomo_m.png'.format(output_folder))
figt.savefig('{}/plots/tomo_t.png'.format(output_folder))

print('Tomographic Bins Printed',flush=True)









