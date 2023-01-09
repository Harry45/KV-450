import argparse
import numpy as np

# input param file
parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()

p_args = args.p

p = np.genfromtxt('{}'.format(p_args), dtype=None, delimiter=',', encoding='ascii')
p_float = np.genfromtxt('{}'.format(p_args), dtype=float, delimiter=',', encoding='ascii')
p_int = np.genfromtxt('{}'.format(p_args), dtype=int, delimiter=',', encoding='ascii')
p_bool = np.genfromtxt('{}'.format(p_args), dtype=bool, delimiter=',', encoding='ascii')


flux_folder, output_folder, filter_folder, template_folder, template_list_folder, filter_list_folder = p[:6]
flux_name, flux_error_name, filter_list, template_list = p[6:10]
zlist = np.loadtxt('{}'.format(p_args), skiprows=32, max_rows=1)
mlist = np.loadtxt('{}'.format(p_args), skiprows=33, max_rows=1)
reffilter, n_objects, nsamples, nsamples_split, n_split, number_of_chains = p_int[12:18]
likelihood_generator, sample_generator, sample_resume, WEIGHTS, RANDOM = p_bool[18:23]
resume_number = p_int[23]

m1 = np.arange(mlist[0], mlist[1]+3*mlist[2], mlist[2])
m2 = np.arange(mlist[0], mlist[1]+2*mlist[3], mlist[3])

z1 = np.round(np.arange(zlist[0],zlist[1]+(3*zlist[2]),zlist[2]),5)


print(f'nsamples: {nsamples}')
print(f'nsamples_split: {nsamples_split}')
print(f'n_split: {n_split}')
print(f'number_of_chains: {number_of_chains}')

# from mpi4py import MPI

# # parallel processing
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# print(size)