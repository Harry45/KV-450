import argparse

import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.patches import Rectangle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


plt.rc("font", **{"family": "sans-serif", "serif": ["Palatino"]})
figSize = (12, 8)
fontSize = 20

# input param file
parser = argparse.ArgumentParser(description="A tutorial of argparse!")
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()

p_args = args.p
p = np.genfromtxt("{}".format(p_args), dtype=None, delimiter=",", encoding="ascii")
p_float = np.genfromtxt(
    "{}".format(p_args), dtype=float, delimiter=",", encoding="ascii"
)
p_int = np.genfromtxt("{}".format(p_args), dtype=int, delimiter=",", encoding="ascii")
p_bool = np.genfromtxt("{}".format(p_args), dtype=bool, delimiter=",", encoding="ascii")


# paralell processing
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

rank = 0
size = 1
# tomograthic bins set to 1 as were looking at whole survey
number_of_tomograthic_bins = 1

# assign parameter file to varibles

(
    flux_folder,
    output_folder,
    filter_folder,
    template_folder,
    template_list_folder,
    filter_list_folder,
) = p[:6]
flux_name, flux_error_name, filter_list, template_list = p[6:10]
zlist = np.loadtxt("{}".format(p_args), skiprows=32, max_rows=1)
mlist = np.loadtxt("{}".format(p_args), skiprows=33, max_rows=1)
reffilter, n_objects, nsamples, nsamples_split, n_split, number_of_chains = p_int[12:18]
likelihood_generator, sample_generator, sample_resume, WEIGHTS, RANDOM = p_bool[18:23]
resume_number = p_int[23]

HPC = True


# load templates
nt = 0
template_names_full = None
template_names = None
if rank == 0:
    template_names_full = np.loadtxt(
        "{}/{}".format(template_list_folder, template_list), dtype=str
    )

    nt = len(template_names_full)
    if nt > size:
        template_names = np.array_split(template_names_full, size, axis=0)
    else:
        template_names = np.array_split(template_names_full, nt, axis=0)

# nt = comm.bcast(nt, root=0)
if rank < nt:
    colour = 0
else:
    colour = 1
key = rank
# sub_comm = comm.Split(colour, key)
# sub_size = sub_comm.Get_size()
# sub_rank = sub_comm.Get_rank()

# template_names = sub_comm.scatter(template_names, root=0)

z1 = np.round(np.arange(zlist[0], zlist[1] + (3 * zlist[2]), zlist[2]), 2)
t1 = np.arange(0, nt)


"""
np.random.shuffle(t1)
template_names = template_names[t1]
template_names = template_names
"""
template_names = template_names[0]
if rank < nt:
    nt_chunk = len(template_names)
if rank == 0:
    print("Inputing Templates and Filters....", flush=True)

interp_x = np.arange(0, 25600, 5)
template_data = np.zeros((nt, len(interp_x)))


if rank < nt:
    for it in range(nt_chunk):

        seddata = np.genfromtxt("{}/{}".format(template_folder, template_names[it]))
        wavelength, template_sed = seddata[:, 0], seddata[:, 1] * seddata[:, 0] ** 2

        fnorm = np.interp(7e3, wavelength, template_sed)
        interp_sed = np.interp(interp_x, wavelength, template_sed)
        template_data[it] = interp_sed

        """
        plt.plot(interp_x, (interp_sed / fnorm), label=template_names[it], lw=2)
        template_data[it] = interp_sed
        plt.legend(loc="lower right", ncol=2)
        plt.ylabel(r"$L_\nu(\lambda)$")
        plt.yscale("log")
        plt.ylim([1e-3, 1e2])
        plt.xlim([1e3, 1.1e4])
        plt.xlabel(r"$\lambda$  [\AA]")
        """
# plt.show()

X = preprocessing.normalize(template_data)
mu = X.mean(0)
std = X.std(0)
"""
plt.plot(interp_x, mu, color='black')
plt.fill_between(interp_x, mu - std, mu + std, color='#CCCCCC')
plt.legend(loc='lower right', ncol=2)
plt.ylabel(r'$L_\nu(\lambda)$')
plt.yscale('log')
#plt.ylim([1e-3, 1e-1])
#plt.xlim([1e3, 1.1e4])
plt.xlabel(r'$\lambda$  [\AA]')
#plt.savefig('mean_diviation')
plt.show()
"""


rpca = PCA(n_components=4, svd_solver="randomized")
X_proj = rpca.fit_transform(X)

fig = plt.figure(constrained_layout=True, figsize=figSize)
subfigs = fig.subfigures(2, 1)

ax = subfigs[0].subplots(1, 1, sharey=True)
# plt.figure()
l = ax.plot(interp_x, rpca.mean_ - 0.15)
c = l[0].get_color()
ax.text(18000, -0.15 + 0.04, "mean - 0.15", color=c, fontsize=fontSize)
for i in range(4):
    l = ax.plot(interp_x, rpca.components_[i] + (0.15 * i))
    c = l[0].get_color()
    ax.text(
        18000,
        (0.15 * i) + 0.04,
        f"component({i + 1}) + {np.round(0.15 * i,2)}",
        color=c,
        fontsize=fontSize,
    )
# ax.ylim(-0.2, 0.6)
ax.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=fontSize)
ax.set_ylabel(r"$F$", fontsize=fontSize, rotation=0)
ax.tick_params(axis="x", labelsize=fontSize)
ax.tick_params(axis="y", labelsize=fontSize)
# ax.xlabel('wavelength (Angstroms)')
# ax.ylabel('scaled flux + offset')
# ax.title('Mean Spectrum and Eigen-spectra')
# plt.savefig('componants')
# plt.show()


orginal_colour_list = np.array(
    ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "black"]
)

# clustering = DBSCAN().fit(X_proj)

clustering = KMeans(n_clusters=6).fit(X_proj)
labels = clustering.labels_
colour_labels = orginal_colour_list[labels]

rows = 4
cols = 4
ax2 = subfigs[1].subplots(nrows=rows, ncols=cols)
for i in range(rows * cols):
    row = i // cols
    col = i % cols
    ax2[row, col].scatter(X_proj[:, row], X_proj[:, col], color=colour_labels)
    ax2[row, col].set_xticks([])
    ax2[row, col].set_yticks([])
    if row == 0:
        ax2[row, col].set_title(f"{col + 1}", fontsize=fontSize)
    if col == 0:
        ax2[row, col].set_ylabel(f"{row + 1}", rotation=0, fontsize=fontSize)
subfigs[1].subplots_adjust(wspace=0, hspace=0)
# fig.tight_layout()
# subfigs[1].tight_layout()
# fig.subplots_adjust(hspace=.0)
# fig.subplots_adjust(wspace=.0)
# plt.savefig('clustering')
# plt.savefig('clustering.pdf', bbox_inches='tight')
plt.show()

cluster_t = np.vstack((template_names, labels)).T
print(cluster_t.shape)
# print(cluster_t[0,1].type())
labels = labels.astype(np.int)
for i in range(6):
    cluster = template_names[labels == i]
    print(len(cluster))
    if 8 > len(cluster):
        choices = np.random.choice(cluster, len(cluster), replace=True)
    else:
        choices = np.random.choice(cluster, 8, replace=False)
    # print(choices.shape)
    if i == 0:
        choices_full = np.vstack(choices)
    else:
        choices_full = np.vstack((choices_full, np.vstack(choices)))

print(choices_full.shape)
# np.savetxt('TEMPLATE_LIST/PCA.list', choices_full, fmt='%s')
