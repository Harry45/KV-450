# def cleaning(config: ConfigDict, save: bool = False) -> None:
#     """Process the data and store them. We have 5 catalogues - we combine them into a single file.

#     Args:
#         config (ConfigDict): a configuration files.
#         save (bool): option to save the files. Default to False.
#     """
#     fobs_complete = list()
#     fobserr_complete = list()
#     bpz_complete = list()
#     ex_complete = list()
#     lim_complete = list()
#     name_complete = list()
#     mag_0_complete = list()
#     flux_complete = list()
#     flux_e_complete = list()
#     weight_complete = list()

#     for cat in range(config.catnames):
#         fits_file = fits.open(cat, memmap=True)
#         data = fits_file[1].data

#         # important columns
#         fobs = np.asarray([data[config.cols.mag[i]] for i in range(config.nband)]).T
#         fobserr = np.asarray([data[config.cols.mag_err[i]] for i in range(config.nband)]).T
#         flux = np.asarray([data[config.cols.flux[i]] for i in range(config.nband)]).T
#         fluxerr = np.asarray([data[config.cols.flux_err[i]] for i in range(config.nband)]).T
#         ext = np.asarray([data[config.cols.ext[i]] for i in range(config.nband)]).T
#         mag_lim = np.asarray([data[config.cols.mag_lim[i]] for i in range(config.nband)]).T

#         # flags
#         flag = data['GAAP_Flag_ugriZYJHKs']
#         bpz = data['Z_B']
#         name = data['THELI_NAME']
#         mag_0 = data['MAG_AUTO']
#         weight = data['recal_weight']

#         # Filter data by flags
#         fobs = fobs[flag == 0, :]
#         fobserr = fobserr[flag == 0, :]
#         flux = flux[flag == 0]
#         fluxerr = fluxerr[flag == 0]
#         bpz = bpz[flag == 0]
#         ext = ext[flag == 0, :]
#         mag_lim = mag_lim[flag == 0, :]
#         name = name[flag == 0]
#         mag_0 = mag_0[flag == 0]
#         weight = weight[flag == 0]

#         # append all data
#         fobs_complete.append(fobs)
#         fobserr_complete.append(fobserr)
#         flux_complete.append(flux)
#         flux_e_complete.append(fluxerr)
#         ex_complete.append(ext)
#         lim_complete.append(mag_lim)

#         bpz_complete.append(bpz)
#         name_complete.append(name)
#         mag_0_complete.append(mag_0)
#         weight_complete.append(weight)

#     fobs_complete = np.concatenate(fobs_complete, axis=0)
#     fobserr_complete = np.concatenate(fobserr_complete, axis=0)
#     flux_complete = np.concatenate(flux_complete, axis=0)
#     flux_e_complete = np.concatenate(flux, axis=0)
#     ex_complete = np.concatenate(ex_complete, axis=0)
#     lim_complete = np.concatenate(lim_complete, axis=0)

#     bpz_complete = np.asarray(bpz_complete)
#     name_complete = np.asarray(name_complete)
#     mag_0_complete = np.asarray(mag_0_complete)
#     weight_complete = np.asarray(weight_complete)

#     if save:
#         directory = 'data/processed/'
#         os.makedirs(directory, exist_ok=True)
#         np.save(directory + 'flux.npy', flux_complete)
#         np.save(directory + 'flux_err.npy', flux_e_complete)
#         np.save(directory + 'mag.npy', fobs_complete)
#         np.save(directory + 'mag_err.npy', fobserr_complete)
#         np.save(directory + 'ex.npy', ex_complete)
#         np.save(directory + 'lim.npy', lim_complete)

#         np.save(directory + 'bpz.npy', bpz_complete)
#         np.save(directory + 'name.npy', name_complete)
#         np.save(directory + 'mag_0.npy', mag_0_complete)
#         np.save(directory + 'weight.npy', weight_complete)
