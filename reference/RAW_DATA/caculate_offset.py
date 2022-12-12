import numpy as np 

name = np.load('name.npy')
mag = np.load('Mag.npy')
flux = np.load('Flux.npy')

offset = np.ones_like(flux)

unique_name = np.unique(name)

for i in unique_name :
   
   mag_s = mag[name == i]
   flux_s = flux[name == i]

   for j in range(9):
      flux_s = flux_s[mag_s[:,j]!=99]
      mag_s = mag_s[mag_s[:,j]!=99]


   offset_individual = mag_s + (2.5*np.log10(flux_s))
   offset_median = np.median(offset_individual,axis=0)

   offset[name==i] *= offset_median

np.save('offset.npy',offset)


   

   
