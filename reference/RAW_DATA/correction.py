import numpy as np 

flux = np.load('Flux.npy')
flux_e = np.load('Flux_e.npy')
mag = np.load('Mag.npy')
mag_e = np.load('Mag_e.npy')
lim = np.load('lim.npy')
ex = np.load('ex.npy')
offset = np.load('offset.npy')

# apply extraction
for i in range(9):
   mag[mag[:,i]!=99.,i] -= ex[mag[:,i]!=99.,i]
flux *= 10**(0.4*ex)
flux_e *= 10**(0.4*ex)

# correct offset 
flux *= 10**(-0.4*offset)
flux_e *= 10**(-0.4*offset)

np.save('flux_corrected.npy',flux)
np.save('flux_e_corrected.npy',flux_e)


# apply undetected conditions 
for i in range(9):
   mag[mag[:,i]>lim[:,i],i] = 99.
   mag[mag_e[:,i]>1.,i] = 99.
   mag_e[mag[:,i]==99,i] = lim[mag[:,i]==99,i]

np.save('mag_corrected.npy',mag)
np.save('mag_e_corrected.npy',mag_e)



            
stop

#set up for BPZ
#convert undetcted to unobserved
mag_e[mag==99.] = 0.
mag[mag==99.] = -99
#correct absalute magnitude M_0
M_0 = np.load('M_0.npy')
M_0 = np.where((mag[:,2]==99.)|(mag[:,2]==-99.)|(mag[:,3]==99.)|(mag[:,3]==-99),M_0-ex[:,0],(M_0-ex[:,0])-mag[:,2]+mag[:,3])
# truncate values to 4s.f
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
mag = trunc(mag,4)
mag_e = trunc(mag_e,4)
M_0 = trunc(M_0,4)# reformat to 1 file 
bpz_file = np.concatenate((mag,mag_e,np.array([M_0]).T),axis=1)
bpz_file = bpz_file[:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17,18]]
np.savetxt('BPZ.cat',bpz_file)
print('Data Gathered',bpz_file.shape)


