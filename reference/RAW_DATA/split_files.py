import numpy as np 

file_list = ['BPZ','flux_e_corrected','flux_corrected','weight']

x = np.load('BPZ.npy')
print(x.shape)
random = np.arange(len(x))
np.random.shuffle(random)



for i in range(len(file_list)):
   full_file  = np.load('{}.npy'.format(file_list[i]))
   full_file = full_file[random]
   print(full_file.shape)
   full_file =  np.array_split(full_file,5,axis=0)
   for j in range(5):
      np.save('split_100000/{}/{}.npy'.format(j+1,file_list[i]),full_file[j][:250000])
      print(full_file[j][:100000].shape)
      
