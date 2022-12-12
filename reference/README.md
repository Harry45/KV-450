Pipeline 

1. Download files from https://kids.strw.leidenuniv.nl/DR3/kv450data.php and 
move to RAW_DATA folder 
2. run the bash script process_data.sh to deal with corrections and process
data to the condition needed for rest of code
3. run main code with mpiexec -np 4 python Nz.py --p param_kids.txt ,
paramiters and paths can be changed in param_kids.txt, this will produce
samples in OUTPUT/KiDs/sample_chains_counts/
4. run plotting code with python Plot_kids.py ==p param_kids.txt , this will 
also produce samples in the format needed to extract cosmology

processed data has also been inlcude in DATA/kids_450/1 , if you wish to use
this data skip steps 1 and 2 and chnage the parameter files to point to 
relavant data path 

repeat the above if you wish to look at the simulated data in DATA/simulation/1






