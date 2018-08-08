from C_fit_full import Stich_grids
import numpy as np
import pandas as pd
import os
from glob import glob

metal=np.round(np.arange(0.002,0.031,0.001),3)
age=np.round(np.arange(.5,6.1,.1),1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

flist = glob('/home/vestrada78840/chidat/*_full_fit_tZ_pos.npy')
glist = [os.path.basename(U).replace('_full_fit_tZ_pos.npy','') for U in flist]

def Best_fit_model(name, metal, age, tau, redshift, dust = np.arange(0,1.1,.1)):
    grids = ['/fdata/scratch/vestrada78840/chidat/{0}_d{1}_chidata.npy'.format(name,U) for U in range(11)]
    chi = Stich_grids(grids)
    x = np.argwhere(chi == np.min(chi))[0]
    print(x)
    print(dust[x[0]],metal[x[1]], age[x[2]], tau[x[3]],redshift[x[4]])
    return dust[x[0]],metal[x[1]], age[x[2]], tau[x[3]],redshift[x[4]]

bfZ,bft,bftau,bfz,bfd  = np.zeros([5,len(glist)])
for i in range(len(glist)):
    z,Pz = np.save('/home/vestrada78840/chidat/{0}_rs_pos.npy'.format(glist[i]))

    bfd[i],bfZ[i],bft[i],bftau[i],bfz[i] = Best_fit_model(glist[i] + '_full_fit', Z, t, tau, z, d)
    
DF = pd.DataFrame({'gids':glist,'bfZ':bfZ,'bft':bft,'bftau':bftau,'bfz':bfz,'bfd':bfd})

DF.to_pickle('/home/vestrada78840/chidat/BF_fullfit.pkl')