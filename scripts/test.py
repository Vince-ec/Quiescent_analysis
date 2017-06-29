import numpy as np
import pandas as pd
from spec_id import Stack,Median_w_Error
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
from time import time
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

galDB = pd.read_pickle('../data/sgal_param_DB.pkl')
metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

lzDB = galDB.query('hi_res_specz < 1.16')
mzDB = galDB.query('1.16 < hi_res_specz < 1.3')
hzDB = galDB.query('1.3 < hi_res_specz')

PZ = np.ones(len(metal))
PT= np.ones(len(age))

for i in lzDB.index:
    z, pz = np.load('../chidat/%s_Z_pos.npy' % lzDB['gids'][i])
    t, pt = np.load('../chidat/%s_t_pos.npy' % lzDB['gids'][i])
    PZ = PZ * pz
    PT = PT * pt

CZ = np.trapz(PZ, metal)
CT = np.trapz(PT, age)

PZ /= CZ
PT /= CT

Zmed, Zler, Zher = Median_w_Error(PZ, metal)
tmed, tler, ther = Median_w_Error(PT, age)

gids = np.array(lzDB['gids'])
specz = np.array(lzDB['hi_res_specz'])

lzstack = Stack(gids,specz,np.arange(3500,6000,10))
lzstack.Stack_normwmean()
lzstack.Stack_normwmean_model(Zmed,tmed,tau)

plt.figure(figsize=[12,5])
plt.errorbar(Stack.wv,Stack.fl,Stack.er,fmt='o',ms=5)
plt.plot(Stack.mwv,Stack.mfl)
plt.show()