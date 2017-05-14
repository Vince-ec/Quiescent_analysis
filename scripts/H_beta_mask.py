import numpy as np
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d,interp2d
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

ids,speclist,lmass,rshift,rad,sig,comp=np.array(Readfile('masslist_mar22.dat',is_float=False))
lmass,rshift,rad,sig,comp=np.array([lmass,rshift,rad,sig,comp]).astype(float)

IDc=[]  # compact sample
IDd=[]  # diffuse sample

IDmL=[]  # low mass sample
IDmH=[]  # high mass sample

for i in range(len(ids)):
    if 0.11 < comp[i]:
        IDd.append(i)
    if 0.11 > comp[i]:
        IDc.append(i)
    if 10.931 > lmass[i]:
        IDmL.append(i)
    if 10.931 < lmass[i]:
        IDmH.append(i)

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
M,A=np.meshgrid(metal,age)

wv,fl,er=Readfile('spec_stacks_jan24/n34694_stack.dat')
wv,fl,er=np.array([wv[wv<11300],fl[wv<11300],er[wv<11300]])

idx=np.argwhere(ids=='n34694')[0][0]

IDb=[U for U in range(len(wv)) if 4855 <= wv[U]/(1+rshift[idx]) <= 4880]


plt.figure(figsize=[12,8])
plt.errorbar(wv,fl,er,fmt='o')
# plt.errorbar(wv[IDb]/(1+rshift[idx]),fl[IDb],er[IDb],fmt='ro')
# plt.axvline(4862.68,linestyle='--', alpha=.1)
# plt.axvspan(4830, 4930, color='k', alpha=.1)
# plt.axvspan(4862.68 - 20, 4862.68 + 20, color='r', alpha=.1)
# plt.xlim(4830,4930)
# plt.ylim(4E-18,5.5E-18)
plt.show()
plt.close()