from spec_id import Stack_model
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_zps2.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]

for i in range(len(zsmax)):
    zlist.append(int(zsmax[i] * 100) / 5 / 20.)
for i in range(len(bins)):
    b = []
    for ii in range(len(zlist)):
        if bins[i] == zlist[ii]:
            b.append(ii)
    if len(b) > 0:
        zcount.append(len(b))
zbin = sorted(set(zlist))

modlist=[]
fp = '../../../bc03_models_for_fit/models/'

for i in range(len(zbin)):
    modlist.append(fp+'m0.02_a2.74_z%s_model.dat' % zbin[i])

w,s,e=Stack_model(modlist,zbin,zcount,np.arange(3250,5550,5))
# wv,fl,er=np.array(Readfile(modlist[3],1))
# print  modlist[3]
plt.plot(w,s)
plt.plot(w,e)
# plt.plot(wv,fl)
# plt.xlim(8000,11300)
# plt.ylim(0,1.2)
plt.show()