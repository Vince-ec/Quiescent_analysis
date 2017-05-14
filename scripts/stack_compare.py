from vtl.Readfile import Readfile
from spec_id import Scale_model,Stack_spec
from astropy.io import ascii
from astropy.table import Table
from scipy.interpolate import interp1d
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

modellist1=[]
modellist2=[]
z = zsmax
for i in range(len(z)):
    modellist1.append('spec_stacks/m0.019_a2.11_z%ssim_werr.dat' % z[i])
    modellist2.append('spec_stacks/m0.019_a2.11_z%ssim.dat' % z[i])

w,s,e=Stack_spec(speclist,zsmax,np.arange(3200,5500,10))
ws1,ss1,es1=Stack_spec(modellist1,zsmax,np.arange(3200,5500,10))
ws2,ss2,es2=Stack_spec(modellist2,zsmax,np.arange(3200,5500,10))

# plt.plot(w,s)
# # plt.plot(ws1,ss1)
# # plt.plot(ws2,ss2)
# plt.plot(w,e)
# plt.plot(w,er)
# # plt.plot(ws1,es1)
# # plt.plot(ws2,es2)
plt.hist(ss1-ss2,50)
plt.show()