import numpy as np
from spec_id import Stack_spec_normwmean,Stack_model_normwmean
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits,ascii
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871>=lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

print rshift[IDS]
metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
        # 2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]

age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
        2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.4, 3.45, 3.5, 3.56, 3.6, 3.7, 3.8, 4.62, 6.0]

tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]


wver,flxerr=np.array(Readfile('flx_err/LM_10_3400-5250.dat'))
wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3400,5250,10))
err= np.sqrt(er ** 2 + (flxerr * fl) ** 2)

chi0=np.zeros(len(age))
chi8=np.zeros(len(age))
for i in range(len(age)):
    flist0=[]
    flist8=[]
    for ii in range(len(rshift[IDS])):
        flist0.append('../../../fsps_models_for_fit/models/m0.0049_a%s_t0_z%s_model.dat' % (age[i],rshift[IDS][ii]))
        flist8.append('../../../fsps_models_for_fit/models/m0.0049_a%s_t8.0_z%s_model.dat' % (age[i],rshift[IDS][ii]))
    mwv0, mfl0, mer0 = Stack_model_normwmean(speclist[IDS], flist0, rshift[IDS], np.arange(wv[0], wv[-1] + 10, 10))
    mwv8, mfl8, mer8 = Stack_model_normwmean(speclist[IDS], flist8, rshift[IDS], np.arange(wv[0], wv[-1] + 10, 10))
    chi0[i]=sum(((fl - mfl0) / err) ** 2)
    chi8[i]=sum(((fl - mfl8) / err) ** 2)

print age[np.argmin(chi0)]
print age[np.argmin(chi8)]
print np.min(chi0)
print np.min(chi8)


plt.plot(age,chi0,label='$\\tau=0$')
plt.plot(age,chi0,'o')
plt.plot(age,chi8,label='$\\tau=8$')
plt.plot(age,chi8,'o')
plt.show()