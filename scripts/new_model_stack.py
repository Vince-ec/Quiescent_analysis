from spec_id  import Stack_spec_normwmean,Stack_model_normwmean,Model_fit_stack_normwmean,Analyze_Stack_avgage,\
    Likelihood_contours
import seaborn as sea
from glob import glob
import numpy as np
from scipy.interpolate import interp1d, interp2d
import sympy as sp
import matplotlib.pyplot as plt
from astropy.io import fits,ascii
from vtl.Readfile import Readfile
from astropy.table import Table
import cPickle
import os
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""get galaxies"""
###get list of spectra
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871<lmass[i] and 1<=rshift[i]<=1.75:
        IDS.append(i)

metal = np.array([0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0068, 0.0077, 0.0085, 0.0096, 0.0106,
                  0.012, 0.0132, 0.014, 0.0150, 0.0164, 0.018, 0.019, 0.021, 0.024, 0.027, 0.03])
age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau = [0, 8.0, 8.15, 8.28, 8.43, 8.57, 8.72, 8.86, 9.0, 9.14, 9.29, 9.43, 9.57, 9.71, 9.86, 10.0]

# Model_fit_stack_normwmean(speclist[IDS],tau,metal,age,rshift[IDS],np.arange(3250,5500,10),
#                          'gt10.87_fsps_newdata_stackfit','gt10.87_fsps_newdata_spec',res=10,fsps=True)

# M,A=np.meshgrid(metal,age)
#
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_newdata_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# # levels=np.array([201.66782781,  908.392327])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# plt.show()

flist=[]
for i in range(len(rshift[IDS])):
    flist.append('../../../fsps_models_for_fit/models/m0.012_a2.4_t0_z%s_model.dat' % rshift[IDS][i])

wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3250,5500,10))
mwv,mfl,mer=Stack_model_normwmean(speclist[IDS],flist,rshift[IDS],np.arange(wv[0],wv[-1]+10,10))

plt.errorbar(wv,fl,er,fmt='o',ms=5)
# plt.plot(wv,fl)
plt.plot(mwv,mfl)
plt.xlim(3250,5500)
plt.show()