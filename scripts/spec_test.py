from vtl.Readfile import Readfile
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import sympy as sp
from spec_id import Analyze_Stack,Stack_spec,Model_fit_stack,Likelihood_contours,Model_fit_stack_2,Scale_model,P
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from glob import glob
from time import time
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

metal =np.array( [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
age=np.array([0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0])
tau=np.array([0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0])
taulist=np.array(['0','8.0','8.15','8.28','8.43','8.57','8.72','8.86',
              '9.0','9.14','9.29','9.43','9.57','9.71','9.86','10.0'])

M,A=np.meshgrid(metal,age)

###############create set of data that matches data
##
# ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
# lmass,rshift=np.array([lmass,rshift]).astype(float)
# nlist=glob('spec_stacks/*')
# IDS=[]
#
# for i in range(len(ids)):
#     if 10.87<lmass[i]:
#         IDS.append(i)
#
# speclist=[]
# for i in range(len(ids[IDS])):
#     for ii in range(len(nlist)):
#         if ids[IDS][i] == nlist[ii][12:18]:
#             speclist.append(nlist[ii])
# ###
#
# zlist=[]
# speczs = np.round(rshift[IDS], 2)
#
# for i in range(len(speczs)):
#     zinput = int(speczs[i] * 100) / 5 / 20.
#     if zinput < 1:
#         zinput = 1.0
#     if zinput > 1.8:
#         zinput = 1.8
#     zlist.append(zinput)
#
# T=np.random.choice(taulist,len(zlist))
#
# for i in range(len(speclist)):
#     ipwv,ipfl,iperr=np.array(Readfile(speclist[i],1))
#     IDX=[U for U in range(len(ipwv)) if ipwv[U] <11500]
#     ipwv, ipfl, iperr=ipwv[IDX],ipfl[IDX],iperr[IDX]
#     mwv, mfl, merr = np.array(
#         Readfile('../../../fsps_models_for_fit/models/m0.015_a2.11_t8.0_z%s_model.dat' % zlist[i], 1))
#     nmwv=mwv/(1+zlist[i])*(1+rshift[IDS][i])
#     imfl=interp1d(nmwv,mfl)(ipwv)
#     C=Scale_model(ipfl,iperr,imfl)
#     imfl=C*imfl
#     imfl=imfl+np.random.normal(0,1,len(iperr))*iperr
#     dat=Table([ipwv,imfl,iperr],names=('wv','fl','err'))
#     ascii.write(dat,'test_spec/gal_%s.dat' % rshift[IDS][i])
#
# print T

""""""
##############Test fits
splist=glob('test_spec/*.dat')
rshifts=[]
for i in range(len(splist)):
    d=splist[i].replace('.dat','')
    rshifts.append(float(d[14:]))
rshifts=np.array(rshifts)

wv,st,er=Stack_spec(splist,rshifts,np.arange(2000,6000,5))
plt.plot(wv,st)
plt.plot(wv,er)
plt.show()

#####old way
# Model_fit_stack(splist,tau,metal,age,rshifts,np.arange(3400,5250,5),'test1','ls10.87_fsps_spec',fsps=True)
# Pr,bfage,bfmetal=Analyze_Stack('chidat/test1_chidata.fits', tau,metal,age)
#
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# # levels=np.array([ 11.42137475 , 79.64640091])
# print levels
# plt.contour(M,A,Pr,levels,colors='w',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap='cubehelix')
# plt.plot(bfmetal,bfage,'cp')
# plt.savefig('test_spec/test1.png')
# plt.close()

#####new way
# # Model_fit_stack_2(splist,tau,metal,age,rshifts,np.arange(3400,5250,5),'test2','ls10.87_fsps2_spec',fsps=True)
# Pr,bfage,bfmetal=Analyze_Stack('chidat/test2_chidata.fits', tau,metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# # levels=np.array([  3.7904828   28.03918784])
# print levels
# plt.contour(M,A,Pr,levels,colors='w',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap='cubehelix')
# plt.plot(bfmetal,bfage,'cp')
# plt.savefig('test_spec/test2.png')
# plt.close()
#
