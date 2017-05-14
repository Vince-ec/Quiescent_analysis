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

def Stack_spec_normwmean(spec, redshifts, wv):
    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl = interp1d(wave, flux)
        ier = interp1d(wave, error)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
    ################

    flgrid = np.transpose(flgrid)
    errgrid = np.transpose(errgrid)
    weigrid = errgrid ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack, err = np.zeros([2, len(wv)])
    for i in range(len(wv)):
        stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / np.sum(weigrid[i])
        err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
    ################
    ###take out nans

    # IDX = [U for U in range(len(wv)) if stack[U] > 0]

    return wv, stack, err

"""get galaxies"""
###get list of spectra
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 1<=rshift[i]<=1.75:
        IDS.append(i)

metal = np.array([0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0068, 0.0077, 0.0085, 0.0096, 0.0106,
                  0.012, 0.0132, 0.014, 0.0150, 0.0164, 0.018, 0.019, 0.021, 0.024, 0.027, 0.03])
age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau = [0, 8.0, 8.15, 8.28, 8.43, 8.57, 8.72, 8.86, 9.0, 9.14, 9.29, 9.43, 9.57, 9.71, 9.86, 10.0]

wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3250,5500,10))

"""jacknife"""
# err_grid=np.zeros([len(IDS),len(er)])
# for i in range(len(speclist[IDS])):
#     new_list=[]
#     new_rshift=[]
#     for ii in range(len(speclist[IDS])):
#         if ii != i:
#             new_list.append(speclist[IDS][ii])
#             new_rshift.append(rshift[IDS][ii])
#     w,f,e=Stack_spec_normwmean(new_list,new_rshift,np.arange(3250,5500,10))
#     err_grid[i]=e
#
# jk_errs=[np.std(U) for U in err_grid.T]
#
# plt.plot(wv,fl)
# plt.plot(wv,er)
# plt.plot(wv,jk_errs)
# plt.show()

print len(IDS)
"""bootstrap"""
err_grid=np.zeros([1000,len(er)])
for i in range(1000):
    new_list=[]
    new_rshift=[]
    for ii in range(len(IDS)):
        rid=np.random.choice(IDS)
        new_list.append(speclist[rid])
        new_rshift.append(rshift[rid])
    # print len(new_rshift)
    w,f,e=Stack_spec_normwmean(new_list,new_rshift,np.arange(3250,5500,10))
    err_grid[i]=e
    # plt.plot(wv,fl)
    # plt.plot(w,f)
    # plt.show()

bs_errs=[np.std(U) for U in err_grid.T]

plt.plot(wv,fl)
plt.plot(wv,er)
plt.plot(wv,bs_errs)
plt.show()