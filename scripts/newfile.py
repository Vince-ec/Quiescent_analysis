import numpy as np
from vtl.Readfile import Readfile
from spec_id import Stack_spec_normwmean,Stack_model_normwmean,Make_model_list,Scale_model
from scipy.interpolate import interp1d
from time import time
from astropy.io import ascii
from astropy.table import Table

ids,speclist,lmass,rshift,rad,sig,comp=np.array(Readfile('masslist_feb28.dat',is_float=False))
lmass,rshift,rad,sig,comp=np.array([lmass,rshift,rad,sig,comp]).astype(float)

nspeclist=[]
for i in range(len(speclist)):
    nspeclist.append(speclist[i].replace('dat','npy'))

nspeclist=np.array(nspeclist)

IDA=[]

for i in range(len(ids)):
    if 1 < rshift[i] < 1.75:
        IDA.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=np.arange(.5,6.1,.1)
tau=[0,8.0,8.93,9.21,9.37,9.5,9.6,9.66,9.73,9.8,9.84,9.88,9.93,9.97,10.0]

def Stack_spec_normwmean(spec, redshifts, wv):
    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.load(spec[i])
        if spec[i] == 'spec_stacks_jan24/s40597_stack.npy':
            IDW = []
            for ii in range(len(wave)):
                if 7950 < wave[ii] < 11000:
                    IDW.append(ii)

        else:
            IDW = []
            for ii in range(len(wave)):
                if 7950 < wave[ii] < 11300:
                    IDW.append(ii)

        wave, flux, error = np.array([wave[IDW], flux[IDW], error[IDW]])
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

    IDX = [U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_normwmean_in_mfit(modellist, redshifts,wave_grid,flux_grid,err_grid, wv_range):
    flgrid = []
    errgrid = []

    for i in range(len(modellist)):
        #######read in spectra
        wave, flux, error = np.array([wave_grid[i],flux_grid[i],err_grid[i]])

        #######read in corresponding model, and interpolate flux
        W, F,= np.load(modellist[i])
        W = W / (1 + redshifts[i])
        iF = interp1d(W, F)(wave)

        #######scale the model
        C = Scale_model(flux, error, iF)
        F *= C
        Er = error

        ########interpolate spectra
        flentry = np.zeros(len(wv_range))
        errentry = np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl = interp1d(W, F)
        ier = interp1d(wave, Er)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flentry[mask] = ifl(wv_range[mask]) / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    wv = np.array(wv_range)

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

    return wv, stack

wgrid=[]
fgrid=[]
egrid=[]

for i in range(len(nspeclist[IDA])):
    #######read in spectra
    wave, flux, error = np.load(nspeclist[i])
    if nspeclist[i] == 'spec_stacks_jan24/s40597_stack.npy':
        IDW = []
        for ii in range(len(wave)):
            if 7950 < wave[ii] < 11000:
                IDW.append(ii)
    else:
        IDW = []
        for ii in range(len(wave)):
            if 7950 < wave[ii] < 11300:
                IDW.append(ii)

    wave, flux, error = np.array([wave[IDW], flux[IDW], error[IDW]])

    wave = wave / (1 + rshift[IDA][i])
    wgrid.append(wave)
    fgrid.append(flux)
    egrid.append(error)

mlist=Make_model_list(ids[IDA],0.0132,3.4,0,rshift[IDA])

start=time()
wv,fl=Stack_model_normwmean(speclist[IDA],mlist,rshift[IDA],np.arange(3250,5350,10))
end=time()
print end-start

start=time()
wv,fl=Stack_model_normwmean_in_mfit(mlist,rshift[IDA],wgrid,fgrid,egrid,np.arange(3250,5350,10))
end=time()
print end-start
#
# start=time()
# wv,fl,er=Readfile(speclist[0])
# end=time()
# print end-start
#
# start=time()
# wv,fl,er=np.load(nspeclist[0])
# end=time()
# print end-start