from multiprocessing import Process
from spec_id import Stack_model
from scipy.interpolate import interp1d
import numpy as np
from time import time
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_fsps.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

def Stack_model(spec,redshifts,redshift_counts, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(spec)):
        #######read in spectra
        W,F,E=np.array(Readfile(spec[i],1))

        ######clip spectra
        idx=[]
        for ii in range(len(W)):
            if W[ii] < 12000:
                idx.append(ii)
        W,F,E=[W[idx],F[idx],E[idx]]

        ######de-redshift
        W /= (1 + redshifts[i])

        ######divide out continuum
        m2r = [3910, 3990, 4082, 4122, 4250, 4330, 4830, 4890, 4990, 5030]
        Mask = np.zeros(len(W))
        for ii in range(len(Mask)):
            if m2r[0] <= W[ii] <= m2r[1]:
                Mask[ii] = 1
            if m2r[2] <= W[ii] <= m2r[3]:
                Mask[ii] = 1
            if m2r[4] <= W[ii] <= m2r[5]:
                Mask[ii] = 1
            if m2r[6] <= W[ii] <= m2r[7]:
                Mask[ii] = 1
            if m2r[8] <= W[ii] <= m2r[9]:
                Mask[ii] = 1
            if W[ii]>m2r[9]:
                break

        maskw = np.ma.masked_array(W, Mask)

        x3, x2, x1, x0 = np.ma.polyfit(maskw, F, 3,w=1/E**2)
        C0 = x3 * W ** 3 + x2 * W ** 2 + x1 * W + x0

        F /= C0
        E /= C0

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([W[0] < U < W[-1] for U in wv_range])
        ifl=interp1d(W,F)
        ier=interp1d(W,E)
        flentry[mask]=ifl(wv_range[mask])
        errentry[mask]=ier(wv_range[mask])

        for ii in range(redshift_counts[i]):
            flgrid.append(flentry)
            errgrid.append(errentry)

    wv = np.array(wv_range)

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=1/np.sqrt(np.sum(weigrid[i]))
    ################

    return wv, stack, err

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]

zps = np.round(zps, 2)

for i in range(len(zps)):
    zinput=int(zps[i] * 100) / 5 / 20.
    if zinput<1:
        zinput=1.0
    zlist.append(zinput)
for i in range(len(bins)):
    b = []
    for ii in range(len(zlist)):
        if bins[i] == zlist[ii]:
            b.append(ii)
    if len(b) > 0:
        zcount.append(len(b))
zbin = sorted(set(zlist))

flist=[]
for i in range(len(zbin)):
    flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t8.0_z%s_model.dat' % zbin[i])

print len(zbin)
start=time()
fwv,fs,fe=Stack_model(flist,zbin,zcount,np.arange(3250,5550,5))
end=time()
print end-start

start=time()
fwv1,fs1,fe1=Stack_model1(flist,zbin,zcount,np.arange(3250,5550,5))
end=time()
print end-start

plt.plot(fwv,fs)
plt.plot(fwv1,fs1)
plt.show()
