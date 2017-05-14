from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from astropy.table import Table
import os
from astropy.io import fits, ascii
from scipy.interpolate import interp1d
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

def Get_flux(FILE):

    observ = fits.open(FILE)
    w = np.array(observ[1].data.field('wave'))
    f = np.array(observ[1].data.field('flux'))*1E-17
    sens = np.array(observ[1].data.field('sensitivity'))
    contam = np.array(observ[1].data.field('contam'))*1E-17
    e = np.array(observ[1].data.field('error'))*1E-17
    f -= contam
    f /=sens
    e/=sens

    INDEX = []
    for i in range(len(w)):
        if 7900 < w[i] < 11600:
            INDEX.append(i)

    w = w[INDEX]
    f = f[INDEX]
    e = e[INDEX]

    for i in range(len(f)):
        if f[i] <= 0:
            f[i] = 0

    return w, f, e
def Stack_spec1(spec, wv):

    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])

    for i in range(len(spec)):
        wave, flux, error = np.array(Get_flux(spec[i]))
        if sum(flux) != 0:
            mask = np.array([wave[0] < U < wave[-1] for U in wv])
            ifl = interp1d(wave, flux)
            ier = interp1d(wave, error)
            flgrid[i][mask] = ifl(wv[mask])
            errgrid[i][mask] = ier(wv[mask])
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
    ###############

    return wv, stack, err
def Stack_galaxy(field, num):

    fn=glob('../../../Clear_data/%s_extractions_quiescent/*%s*1D.fits' % (field,num))
    print fn

    fln=np.array(fn[:-1])

    wv,fl,er=np.array(Stack_spec1(fln,np.arange(7800,11600,5)))

    mask=np.isnan(fl)

    w,f,e=[wv[~mask],fl[~mask],er[~mask]]

    return w,f,e

# southfields=['GS1','GS2','GS3','GS4','GS5']
# northfields=['GN5','GN7']
#
# southnms=np.array([])
# northnms=[]
#
# for i in  range(len(southfields)):
#     snms=glob('../../../Clear_data/%s_extractions_quiescent/*stack.png' % southfields[i])
#     southnms=np.append(southnms,snms)
#
# for i in  range(len(northfields)):
#     nnms=glob('../../../Clear_data/%s_extractions_quiescent/*stack.png' % northfields[i])
#     northnms=np.append(northnms,nnms)
#
# print southnms
#
# southids=np.array(sorted(set([int(i[55:60]) for i in southnms])))
# northids=np.array(sorted(set([int(i[55:60]) for i in northnms])))

# print southids
# print northids

# for i in range(len(southfields)):
#     for ii in range(len(southids)):
#         fn = ('../../../Clear_data/%s_extractions_quiescent/%s-G102_%s.1D.fits'
#             % (southfields[i], southfields[i], southids[ii]))
#         if os.path.isfile(fn):
#             wv,fl,err=Stack_galaxy(southfields[i],southids[ii])
#             plt.plot(wv,fl)
#             plt.plot(wv,err)
#             plt.title('%s-%s' %(southfields[i], southids[ii]))
#             plt.show()
#             # plt.savefig('../spcplots/%s-%s_stack.png' %(southfields[i], southids[ii]))
#             plt.close()
#
# for i in range(len(northfields)):
#     for ii in range(len(northids)):
#         fn = ('../../../Clear_data/%s_extractions_quiescent/%s-G102_%s.1D.fits'
#                 % (northfields[i], northfields[i], northids[ii]))
#         if os.path.isfile(fn):
#             wv, fl, err = Stack_galaxy(northfields[i],northids[ii])
#             plt.plot(wv, fl)
#             plt.plot(wv, err)
#             plt.title('%s-%s' % (northfields[i], northids[ii]))
#             plt.show()
#             # plt.savefig('../spcplots/%s-%s_stack.png' % (northfields[i], northids[ii]))
#             plt.close()



# bad=[45775,47223]
# badloc=[np.argwhere(southids==i)[0][0] for i in bad]
# southids=np.delete(southids,badloc)
#
# nbad=[14140]
# nbadloc=[np.argwhere(northids==i)[0][0] for i in nbad]
# northids=np.delete(northids,nbadloc)

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_9-26.dat',1,is_float=False))
zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

specids=[U[12:18] for U in speclist]
southids=[int(U[1:]) for U in specids if U[0]=='s']
northids=[int(U[1:]) for U in specids if U[0]=='n']

goodss=fits.open('../../../Clear_data/goodss_3dhst.v4.1.fout.FITS')[1].data
goodsn=fits.open('../../../Clear_data/goodsn_3dhst.v4.1.fout.FITS')[1].data
slmass=goodss.field('lmass')
nlmass=goodsn.field('lmass')

smasses=slmass[np.array(southids)-1]
nmasses=nlmass[np.array(northids)-1]

nms_exist=glob('spec_stacks/*')
ids_exist=np.array(sorted(set([int(i[13:18]) for i in nms_exist])))

"""list all galaxies and masses"""

sids=['s%s' % i for i in southids]
nids=['n%s' % i for i in northids]

idlist=np.append(sids,nids)
masslist=np.append(smasses,nmasses)

newz=np.append(zps[7:],zps[:7])

dat=Table([idlist,masslist,newz],names=['ID','lmass','z'])
print dat
ascii.write(dat,'masslist_sep28.dat')

"""script to add spectra that is missing"""
# needed=[i for i in southids if i not in ids_exist[3:]]
#
# print needed
#
# IDS=needed[0]
# w,f,e=Stack_galaxy(IDS)
#
# plt.fill_between(w,f-e,f+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(w,f, label='%s' % IDS)
# plt.plot(w,e)
# plt.legend()
# plt.show()
# plt.close()
#
# dat=Table([w,f,e],names=['wavelength','Flam','error'])
# ascii.write(dat,'spec_stacks/s%s_stack.dat' % IDS)
