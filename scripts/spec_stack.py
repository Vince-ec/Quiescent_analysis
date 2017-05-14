from astropy.io import ascii
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

num=46066

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
            e[i] = 0

    return w, f, e

fn=glob('../../../Clear_data/extractions_nov_22/ERSPRIME/*%s*1D.fits' % num)
print fn
fln=np.array(fn[:-1])

# for i in range(len(fn)):
#     ww,ff,ee=Get_flux(fn[i])
#     plt.plot(ww,ff)
#     plt.plot(ww,ee)
#     plt.show()

def Stack_spec1(spec, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave,flux,error=np.array(Get_flux(spec[i]))
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        flgrid[i][mask]=ifl(wv[mask])
        errgrid[i][mask]=ier(wv[mask])
    ################

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

wv,fl,er=np.array(Stack_spec1(fln,np.arange(7800,11600,5)))

mask=np.isnan(fl)

w,f,e=[wv[~mask],fl[~mask],er[~mask]]
ww,ff,ee=Get_flux(fn[-1])
#
# plt.fill_between(w,f-e,f+e,color=sea.color_palette('muted')[5],alpha=.9)
plt.plot(w,f)
plt.plot(ww,ff)
# plt.plot(w,e)
# plt.plot(ww,ee)
plt.show()
#
# dat=Table([w,f,e],names=['wavelength','Flam','error'])
# ascii.write(dat,'spec_stacks/s%s_stack.dat' % num)