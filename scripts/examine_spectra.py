import numpy as np
from spec_id import Get_flux
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from astropy.io import fits,ascii
from astropy.table import Table
import os
from scipy.interpolate import interp1d
from glob import glob
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Stack_gal_spec(spec, wv, mregion):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave,flux,error=np.array(Get_flux(spec[i]))
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)(wv[mask])
        ier=interp1d(wave,error)(wv[mask])
        if sum(mregion[i])>0:
            # flmask=np.array([mregion[i][0] < U < mregion[i][1] for U in wv[mask]])
            for ii in range(len(wv[mask])):
                if mregion[i][0] < wv[mask][ii] <mregion[i][1]:
                    ifl[ii]=0
                    ier[ii]=0
            # flgrid[i][mask]=ifl[flmask]
            # errgrid[i][mask]=ier[flmask]
        # else:
        flgrid[i][mask] = ifl
        errgrid[i][mask] = ier
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

"""Create file path"""
### field directory
field='GN7'
loc='n'

### file path
fp='../../../Clear_data/extractions_nov_22/%s/' % field

"""Get files"""
### ID num
galaxy=19442

### get list of individual spectra and stack
s=glob(fp+'*%s.1D.fits' % galaxy)
spectra=s[:-1]

### get png's of individual pointings
ind_gal_img=glob(fp+'*%s.2D.png' % galaxy)

### get png of all 2d spectra plus stack
all_gal=glob(fp+'*%s_stack.png' % galaxy)

"""Examine spectra and assign value"""
### open all spec png
os.system("open "+ all_gal[0])

### initialize quality and pointing name arrays
quality=np.repeat(True,len(ind_gal_img))
Mask=np.zeros([len(ind_gal_img),2])
p_names=[]

### examine each spectra and assign quality
for i in range(len(ind_gal_img)):
    os.system("open " + ind_gal_img[i])
    p_names.append(ind_gal_img[i][48:74])
    wv,fl,err= Get_flux(spectra[i])
    plt.plot(wv,fl)
    plt.plot(wv,err)
    plt.xlim(8000,11500)
    plt.show()
    quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
    if quality[i] != 0:
        minput = int(input('Mask region: (0 if no mask needed)'))
        if minput != 0:
            rinput= int(input('Lower bounds'))
            linput= int(input('Upper bounds'))
            Mask[i]=[rinput,linput]

"""Stack spectra and save quality file"""
### make data table and assign file name
qual_dat=Table([p_names,quality],names=['id','good_spec'])
fn='spec_stacks_nov29/%s%s_quality.txt' % (loc,galaxy)

### save quality file
ascii.write(qual_dat,fn)

### select good galaxies
new_speclist=[]
new_mask=[]
for i in range(len(quality)):
    if quality[i]==True:
        new_speclist.append(spectra[i])
        new_mask.append(Mask[i])

### get wavelength coverage
wvcover=[]
wvsum=[]
for i in range(len(new_speclist)):
    w,f,e=Get_flux(new_speclist[i])
    wvsum.append(sum(w))
    wvcover.append(w)

IDX = np.argwhere(wvsum==np.max(wvsum))[0]

### get stack and compare to old stack
swv,sfl,ser=Stack_gal_spec(new_speclist,wvcover[IDX][:-1],new_mask)
swv2,sfl2,ser2=Get_flux(s[-1])


plt.plot(swv2,sfl2,'k',alpha=.5)
plt.plot(swv2,ser2,'k',alpha=.5)
plt.plot(swv,sfl,'b')
plt.plot(swv,ser,'b')
plt.show()

### make data table for new stack
stack_dat=Table([swv,sfl,ser],names=['wv','flam','err'])
fn='spec_stacks_nov29/%s%s_stack.dat' % (loc,galaxy)
ascii.write(stack_dat,fn)


