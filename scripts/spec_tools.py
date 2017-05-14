import numpy as np
from astropy.io import fits

def Simspec(f):
    spec = fits.open(f)
    simwave=np.array(spec[1].data.field('LAMBDA'))
    simflux=np.array(spec[1].data.field('FLUX'))
    Index=[]
    for i in range(len(simwave)):
        if simwave[i]<12000:
            Index.append(i)
    return simwave[Index],simflux[Index]

def Simspectest(f):
    spec = fits.open(f)
    simwave=np.array(spec[1].data.field('LAMBDA'))
    simflux=np.array(spec[1].data.field('FLUX'))
    simerr=np.array(spec[1].data.field('FERROR'))
    Index=[]
    for i in range(len(simwave)):
        if simwave[i]<12000:
            Index.append(i)
    return simwave[Index],simflux[Index],simerr[Index]

##################################################################################################
# Routines to fit models with bins
##################################################################################################
def Bin_files(filename):
    dat1=Simspec(filename)
    dat=Normalize(dat1[0],dat1[1])
    interp=interp1d(dat[0],dat[1])
    lims = np.linspace(8250, 11400, 100)
    bin =[]
    bnwave=[]
    for i in range(len(lims)-1):
        rng=np.linspace(lims[i],lims[i+1],500)
        tp=np.trapz(interp(rng),rng)
        bt=np.trapz(np.ones(len(rng)),rng)
        bin.append(tp/bt)
        top=np.trapz(rng, rng)
        bm=np.trapz(np.ones(len(rng)),rng)
        bnwave.append(top/bm)

    # fn = 'BINS/' + filename.replace('_slitless_2.SPC.fits', '_bin_norm.dat')
    # ascii.write([bnwave, bin], fn)

    return bnwave,bin

def Bin(wv,fl):
    dat=Normalize(wv,fl)
    interp=interp1d(dat[0],dat[1])
    lims = np.linspace(8250, 11400, 100)
    bin =[]
    bnwave=[]
    for i in range(len(lims)-1):
        rng=np.linspace(lims[i],lims[i+1],500)
        tp=np.trapz(interp(rng),rng)
        bt=np.trapz(np.ones(len(rng)),rng)
        bin.append(tp/bt)
        top=np.trapz(rng, rng)
        bm=np.trapz(np.ones(len(rng)),rng)
        bnwave.append(top/bm)

    return bnwave,bin

def Identify_with_bins(wv,spec,sens,filetocheck):
    weight=interp1d(wv,sens)
    dat=Bin(wv,spec)
    chi=[]

    for i in range(len(filetocheck)):
        checkdat=Readfile(filetocheck[i],1)
        x = []
        for ii in range(len(dat[1])):
            x.append(weight(dat[0])*((dat[1][ii] - checkdat[1][ii]) ** 2) / checkdat[1][ii])
        chi.append(np.sum(x))

    chi_id=filetocheck[chi.index(min(chi))]

    return chi,chi_id

###################################################################################################
#Identify with normalization to 1
##################################################################################################

def Normalize(wl,flux):
    sdat=Select_range(wl,flux)
    coeff=np.trapz(sdat[1],sdat[0])
    Nflux=np.multiply(1/coeff,sdat[1])

    return sdat[0],Nflux

def Normalize2(wl,flux):
    sdat=Select_range2(wl,flux)
    coeff=np.trapz(sdat[1],sdat[0])
    Nflux=np.multiply(1/coeff,sdat[1])

    return sdat[0],Nflux

def Identify1(wv,spec,sens,filetocheck):
    weight=interp1d(wv,sens)
    dat=Normalize(wv,spec)
    chi=[]

    for i in range(len(filetocheck)):
        checkdat1=Simspec(filetocheck[i])
        checkdat2=Normalize2(checkdat1[0],checkdat1[1])
        checkdat=interp1d(checkdat2[0],checkdat2[1])
        x = []
        for ii in range(len(dat[1])):
            x.append(weight(dat[0])*((dat[1][ii] - checkdat(dat[0][ii])) ** 2) / checkdat(dat[0][ii]))
        chi.append(np.sum(x))

    chi_id=filetocheck[chi.index(min(chi))]

    return chi,chi_id