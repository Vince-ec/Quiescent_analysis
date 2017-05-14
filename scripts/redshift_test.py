import numpy as np
from vtl.Readfile import Readfile
from scipy.interpolate import interp1d
from glob import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import seaborn as sea
from spec_id import Scale_model,Identify_stack,Analyze_specz

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})

def Specz_fit2(galaxy, spec, rshift,name):
    #############Parameters#####################
    metal = np.array([0.0031, 0.0061, 0.0085, 0.012, 0.015, 0.019, 0.024])
    A = [0.5,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]

    #############Read in spectra#################
    wv,fl,err=np.load(spec)
    wv, fl, err = np.array([wv[wv<11100], fl[wv<11100], err[wv<11100]])

    #############Prep output file###############
    chifile='rshift_dat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############
    filepath = '../../../fsps_models_for_fit/rshift_models/'
    modellist = []
    for i in range(len(metal)):
        m = []
        for ii in range(len(A)):
            a = []
            for iii in range(len(rshift)):
                a.append(filepath + 'm%s_a%s_t8.0_z%s_%s_model.npy' % (metal[i], A[ii], rshift[iii],galaxy))
            m.append(a)
        modellist.append(m)

    ##############Create chigrid and add to file#################
    chigrid=np.zeros([len(metal),len(A),len(rshift)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(rshift)):
                IDer = [U for U in range(len(wv)) if 4860 * (1 + rshift[iii]) <= wv[U] <= 4880 * (1 + rshift[iii])]
                error = np.array(err)
                flux = np.array(fl)
                error[IDer] = 1E8
                flux[IDer] = 0
                mwv,mf= np.load(modellist[i][ii][iii])
                imf=interp1d(mwv,mf)(wv)
                C=Scale_model(fl,err,imf)
                chigrid[i][ii][iii]=Identify_stack(flux,error,imf*C)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)
    ################Write chigrid file###############

    hdulist.writeto(chifile)

    Analyze_specz(chifile,rshift,metal,A,name)

    print 'Done!'

    return

rshift=np.arange(1.19,1.231,0.001)

# Specz_fit2('s40597','spec_stacks_jan24/s40597_stack.npy',rshift,'s40597_hb')

metal = np.array([0.0031, 0.0061, 0.0085, 0.012, 0.015, 0.019, 0.024])
A = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

# Analyze_specz('rshift_dat/s40597_hb_chidata.fits',rshift,metal,A,'s40597_hb')

z,pz=Readfile('rshift_dat/s40597_specz_Pofz.dat')
zb,pzb=Readfile('rshift_dat/s40597_hb_Pofz.dat')
pzp,zp=Readfile('Pofz/s40597_pofz.dat')

pzp=np.exp(-pzp/2)
C=np.trapz(pzp,zp)

plt.plot(z,pz)
plt.plot(zb,pzb)
plt.plot(zp,pzp/C)
plt.show()