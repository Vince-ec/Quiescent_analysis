import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from astropy.io import fits
import cPickle
import os
from spec_id import Stack_spec_normwmean,Analyze_Stack_avgage,Likelihood_contours,Identify_stack, Analyze_Stack
from glob import glob
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Stack_spec_normwmean_nsig(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    numgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        inm=interp1d(wave,np.ones(len(wave)))
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
        numgrid[i][mask] = inm(wv[mask])
    ################

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    numgrid=np.transpose(numgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=np.sqrt(np.sum(numgrid[i]))/np.sqrt(np.sum(weigrid[i]))
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Model_fit_stack_normwmean_nsig(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

    #############Get redshift info###############

    zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
    speczs = np.round(speczs, 2)

    for i in range(len(speczs)):
        zinput = int(speczs[i] * 100) / 5 / 20.
        if zinput < 1:
            zinput = 1.0
        if zinput >1.8:
            zinput = 1.8
        zlist.append(zinput)
    for i in range(len(bins)):
        b = []
        for ii in range(len(zlist)):
            if bins[i] == zlist[ii]:
                b.append(ii)
        if len(b) > 0:
            zcount.append(len(b))
    zbin = sorted(set(zlist))

    ##############Stack spectra################

    wv,fl,err=Stack_spec_normwmean_nsig(speclist,speczs,wv_range)

    #############Prep output file###############

    chifile='chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps==False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m=[]
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    else:
        filepath = '../../../fsps_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname='%s.pkl' % pkl_name

    if os.path.isfile(pklname)==False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid=np.zeros([len(metal),len(A),len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii]=Identify_stack(fl,err,mf)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

""">10.87 Galaxies"""
###get list of spectra
ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)
nlist=glob('spec_stacks/*')

IDS=[]

for i in range(len(ids)):
    if 10.87<lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

speclist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i]==nlist[ii][12:18]:
            speclist.append(nlist[ii])

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput=int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
# metal = np.array([.0001, .0004, .004, .008, .02, ])
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

M,A=np.meshgrid(metal,age)

# Model_fit_stack_normwmean_nsig(speclist,tau,metal,age,speczs,np.arange(3250,5500,5),
#                          'gt10.87_fsps_nsignwm_stackfit','gt10.87_fsps_nwmeannm_spec',fsps=True)

# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_nsignwm_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# # levels=np.array([46.67813219, 418.99838181])
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
# # plt.show()
# plt.savefig('../research_plots/nsig_likelihood.png')
# plt.close()

wvs,fls,ers=Stack_spec_normwmean_nsig(speclist,rshift[IDS],np.arange(3250,5500,5))
wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,5))

plt.plot(wv,fl,label='>10.87 Stack')
plt.plot(wvs,fls)
plt.plot(wv,er)
plt.plot(wvs,ers)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
plt.ylabel('Relative Flux',size=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.minorticks_on()
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4)
plt.show()
# plt.savefig('../research_plots/nsig_stack.png')
# plt.close()