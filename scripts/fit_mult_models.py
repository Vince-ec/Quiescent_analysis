from spec_id  import Stack_spec_normwmean, Stack_model_normwmean, Scale_model, Identify_stack,\
    Analyze_Stack_avgage, Likelihood_contours
from time import time
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

def Model_fit_stack_normwmean_flxerr(speclist, tau, metal, A, speczs, wv_range, name, pkl_name, window, flxerr, res=5, fsps=False):
    ##############Stack spectra################
    wv, fl, er = Stack_spec_normwmean(speclist, speczs, wv_range)
    IDW=[U for U in range(len(wv)) if window[0]<=wv[U]<=window[1]]
    wv=wv[IDW]
    fl=fl[IDW]
    er=er[IDW]

    err = np.sqrt(er ** 2 + (flxerr * fl) ** 2)

    #############Prep output file###############

    chifile = 'chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps == False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
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
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname = 'pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist, modellist[i][ii][iii], speczs,
                                                       np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid = np.zeros([len(metal), len(A), len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii] = Identify_stack(fl, err, mf[IDW])
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

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

def Model_fit_stack_normwmean_std(speclist, tau, metal, A, speczs, wv_range, name, pkl_name, window, flxerr, res=5, fsps=False):
    ##############Stack spectra################
    wv, fl, er = Stack_spec_normwmean_nsig(speclist, speczs, wv_range)
    IDW=[U for U in range(len(wv)) if window[0]<=wv[U]<=window[1]]
    wv=wv[IDW]
    fl=fl[IDW]
    er=er[IDW]

    err = np.sqrt(er ** 2 + (flxerr * fl) ** 2)

    #############Prep output file###############

    chifile = 'chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps == False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
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
                    for iv in range(len(speczs)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], speczs[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname = 'pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist, modellist[i][ii][iii], speczs,
                                                       np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid = np.zeros([len(metal), len(A), len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii] = Identify_stack(fl, err, mf[IDW])
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

"""galaxy selection"""
ids, speclist, lmass, rshift = np.array(Readfile('masslist_dec8.dat', 1, is_float=False))
lmass, rshift = np.array([lmass, rshift]).astype(float)

IDA = []  # all masses in sample
IDL = []  # low mass sample
IDH = []  # high mass sample

for i in range(len(ids)):
    if 10.0 <= lmass[i] and 1 < rshift[i] < 1.75:
        IDA.append(i)
    if 10.871 > lmass[i] and 1 < rshift[i] < 1.75:
        IDL.append(i)
    if 10.871 < lmass[i] and 1 < rshift[i] < 1.75:
        IDH.append(i)

metal = np.array([0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0068, 0.0077, 0.0085, 0.0096, 0.0106,
                  0.012, 0.0132, 0.014, 0.0150, 0.0164, 0.018, 0.019, 0.021, 0.024, 0.027, 0.03])
age = [0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
       2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau = [0, 8.0, 8.15, 8.28, 8.43, 8.57, 8.72, 8.86, 9.0, 9.14, 9.29, 9.43, 9.57, 9.71, 9.86, 10.0]
M,A=np.meshgrid(metal,age)

"""high mass 5 and 10 res"""
# R=[5,10]
# W=[[3250,5500], [3400,5500], [3500,5500], [3250,5250], [3400,5250], [3500,5250]]
#
# for i in range(len(R)):
#     for ii in range(len(W)):
#         wv_flxerr,flxer=np.array(Readfile('flx_err/HM_%s_%s-%s.dat' %(R[i], W[ii][0], W[ii][1])))
#         flxer=np.zeros(len(flxer))
#         Model_fit_stack_normwmean_std(speclist[IDH] , tau, metal, age, rshift[IDH], np.arange(W[ii][0],W[ii][1],R[i]),
#                                  'gt10.87_fsps_std_nfe_%s_%s-%s_stackfit' %(R[i], W[ii][0], W[ii][1]),
#                                   'gt10.87_fsps_10-13_%s_spec' % R[i], W[ii], flxer,res=R[i],fsps=True)
#         Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_std_nfe_%s_%s-%s_stackfit_chidata.fits' %
#                                               (R[i], W[ii][0], W[ii][1]), np.array(tau),metal,age)
#         onesig,twosig=Likelihood_contours(age,metal,Pr)
#         levels=np.array([twosig,onesig])
#         plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
#         plt.contourf(M,A,Pr,40,cmap=colmap)
#         plt.plot(bfmetal,bfage,'cp',ms=2,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
#         plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
#         plt.tick_params(axis='both', which='major', labelsize=17)
#         plt.gcf().subplots_adjust(bottom=0.16)
#         plt.minorticks_on()
#         plt.xlabel('Metallicity (Z$_\odot$)')
#         plt.ylabel('Age (Gyrs)')
#         plt.legend()
#         plt.savefig('../poster_plots/gt10.87_fsps_std_nfe_%s_%s-%s_LH.png' % (R[i], W[ii][0], W[ii][1]))
#         plt.close()

"""high mass 12 res"""
# W=[[3250,5500], [3394,5500], [3490,5500], [3250,5254], [3394,5254], [3490,5254]]
#
# for i in range(len(W)):
#     wv_flxerr,flxer=np.array(Readfile('flx_err/HM_12_%s-%s.dat' %(W[i][0], W[i][1])))
#     flxer = np.zeros(len(flxer))
#     Model_fit_stack_normwmean_std(speclist[IDH] , tau, metal, age, rshift[IDH], np.arange(W[i][0],W[i][1],12),
#                              'gt10.87_fsps_std_nfe_12_%s-%s_stackfit' %(W[i][0], W[i][1]),
#                               'gt10.87_nage_flxer_spec', W[i], flxer,res=12,fsps=True)
#     Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_std_nfe_12_%s-%s_stackfit_chidata.fits' %
#                                           (W[i][0], W[i][1]), np.array(tau),metal,age)
#     onesig,twosig=Likelihood_contours(age,metal,Pr)
#     levels=np.array([twosig,onesig])
#     plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
#     plt.contourf(M,A,Pr,40,cmap=colmap)
#     plt.plot(bfmetal,bfage,'cp',ms=2,label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
#     plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
#     plt.tick_params(axis='both', which='major', labelsize=17)
#     plt.gcf().subplots_adjust(bottom=0.16)
#     plt.minorticks_on()
#     plt.xlabel('Metallicity (Z$_\odot$)')
#     plt.ylabel('Age (Gyrs)')
#     plt.legend()
#     plt.savefig('../poster_plots/gt10.87_fsps_std_nfe_12_%s-%s_LH.png' % (W[i][0], W[i][1]))
#     plt.close()

"""low mass 5 and 10 res"""
# R = [5, 10]
# W = [[3400, 5500], [3500, 5500], [3400, 5250], [3500, 5250]]
R = [10]
W = [[3600, 5350]]

for i in range(len(R)):
    for ii in range(len(W)):
        wv_flxerr, flxer = np.array(Readfile('flx_err/LM_%s_%s-%s.dat' % (R[i], W[ii][0], W[ii][1])))
        # flxer = np.zeros(len(flxer))
        Model_fit_stack_normwmean_std(speclist[IDL], tau, metal, age, rshift[IDL], np.arange(W[ii][0], W[ii][1], R[i]),
                                  'lt10.87_fsps_std__%s_%s-%s_stackfit' % (R[i], W[ii][0], W[ii][1]),
                                  'ls10.87_fsps_10-13_%s_spec' % R[i], W[ii], flxer, res=R[i], fsps=True)
        Pr, bfage, bfmetal = Analyze_Stack_avgage('chidat/lt10.87_fsps_std__%s_%s-%s_stackfit_chidata.fits' %
                                                  (R[i], W[ii][0], W[ii][1]), np.array(tau), metal, age)
        onesig, twosig = Likelihood_contours(age, metal, Pr)
        levels = np.array([twosig, onesig])
        plt.contour(M, A, Pr, levels, colors='k', linewidths=2)
        plt.contourf(M, A, Pr, 40, cmap=colmap)
        plt.plot(bfmetal, bfage, 'cp', ms=2,
                 label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage, np.round(bfmetal / 0.019, 2)))
        plt.xticks([0, .005, .01, .015, .02, .025, .03],
                   np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.02, 2))
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.gcf().subplots_adjust(bottom=0.16)
        plt.minorticks_on()
        plt.xlabel('Metallicity (Z$_\odot$)')
        plt.ylabel('Age (Gyrs)')
        plt.legend()
        plt.savefig('../poster_plots/lt10.87_fsps_std__%s_%s-%s_LH.png' % (R[i], W[ii][0], W[ii][1]))
        plt.close()

"""low mass 12 res"""
# W = [[3400,5500], [3496,5500], [3400,5248], [3496,5248]]
#
# for i in range(len(W)):
#     wv_flxerr, flxer = np.array(Readfile('flx_err/LM_12_%s-%s.dat' % (W[i][0], W[i][1])))
#     flxer = np.zeros(len(flxer))
#     Model_fit_stack_normwmean_std(speclist[IDL], tau, metal, age, rshift[IDL], np.arange(W[i][0], W[i][1], 12),
#                               'lt10.87_fsps_std_nfe_12_%s-%s_stackfit' % (W[i][0], W[i][1]),
#                               'ls10.87_fsps_newdata_12_spec', W[i], flxer, res=12, fsps=True)
#     Pr, bfage, bfmetal = Analyze_Stack_avgage('chidat/lt10.87_fsps_std_nfe_12_%s-%s_stackfit_chidata.fits' %
#                                               (W[i][0], W[i][1]), np.array(tau), metal, age)
#     onesig, twosig = Likelihood_contours(age, metal, Pr)
#     levels = np.array([twosig, onesig])
#     plt.contour(M, A, Pr, levels, colors='k', linewidths=2)
#     plt.contourf(M, A, Pr, 40, cmap=colmap)
#     plt.plot(bfmetal, bfage, 'cp', ms=2,
#              label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage, np.round(bfmetal / 0.019, 2)))
#     plt.xticks([0, .005, .01, .015, .02, .025, .03], np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.02, 2))
#     plt.tick_params(axis='both', which='major', labelsize=17)
#     plt.gcf().subplots_adjust(bottom=0.16)
#     plt.minorticks_on()
#     plt.xlabel('Metallicity (Z$_\odot$)')
#     plt.ylabel('Age (Gyrs)')
#     plt.legend()
#     plt.savefig('../poster_plots/lt10.87_fsps_std_nfe_12_%s-%s_LH.png' % (W[i][0], W[i][1]))
#     plt.close()