from spec_id  import Identify_stack, Scale_model, Stack_spec, Stack_model, Analyze_Stack, Likelihood_contours, Model_fit_stack_features
from time import time
import seaborn as sea
from glob import glob
import numpy as np
from scipy.interpolate import interp1d, interp2d
import sympy as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import cPickle
import os
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)


def Stack_spec_add(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave,flux,error=np.array(Readfile(spec[i],1))
        wave/=(1+redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        flgrid[i][mask]=ifl(wv[mask])
        errgrid[i][mask]=ier(wv[mask])
    ################

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    # weigrid=errgrid**(-2)
    # infmask=np.isinf(weigrid)
    # weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i])
        err[i]=np.sum(errgrid[i])
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_add(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        C=Scale_model(flux,error,iF)
        mflux=C*iF

        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        flentry[mask]=ifl(wv_range[mask])
        errentry[mask]=ier(wv_range[mask])
        flgrid.append(flentry)
        errgrid.append(errentry)

    wv = np.array(wv_range)

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    # weigrid=errgrid**(-2)
    # infmask=np.isinf(weigrid)
    # weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i])
        err[i]=np.sum(errgrid[i])
    ################

    return wv, stack, err

def Model_fit_stack_add(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_add(speclist,speczs,wv_range)

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
                    mw, mf, me = Stack_model_add(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
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

def Stack_spec_wmean(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
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
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_wmean(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        C=Scale_model(flux,error,iF)
        mflux=C*iF

        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        flentry[mask]=ifl(wv_range[mask])
        errentry[mask]=ier(wv_range[mask])
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

def Model_fit_stack_wmean(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_wmean(speclist,speczs,wv_range)

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
                    mw, mf, me = Stack_model_wmean(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
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

def Stack_spec_normadd(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave,flux,error=np.array(Readfile(spec[i],1))
        wave/=(1+redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        reg=np.arange(4150,4180,1)
        Cr=np.trapz(ifl(reg),reg)
        flgrid[i][mask]=ifl(wv[mask])/Cr
        errgrid[i][mask]=ier(wv[mask])/Cr
    ################

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    # weigrid=errgrid**(-2)
    # infmask=np.isinf(weigrid)
    # weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i])
        err[i]=np.sum(errgrid[i])
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_normadd(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        C=Scale_model(flux,error,iF)
        mflux=C*iF

        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        reg = np.arange(4150, 4180, 1)
        Cr = np.trapz(ifl(reg), reg)
        flentry[mask]=ifl(wv_range[mask])/ Cr
        errentry[mask]=ier(wv_range[mask])/ Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    wv = np.array(wv_range)

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    # weigrid=errgrid**(-2)
    # infmask=np.isinf(weigrid)
    # weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i])
        err[i]=np.sum(errgrid[i])
    ################

    return wv, stack, err

def Model_fit_stack_normadd(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_normadd(speclist,speczs,wv_range)

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
                    mw, mf, me = Stack_model_normadd(speclist,modellist[i][ii][iii], speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
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

def Stack_spec_normwmean_WC(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    cogrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        co=Get_Cont(wave,flux,error)
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ico=interp1d(wave,co)
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        cogrid[i][mask] = ico(wv[mask]) / Cr
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
    ################

    cogrid=np.transpose(cogrid)
    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err,cont=np.zeros([3,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        cont[i]=np.sum(cogrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=1/np.sqrt(np.sum(weigrid[i]))
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX], cont[IDX]

def Model_fit_stack_normwmean_NC(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_normwmean(speclist,speczs,wv_range)
    IDX = [U for U in range(len(wv)) if 3700 <= wv[U]]
    co = Get_Cont(wv, fl, err)

    ncfl=fl[IDX]/co[IDX]
    ncerr=err[IDX]/co[IDX]

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
                mco=Get_Cont(wv,mf,err)
                chigrid[i][ii][iii]=Identify_stack(ncfl,ncerr,mf[IDX]/mco[IDX])
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

def Get_Cont(w,f,e):
    m2r = [3175, 3280, 3340, 3515, 3550, 3650, 3710, 3770, 3800, 3850,
            3910, 3989, 3991, 4030, 4082, 4122, 4250, 4385, 4830, 4930, 4990, 5030, 5109, 5250]

    Mask=np.zeros(len(w))
    for i in range(len(Mask)):
        if m2r[0]<=w[i]<=m2r[1]:
            Mask[i]=1
        if m2r[2]<=w[i]<=m2r[3]:
            Mask[i]=1
        if m2r[4]<=w[i]<=m2r[5]:
            Mask[i]=1
        if m2r[6]<=w[i]<=m2r[7]:
            Mask[i]=1
        if m2r[8]<=w[i]<=m2r[9]:
            Mask[i]=1
        if m2r[8]<=w[i]<=m2r[9]:
            Mask[i]=1
        if m2r[10]< w[i]<=m2r[11]:
            Mask[i] = 1
        if m2r[12]<=w[i]<=m2r[13]:
            Mask[i]=1
        if m2r[14]<=w[i]<=m2r[15]:
            Mask[i]=1
        if m2r[16]<=w[i]<=m2r[17]:
            Mask[i]=1
        if m2r[18] <= w[i] <= m2r[19]:
            Mask[i] = 1
        if m2r[20] <= w[i] <= m2r[21]:
            Mask[i] = 1
        if m2r[22] <= w[i] <= m2r[23]:
            Mask[i] = 1


    maskw = np.ma.masked_array(w, Mask)

    conts = np.ma.polyfit(maskw, f, 3, w=1/e**2)
    # C0 = x3 * w ** 3 + x2 * w ** 2 + x1 * w + x0
    C0 = np.polyval(conts,w)

    return C0

def Stack_spec_normwmean(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
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
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

def Stack_model_normwmean(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        C=Scale_model(flux,error,iF)
        mflux=C*iF

        # Fl = iF
        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flentry[mask] = ifl(wv_range[mask]) / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
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

def Stack_model_normwmean_2(speclist, modellist, redshifts, redshiftbins, wv_range):

    flgrid =[]
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave,flux,error=np.array(Readfile(speclist[i],1))
        wave=wave/(1+redshifts[i])

        #######read in corresponding model, and interpolate flux
        W,F,E=np.array(Readfile(modellist[i],1))
        W=W/(1+redshiftbins[i])
        iF=interp1d(W,F)(wave)

        #######scale the model
        # C=Scale_model(flux,error,iF)
        C=1
        mflux=C*iF

        # Fl = iF
        Fl = mflux
        Er = error

        ########interpolate spectra
        flentry=np.zeros(len(wv_range))
        errentry=np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl=interp1d(wave,Fl)
        ier=interp1d(wave,Er)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flentry[mask] = ifl(wv_range[mask]) / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
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

def Model_fit_stack_normwmean(speclist, tau, metal, A, speczs, wv_range,name, pkl_name, fsps=False):

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

    wv,fl,err=Stack_spec_normwmean(speclist,speczs,wv_range)

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


"""galaxy selection"""
ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)
nlist=glob('spec_stacks/*')

IDS=[]

for i in range(len(ids)):
    if 10.87>lmass[i] and 1<=rshift[i]<1.75:
        IDS.append(i)

print len(IDS)

speclist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i]==nlist[ii][12:18]:
            speclist.append(nlist[ii])

metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput=int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

"""1D bestfit"""
###########################
# flist=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.012_a1.62_t8.0_z%s_model.dat' % zlist[i])
# wv,fl,er=Stack_spec(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# print sum(((fl - fs) / er) ** 2)
#
# plt.plot(wv,np.ones(len(wv)),'k--', alpha=.2)
# plt.plot(wv,fl,label='>10.87 Stack')
# plt.plot(fwv,fs, label='Best fit t=1.62 Gyrs Z=0.63 Z$_\odot$')
# plt.plot(wv,er)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/stack_nocont.png')
# plt.close()
#
# ##
# flist=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.0016_a5.26_t8.0_z%s_model.dat' % zlist[i])
# wv,fl,er=Stack_spec_add(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model_add(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
#
# print sum(((fl - fs) / er) ** 2)
#
# plt.plot(wv,fl,label='>10.87 Stack')
# plt.plot(fwv,fs, label='Best fit t=5.62 Gyrs Z=0.08 Z$_\odot$')
# plt.plot(wv,er)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('F$_\lambda$ (erg/s/cm$^2$/$\AA$)',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/stack_add.png')
# plt.close()
#
# #
# flist=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.0077_a1.42_t8.0_z%s_model.dat' % zlist[i])
# wv,fl,er=Stack_spec_wmean(speclist,rshift[IDS],np.arange(3250,5500,5))
# # fwv,fs,fe=Stack_model_wmean(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
# # print sum(((fl - fs) / er) ** 2)
#
# plt.plot(wv,fl,label='>10.87 Stack')
# # plt.plot(fwv,fs, label='Best fit t=1.42 Gyrs Z=0.41 Z$_\odot$')
# plt.plot(wv,er)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('F$_\lambda$ (erg/s/cm$^2$/$\AA$)',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4)
# plt.show()
# plt.savefig('../research_plots/stack_mean.png')
# plt.close()
#
# #
# flist=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.002_a5.26_t8.0_z%s_model.dat' % zlist[i])
# wv,fl,er=Stack_spec_normadd(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model_normadd(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
#
#
# print sum(((fl - fs) / er) ** 2)
#
# plt.plot(wv,fl,label='>10.87 Stack')
# plt.plot(fwv,fs, label='Best fit t=5.26 Gyrs Z=0.11 Z$_\odot$')
# plt.plot(wv,er)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/stack_normadd.png')
# plt.close()
#
# #
flist=[]
for i in range(len(zlist)):
    flist.append('../../../fsps_models_for_fit/models/m0.012_a2.11_t0_z%s_model.dat' % zlist[i])
wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,5))
fwv,fs,fe=Stack_model_normwmean(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
fwv2,fs2,fe2=Stack_model_normwmean_2(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))

print sum(((fl - fs) / er) ** 2)

# plt.plot(wv,fl,label='>10.87 Stack')
plt.plot(fwv,fs, label='Best fit t=1.62 Gyrs Z=0.79 Z$_\odot$')
plt.plot(fwv2,fs2, label='Best fit t=1.62 Gyrs Z=0.79 Z$_\odot$')
# plt.plot(wv,er)
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
# plt.savefig('../research_plots/stack_normmean.png')
# plt.close()

# plt.hist(rshift[IDS])
# plt.show()

"""Likelihood contours"""
# M,A=np.meshgrid(metal,age)

# Model_fit_stack_features(speclist,tau,metal,age,speczs,np.arange(3250,5500,5),
#                          'gt10.87_fsps_feat_stackfit','gt10.87_fsps_spec',fsps=True)
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_feat_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# levels=np.array([46.67813219, 418.99838181])
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
# plt.show()

# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# levels=np.array([46.67813219, 418.99838181])
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
# plt.show()
# plt.savefig('../research_plots/stack_nocont_lhood.png')
# plt.close()
#
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_add_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# levels=np.array([17.40797549, 148.76877216])
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
# plt.savefig('../research_plots/stack_add_lhood.png')
# plt.close()
#
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_wmean_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# levels=np.array([261.690917, 956.85758607])
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
# plt.savefig('../research_plots/stack_wmean_lhood.png')
# plt.close()
#
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_normadd_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# levels=np.array([18.07512291, 168.9915106])
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
# plt.savefig('../research_plots/stack_normadd_lhood.png')
# plt.close()
#
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_normwmean_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# levels=np.array([67.55856273, 296.48441537])
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
# plt.savefig('../research_plots/stack_normwmean_lhood.png')
# plt.close()

"""norm wmean exam"""
# M,A=np.meshgrid(metal,age)
#
# flist=[]
# flist2=[]
# for i in range(len(zlist)):
#     flist.append('../../../fsps_models_for_fit/models/m0.015_a1.62_t8.0_z%s_model.dat' % zlist[i])
#     flist2.append('../../../fsps_models_for_fit/models/m0.0096_a1.85_t8.0_z%s_model.dat' % zlist[i])
#
# wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],np.arange(3250,5500,5))
# # wvc,flc,erc,con=Stack_spec_normwmean_WC(speclist,rshift[IDS],np.arange(3250,5500,5))
# # wvnc,flnc,ernc=Stack_spec(speclist,rshift[IDS],np.arange(3250,5500,5))
# fwv,fs,fe=Stack_model_normwmean(speclist,flist, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# fwv2,fs2,fe2=Stack_model_normwmean(speclist,flist2, speczs, zlist, np.arange(wv[0],wv[-1]+5,5))
# co=Get_Cont(wv,fl,er)
#
# plt.plot(wvc,flc,label='>10.87 Stack')
# plt.plot(wv,fl)
# # plt.plot(wvnc,flnc)
# plt.plot(wv,fl)
# # plt.plot(wvc,flc/con)
# plt.plot(fwv,fs, label='Best fit t=1.62 Gyrs Z=0.79 Z$_\odot$')
# plt.plot(fwv2,fs2, label='No continuum Best fit t=1.85 Gyrs Z=0.51 Z$_\odot$')
# # plt.plot(wv,er)
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# # plt.ylim(0,1.4)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.minorticks_on()
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4)
# plt.show()

# Model_fit_stack_normwmean_NC(speclist,tau,metal,age,speczs,np.arange(3250,5500,5),
#                          'gt10.87_fsps_feat_normwmean_nc_stackfit','gt10.87_fsps_normwmean_spec',fsps=True)
# Pr,bfage,bfmetal=Analyze_Stack('chidat/gt10.87_fsps_feat_normwmean_nc_stackfit_chidata.fits', np.array(tau),metal,age)
# # onesig,twosig=Likelihood_contours(age,metal,Pr)
# # levels=np.array([twosig,onesig])
# levels=np.array([37.89456211, 329.82304056])
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
# plt.show()