__author__ = 'vestrada'

import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import fits
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import cPickle
import os
from time import time
import seaborn as sea

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Oldest_galaxy(z):
    return cosmo.age(z).value


def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    C = np.trapz(G, x)
    G /= C
    return G


def Select_range(wv, flux, error):
    w = np.array(wv)
    f = np.array(flux)
    e = np.array(error)

    INDEX = []
    for i in range(len(w)):
        if 8000 < w[i] < 11500:
            INDEX.append(i)

    W = w[INDEX]
    F = f[INDEX]
    E = e[INDEX]

    return W, F, E


def Select_range_model(wv, flux):
    w = np.array(wv)
    f = np.array(flux)

    INDEX = []
    for i in range(len(w)):
        if 7900 < w[i] < 11600:
            INDEX.append(i)

    W = w[INDEX]
    F = f[INDEX]

    return W, F


def Get_flux(FILE):
    observ = fits.open(FILE)
    w = np.array(observ[1].data.field('wave'))
    f = np.array(observ[1].data.field('flux')) * 1E-17
    sens = np.array(observ[1].data.field('sensitivity'))
    contam = np.array(observ[1].data.field('contam')) * 1E-17
    e = np.array(observ[1].data.field('error')) * 1E-17
    f -= contam
    f /= sens
    e /= sens

    INDEX = []
    for i in range(len(w)):
        if 7950 < w[i] < 11350:
            INDEX.append(i)

    w = w[INDEX]
    f = f[INDEX]
    e = e[INDEX]

    for i in range(len(f)):
        if f[i] < 0:
            f[i] = 0

    return w, f, e


def Get_flux_nocont(FILE, z):
    w, f, e = np.load(FILE)
    if FILE == '../spec_stacks_jan24/s40597_stack.npy':
        IDW = []
        for ii in range(len(w)):
            if 7950 <  w[ii] < 11000:
                IDW.append(ii)

    else:
        IDW = []
        for ii in range(len(w)):
            if 7950 < w[ii] < 11300:
                IDW.append(ii)

    w, f, e = np.array([w[IDW], f[IDW], e[IDW]])
    w /= (1 + z)

    m2r = [3175, 3280, 3340, 3515, 3550, 3650, 3710, 3770, 3800, 3850,
           3910, 3989, 3991, 4030, 4082, 4122, 4250, 4385, 4830, 4930, 4990, 5030, 5109, 5250]

    Mask = np.zeros(len(w))
    for i in range(len(Mask)):
        if m2r[0] <= w[i] <= m2r[1]:
            Mask[i] = 1
        if m2r[2] <= w[i] <= m2r[3]:
            Mask[i] = 1
        if m2r[4] <= w[i] <= m2r[5]:
            Mask[i] = 1
        if m2r[6] <= w[i] <= m2r[7]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[10] < w[i] <= m2r[11]:
            Mask[i] = 1
        if m2r[12] <= w[i] <= m2r[13]:
            Mask[i] = 1
        if m2r[14] <= w[i] <= m2r[15]:
            Mask[i] = 1
        if m2r[16] <= w[i] <= m2r[17]:
            Mask[i] = 1
        if m2r[18] <= w[i] <= m2r[19]:
            Mask[i] = 1
        if m2r[20] <= w[i] <= m2r[21]:
            Mask[i] = 1
        if m2r[22] <= w[i] <= m2r[23]:
            Mask[i] = 1

    maskw = np.ma.masked_array(w, Mask)

    x3, x2, x1, x0 = np.ma.polyfit(maskw, f, 3, w=1 / e ** 2)
    C0 = x3 * w ** 3 + x2 * w ** 2 + x1 * w + x0

    f /= C0
    e /= C0

    return w, f, e


def NormalP(dependent, p):
    ncoeff = np.trapz(p, dependent)
    p /= ncoeff
    return p


def P(chisqr):
    p1 = np.exp(-chisqr / 2.)
    return p1


def Get_repeats(x,y):
    z=[x,y]
    tz=np.transpose(z)
    size=np.zeros(len(tz))
    for i in range(len(size)):
        size[i]=len(np.argwhere(tz==tz[i]))/2
    size/=5
    return size


def Error(p, y):
    NP = interp1d(y, p)
    x = np.linspace(y[0], y[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(x)):
        e = np.trapz(NP(x[0:i + 1]), x[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = x[i]
        if herr == 0:
            if e >= .84:
                herr = x[i]
                break

    return lerr, herr


def Scale_model(D, sig, M):
    C = np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))
    return C


def Identify_stack(fl, err, mfl):
    x = ((fl - mfl) / err) ** 2
    chi = np.sum(x)
    return chi


def Likelihood_contours(age, metallicty, prob):
    ####### Create fine resolution ages and metallicities
    ####### to integrate over
    m2 = np.linspace(min(metallicty), max(metallicty), 50)

    ####### Interpolate prob
    P2 = interp2d(metallicty, age, prob)(m2, age)

    ####### Create array from highest value of P2 to 0
    pbin = np.linspace(0, np.max(P2), 1000)
    pbin = pbin[::-1]

    ####### 2d integrate to find the 1 and 2 sigma values
    prob_int=np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i]=np.trapz(np.trapz(p,m2,axis=1),age)

    ######## Identify 1 and 2 sigma values
    onesig = np.abs(np.array(prob_int) - 0.68)
    twosig = np.abs(np.array(prob_int) - 0.95)

    return pbin[np.argmin(onesig)], pbin[np.argmin(twosig)]


def Norm_P_stack(tau, metal, age, chi):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    prob = []
    for i in range(len(metal)):
        preprob1 = []
        for ii in range(len(age)):
            preprob2 = []
            for iii in range(len(tau)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii] / 2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all tau
    ######## End up with age vs metallicity matricies
    ######## use unlogged tau
    ultau = np.append(0, np.power(10, tau[1:] - 9))
    M = []
    for i in range(len(metal)):
        A = []
        for ii in range(len(age)):
            T = []
            for iii in range(len(tau) - 1):
                T.append(sp.N((ultau[iii + 1] - ultau[iii]) * (prob[i][ii][iii] + prob[i][ii][iii + 1]) / 2))
            A.append(sp.mpmath.fsum(T))
        M.append(A)

    ######## Integrate over metallicity to get age prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(metal)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (M[i][ii] + M[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(metal) - 1):
        preC3.append(sp.N((metal[i + 1] - metal[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    ######## Create normal prob grid
    P = []
    for i in range(len(metal)):
        preP = []
        for ii in range(len(age)):
            preP.append(M[i][ii] / C)
        P.append(np.array(preP).astype(np.float128))

    return P


def Analyze_Stack(chifits, tau, metal, age):
    ####### Read in file
    dat = fits.open(chifits)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    ####### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P_stack(tau, metal, age, chi)).astype(np.float128)

    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]


def Analyze_Stack_avgage(chifits, tau, metal, age,age_conv='../data/tau_scale_ntau.dat'):
    ####### Read in file
    dat = fits.open(chifits)
    chi = np.zeros([len(metal),len(age),len(tau)])
    for i in range(len(metal)):
        chi[i]=dat[i + 1].data
    chi = chi.T

    scale = Readfile(age_conv, 1)

    overhead = np.zeros(len(scale))
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i]=sum(amt)

    newchi = np.zeros(chi.shape)
    for i in range(len(chi)):
        if i == 0:
            newchi[i]=chi[i]
        else:
            iframe = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            newchi[i]=np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau
    ultau = np.append(0, np.power(10, tau[1:] - 9))
    prob = np.exp(-newchi.T.astype(np.float128)/2)

    P=np.trapz(prob,ultau,axis=2)
    C=np.trapz(np.trapz(P,age,axis=1),metal)

    prob=P/C
    ####### get best fit values
    [idmax] = np.argwhere(prob==np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]


def Analyze_Stack_avgage_cont_feat(contfits, featfits, tau, metal, age, age_conv='../data/tau_scale_ntau.dat'):
    ####### Read in file
    Cdat = fits.open(contfits)
    Cchi = np.zeros([len(metal), len(age), len(tau)])

    Fdat = fits.open(featfits)
    Fchi = np.zeros([len(metal), len(age), len(tau)])

    for i in range(len(metal)):
        Fchi[i] = Fdat[i + 1].data
        Cchi[i] = Cdat[i + 1].data

    Fchi = Fchi.T
    Cchi = Cchi.T

    scale = Readfile(age_conv)

    overhead = np.zeros(len(scale))
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    newCchi = np.zeros(Cchi.shape)
    newFchi = np.zeros(Fchi.shape)

    for i in range(len(Cchi)):
        if i == 0:
            newCchi[i] = Cchi[i]
            newFchi[i] = Fchi[i]
        else:
            cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
            newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

            fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
            newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau

    ultau = np.append(0, np.power(10, tau[1:] - 9))

    cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

    Pc = np.trapz(cprob, ultau, axis=2)
    Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

    fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

    Pf = np.trapz(fprob, ultau, axis=2)
    Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

    comb_prob = cprob / Cc * fprob / Cf

    prob = np.trapz(comb_prob, ultau, axis=2)
    C0 = np.trapz(np.trapz(prob, age, axis=1), metal)
    prob /= C0

    ##### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]


def Analyze_Stack_avgage_cont_feat_combine(cont_chifits, feat_chifits, tau, metal, age, age_conv='../data/tau_scale_ntau.dat'):

    Cprob = np.ones([len(metal), len(age), len(tau)])
    scale = Readfile(age_conv, 1)
    overhead = np.zeros(len(scale))
    ultau = np.append(0, np.power(10, tau[1:] - 9))

    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for i in range(len(cont_chifits)):
        ####### Read in file
        Cdat = fits.open(cont_chifits[i])
        Cchi = np.zeros([len(metal), len(age), len(tau)])

        Fdat = fits.open(feat_chifits[i])
        Fchi = np.zeros([len(metal), len(age), len(tau)])

        for ii in range(len(metal)):
            Fchi[ii] = Fdat[ii + 1].data
            Cchi[ii] = Cdat[ii + 1].data

        Fchi = Fchi.T
        Cchi = Cchi.T

        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for ii in range(len(Cchi)):
            if ii == 0:
                newCchi[ii] = Cchi[ii]
                newFchi[ii] = Fchi[ii]
            else:
                cframe = interp2d(metal, scale[ii], Cchi[ii])(metal, age[:-overhead[ii]])
                newCchi[ii] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[ii], axis=0), axis=0)

                fframe = interp2d(metal, scale[ii], Fchi[ii])(metal, age[:-overhead[ii]])
                newFchi[ii] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[ii], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

        comb_prob = cprob.astype(np.float128) / Cc * fprob.astype(np.float128) / Cf

        Cprob *= comb_prob.astype(np.float128)

    #########################
    P = np.trapz(Cprob, ultau, axis=2)
    C = np.trapz(np.trapz(P, age, axis=1), metal)
    prob = P / C
    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]

"""Stack Fit"""

def Stack_spec(spec, redshifts, wv):
    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Get_flux_nocont(spec[i], redshifts[i]))
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
    ################
    ###take out nans

    IDX = [U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]


def Stack_model(speclist, modellist, redshifts, wv_range):
    flgrid = []
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave, flux, error = np.load(speclist[i])
        if speclist[i] == '../data/spec_stacks_jan24/s40597_stack.npy':
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

        wave = wave / (1 + redshifts[i])

        #######read in corresponding model, and interpolate flux
        W, F = np.load(modellist[i])
        W = W / (1 + redshifts[i])
        iF = interp1d(W, F)(wave)

        #######scale the model
        C = Scale_model(flux, error, iF)
        mflux = C * iF

        ######divide out continuum
        m2r = [3910, 3990, 4082, 4122, 4250, 4330, 4830, 4890, 4990, 5030]
        Mask = np.zeros(len(wave))
        for ii in range(len(Mask)):
            if m2r[0] <= wave[ii] <= m2r[1]:
                Mask[ii] = 1
            if m2r[2] <= wave[ii] <= m2r[3]:
                Mask[ii] = 1
            if m2r[4] <= wave[ii] <= m2r[5]:
                Mask[ii] = 1
            if m2r[6] <= wave[ii] <= m2r[7]:
                Mask[ii] = 1
            if m2r[8] <= wave[ii] <= m2r[9]:
                Mask[ii] = 1
            if wave[ii] > m2r[9]:
                break

        maskw = np.ma.masked_array(wave, Mask)

        coeff = np.ma.polyfit(maskw, mflux, 3, w=1 / error ** 2)
        C0 = np.polyval(coeff, wave)

        Fl = mflux / C0
        Er = error / C0

        ########interpolate spectra
        flentry = np.zeros(len(wv_range))
        errentry = np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl = interp1d(wave, Fl)
        ier = interp1d(wave, Er)
        flentry[mask] = ifl(wv_range[mask])
        errentry[mask] = ier(wv_range[mask])
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
    ################

    return wv, stack


def Stack_model_in_mfit(modellist, redshifts, wave_grid, flux_grid, err_grid, wv_range):
    flgrid = []
    errgrid = []

    for i in range(len(modellist)):
        #######read in spectra
        wave, flux, error = np.array([wave_grid[i], flux_grid[i], err_grid[i]])

        #######read in corresponding model, and interpolate flux
        W, F = np.load(modellist[i])
        W = W / (1 + redshifts[i])
        iF = interp1d(W, F)(wave)

        #######scale the model
        C = Scale_model(flux, error, iF)
        mflux = C * iF

        ######divide out continuum
        m2r = [3910, 3990, 4082, 4122, 4250, 4330, 4830, 4890, 4990, 5030]
        Mask = np.zeros(len(wave))
        for ii in range(len(Mask)):
            if m2r[0] <= wave[ii] <= m2r[1]:
                Mask[ii] = 1
            if m2r[2] <= wave[ii] <= m2r[3]:
                Mask[ii] = 1
            if m2r[4] <= wave[ii] <= m2r[5]:
                Mask[ii] = 1
            if m2r[6] <= wave[ii] <= m2r[7]:
                Mask[ii] = 1
            if m2r[8] <= wave[ii] <= m2r[9]:
                Mask[ii] = 1
            if wave[ii] > m2r[9]:
                break

        maskw = np.ma.masked_array(wave, Mask)

        coeff = np.ma.polyfit(maskw, mflux, 3, w=1 / error ** 2)
        C0 = np.polyval(coeff, wave)

        Fl = mflux / C0
        Er = error / C0

        ########interpolate spectra
        flentry = np.zeros(len(wv_range))
        errentry = np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl = interp1d(wave, Fl)
        ier = interp1d(wave, Er)
        flentry[mask] = ifl(wv_range[mask])
        errentry[mask] = ier(wv_range[mask])
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
    ################

    return wv, stack


def Stack_sim_model(speclist, modellist, redshifts, wv_range):
    flgrid = []
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave, flux, error = np.array(Readfile(speclist[i], 1))
        if speclist[i] == '../data/spec_stacks_jan24/s40597_stack.dat':
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

        wave = wave / (1 + redshifts[i])

        #######read in corresponding model, and interpolate flux
        W, F = np.load(modellist[i])
        W = W / (1 + redshifts[i])
        iF = interp1d(W, F)(wave)

        #######scale the model
        C = Scale_model(flux, error, iF)
        mflux = C * iF + np.random.normal(0,error)

        ######divide out continuum
        m2r = [3910, 3990, 4082, 4122, 4250, 4330, 4830, 4890, 4990, 5030]
        Mask = np.zeros(len(wave))
        for ii in range(len(Mask)):
            if m2r[0] <= wave[ii] <= m2r[1]:
                Mask[ii] = 1
            if m2r[2] <= wave[ii] <= m2r[3]:
                Mask[ii] = 1
            if m2r[4] <= wave[ii] <= m2r[5]:
                Mask[ii] = 1
            if m2r[6] <= wave[ii] <= m2r[7]:
                Mask[ii] = 1
            if m2r[8] <= wave[ii] <= m2r[9]:
                Mask[ii] = 1
            if wave[ii] > m2r[9]:
                break

        maskw = np.ma.masked_array(wave, Mask)

        coeff = np.ma.polyfit(maskw, mflux, 3, w=1 / error ** 2)
        C0 = np.polyval(coeff, wave)

        Fl = mflux / C0
        Er = error / C0

        ########interpolate spectra
        flentry = np.zeros(len(wv_range))
        errentry = np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl = interp1d(wave, Fl)
        ier = interp1d(wave, Er)
        flentry[mask] = ifl(wv_range[mask])
        errentry[mask] = ier(wv_range[mask])
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

    return wv, stack, err


def Model_fit_stack(speclist, tau, metal, A, speczs, ids, wv_range, name, pkl_name, res=10,fsps=True):
    ##############Stack spectra################
    wv, fl, err = Stack_spec(speclist, speczs, wv_range)

    #############Prep output file###############

    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ###############Pickle spectra##################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:


        wgrid = []
        fgrid = []
        egrid = []

        for i in range(len(speclist)):
            #######read in spectra
            wave, flux, error = np.load(speclist[i])
            if speclist[i] == '../spec_stacks_jan24/s40597_stack.npy':
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

            wave = wave / (1 + speczs[i])
            wgrid.append(wave)
            fgrid.append(flux)
            egrid.append(error)


        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    if fsps == True:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs)
                    else:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs,fsps=False)
                    mw, mf = Stack_model_in_mfit(mlist, speczs, wgrid, fgrid, egrid,
                                                 np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################
    outspec = open(pklname, 'rb')

    mf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(cPickle.load(outspec))

    mf = np.array(mf)

    outspec.close()

    chigrid = np.sum(((fl - mf) / err) ** 2, axis=1).reshape([len(metal), len(A), len(tau)])

    ###############
    for i in range(len(metal)):
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return


def Model_fit_stack_features(speclist, tau, metal, A, speczs,ids, wv_range, name, pkl_name, res=10, fsps=True):
    ##############Stack spectra################
    wv, fl, er = Stack_spec(speclist, speczs, wv_range)
    IDM=[]
    for i in range(len(wv)):
        if 3800<=wv[i]<=3850 or 3910<=wv[i]<=4030 or 4080<=wv[i]<=4125 or 4250<=wv[i]<=4385 or 4515<=wv[i]<=4570 or 4810<=wv[i]<=4910 or 4975<=wv[i]<=5055 or 5110<=wv[i]<=5285:
            IDM.append(i)


    #############Prep output file###############

    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ###############Pickle spectra##################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:


        wgrid = []
        fgrid = []
        egrid = []

        for i in range(len(speclist)):
            #######read in spectra
            wave, flux, error = np.load(speclist[i])
            if speclist[i] == '../spec_stacks_jan24/s40597_stack.npy':
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

            wave = wave / (1 + speczs[i])
            wgrid.append(wave)
            fgrid.append(flux)
            egrid.append(error)


        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    if fsps == True:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs)
                    else:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs,fsps=False)
                    mw, mf = Stack_model_in_mfit(mlist, speczs, wgrid, fgrid, egrid,
                                                 np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    mf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(cPickle.load(outspec)[IDM])

    mf = np.array(mf)

    outspec.close()

    chigrid = np.sum(((fl[IDM] - mf) / er[IDM]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)])

    ###############
    for i in range(len(metal)):
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return


def Model_fit_stack_MCerr_bestfit(speclist, tau, metal, A, speczs, wv_range, name, pkl_name, repeats=100, fsps=True):
    #############Get redshift info###############

    zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]

    for i in range(len(speczs)):
        zlist.append(int(speczs[i] * 100) / 5 / 20.)
    for i in range(len(bins)):
        b = []
        for ii in range(len(zlist)):
            if bins[i] == zlist[ii]:
                b.append(ii)
        if len(b) > 0:
            zcount.append(len(b))
    zbin = sorted(set(zlist))

    #############Get list of models to fit againts##############

    if fsps == False:

        ml = [22, 32, 42, 52, 62, 72]
        mv = np.array([.0001, .0004, .004, .008, .02, .05])
        metalval = np.zeros(len(metal))
        for i in range(len(metal)):
            for ii in range(len(ml)):
                if metal[i] == ml[ii]:
                    metalval[i] = mv[ii]

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zbin)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metalval[i], A[ii], tau[iii], zbin[iv]))
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
                    for iv in range(len(zbin)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zbin[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ##############Pickle spectra#################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model(modellist[i][ii][iii], zbin, zcount, wv_range)
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Stack spectra################

    wv, flx, err = Stack_spec(speclist, speczs, wv_range)

    ##############Start loop and add error#############

    mlist = []
    alist = []

    for i in range(repeats):

        outspec = open(pklname, 'rb')

        fl = flx + np.random.normal(0, err)

        ##############Create chigrid#################

        chigrid = np.zeros([len(metal), len(A), len(tau)])
        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mf = np.array(cPickle.load(outspec))
                    chigrid[i][ii][iii] = Identify_stack(fl, err, mf)

        chigrid = np.array(chigrid, dtype=np.float128)

        ################Find best fit##################

        prob = np.exp(-chigrid / 2)

        tau = np.array(tau)
        chigr = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(prob[i][ii], np.power(10, tau - 9)))
            chigr.append(acomp)
        prob = np.array(chigr)

        [idmax] = np.argwhere(prob == np.max(prob))
        alist.append(A[idmax[1]])
        mlist.append(metal[idmax[0]])

        outspec.close()

    fn = name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


def Model_fit_sim_stack_MCerr_bestfit(speclist, tau, metal, A, sim_m, sim_a, sim_t, speczs, ids,
                                      wv_range, name, pkl_name, repeats=100):

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    ##############Start loop and add error#############

    mlist = []
    alist = []
    rmlist = Make_model_list(ids, sim_m, sim_a, sim_t, speczs)

    outspec = open(pklname, 'rb')

    mf=[]
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(np.array(cPickle.load(outspec)))

    mf=np.array(mf)

    outspec.close()
    scale = Readfile('../data/tau_scale_nage.dat', 1)

    for xx in range(repeats):
        ##############Stack spectra################
        wv, fl, err = Stack_sim_model(speclist,rmlist,speczs,wv_range)

        ##############Create chigrid#################
        chigrid = np.sum(((fl - mf) / err) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128)
        chi = np.transpose(chigrid)
        ################Find best fit##################

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        newchi = []
        for i in range(len(chi)):
            if i == 0:
                iframe = chi[i]
            else:
                iframe = interp2d(metal, scale[i], chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newchi.append(iframe)
        newchi = np.transpose(newchi)

        prob = np.exp(-newchi / 2)

        tau = np.array(tau)
        chigr = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(prob[i][ii], np.power(10, tau - 9)))
            chigr.append(acomp)
        prob = np.array(chigr)

        [idmax] = np.argwhere(prob == np.max(prob))
        alist.append(A[idmax[1]])
        mlist.append(metal[idmax[0]])

    fn = '../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


"""Grism Fit"""


def Make_model_list(galaxy, Metal, Age, Tau, Rshift, fsps=True):
    flist = []

    if fsps == True:
        for i in range(len(Rshift)):
            flist.append('../../../fsps_models_for_fit/galaxy_models/m%s_a%s_t%s_z%s_%s_model.npy' % (Metal, Age, Tau, Rshift[i], galaxy[i]))

    else:
        for i in range(len(Rshift)):
            flist.append('../../../bc03_models_for_fit/galaxy_models/m%s_a%s_t%s_z%s_%s_model.npy' % (Metal, Age, Tau, Rshift[i], galaxy[i]))

    return flist


def Norm_P(tau, metal, age, chi):
    ####### transpose chi cube for integration of tau
    ####### heirarchy is age_-> metallicity -> tau
    chi = chi.T

    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    prob = []
    for i in range(len(age)):
        preprob1 = []
        for ii in range(len(metal)):
            preprob2 = []
            for iii in range(len(tau)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii] / 2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all tau
    ######## End up with age vs metallicity matricies
    ######## use unlogged tau
    ultau = np.append(0, np.power(10, tau[1:] - 9))
    A = []
    for i in range(len(age)):
        M = []
        for ii in range(len(metal)):
            T = []
            for iii in range(len(tau) - 1):
                T.append(sp.N((ultau[iii + 1] - ultau[iii]) * (prob[i][ii][iii] + prob[i][ii][iii + 1]) / 2))
            M.append(sp.mpmath.fsum(T))
        A.append(M)

    ######## Integrate over metallicity to get age prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(age)):
        preC2 = []
        for ii in range(len(metal) - 1):
            preC2.append(sp.N((metal[ii + 1] - metal[ii]) * (A[i][ii] + A[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(age) - 1):
        preC3.append(sp.N((age[i + 1] - age[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    ######## Create normal prob grid
    P = []
    for i in range(len(age)):
        preP = []
        for ii in range(len(metal)):
            preP.append(A[i][ii] / C)
        P.append(np.array(preP).astype(np.float128))

    return P


def Identify_grism(galimg, galerr, modelimg):
    C = Scale_model(galimg, galerr, modelimg)
    x = ((galimg - C * modelimg) / galerr) ** 2
    return np.sum(x)


def Model_fit_grism(grismfile, tau, metal, age, name):
    #############Open 2d fits file##############
    grismdata = fits.open(grismfile)
    grismflx = grismdata['DATA'].data
    grismerr = grismdata['ERR'].data
    # grismflx[grismflx < 0] = 0

    #############Prep output file###############
    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ##############Create chigrid and add to file#################
    chigrid = np.zeros([len(tau), len(metal), len(age)])
    for i in range(len(tau)):
        for ii in range(len(metal)):
            for iii in range(len(age)):
                m_grism = grismdata[i * (len(metal) * len(age)) + ii * (len(age)) + iii + 4].data
                chigrid[i][ii][iii] = Identify_grism(grismflx, grismerr, m_grism)
        inputgrid = np.array(chigrid[i])
        spc = 'tau_%s' % tau[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############
    hdulist.writeto(chifile)

    return


def Analyze_grism(chifits, tau, metal, age):
    ####### Read in file
    dat = fits.open(chifits)
    chi = []
    for i in range(len(tau)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    ####### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P(tau, metal, age, chi)).astype(np.float128)

    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[0]], metal[idmax[1]])

    return prob, age[idmax[0]], metal[idmax[1]]


"""Cluster"""


def Divide_cont(wave, flux, error, z):
    if z == 1.35:
        IDx = [U for U in range(len(wave)) if 8000 < wave[U] < 11500]
    else:
        IDx = [U for U in range(len(wave)) if 7500 < wave[U] < 11500]

    wv = wave[IDx]
    fl = flux[IDx]
    er = error[IDx]

    wi = np.array(wv)
    w = wv / (1 + z)

    m2r = [3800, 3850, 3910, 4030, 4080, 4125, 4250, 4385, 4515, 4570, 4810, 4910, 4975, 5055, 5110, 5285]

    Mask = np.zeros(len(w))
    for i in range(len(Mask)):
        if m2r[0] <= w[i] <= m2r[1]:
            Mask[i] = 1
        if m2r[2] <= w[i] <= m2r[3]:
            Mask[i] = 1
        if m2r[4] <= w[i] <= m2r[5]:
            Mask[i] = 1
        if m2r[6] <= w[i] <= m2r[7]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[10] < w[i] <= m2r[11]:
            Mask[i] = 1
        if m2r[12] <= w[i] <= m2r[13]:
            Mask[i] = 1
        if m2r[14] <= w[i] <= m2r[15]:
            Mask[i] = 1

    maskw = np.ma.masked_array(w, Mask)

    params = np.ma.polyfit(maskw, fl, 3)
    C0 = np.polyval(params, w)

    flx = fl / C0
    err = er / C0
    if z == 1.35:
        IDr = [U for U in range(len(wi)) if 8000 < wi[U] < 11100]
    else:
        IDr = [U for U in range(len(wi)) if 7900 < wi[U] < 11100]
    return w[IDr], flx[IDr], err[IDr]


def Divide_cont_model(wave,flux, z):
    IDx = [U for U in range(len(wave)) if 7500 < wave[U] < 11500]

    wv = wave[IDx]
    fl = flux[IDx]

    wi = np.array(wv)
    w = wv / (1 + z)

    m2r = [3800, 3850, 3910, 4030, 4080, 4125, 4250, 4385, 4515, 4570, 4810, 4910, 4975, 5055, 5110, 5285]

    Mask = np.zeros(len(w))
    for i in range(len(Mask)):
        if m2r[0] <= w[i] <= m2r[1]:
            Mask[i] = 1
        if m2r[2] <= w[i] <= m2r[3]:
            Mask[i] = 1
        if m2r[4] <= w[i] <= m2r[5]:
            Mask[i] = 1
        if m2r[6] <= w[i] <= m2r[7]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[8] <= w[i] <= m2r[9]:
            Mask[i] = 1
        if m2r[10] < w[i] <= m2r[11]:
            Mask[i] = 1
        if m2r[12] <= w[i] <= m2r[13]:
            Mask[i] = 1
        if m2r[14] <= w[i] <= m2r[15]:
            Mask[i] = 1

    maskw = np.ma.masked_array(w, Mask)

    params = np.ma.polyfit(maskw, fl, 3)
    C0 = np.polyval(params, w)

    flx = fl / C0

    if z == 1.35:
        IDr = [U for U in range(len(wi)) if 8000 < wi[U] < 11100]
    else:
        IDr = [U for U in range(len(wi)) if 7900 < wi[U] < 11100]
    return w[IDr], flx[IDr]


def Cluster_fit(spec, metal, age, tau, rshift, name):
    #############Define cluster#################
    cluster = Cluster(spec,rshift)
    cluster.Remove_continuum()

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1='../chidat/%s_chidata.fits' % name
    prihdr1 = fits.Header()
    prihdu1 = fits.PrimaryHDU(header=prihdr1)
    hdulist1 = fits.HDUList(prihdu1)


    ##############Create chigrid and add to file#################
    chigrid1=np.zeros([len(metal),len(age),len(tau)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                cmodel = Cluster_model(metal[i], age[ii], tau[iii], rshift, cluster.wv, cluster.fl, cluster.er)
                cmodel.Remove_continuum()
                chigrid1[i][ii][iii] = Identify_stack(cluster.nc_fl, cluster.nc_er, cmodel.nc_fl)
        inputgrid1 = np.array(chigrid1[i])
        spc1 = 'metal_%s' % metal[i]
        mchi1 = fits.ImageHDU(data=inputgrid1, name=spc1)
        hdulist1.append(mchi1)


    ################Write chigrid file###############
    hdulist1.writeto(chifile1)

    print 'Done!'
    return


class Cluster(object):

    def __init__(self,cluster_spec,redshift):
        wv, fl, er = np.load(cluster_spec)
        self.wv = wv
        self.fl = fl
        self.er = er
        self.contour = [0]
        self.redshift = redshift

    def Remove_continuum(self):
        nc_wv, nc_fl, nc_er = Divide_cont(self.wv,self.fl,self.er,self.redshift)
        self.nc_wv = nc_wv
        self.nc_fl = nc_fl
        self.nc_er = nc_er

    def Analyze_fit(self, chigrid, metal, age, tau, cut_tau = False,tau_new=[] , age_conv='../data/tau_scale_cluster.dat'):
        self.metal = metal
        self.age = age
        if cut_tau == False:
            self.tau = tau
        else:
            self.tau = tau_new

        ultau = np.append(0, np.power(10, np.array(self.tau)[1:] - 9))

        ####### Read in file
        dat = fits.open(chigrid)

        chi = np.zeros([len(self.metal), len(self.age), len(tau)])
        for i in range(len(self.metal)):
            chi[i] = dat[i + 1].data

        if len(tau) == 1:
            chi = chi.reshape([len(self.metal), len(self.age)])

        self.chi = chi

        ####### Get scaling factor for tau reshaping
        convtau = np.array([0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23,
                   9.26, 9.28, 9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48])
        convage = np.arange(.5,14.1,.1)

        mt = [U for U in range(len(convtau)) if convtau[U] in self.tau]
        ma = [U for U in range(len(convage)) if np.round(convage[U],1) in np.round(self.age,1)]

        if cut_tau == True:
            self.chi = self.chi[:,:,mt[0]:mt[-1]+1]

        if len(self.tau) == 1:
            self.chi = self.chi.reshape([len(self.metal), len(self.age)])

        convtable = Readfile(age_conv)
        scale = convtable[mt[0]:mt[-1]+1,ma[0]:ma[-1]+1]

        overhead = np.zeros(len(scale)).astype(int)
        for i in range(len(scale)):
            amt = []
            for ii in range(len(self.age)):
                if self.age[ii] > scale[i][-1]:
                    amt.append(1)
            overhead[i] = sum(amt)

        ######## Reshape likelihood to get average age instead of age when marginalized
        newchi = np.zeros(self.chi.T.shape)

        for i in range(len(scale)):
            if i == 0 and len(self.tau) == 1:
                newchi = self.chi.T
            if i == 0 and len(self.tau) > 1:
                newchi[i] = self.chi.T[i]
            if i > 0 and len(self.tau) > 1:
                if max(scale[i]) >= min(self.age):
                    frame = interp2d(self.metal, scale[i], self.chi.T[i])(self.metal, self.age[:-overhead[i]])
                if len(frame) == len(metal):
                    newchi[i] = np.repeat([np.repeat(1E5, len(self.metal))], len(self.age), axis=0)
                else:
                    newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(self.metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        prob = np.exp(-newchi.T.astype(np.float128) / 2)

        if len(self.tau) == 1:
            TP = prob
        else:
            TP = np.trapz(prob, ultau, axis=2)

        print TP.shape
        print self.age.shape

        AP = np.trapz(TP.T, self.metal)
        MP = np.trapz(TP, self.age)
        C = np.trapz(AP, self.age)

        self.prob = TP.T / C
        self.AP = AP/C
        self.MP = MP/C

        # ####### get best fit values
        print np.argwhere(self.prob == np.max(self.prob))
        [idmax] = np.argwhere(self.prob == np.max(self.prob))

        self.bfage = self.age[idmax[0]]
        self.bfmetal = self.metal[idmax[1]]

        print 'Best fit model is %s Gyr and %s Z' % (self.bfage, self.bfmetal)

    def Get_contours(self):
        onesig, twosig = Likelihood_contours(self.age, self.metal, self.prob)

        self.contour = np.array([twosig,onesig])

    def Plot_2D_likelihood(self,save_plot=False,plot_name=''):
        M, A = np.meshgrid(self.metal, self.age)

        self.Get_contours()

        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],wspace=0.0, hspace=0.0)

        plt.figure(figsize=[8, 8])
        plt.subplot(gs[1, 0])
        plt.contour(M, A, self.prob, self.contour, colors='k', linewidths=2)
        plt.contourf(M, A, self.prob, 40, cmap=colmap)
        plt.xticks([0, .005, .01, .015, .02, .025, .03],
                   np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.02, 2))
        plt.plot(self.bfmetal, self.bfage, 'cp',
                 label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (self.bfage, np.round(self.bfmetal / 0.019, 2)))
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.minorticks_on()
        plt.xlabel('Z/Z$_\odot$', size=20)
        plt.ylabel('Average Age (Gyrs)', size=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.legend()

        plt.subplot(gs[1, 1])
        plt.plot(self.AP, self.age)
        plt.ylim(min(self.age),max(self.age))
        plt.yticks([])
        plt.xticks([])

        plt.subplot(gs[0, 0])
        plt.plot(self.metal / 0.019, self.MP)
        plt.xlim(min(self.metal) / 0.019, max(self.metal) / 0.019)
        plt.yticks([])
        plt.xticks([])
        plt.gcf().subplots_adjust(bottom=0.165, left=0.12)
        if save_plot == True:
            plt.savefig(plot_name)
        else:
            plt.show()
        plt.close()

    def Best_fit_spec(self):

        mspec = Cluster_model(self.bfmetal,self.bfage, 0, self.redshift ,
                              self.nc_wv*(1 + self.redshift),self.nc_fl,self.nc_er)
        mspec.Remove_continuum()

        self.mwv = mspec.wv
        self.mfl = mspec.fl

        self.nc_mwv = mspec.nc_wv
        self.nc_mfl = mspec.nc_fl


class Cluster_model(object):
    def __init__(self, metal, age, tau, redshift, cluster_wv, cluster_fl, cluster_er):
        self.metal = metal
        self.age = age
        self.tau = tau
        self.redshift = redshift
        mwv, mfl = np.load('../../../fsps_models_for_fit/cluster_models/m%s_a%s_t%s_z%s_clust_model.npy' %
                           (self.metal, self.age, self.tau, self.redshift))
        self.mwv = mwv
        self.mfl = mfl
        imfl = interp1d(self.mwv,self.mfl)(cluster_wv)
        C=Scale_model(cluster_fl, cluster_er, imfl)
        self.fl = imfl*C
        self.wv=cluster_wv

    def Remove_continuum(self):
        nc_wv, nc_fl = Divide_cont_model( self.mwv, self.mfl, self.redshift)
        self.nc_wv = nc_wv
        self.nc_fl = nc_fl


"""Single Gal fit"""

def Single_gal_fit_full(spec, tau, metal, A, specz, galaxy,name, fsps=True):
    #############Read in spectra#################
    wv, fl, err = np.load(spec)
    wv, fl, err = np.array([wv[wv <= 11100], fl[wv <= 11100], err[wv <= 11100]])
    if galaxy == 'n21156' or galaxy == 's39170' or galaxy == 'n34694' or galaxy == 's45792':
        IDer = []
        for ii in range(len(wv)):
            if 4855 * (1 + specz) <= wv[ii] <= 4880 * (1 + specz):
                IDer.append(ii)
        err[IDer] = 1E8
        fl[IDer] = 0

    wv /= (1 + specz)

    IDF = []
    for i in range(len(wv)):
        if 3800 <= wv[i] <= 3850 or 3910 <= wv[i] <= 4030 or 4080 <= wv[i] <= 4125 or 4250 <= wv[i] <= 4385 or 4515 <= \
                wv[i] <= 4570 or 4810 <= wv[i] <= 4910 or 4975 <= wv[i] <= 5055 or 5110 <= wv[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(wv)):
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 <= \
                wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110 or 5285 <= wv[i] <= wv[-1]:
            IDC.append(i)

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1='../chidat/%s_chidata.fits' % name
    prihdr1 = fits.Header()
    prihdu1 = fits.PrimaryHDU(header=prihdr1)
    hdulist1 = fits.HDUList(prihdu1)

    chifile2='../chidat/%s_cont_chidata.fits' % name
    prihdr2 = fits.Header()
    prihdu2 = fits.PrimaryHDU(header=prihdr2)
    hdulist2 = fits.HDUList(prihdu2)

    chifile3='../chidat/%s_feat_chidata.fits' % name
    prihdr3 = fits.Header()
    prihdu3 = fits.PrimaryHDU(header=prihdr3)
    hdulist3 = fits.HDUList(prihdu3)

    ##############Create chigrid and add to file#################
    chigrid1=np.zeros([len(metal),len(A),len(tau)])
    chigrid2=np.zeros([len(metal),len(A),len(tau)])
    chigrid3=np.zeros([len(metal),len(A),len(tau)])

    if fsps==False:

        filepath = '../../../bc03_models_for_fit/galaxy_models/'

    else:
        filepath = '../../../fsps_models_for_fit/galaxy_models/'

    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv,mf = np.load(filepath + 'm%s_a%s_t%s_z%s_%s_model.npy' % (metal[i], A[ii], tau[iii], specz, galaxy))
                imf = interp1d(mwv/(1 + specz), mf)(wv)
                C = Scale_model(fl, err, imf)
                chigrid1[i][ii][iii] = Identify_stack(fl, err, imf * C)
                chigrid2[i][ii][iii] = Identify_stack(fl[IDC], err[IDC], imf[IDC] * C)
                chigrid3[i][ii][iii] = Identify_stack(fl[IDF], err[IDF], imf[IDF] * C)
        inputgrid1 = np.array(chigrid1[i])
        spc1 = 'metal_%s' % metal[i]
        mchi1 = fits.ImageHDU(data=inputgrid1, name=spc1)
        hdulist1.append(mchi1)

        inputgrid2 = np.array(chigrid2[i])
        spc2 = 'metal_%s' % metal[i]
        mchi2 = fits.ImageHDU(data=inputgrid2, name=spc2)
        hdulist2.append(mchi2)

        inputgrid3 = np.array(chigrid3[i])
        spc3 = 'metal_%s' % metal[i]
        mchi3 = fits.ImageHDU(data=inputgrid3, name=spc3)
        hdulist3.append(mchi3)

    ################Write chigrid file###############

    hdulist1.writeto(chifile1)
    hdulist2.writeto(chifile2)
    hdulist3.writeto(chifile3)
    print 'Done!'
    return


def Single_gal_fit(spec, tau, metal, A, specz, galaxy,name, fsps=True):
    #############Read in spectra#################
    wv, fl, err = np.load(spec)
    wv, fl, err = np.array([wv[7950 <= wv[wv <= 11000]], fl[7950 <= wv[wv <= 11000]], err[7950 <= wv[wv <= 11000]]])

    #############Prep output file###############
    chifile='../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ##############Create chigrid and add to file#################
    chigrid=np.zeros([len(metal),len(A),len(tau)])

    if fsps==False:

        filepath = '../../../bc03_models_for_fit/galaxy_models/'

    else:
        filepath = '../../../fsps_models_for_fit/galaxy_models/'

    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv,mf = np.load(filepath + 'm%s_a%s_t%s_z%s_%s_model.npy' % (metal[i], A[ii], tau[iii], specz, galaxy))
                imf = interp1d(mwv, mf)(wv)
                C = Scale_model(fl, err, imf)
                chigrid[i][ii][iii] = Identify_stack(fl, err, imf * C)
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    print 'Done!'
    return


def Single_gal_fit_feat(spec, tau, metal, A, specz, galaxy, name, fsps=True):
    #############Read in spectra#################
    wv, fl, err = np.load(spec)
    wv, fl, err = np.array([wv[7950 <= wv[wv <= 11000]], fl[7950 <= wv[wv <= 11000]], err[7950 <= wv[wv <= 11000]]])

    wv /= (1 + specz)

    IDM = []
    for i in range(len(wv)):
        if 3800 <= wv[i] <= 3850 or 3910 <= wv[i] <= 4030 or 4080 <= wv[i] <= 4125 or 4250 <= wv[i] <= 4385 or 4515 <= \
                wv[i] <= 4570 or 4810 <= wv[i] <= 4910 or 4975 <= wv[i] <= 5055 or 5110 <= wv[i] <= 5285:
            IDM.append(i)
    #############Prep output file###############
    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ##############Create chigrid and add to file#################
    chigrid = np.zeros([len(metal), len(A), len(tau)])

    if fsps == False:

        filepath = '../../../bc03_models_for_fit/galaxy_models/'

    else:
        filepath = '../../../fsps_models_for_fit/galaxy_models/'

    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv, mf = np.load(
                    filepath + 'm%s_a%s_t%s_z%s_%s_model.npy' % (metal[i], A[ii], tau[iii], specz, galaxy))
                imf = interp1d(mwv/(1 + specz), mf)(wv)
                C = Scale_model(fl, err, imf)
                chigrid[i][ii][iii] = Identify_stack(fl[IDM], err[IDM], imf[IDM] * C)
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    print 'Done!'
    return


def Single_gal_fit_cont(spec, tau, metal, A, specz, galaxy, name, fsps=True):
    #############Read in spectra#################
    wv, fl, err = np.load(spec)
    wv, fl, err = np.array([wv[7950 <= wv[wv <= 11000]], fl[7950 <= wv[wv <= 11000]], err[7950 <= wv[wv <= 11000]]])

    wv /= (1 + specz)

    IDM=[]
    for i in range(len(wv)):
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 <= \
                wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110:
            IDM.append(i)
    #############Prep output file###############
    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ##############Create chigrid and add to file#################
    chigrid = np.zeros([len(metal), len(A), len(tau)])

    if fsps == False:

        filepath = '../../../bc03_models_for_fit/galaxy_models/'

    else:
        filepath = '../../../fsps_models_for_fit/galaxy_models/'

    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv, mf = np.load(filepath + 'm%s_a%s_t%s_z%s_%s_model.npy' % (metal[i], A[ii], tau[iii], specz, galaxy))
                imf = interp1d(mwv/(1 + specz), mf)(wv)
                C = Scale_model(fl, err, imf)
                chigrid[i][ii][iii] = Identify_stack(fl[IDM], err[IDM], imf[IDM] * C)
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    print 'Done!'
    return


def Specz_fit(galaxy, spec, rshift, name):
    #############Parameters#####################
    metal = np.array([0.0031, 0.0061, 0.0085, 0.012, 0.015, 0.019, 0.024])
    A = [0.5,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]

    #############Read in spectra#################
    wv,fl,err=np.array(Readfile(spec,1))
    IDW=[]
    for i in range(len(wv)):
        if 7950<wv[i]<11000:
            IDW.append(i)
    wv, fl, err = np.array([wv[IDW], fl[IDW], err[IDW]])

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
                mwv,mf= np.load(modellist[i][ii][iii])
                imf=interp1d(mwv,mf)(wv)
                C=Scale_model(fl,err,imf)
                chigrid[i][ii][iii]=Identify_stack(fl,err,imf*C)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)
    ################Write chigrid file###############

    hdulist.writeto(chifile)

    Analyze_specz(chifile,rshift,name)

    print 'Done!'

    return


def Specz_fit_feat(galaxy, spec, rshift, name):
    #############Parameters#####################
    metal = np.array([0.0031, 0.0061, 0.0085, 0.012, 0.015, 0.019, 0.024])
    A = [0.5,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]

    #############Read in spectra#################
    wv,fl,err=np.array(Readfile(spec,1))
    IDW=[]
    for i in range(len(wv)):
        if 7950<wv[i]<11300:
            IDW.append(i)
    wv, fl, err = np.array([wv[IDW], fl[IDW], err[IDW]])

    #############Prep output file###############
    chifile='../rshift_dat/%s_chidata.fits' % name
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
                mwv,mf= np.load(modellist[i][ii][iii])
                ww=wv/(1+rshift[iii])
                IDF=[]
                for iv in range(len(ww)):
                    if 3910 <= ww[iv] <= 3980 or 4082 <= ww[iv] <= 4122 or 4250 <= ww[iv] <=4400 or 4830 <= ww[iv] <=4930 or 5109 <= ww[iv] <=5250:
                        IDF.append(iv)
                imf=interp1d(mwv,mf)(wv[IDF])
                C=Scale_model(fl[IDF],err[IDF],imf)
                chigrid[i][ii][iii]=Identify_stack(fl[IDF],err[IDF],imf*C)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)
    ################Write chigrid file###############

    hdulist.writeto(chifile)

    Analyze_specz(chifile,rshift,name)

    print 'Done!'

    return


def Norm_P_specz(rshift, metal, age, chi):
    ####### Heirarchy is rshift_-> age -> metal
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    prob = []
    for i in range(len(rshift)):
        preprob1 = []
        for ii in range(len(age)):
            preprob2 = []
            for iii in range(len(metal)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii] / 2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all metal
    ######## End up with age vs rshift matricies
    R = []
    for i in range(len(rshift)):
        A = []
        for ii in range(len(age)):
            M = []
            for iii in range(len(metal) - 1):
                M.append(sp.N((metal[iii + 1] - metal[iii]) * (prob[i][ii][iii] + prob[i][ii][iii + 1]) / 2))
            A.append(sp.mpmath.fsum(M))
        R.append(A)

    ######## Integrate over age to get rshift prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(rshift)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (R[i][ii] + R[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(rshift) - 1):
        preC3.append(sp.N((rshift[i + 1] - rshift[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    ######## Create normal prob grid
    P = []
    for i in range(len(rshift)):
        P.append(preC1[i] / C)

    return np.array(P).astype(np.float128)


def Analyze_specz(chifits, rshift,metal,age, name):
    ####### Read in file
    dat = fits.open(chifits)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    ####### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P_specz(rshift, metal, age, chi.T)).astype(np.float128)

    ####### get best fit values
    print 'Best fit specz is %s' % rshift[np.argmax(prob)]

    rshiftdat=Table([rshift,prob],names=['rshifts','Pz'])
    ascii.write(rshiftdat,'../rshift_dat/%s_Pofz.dat' % name)

    return


def Highest_likelihood_model_galaxy(galaxy,rshift, bfmetal, bfage, tau):
    wv,fl,er=np.load('../spec_stacks_jan24/%s_stack.npy' % galaxy)
    fp='../../../fsps_models_for_fit/galaxy_models/'

    chi=[]
    for i in range(len(tau)):
        mwv,mfl=np.load(fp + 'm%s_a%s_t%s_z%s_%s_model.npy' % (bfmetal,bfage,tau[i],rshift,galaxy))
        imfl=interp1d(mwv,mfl)(wv)
        C=Scale_model(fl,er,imfl)
        chi.append(Identify_stack(fl,er,C*imfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def Analyze_Stack_avgage_single_gal_combine(cont_chifits,feat_chifits, speclist, rshift, tau, metal, age,
                                            stack_scale=True,age_conv='../data/tau_scale_ntau.dat'):
    reg = np.arange(4000, 4210, 1)
    regint = np.zeros(len(speclist))

    if stack_scale == True:
        for i in range(len(speclist)):
            wv, fl, er = np.load(speclist[i])
            ifl = interp1d(wv / (1 + rshift[i]), fl)
            regint[i] = np.trapz(ifl(reg), reg)

        scale_fact = min(regint) / regint

    else:
        scale_fact = np.ones(len(speclist))

    Cprob = np.ones([len(metal), len(age), len(tau)])
    scale = Readfile(age_conv, 1)
    overhead = np.zeros(len(scale))
    ultau = np.append(0, np.power(10, tau[1:] - 9))

    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for i in range(len(cont_chifits)):
        ####### Read in file
        Cdat = fits.open(cont_chifits[i])
        Cchi = np.zeros([len(metal), len(age), len(tau)])

        Fdat = fits.open(feat_chifits[i])
        Fchi = np.zeros([len(metal), len(age), len(tau)])

        for ii in range(len(metal)):
            Fchi[ii] = Fdat[ii + 1].data
            Cchi[ii] = Cdat[ii + 1].data

        Fchi = Fchi.T
        Cchi = Cchi.T

        max_age = Oldest_galaxy(rshift[i])

        Fchi[:, len(age[age <= max_age]):, :] = 1E5
        Cchi[:, len(age[age <= max_age]):, :] = 1E5

        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for ii in range(len(Cchi)):
            if ii == 0:
                newCchi[ii] = Cchi[ii]
                newFchi[ii] = Fchi[ii]
            else:
                cframe = interp2d(metal, scale[ii], Cchi[ii])(metal, age[:-overhead[ii]])
                newCchi[ii] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[ii], axis=0), axis=0)

                fframe = interp2d(metal, scale[ii], Fchi[ii])(metal, age[:-overhead[ii]])
                newFchi[ii] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[ii], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

        comb_prob = cprob.astype(np.float128) / Cc * fprob.astype(np.float128) / Cf

        Cprob *= (comb_prob ** scale_fact[i]).astype(np.float128)

    #########################
    P = np.trapz(Cprob, ultau, axis=2)
    C = np.trapz(np.trapz(P, age, axis=1), metal)
    prob = P / C
    ####### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]


def Single_gal_fit_MCerr_bestfit_normwmean_cont_feat(spec, tau, metal, A, sim_m, sim_a, sim_t, specz, gal_id,
                                                name, f_weight=1, repeats=100,tau_scale='../data/tau_scale_ntau.dat'):

    mlist = []
    alist = []

    wv, flux, err = np.load(spec)
    wv,flux,err=np.array([wv[wv<11100],flux[wv<11100],err[wv<11100]])
    mwv, mfl = np.load('../../../fsps_models_for_fit/galaxy_models/m%s_a%s_t%s_z%s_%s_model.npy'
                       % (sim_m,sim_a,sim_t,specz,gal_id))

    imfl = interp1d(mwv, mfl)(wv)
    C = Scale_model(flux, err, imfl)

    FL=imfl*C
    wv/=(1+specz)

    ###############Get indicies
    IDf = []
    IDc = []
    for i in range(len(wv)):
        if 3800 <= wv[i] <= 3850 or 3910 <= wv[i] <= 4030 or 4080 <= wv[i] <= 4125 or 4250 <= wv[i] <= 4385 or 4515 \
                <= wv[i] <= 4570 or 4810 <= wv[i] <= 4910 or 4975 <= wv[i] <= 5055 or 5110 <= wv[i] <= 5285:
            IDf.append(i)
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 \
                <= wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110:
            IDc.append(i)

    ###############Get model list

    fmf=[]
    cmf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                W,MF = np.load('../../../fsps_models_for_fit/galaxy_models/m%s_a%s_t%s_z%s_%s_model.npy'
                       % (metal[i],A[ii],tau[iii],specz,gal_id))
                iMF = interp1d(W/(1+specz), MF)(wv)
                Cm = Scale_model(FL, err, iMF)
                fmf.append(Cm*iMF[IDf])
                cmf.append(Cm*iMF[IDc])

    fmf=np.array(fmf)
    cmf = np.array(cmf)

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))

    scale = Readfile(tau_scale)
    overhead = []
    for i in range(len(scale)):
        amt = []
        for ii in range(len(A)):
            if A[ii] > scale[i][-1]:
                amt.append(1)
        overhead.append(sum(amt))

    for xx in range(repeats):
        fl = FL + np.random.normal(0, err)

        Fchi = np.sum(((fl[IDf] - fmf) / err[IDf]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128).T
        Cchi = np.sum(((fl[IDc] - cmf) / err[IDc]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128).T

        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                cframe = interp2d(metal, scale[i], Cchi[i])(metal, A[:-overhead[i]])
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, A[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, A, axis=1), metal)

        Cprob = cprob / Cc

        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, A, axis=1), metal)

        Fprob = fprob / Cf

        prob = Cprob*(Fprob**f_weight)

        P0 = np.trapz(prob, ultau, axis=2).T
        C0 = np.trapz(np.trapz(P0, metal, axis=1), A)
        prob = P0/C0

        idmax = np.argwhere(prob == np.max(prob))

        for i in range(len(idmax)):
            alist.append(A[idmax[i][0]])
            mlist.append(metal[idmax[i][1]])

        if repeats <= len(alist):
            break

    fn ='../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return

def Analyze_Stack_avgage_cont_feat_gal_age_correct(contfits, featfits, specz,
                                                   tau, metal, age, age_conv='../data/tau_scale_ntau.dat'):
    ####### Get maximum age
    max_age=Oldest_galaxy(specz)

    ####### Read in file
    Cdat = fits.open(contfits)
    Cchi = np.zeros([len(metal), len(age), len(tau)])

    Fdat = fits.open(featfits)
    Fchi = np.zeros([len(metal), len(age), len(tau)])

    for i in range(len(metal)):
        Fchi[i] = Fdat[i + 1].data
        Cchi[i] = Cdat[i + 1].data

    Fchi = Fchi.T
    Cchi = Cchi.T

    Fchi[:, len(age[age <= max_age]):, :] = 1E5
    Cchi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    scale = Readfile(age_conv)

    overhead = np.zeros(len(scale))
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newCchi = np.zeros(Cchi.shape)
    newFchi = np.zeros(Fchi.shape)

    for i in range(len(Cchi)):
        if i == 0:
            newCchi[i] = Cchi[i]
            newFchi[i] = Fchi[i]
        else:
            cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
            newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

            fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
            newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau

    ultau = np.append(0, np.power(10, tau[1:] - 9))

    cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

    Pc = np.trapz(cprob, ultau, axis=2)
    Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

    fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

    Pf = np.trapz(fprob, ultau, axis=2)
    Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

    comb_prob = cprob / Cc * fprob / Cf

    prob = np.trapz(comb_prob, ultau, axis=2)
    C0 = np.trapz(np.trapz(prob, age, axis=1), metal)
    prob /= C0

    ##### get best fit values
    [idmax] = np.argwhere(prob == np.max(prob))
    print 'Best fit model is %s Gyr and %s Z' % (age[idmax[1]], metal[idmax[0]])

    return prob.T, age[idmax[1]], metal[idmax[0]]


class Galaxy(object):

    def __init__(self,galaxy_id):
        self.galaxy_id = galaxy_id
        wv,fl,er = np.load('../spec_stacks_jan24/%s_stack.npy' % self.galaxy_id)
        self.wv = wv[wv < 11300]
        self.fl = fl[wv < 11300]
        self.er = er[wv < 11300]
        self.contour = 0

    def Get_best_fit(self,contfits,featfits,metal,age,tau,specz,age_conv='../data/tau_scale_ntau.dat'):
        self.metal = metal
        self.age = age
        self.tau = tau

        self.specz = specz

        ####### Get maximum age
        max_age = Oldest_galaxy(self.specz)

        ####### Read in file
        Cdat = fits.open(contfits)
        Cchi = np.zeros([len(metal), len(age), len(tau)])

        Fdat = fits.open(featfits)
        Fchi = np.zeros([len(metal), len(age), len(tau)])

        for i in range(len(metal)):
            Fchi[i] = Fdat[i + 1].data
            Cchi[i] = Cdat[i + 1].data

        Fchi = Fchi.T
        Cchi = Cchi.T

        Fchi[:, len(age[age <= max_age]):, :] = 1E5
        Cchi[:, len(age[age <= max_age]):, :] = 1E5

        ####### Get scaling factor for tau reshaping
        scale = Readfile(age_conv)

        overhead = np.zeros(len(scale))
        for i in range(len(scale)):
            amt = []
            for ii in range(len(age)):
                if age[ii] > scale[i][-1]:
                    amt.append(1)
            overhead[i] = sum(amt)

        ######## Reshape likelihood to get average age instead of age when marginalized
        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau

        ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

        comb_prob = cprob / Cc * fprob / Cf

        prob = np.trapz(comb_prob, ultau, axis=2)
        C0 = np.trapz(np.trapz(prob, age, axis=1), metal)
        prob /= C0

        ##### get best fit values
        [idmax] = np.argwhere(prob == np.max(prob))

        self.prob = prob.T

        AP = np.trapz(self.prob, self.metal)
        MP = np.trapz(self.prob.T, self.age)
        #
        self.AP = AP/C0
        self.MP = MP/C0

        self.bfage = age[idmax[1]]
        self.bfmetal = metal[idmax[0]]

        fp = '../../../fsps_models_for_fit/galaxy_models/'

        chi = []
        for i in range(len(tau)):
            mwv, mfl = np.load(fp + 'm%s_a%s_t%s_z%s_%s_model.npy' % (self.bfmetal, self.bfage, tau[i], self.specz, self.galaxy_id))
            imfl = interp1d(mwv, mfl)(self.wv)
            C = Scale_model(self.fl, self.er, imfl)
            chi.append(Identify_stack(self.fl, self.er, C * imfl))

        self.bftau = tau[np.argmin(chi)]

        mwv, mfl = np.load(
            fp + 'm%s_a%s_t%s_z%s_%s_model.npy' % (self.bfmetal, self.bfage, self.bftau, self.specz, self.galaxy_id))
        imfl = interp1d(mwv, mfl)(self.wv)
        C = Scale_model(self.fl, self.er, imfl)

        self.mfl = C * imfl


    def Get_contours(self):
        onesig, twosig = Likelihood_contours(self.age, self.metal, self.prob)

        self.contour = np.array([twosig,onesig])


    def Plot_2D_likelihood(self,save_plot=False,plot_name=''):
        M, A = np.meshgrid(self.metal, self.age)

        if self.contour == 0:
            self.Get_contours()

        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],wspace=0.0, hspace=0.0)

        plt.figure(figsize=[8, 8])
        plt.subplot(gs[1, 0])
        plt.contour(M, A, self.prob, self.contour, colors='k', linewidths=2)
        plt.contourf(M, A, self.prob, 40, cmap=colmap)
        plt.xticks([0, .005, .01, .015, .02, .025, .03],
                   np.round(np.array([0, .005, .01, .015, .02, .025, .03]) / 0.02, 2))
        plt.plot(self.bfmetal, self.bfage, 'cp',
                 label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (self.bfage, np.round(self.bfmetal / 0.019, 2)))
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.minorticks_on()
        plt.xlabel('Z/Z$_\odot$', size=20)
        plt.ylabel('Average Age (Gyrs)', size=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.legend()

        plt.subplot(gs[1, 1])
        plt.plot(self.AP, self.age)
        plt.ylim(min(self.age),max(self.age))
        plt.yticks([])
        plt.xticks([])

        plt.subplot(gs[0, 0])
        plt.plot(self.metal / 0.019, self.MP)
        plt.xlim(min(self.metal) / 0.019, max(self.metal) / 0.019)
        plt.yticks([])
        plt.xticks([])
        plt.gcf().subplots_adjust(bottom=0.165, left=0.12)
        if save_plot == True:
            plt.savefig(plot_name)
        else:
            plt.show()
        plt.close()


"""Spec normmean"""

def Stack_spec_normwmean(spec, redshifts, wv):
    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.load(spec[i])

        wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

        if spec[i] == '../spec_stacks_jan24/n21156_stack.npy':
            IDer = []
            for ii in range(len(wave)):
                if 4855 * (1 + redshifts[i]) <= wave[ii] <= 4880 * (1 + redshifts[i]):
                    IDer.append(ii)
            error[IDer] = 1E8
            flux[IDer] = 0

        if spec[i] == '../spec_stacks_jan24/s39170_stack.npy':
            IDer = []
            for ii in range(len(wave)):
                if 4860 * (1 + redshifts[i]) <= wave[ii] <= 4880 * (1 + redshifts[i]):
                    IDer.append(ii)
            error[IDer] = 1E8
            flux[IDer] = 0

        if spec[i] == '../spec_stacks_jan24/n34694_stack.npy':
            IDer = []
            for ii in range(len(wave)):
                if 4860 * (1 + redshifts[i]) <= wave[ii] <= 4880 * (1 + redshifts[i]):
                    IDer.append(ii)
            error[IDer] = 1E8
            flux[IDer] = 0

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


def Stack_model_normwmean(speclist, modellist, redshifts, wv_range):
    flgrid ,errgrid= [[],[]]
    reg = np.arange(4000, 4210, 1)

    for i in range(len(speclist)):
        #######read in spectra
        wave, flux, error = np.load(speclist[i])
        wave, flux, error = np.array([wave[wave<=11100], flux[wave<=11100], error[wave<=11100]])

        wave = wave / (1 + redshifts[i])

        #######read in corresponding model, and interpolate flux
        W, F,= np.load(modellist[i])
        ifl = interp1d(W/ (1 + redshifts[i]), F)
        ier = interp1d(wave, error)

        #######scale the model
        C = Scale_model(flux, error, ifl(wave))

        ########interpolate spectra
        flentry,errentry = np.zeros([2,len(wv_range)])
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        Cr = np.trapz(ifl(reg)*C, reg)
        flentry[mask] = ifl(wv_range[mask]) * C / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    weigrid = np.array(errgrid).T ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack = np.sum(np.array(flgrid).T * weigrid, axis=1) / np.sum(weigrid,axis=1)

    return wv_range, stack


def Stack_model_normwmean_in_mfit(modellist, redshifts, wave_grid, flux_grid, err_grid, wv_range):
    flgrid ,errgrid= [[],[]]
    reg = np.arange(4000, 4210, 1)

    for i in range(len(modellist)):
        #######read in spectra
        wave, flux, error = np.array([wave_grid[i],flux_grid[i],err_grid[i]])

        #######read in corresponding model, and interpolate flux
        W, F, = np.load(modellist[i])
        ifl = interp1d(W / (1 + redshifts[i]), F)
        ier = interp1d(wave, error)

        #######scale the model
        C = Scale_model(flux, error, ifl(wave))

        ########interpolate spectra
        flentry, errentry = np.zeros([2, len(wv_range)])
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        Cr = np.trapz(ifl(reg) * C, reg)
        flentry[mask] = ifl(wv_range[mask]) * C / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    weigrid = np.array(errgrid).T ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack = np.sum(np.array(flgrid).T * weigrid, axis=1) / np.sum(weigrid,axis=1)

    return wv_range, stack


def Stack_sim_model_normwmean(speclist, modellist, redshifts, wv_range):
    flgrid = []
    errgrid = []

    for i in range(len(speclist)):
        #######read in spectra
        wave, flux, error = np.load(speclist[i])
        if speclist[i] == '../spec_stacks_jan24/s40597_stack.npy':
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

        wave = wave / (1 + redshifts[i])

        #######read in corresponding model, and interpolate flux
        W, F,= np.load(modellist[i])
        W = W / (1 + redshifts[i])
        iF = interp1d(W, F)(wave)

        #######scale the model
        C = Scale_model(flux, error, iF)
        F = C*iF + np.random.normal(0,error)
        Er = error

        ########interpolate spectra
        flentry = np.zeros(len(wv_range))
        errentry = np.zeros(len(wv_range))
        mask = np.array([wave[0] < U < wave[-1] for U in wv_range])
        ifl = interp1d(wave, F)
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

    return wv, stack, err


def Model_fit_stack_normwmean_features(speclist, tau, metal, A, speczs,ids, wv_range, name, pkl_name, res=10,
                                       fluxerr=False, flxerr_list='', fsps=True):
    ##############Stack spectra################
    wv, fl, er = Stack_spec_normwmean(speclist, speczs, wv_range)

    if fluxerr == True:
        inwv,inerr=Readfile(flxerr_list)
        er == inerr

    IDM=[]
    for i in range(len(wv)):
        if 3800<=wv[i]<=3850 or 3910<=wv[i]<=4030 or 4080<=wv[i]<=4125 or 4250<=wv[i]<=4385 or 4515<=wv[i]<=4570 or 4810<=wv[i]<=4910 or 4975<=wv[i]<=5055 or 5110<=wv[i]<=5285:
            IDM.append(i)


    #############Prep output file###############

    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ###############Pickle spectra##################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        wgrid = []
        fgrid = []
        egrid = []

        for i in range(len(speclist)):
            #######read in spectra
            wave, flux, error = np.load(speclist[i])
            wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

            wave = wave / (1 + speczs[i])
            wgrid.append(wave)
            fgrid.append(flux)
            egrid.append(error)

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    if fsps == True:
                        mlist = Make_model_list(ids, metal[i], A[ii], tau[iii], speczs)
                    else:
                        mlist = Make_model_list(ids, metal[i], A[ii], tau[iii], speczs, fsps=False)
                    mw, mf = Stack_model_normwmean_in_mfit(mlist, speczs, wgrid, fgrid, egrid,
                                                           np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    mf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(cPickle.load(outspec)[IDM])

    mf = np.array(mf)

    outspec.close()

    chigrid = np.sum(((fl[IDM] - mf) / er[IDM]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)])

    ###############
    for i in range(len(metal)):
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return


def Model_fit_stack_normwmean_cont(speclist, tau, metal, A, speczs,ids, wv_range, name, pkl_name, res=10,
                                   fluxerr=False, flxerr_list='',    fsps=True):
    ##############Stack spectra################
    wv, fl, er = Stack_spec_normwmean(speclist, speczs, wv_range)

    if fluxerr == True:
        inwv,inerr=Readfile(flxerr_list)
        er == inerr

    IDM=[]
    for i in range(len(wv)):
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 <= \
                wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110:
            IDM.append(i)

    #############Prep output file###############

    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ###############Pickle spectra##################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        wgrid = []
        fgrid = []
        egrid = []

        for i in range(len(speclist)):
            #######read in spectra
            wave, flux, error = np.load(speclist[i])
            wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

            wave = wave / (1 + speczs[i])
            wgrid.append(wave)
            fgrid.append(flux)
            egrid.append(error)

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    if fsps == True:
                        mlist = Make_model_list(ids, metal[i], A[ii], tau[iii], speczs)
                    else:
                        mlist = Make_model_list(ids, metal[i], A[ii], tau[iii], speczs, fsps=False)
                    mw, mf = Stack_model_normwmean_in_mfit(mlist, speczs, wgrid, fgrid, egrid,
                                                           np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    mf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(cPickle.load(outspec)[IDM])

    mf = np.array(mf)

    outspec.close()

    chigrid = np.sum(((fl[IDM] - mf) / er[IDM]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)])

    ###############
    for i in range(len(metal)):
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return


def Model_fit_stack_normwmean(speclist, tau, metal, A, speczs, ids, wv_range, name, pkl_name, res=10,
                              fluxerr=False, flxerr_list='',fsps=True):
    ##############Stack spectra################
    wv, fl, err = Stack_spec_normwmean(speclist, speczs, wv_range)

    if fluxerr == True:
        inwv,inerr=Readfile(flxerr_list)
        err == inerr

    #############Prep output file###############

    chifile = '../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    ###############Pickle spectra##################

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname) == False:

        wgrid = []
        fgrid = []
        egrid = []

        for i in range(len(speclist)):
            #######read in spectra
            wave, flux, error = np.load(speclist[i])
            wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

            wave = wave / (1 + speczs[i])
            wgrid.append(wave)
            fgrid.append(flux)
            egrid.append(error)

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    if fsps == True:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs)
                    else:
                        mlist=Make_model_list(ids,metal[i],A[ii],tau[iii],speczs,fsps=False)
                    mw, mf = Stack_model_normwmean_in_mfit(mlist, speczs, wgrid, fgrid, egrid,
                                                           np.arange(wv[0], wv[-1] + res, res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    mf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(cPickle.load(outspec))

    mf = np.array(mf)

    outspec.close()

    chigrid = np.sum(((fl - mf) / err) ** 2, axis=1).reshape([len(metal), len(A), len(tau)])

    ###############
    for i in range(len(metal)):
        inputgrid = np.array(chigrid[i])
        spc = 'metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

        ################Write chigrid file###############

    hdulist.writeto(chifile)
    return


def Model_fit_stack_MCerr_bestfit_normwmean(speclist, tau, metal, A, speczs, wv_range, name, pklname, repeats=100):
    ##############Stack spectra################

    wv, flx, err = Stack_spec_normwmean(speclist, speczs, wv_range)

    ##############Start loop and add error#############

    mlist = []
    alist = []

    for i in range(repeats):

        outspec = open(pklname, 'rb')

        fl = flx + np.random.normal(0, err)

        ##############Create chigrid#################

        chigrid = np.zeros([len(metal), len(A), len(tau)])
        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mf = np.array(cPickle.load(outspec))
                    chigrid[i][ii][iii] = Identify_stack(fl, err, mf)

        chigrid = np.array(chigrid, dtype=np.float128)
        chi = np.transpose(chigrid)
        ################Find best fit##################

        scale = Readfile('../data/tau_scale.dat', 1)

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        newchi = []
        for i in range(len(chi)):
            if i == 0:
                iframe = chi[i]
            else:
                iframe = interp2d(metal, scale[i], chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newchi.append(iframe)
        newchi = np.transpose(newchi)

        prob = np.exp(-newchi / 2)

        tau = np.array(tau)
        chigr = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(prob[i][ii], np.power(10, tau - 9)))
            chigr.append(acomp)
        prob = np.array(chigr)

        [idmax] = np.argwhere(prob == np.max(prob))
        alist.append(A[idmax[1]])
        mlist.append(metal[idmax[0]])

        outspec.close()

    fn = name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


def Model_fit_sim_stack_MCerr_bestfit_normwmean(speclist, tau, metal, A, sim_m, sim_a, sim_t, speczs, ids,
                                                wv_range, name, pkl_name, repeats=100, tau_scale='../data/tau_scale_nage.dat'):

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    mlist = []
    alist = []
    rmlist = Make_model_list(ids, sim_m, sim_a, sim_t, speczs)

    outspec = open(pklname, 'rb')

    mf=[]
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf.append(np.array(cPickle.load(outspec)))

    mf=np.array(mf)

    outspec.close()

    wv,flux,err=Stack_spec_normwmean(speclist,speczs,wv_range)
    mwv, mfl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    for xx in range(repeats):

        fl= mfl + np.random.normal(0,err)

        ##############Create chigrid#################
        chigrid=np.sum(((fl - mf) / err) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128)

        chi = np.transpose(chigrid)
        ################Find best fit##################

        scale = Readfile(tau_scale)

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        newchi = []
        for i in range(len(chi)):
            if i == 0:
                iframe = chi[i]
            else:
                iframe = interp2d(metal, scale[i], chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newchi.append(iframe)
        newchi = np.transpose(newchi)

        prob = np.exp(-newchi / 2)

        tau = np.array(tau)
        chigr = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(prob[i][ii], np.power(10, tau - 9)))
            chigr.append(acomp)
        prob = np.array(chigr)

        [idmax] = np.argwhere(prob == np.max(prob))
        alist.append(A[idmax[1]])
        mlist.append(metal[idmax[0]])

    fn ='../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


def Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat(speclist, tau, metal, A, sim_m, sim_a, sim_t, speczs, ids,
                                                wv_range, name, pkl_name, repeats=100,tau_scale='../data/tau_scale_nage.dat'):
    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    mlist = np.zeros(repeats)
    alist = np.zeros(repeats)
    rmlist = Make_model_list(ids, sim_m, sim_a, sim_t, speczs)

    wv, flux, err = Stack_spec_normwmean(speclist, speczs, wv_range)
    mwv, mfl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    ###############Get indicies
    IDf = []
    IDc = []
    for i in range(len(wv)):
        if 3800 <= wv[i] <= 3850 or 3910 <= wv[i] <= 4030 or 4080 <= wv[i] <= 4125 or 4250 <= wv[i] <= 4385 or 4515 \
                <= wv[i] <= 4570 or 4810 <= wv[i] <= 4910 or 4975 <= wv[i] <= 5055 or 5110 <= wv[i] <= 5285:
            IDf.append(i)
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 \
                <= wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110:
            IDc.append(i)

    ###############Get model list
    outspec = open(pklname, 'rb')

    fmf=[]
    cmf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                MF = np.array(cPickle.load(outspec))
                fmf.append(MF[IDf])
                cmf.append(MF[IDc])

    fmf=np.array(fmf)
    cmf = np.array(cmf)

    outspec.close()

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))

    scale = Readfile(tau_scale)
    overhead = []
    for i in range(len(scale)):
        amt = []
        for ii in range(len(A)):
            if A[ii] > scale[i][-1]:
                amt.append(1)
        overhead.append(sum(amt))

    for xx in range(repeats):
        fl = mfl + np.random.normal(0, err)

        Fchi = np.sum(((fl[IDf] - fmf) / err[IDf]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128).T
        Cchi = np.sum(((fl[IDc] - cmf) / err[IDc]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128).T

        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                cframe = interp2d(metal, scale[i], Cchi[i])(metal, A[:-overhead[i]])
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, A[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        cprob = np.exp(-newCchi.T / 2).astype(np.float128)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, A, axis=1), metal)

        Cprob = Pc / Cc

        fprob = np.exp(-newFchi.T / 2).astype(np.float128)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, A, axis=1), metal)

        Fprob = Pf / Cf

        prob = Cprob.T * Fprob.T

        C0 = np.trapz(np.trapz(prob, metal, axis=1), A)
        prob /= C0

        [idmax] = np.argwhere(prob == np.max(prob))
        alist[xx] = A[idmax[0]]
        mlist[xx]= metal[idmax[1]]

    fn ='../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


def Model_fit_sim_stack_MCerr_bestfit_normwmean_cont(speclist, tau, metal, A, sim_m, sim_a, sim_t, speczs, ids,
                                                     wv_range, name, pkl_name, repeats=100,tau_scale='../data/tau_scale_nage.dat'):

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    mlist = []
    alist = []
    rmlist = Make_model_list(ids, sim_m, sim_a, sim_t, speczs)


    wv, fl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    ###############Get indicies
    IDc = []

    for i in range(len(wv)):
        if wv[0] <= wv[i] <= 3800 or 3850 <= wv[i] <= 3910 or 4030 <= wv[i] <= 4080 or 4125 <= wv[i] <= 4250 or 4385 \
                <= wv[i] <= 4515 or 4570 <= wv[i] <= 4810 or 4910 <= wv[i] <= 4975 or 5055 <= wv[i] <= 5110:
            IDc.append(i)

    ###############Get model list

    outspec = open(pklname, 'rb')

    cmf = []
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                cmf.append(np.array(cPickle.load(outspec))[IDc])

    cmf = np.array(cmf)

    outspec.close()

    wv, flux, err = Stack_spec_normwmean(speclist, speczs, wv_range)
    mwv, mfl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    for xx in range(repeats):

        fl = mfl + np.random.normal(0, err)

        contgrid=np.sum(((fl[IDc] - cmf) / err[IDc]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128)

        cont_chi = np.transpose(contgrid)
        ################Find best fit##################

        scale = Readfile(tau_scale)

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        ################Continuum dist###############
        newcontchi = []
        for i in range(len(cont_chi)):
            if i == 0:
                iframe = cont_chi[i]
            else:
                iframe = interp2d(metal, scale[i], cont_chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newcontchi.append(iframe)
        newcontchi = np.transpose(newcontchi)

        cont_prob = np.exp(-newcontchi / 2)

        tau = np.array(tau)
        marginc = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(cont_prob[i][ii], np.power(10, tau - 9)))
            marginc.append(acomp)
        contprob = np.array(marginc)

        ##################Combine probabilities###########
        combprob = contprob.T

        [idmax] = np.argwhere(combprob == np.max(combprob))
        alist.append(A[idmax[0]])
        mlist.append(metal[idmax[1]])

    fn ='../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


def Model_fit_sim_stack_MCerr_bestfit_normwmean_feat(speclist, tau, metal, A, sim_m, sim_a, sim_t, speczs, ids,
                                                wv_range, name, pkl_name, repeats=100,tau_scale='../data/tau_scale_nage.dat'):

    pklname = '../pickled_mstacks/%s.pkl' % pkl_name

    mlist = []
    alist = []
    rmlist = Make_model_list(ids, sim_m, sim_a, sim_t, speczs)


    wv, fl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    ###############Get indicies
    IDf = []
    for i in range(len(wv)):
        if 3800 <= wv[i] <= 3850 or 3910 <= wv[i] <= 4030 or 4080 <= wv[i] <= 4125 or 4250 <= wv[i] <= 4385 or 4515 \
                <= wv[i] <= 4570 or 4810 <= wv[i] <= 4910 or 4975 <= wv[i] <= 5055 or 5110 <= wv[i] <= 5285:
            IDf.append(i)

    ###############Get model list
    outspec = open(pklname, 'rb')

    fmf=[]
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                fmf.append(np.array(cPickle.load(outspec))[IDf])

    fmf=np.array(fmf)

    outspec.close()

    wv, flux, err = Stack_spec_normwmean(speclist, speczs, wv_range)
    mwv, mfl = Stack_model_normwmean(speclist, rmlist, speczs, wv_range)

    for xx in range(repeats):

        fl = mfl + np.random.normal(0, err)

        featgrid=np.sum(((fl[IDf] - fmf) / err[IDf]) ** 2, axis=1).reshape([len(metal), len(A), len(tau)]).astype(np.float128)

        feat_chi = np.transpose(featgrid)

        ################Find best fit##################

        scale = Readfile(tau_scale)

        overhead = []
        for i in range(len(scale)):
            amt = []
            for ii in range(len(A)):
                if A[ii] > scale[i][-1]:
                    amt.append(1)
            overhead.append(sum(amt))

        ###############Feature dist#################
        newfeatchi = []
        for i in range(len(feat_chi)):
            if i == 0:
                iframe = feat_chi[i]
            else:
                iframe = interp2d(metal, scale[i], feat_chi[i])(metal, A[:-overhead[i]])
                iframe = np.append(iframe, np.repeat([np.repeat(1E8, len(metal))], overhead[i], axis=0), axis=0)
            newfeatchi.append(iframe)
        newfeatchi = np.transpose(newfeatchi)

        feat_prob = np.exp(-newfeatchi / 2)

        tau = np.array(tau)
        margin = []
        for i in range(len(metal)):
            acomp = []
            for ii in range(len(A)):
                acomp.append(np.trapz(feat_prob[i][ii], np.power(10, tau - 9)))
            margin.append(acomp)
        featprob = np.array(margin)

        ##################Combine probabilities###########
        combprob = featprob.T

        [idmax] = np.argwhere(combprob == np.max(combprob))
        alist.append(A[idmax[0]])
        mlist.append(metal[idmax[1]])

    fn ='../mcerr/' + name + '.dat'
    dat = Table([mlist, alist], names=['metallicities', 'age'])
    ascii.write(dat, fn)

    return


"""Test Functions"""

def Best_fit_model(input_file,metal,age, tau):
    dat=fits.open(input_file)

    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    x=np.argwhere(chi==np.min(chi))
    print metal[x[0][0]],age[x[0][1]],tau[x[0][2]]
    return metal[x[0][0]],age[x[0][1]],tau[x[0][2]]


def Identify_stack_features(fl, err, mfl, mask):
    ff = np.ma.masked_array(fl, mask)
    mm = np.ma.masked_array(mfl, mask)
    ee = np.ma.masked_array(err, mask)

    x = ((ff - mm) / ee) ** 2
    chi = np.sum(x)
    return chi


def Single_gal_fit_fsps(spec, tau, metal, A, specz, name):
    #############Read in spectra#################
    wv,fl,err=np.array(Readfile(spec,1))
    wv, fl, err = np.array([wv[wv < 11300], fl[wv < 11300], err[wv < 11300]])

    #############Prep output file###############
    chifile='../chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts#############
    filepath = '../../../fsps_models_for_fit/fsps_spec/'
    modellist = []
    for i in range(len(metal)):
        m = []
        for ii in range(len(A)):
            a = []
            for iii in range(len(tau)):
                a.append(filepath + 'm%s_a%s_t%s_spec.dat' % (metal[i], A[ii], tau[iii]))
            m.append(a)
        modellist.append(m)

    ##############Create chigrid and add to file#################
    chigrid=np.zeros([len(metal),len(A),len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mwv,mf= np.array(Readfile(modellist[i][ii][iii],1))
                mwv=mwv*(1+specz)
                imf=interp1d(mwv,mf)(wv)
                C=Scale_model(fl,err,imf)
                chigrid[i][ii][iii]=Identify_stack(fl,err,imf*C)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)
    ################Write chigrid file###############

    hdulist.writeto(chifile)
    print 'Done!'
    return


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


def B_factor(input_chi_file,tau,metal,age):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    dat = fits.open(input_chi_file)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    prob=[]
    for i in range(len(metal)):
        preprob1=[]
        for ii in range(len(age)):
            preprob2=[]
            for iii in range(len(tau)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii]/2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all tau
    ######## End up with age vs metallicity matricies
    ######## use unlogged tau
    ultau=np.append(0,np.power(10,tau[1:]-9))
    M = []
    for i in range(len(metal)):
        A=[]
        for ii in range(len(age)):
            T=[]
            for iii in range(len(tau) - 1):
                T.append(sp.N((ultau[iii + 1] - ultau[iii]) * (prob[i][ii][iii] + prob[i][ii][iii+1]) / 2))
            A.append(sp.mpmath.fsum(T))
        M.append(A)

    ######## Integrate over metallicity to get age prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(metal)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (M[i][ii] + M[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(metal) - 1):
        preC3.append(sp.N((metal[i + 1] - metal[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    return C


def Highest_likelihood_model_rfv(galaxy,speclist,rshift,RF_v, bfmetal, bfage,tau,wv_range):
    wv,fl,er=Stack_spec_normwmean_rfv(speclist,rshift,RF_v,wv_range)

    chi=[]
    for i in range(len(tau)):
        mlist=Make_model_list(galaxy, bfmetal,  bfage, tau[i] ,rshift)
        mwv,mfl=Stack_model_normwmean_rfv(speclist,mlist,rshift,RF_v,wv_range)
        chi.append(Identify_stack(fl,er,mfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def Highest_likelihood_model(galaxy,speclist,rshift,bfmetal, bfage,tau,wv_range):
    wv,fl,er=Stack_spec_normwmean(speclist,rshift,wv_range)

    chi=[]
    for i in range(len(tau)):
        mlist=Make_model_list(galaxy, bfmetal,  bfage, tau[i] ,rshift)
        mwv,mfl=Stack_model_normwmean(speclist,mlist,rshift,wv_range)
        chi.append(Identify_stack(fl,er,mfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def Get_parameters(gal_id, specz, metal, age, tau):
    Pr, bfage, bfmetal = Analyze_Stack_avgage_cont_feat_gal_age_correct(
        '../chidat/%s_apr6_galfit_cont_chidata.fits' % gal_id,
        '../chidat/%s_apr6_galfit_feat_chidata.fits' % gal_id,
        specz, np.array(tau), metal, age)
    a = [np.trapz(U, metal) for U in Pr]
    Ca = np.trapz(a, age)
    a /= Ca
    m = np.array([np.trapz(U, age) for U in Pr.T])
    Cm = np.trapz(m, metal)
    m /= Cm

    ia = interp1d(age, a)
    iage = np.linspace(age[0], age[-1], 500)

    amean = 0
    ale = 0
    ahe = 0

    for i in range(len(iage)):
        e = np.trapz(ia(iage[0:i + 1]), iage[0:i + 1])
        if ale == 0:
            if e >= .16:
                ale = iage[i]
        if amean == 0:
            if e >= .5:
                amean = iage[i]
        if ahe == 0:
            if e >= .84:
                ahe = iage[i]
                break

    im = interp1d(metal, m)
    imetal = np.linspace(metal[0], metal[-1], 500)

    mmean = 0
    mle = 0
    mhe = 0

    for i in range(len(imetal)):
        e = np.trapz(im(imetal[0:i + 1]), imetal[0:i + 1])
        if mle == 0:
            if e >= .16:
                mle = imetal[i]
        if mmean == 0:
            if e >= .5:
                mmean = imetal[i]
        if mhe == 0:
            if e >= .84:
                mhe = imetal[i]
                break

    return bfmetal, mmean, mmean - mle, mhe - mmean, bfage, amean, amean - ale, ahe - amean


class Stack(object):

    def __init__(self,speclist,redshifts,wv_range):
        self.speclist=speclist
        self.redshifts=redshifts
        self.wv_range=wv_range

    def Stack_normwmean(self):
        flgrid = np.zeros([len(self.speclist), len(self.wv_range)])
        errgrid = np.zeros([len(self.speclist), len(self.wv_range)])
        for i in range(len(self.speclist)):
            wave, flux, error = np.load(self.speclist[i])

            wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

            if self.speclist[i] == '../spec_stacks_jan24/n21156_stack.npy':
                IDer = []
                for ii in range(len(wave)):
                    if 4855 * (1 + self.redshifts[i]) <= wave[ii] <= 4880 * (1 + self.redshifts[i]):
                        IDer.append(ii)
                error[IDer] = 1E8
                flux[IDer] = 0

            if self.speclist[i] == '../spec_stacks_jan24/s39170_stack.npy':
                IDer = []
                for ii in range(len(wave)):
                    if 4860 * (1 + self.redshifts[i]) <= wave[ii] <= 4880 * (1 + self.redshifts[i]):
                        IDer.append(ii)
                error[IDer] = 1E8
                flux[IDer] = 0

            if self.speclist[i] == '../spec_stacks_jan24/n34694_stack.npy':
                IDer = []
                for ii in range(len(wave)):
                    if 4860 * (1 + self.redshifts[i]) <= wave[ii] <= 4880 * (1 + self.redshifts[i]):
                        IDer.append(ii)
                error[IDer] = 1E8
                flux[IDer] = 0

            wave /= (1 + self.redshifts[i])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv_range])
            ifl = interp1d(wave, flux)
            ier = interp1d(wave, error)
            reg = np.arange(4000, 4210, 1)
            Cr = np.trapz(ifl(reg), reg)
            flgrid[i][mask] = ifl(self.wv_range[mask]) / Cr
            errgrid[i][mask] = ier(self.wv_range[mask]) / Cr
        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv_range)])
        for i in range(len(self.wv_range)):
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / np.sum(weigrid[i])
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################
        ###take out nans

        IDX = [U for U in range(len(self.wv_range)) if stack[U] > 0]

        self.wv=self.wv_range[IDX]
        self.fl=stack[IDX]
        self.er=err[IDX]

    def Stack_normwmean_model(self,modellist):
        self.modellist=modellist

        flgrid, errgrid = [[], []]
        reg = np.arange(4000, 4210, 1)

        for i in range(len(self.speclist)):
            #######read in spectra
            wave, flux, error = np.load(self.speclist[i])
            wave, flux, error = np.array([wave[wave <= 11100], flux[wave <= 11100], error[wave <= 11100]])

            wave = wave / (1 + self.redshifts[i])

            #######read in corresponding model, and interpolate flux
            W, F, = np.load(self.modellist[i])
            ifl = interp1d(W / (1 + self.redshifts[i]), F)
            ier = interp1d(wave, error)

            #######scale the model
            C = Scale_model(flux, error, ifl(wave))

            ########interpolate spectra
            flentry, errentry = np.zeros([2, len(self.wv_range)])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv_range])
            Cr = np.trapz(ifl(reg) * C, reg)
            flentry[mask] = ifl(self.wv_range[mask]) * C / Cr
            errentry[mask] = ier(self.wv_range[mask]) / Cr
            flgrid.append(flentry)
            errgrid.append(errentry)

        weigrid = np.array(errgrid).T ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack = np.sum(np.array(flgrid).T * weigrid, axis=1) / np.sum(weigrid, axis=1)

        self.mwv = self.wv_range
        self.mfl = stack

    def Highest_likelihood_model_mlist(self,galaxy, bfmetal, bfage, tau):
        self.galaxy = galaxy

        chi = []
        for i in range(len(tau)):
            mlist = Make_model_list(self.galaxy, bfmetal, bfage, tau[i], self.redshifts)
            mwv, mfl = Stack_model_normwmean(self.speclist, mlist, self.redshifts, self.wv)
            chi.append(Identify_stack(self.fl, self.er, mfl))

        print [bfmetal, bfage, tau[np.argmin(chi)]]

        self.mlist = Make_model_list(self.galaxy, bfmetal, bfage, tau[np.argmin(chi)], self.redshifts)


class Galaxy_ids(object):

    def __init__(self,masslist):

        self.masslist = masslist
        self.ids = 0
        self.speclist = 0
        self.lmass = 0
        self.rshift = 0
        self.rad = 0
        self.sig = 0
        self.comp = 0

        self.ids, self.speclist, self.lmass, self.rshift, self.rad, self.sig, self.comp = \
            np.array(Readfile(self.masslist, is_float=False))
        self.lmass, self.rshift, self.rad, self.sig, self.comp = \
            np.array([self.lmass, self.rshift, self.rad, self.sig, self.comp]).astype(float)

        self.IDC = [U for U in range(len(self.ids)) if 0.11 > self.comp[U]]         # compact sample stack
        self.IDD = [U for U in range(len(self.ids)) if 0.11 < self.comp[U]]         # diffuse sample stack
        self.IDML = [U for U in range(len(self.ids)) if 10.931 > self.lmass[U]]     # low mass sample stack
        self.IDMH = [U for U in range(len(self.ids)) if 10.931 < self.lmass[U]]     # high mass sample stack
        self.IDlz = [U for U in range(len(self.ids)) if self.rshift[U] < 1.13]           # low z sample
        self.IDmz = [U for U in range(len(self.ids)) if 1.13 < self.rshift[U] < 1.3]# med z sample
        self.IDhz = [U for U in range(len(self.ids)) if 1.3 < self.rshift[U]]       # hi z sample
        self.IDml = [U for U in range(len(self.ids)) if 10 < self.lmass[U] < 10.7]  # low mass sample
        self.IDmm = [U for U in range(len(self.ids)) if 10.7 <= self.lmass[U] < 10.9]# med mass sample
        self.IDmh = [U for U in range(len(self.ids)) if 10.9 <= self.lmass[U]]      # hi mass sample

        self.ids_C = self.ids[self.IDC]
        self.ids_D = self.ids[self.IDD]
        self.ids_ML = self.ids[self.IDML]
        self.ids_MH = self.ids[self.IDMH]
        self.ids_lz = self.ids[self.IDlz]
        self.ids_mz = self.ids[self.IDmz]
        self.ids_hz = self.ids[self.IDhz]
        self.ids_ml = self.ids[self.IDml]
        self.ids_mm = self.ids[self.IDmm]
        self.ids_mh = self.ids[self.IDmh]

        self.speclist_C = self.speclist[self.IDC]
        self.speclist_D = self.speclist[self.IDD]
        self.speclist_ML = self.speclist[self.IDML]
        self.speclist_MH = self.speclist[self.IDMH]
        self.speclist_lz = self.speclist[self.IDlz]
        self.speclist_mz = self.speclist[self.IDmz]
        self.speclist_hz = self.speclist[self.IDhz]
        self.speclist_ml = self.speclist[self.IDml]
        self.speclist_mm = self.speclist[self.IDmm]
        self.speclist_mh = self.speclist[self.IDmh]

        self.lmass_C = self.lmass[self.IDC]
        self.lmass_D = self.lmass[self.IDD]
        self.lmass_ML = self.lmass[self.IDML]
        self.lmass_MH = self.lmass[self.IDMH]
        self.lmass_lz = self.lmass[self.IDlz]
        self.lmass_mz = self.lmass[self.IDmz]
        self.lmass_hz = self.lmass[self.IDhz]
        self.lmass_ml = self.lmass[self.IDml]
        self.lmass_mm = self.lmass[self.IDmm]
        self.lmass_mh = self.lmass[self.IDmh]

        self.rshift_C = self.rshift[self.IDC]
        self.rshift_D = self.rshift[self.IDD]
        self.rshift_ML = self.rshift[self.IDML]
        self.rshift_MH = self.rshift[self.IDMH]
        self.rshift_lz = self.rshift[self.IDlz]
        self.rshift_mz = self.rshift[self.IDmz]
        self.rshift_hz = self.rshift[self.IDhz]
        self.rshift_ml = self.rshift[self.IDml]
        self.rshift_mm = self.rshift[self.IDmm]
        self.rshift_mh = self.rshift[self.IDmh]

        self.comp_C = self.comp[self.IDC]
        self.comp_D = self.comp[self.IDD]
        self.comp_ML = self.comp[self.IDML]
        self.comp_MH = self.comp[self.IDMH]
        self.comp_lz = self.comp[self.IDlz]
        self.comp_mz = self.comp[self.IDmz]
        self.comp_hz = self.comp[self.IDhz]
        self.comp_ml = self.comp[self.IDml]
        self.comp_mm = self.comp[self.IDmm]
        self.comp_mh = self.comp[self.IDmh]


class Combine_1D_parameters(object):
    ###Given a set of 3D chi square distributions these will return a 1D parameter distribution with mean and 1-sigma

    def __init__(self,cont_chifits,feat_chifits, speclist, rshift, tau, metal, age,
                stack_scale=False , age_conv='../data/tau_scale_ntau.dat'):
        self.cont_chifits = cont_chifits
        self.feat_chifits = feat_chifits
        self.speclist = speclist
        self.rshift = rshift
        self.tau = np.array(tau)
        self.metal = metal
        self.age = age
        self.stack_scale = stack_scale
        self.age_conv = age_conv
        self.a_dist = np.zeros(len(self.age))
        self.amean = 0
        self.ale = 0
        self.ahe = 0
        self.m_dist = np.zeros(len(self.metal))
        self.mmean = 0
        self.mle = 0
        self.mhe = 0

    def Get_age(self):
        ### If scaling with stack this bit will provide the scaling factor
        reg = np.arange(4000, 4210, 1)                      # region that will be used to normalize
        regint = np.zeros(len(self.speclist))                    # initialize array which will contain the norm factors

        if self.stack_scale == True:
            for i in range(len(self.speclist)):
                wv, fl, er = np.load(self.speclist[i])           # read in data
                ifl = interp1d(wv / (1 + self.rshift[i]), fl)    # interpolate flux in restframe
                regint[i] = np.trapz(ifl(reg), reg)         # get norm factor

            scale_fact = min(regint) / regint               # adjust values so that its the amount each sed needs to be
                                                            # divided by to be normalized to the lowest amount
        else:
            scale_fact = np.ones(len(self.speclist))             # if not using stack scaling factor this will set array to ones

        ### This is used to get information necessary for interpolation later to change age to average age
        scale = Readfile(self.age_conv, 1)                       # read in scaling table
        overhead = np.zeros(len(scale))                     # initialize array which will contain how many lines to erase
        ultau = np.append(0, np.power(10, self.tau[1:] - 9))     # un-log tau for calculations

        for i in range(len(scale)):
            amt = []
            for ii in range(len(self.age)):
                if self.age[ii] > scale[i][-1]:
                    amt.append(1)
            overhead[i] = sum(amt)

        ### Iterate over data to get 1D distribution
        Cprob = np.ones(len(self.age))

        for i in range(len(self.cont_chifits)):
            ####### Read in file
            Cdat = fits.open(self.cont_chifits[i])
            Cchi = np.zeros([len(self.metal), len(self.age), len(self.tau)])

            Fdat = fits.open(self.feat_chifits[i])
            Fchi = np.zeros([len(self.metal), len(self.age), len(self.tau)])

            for ii in range(len(self.metal)):
                Fchi[ii] = Fdat[ii + 1].data
                Cchi[ii] = Cdat[ii + 1].data

            Fchi = Fchi.T
            Cchi = Cchi.T

            max_age = Oldest_galaxy(self.rshift[i])

            Fchi[:, len(self.age[self.age <= max_age]):, :] = 1E5
            Cchi[:, len(self.age[self.age <= max_age]):, :] = 1E5

            newCchi = np.zeros(Cchi.shape)
            newFchi = np.zeros(Fchi.shape)

            for ii in range(len(Cchi)):
                if ii == 0:
                    newCchi[ii] = Cchi[ii]
                    newFchi[ii] = Fchi[ii]
                else:
                    cframe = interp2d(self.metal, scale[ii], Cchi[ii])(self.metal, self.age[:-overhead[ii]])
                    newCchi[ii] = np.append(cframe, np.repeat([np.repeat(1E5, len(self.metal))], overhead[ii], axis=0), axis=0)

                    fframe = interp2d(self.metal, scale[ii], Fchi[ii])(self.metal, self.age[:-overhead[ii]])
                    newFchi[ii] = np.append(fframe, np.repeat([np.repeat(1E5, len(self.metal))], overhead[ii], axis=0), axis=0)

            ####### Create normalize probablity marginalized over tau
            cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

            Pc = np.trapz(cprob, ultau, axis=2)
            Ac = [np.trapz(U, self.metal) for U in Pc.T]
            Cc = np.trapz(np.trapz(Pc, self.age, axis=1), self.metal)


            fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

            Pf = np.trapz(fprob, ultau, axis=2)
            Af = [np.trapz(U, self.metal) for U in Pf.T]
            Cf = np.trapz(np.trapz(Pf, self.age, axis=1), self.metal)

            comb_prob = np.array(Ac).astype(np.float128) / Cc * np.array(Af).astype(np.float128) / Cf

            Cprob *= (comb_prob ** scale_fact[i]).astype(np.float128)

        #########################
        Ca = np.trapz(Cprob, self.age)
        a = Cprob / Ca

        ia = interp1d(self.age, a)
        iage = np.linspace(self.age[0], self.age[-1], 500)

        amean = 0
        ale = 0
        ahe = 0

        for i in range(len(iage)):
            e = np.trapz(ia(iage[0:i + 1]), iage[0:i + 1])
            if ale == 0:
                if e >= .16:
                    ale = iage[i]
            if amean == 0:
                if e >= .5:
                    amean = iage[i]
            if ahe == 0:
                if e >= .84:
                    ahe = iage[i]
                    break

        self.a_dist = a
        self.amean = amean
        self.ale = amean - ale
        self.ahe = ahe - amean

        # return a, amean, amean - ale, ahe - amean

    def Get_metallicity(self):
        ### If scaling with stack this bit will provide the scaling factor
        reg = np.arange(4000, 4210, 1)                      # region that will be used to normalize
        regint = np.zeros(len(self.speclist))                    # initialize array which will contain the norm factors

        if self.stack_scale == True:
            for i in range(len(self.speclist)):
                wv, fl, er = np.load(self.speclist[i])           # read in data
                ifl = interp1d(wv / (1 + self.rshift[i]), fl)    # interpolate flux in restframe
                regint[i] = np.trapz(ifl(reg), reg)         # get norm factor

            scale_fact = min(regint) / regint               # adjust values so that its the amount each sed needs to be
                                                            # divided by to be normalized to the lowest amount
        else:
            scale_fact = np.ones(len(self.speclist))             # if not using stack scaling factor this will set array to ones

        ### This is used to get information necessary for interpolation later to change age to average age
        scale = Readfile(self.age_conv, 1)                       # read in scaling table
        overhead = np.zeros(len(scale))                     # initialize array which will contain how many lines to erase
        ultau = np.append(0, np.power(10, self.tau[1:] - 9))     # un-log tau for calculations

        for i in range(len(scale)):
            amt = []
            for ii in range(len(self.age)):
                if self.age[ii] > scale[i][-1]:
                    amt.append(1)
            overhead[i] = sum(amt)

        ### Iterate over data to get 1D distribution
        Cprob = np.ones(len(self.metal))

        for i in range(len(self.cont_chifits)):
            ####### Read in file
            Cdat = fits.open(self.cont_chifits[i])
            Cchi = np.zeros([len(self.metal), len(self.age), len(self.tau)])

            Fdat = fits.open(self.feat_chifits[i])
            Fchi = np.zeros([len(self.metal), len(self.age), len(self.tau)])

            for ii in range(len(self.metal)):
                Fchi[ii] = Fdat[ii + 1].data
                Cchi[ii] = Cdat[ii + 1].data

            Fchi = Fchi.T
            Cchi = Cchi.T

            max_age = Oldest_galaxy(self.rshift[i])

            Fchi[:, len(self.age[self.age <= max_age]):, :] = 1E5
            Cchi[:, len(self.age[self.age <= max_age]):, :] = 1E5

            newCchi = np.zeros(Cchi.shape)
            newFchi = np.zeros(Fchi.shape)

            for ii in range(len(Cchi)):
                if ii == 0:
                    newCchi[ii] = Cchi[ii]
                    newFchi[ii] = Fchi[ii]
                else:
                    cframe = interp2d(self.metal, scale[ii], Cchi[ii])(self.metal, self.age[:-overhead[ii]])
                    newCchi[ii] = np.append(cframe, np.repeat([np.repeat(1E5, len(self.metal))], overhead[ii], axis=0),
                                            axis=0)

                    fframe = interp2d(self.metal, scale[ii], Fchi[ii])(self.metal, self.age[:-overhead[ii]])
                    newFchi[ii] = np.append(fframe, np.repeat([np.repeat(1E5, len(self.metal))], overhead[ii], axis=0),
                                            axis=0)

            ####### Create normalize probablity marginalized over tau
            cprob = np.exp(-newCchi.T.astype(np.float128) / 2)

            Pc = np.trapz(cprob, ultau, axis=2)
            Mc = [np.trapz(U, self.age) for U in Pc]
            Cc = np.trapz(np.trapz(Pc, self.age, axis=1), self.metal)

            fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

            Pf = np.trapz(fprob, ultau, axis=2)
            Mf = [np.trapz(U, self.age) for U in Pf]
            Cf = np.trapz(np.trapz(Pf, self.age, axis=1), self.metal)

            comb_prob = np.array(Mc).astype(np.float128) / Cc * np.array(Mf).astype(np.float128) / Cf

            Cprob *= (comb_prob ** scale_fact[i]).astype(np.float128)

        #########################
        Cm = np.trapz(Cprob, self.metal)
        m = Cprob / Cm

        im = interp1d(self.metal, m)
        imetal = np.linspace(self.metal[0], self.metal[-1], 500)

        mmean = 0
        mle = 0
        mhe = 0

        for i in range(len(imetal)):
            e = np.trapz(im(imetal[0:i + 1]), imetal[0:i + 1])
            if mle == 0:
                if e >= .16:
                    mle = imetal[i]
            if mmean == 0:
                if e >= .5:
                    mmean = imetal[i]
            if mhe == 0:
                if e >= .84:
                    mhe = imetal[i]
                    break

        self.m_dist = m
        self.mmean = mmean
        self.mle = mmean - mle
        self.mhe = mhe - mmean