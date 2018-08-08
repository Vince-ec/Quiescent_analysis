__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import grizli
from grizli import model as griz_model
import matplotlib.image as mpimg
from astropy.io import fits
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import os
from glob import glob
from time import time
import seaborn as sea

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()

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

def Calzetti(Av,lam):
    lam = lam * 1E-4
    Rv=4.05
    k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
    cal = 10**(-0.4*k*Av/Rv)    
    
    return cal

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

#    for i in range(len(f)):
#        if f[i] < 0:
#            f[i] = 0

    return w, f, e


def Get_flux_nocont(FILE, z):
    w, f, e = np.load(FILE)
    if FILE == '../spec_stacks_jan24/s40597_stack.npy':
        IDW = []
        for ii in range(len(w)):
            if 7950 < w[ii] < 11000:
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


def Get_repeats(x, y):
    z = [x, y]
    tz = np.transpose(z)
    size = np.zeros(len(tz))
    for i in range(len(size)):
        size[i] = len(np.argwhere(tz == tz[i])) / 2
    size /= 5
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


def Median_w_Error(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_95(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .025:
                lerr = ix[i]
        if herr == 0:
            if e >= .975:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_cont(Pofx, x):
    ix = np.linspace(x[0], x[-1], 500)
    iP = interp1d(x, Pofx)(ix)

    C = np.trapz(iP,ix)

    iP/=C


    lerr = 0
    herr = 0
    med = 0

    for i in range(len(ix)):
        e = np.trapz(iP[0:i + 1], ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if med == 0:
            if e >= .50:
                med = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    return med, med - lerr, herr - np.abs(med)


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
    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(np.trapz(p, m2, axis=1), age)

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



"""Stack Fit"""


def Stack_model_normwmean(speclist, redshifts, bfmetal, bfage, bftau, wv_range, n_win):
    flgrid, errgrid = [[], []]

    for i in range(len(speclist)):
    #######read in spectra
        spec = Gen_spec(speclist[i],redshifts[i])

        #######read in corresponding model, and interpolate flux
        spec.Sim_spec(bfmetal,bfage,bftau)
        ifl = interp1d(spec.gal_wv_rf, spec.fl)
        ier = interp1d(spec.gal_wv_rf, spec.gal_er)

        ########interpolate spectra
        flentry, errentry = np.zeros([2, len(wv_range)])
        mask = np.array([spec.gal_wv_rf[0] < U < spec.gal_wv_rf[-1] for U in wv_range])
        Cr = np.trapz(ifl(n_win), n_win)
        flentry[mask] = ifl(wv_range[mask]) / Cr
        errentry[mask] = ier(wv_range[mask]) / Cr
        flgrid.append(flentry)
        errgrid.append(errentry)

    weigrid = np.array(errgrid).T ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack = np.sum(np.array(flgrid).T * weigrid, axis=1) / np.sum(weigrid, axis=1)

    return wv_range, stack


"""Single Galaxy"""
class Gen_spec(object):
    def __init__(self, galaxy_id, redshift,minwv = 7900, maxwv = 11200, shift = 1):
        self.galaxy_id = galaxy_id
        self.gid = int(self.galaxy_id[1:])
        self.redshift = redshift
        self.shift = shift

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        gal_wv, gal_fl, gal_er = np.load(glob('../spec_stacks/*{0}*'.format(self.gid))[0])
        self.flt_input = glob('../beams/*{0}*'.format(self.gid))[0]

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        WV,TEF = np.load('../data/template_error_function.npy')
        iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
        self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.gal_fl)**2)

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(fits_file=self.flt_input)

        ## Get sensitivity function

        flat = self.beam.flat_flam.reshape(self.beam.beam.sh_beam)
        fwv, ffl, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,flat.shape[0]]),flat.T[:-1],axis=0).T , bin=0)
        
        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
    def Sim_spec(self, metal, age, tau, model_redshift = 0, dust = 0):
        if model_redshift ==0:
            model_redshift = self.redshift
            
        model = '../../../fsps_models_for_fit/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

        wave, fl = np.load(model)

        cal = 1
        if dust !=0:
            lam = wave * 1E-4
            Rv = 4.05
            k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
            cal = 10**(-0.4 * k * dust / Rv)  
        
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[wave*(1+model_redshift),fl * cal])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,self.beam.model.shape[0]]),
                                                           self.beam.model.T[:-1],axis=0).T , bin=0)

        ifl = interp1d(w, f)(self.gal_wv)
        adj_ifl = ifl /self.filt
        
        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl
        
    def Sim_spec_mult(self, wave, fl, model_redshift = 0):
        if model_redshift ==0:
            model_redshift = self.redshift

        ## Compute the models
        self.beam.compute_model(spectrum_1d=[wave*(1+model_redshift), fl])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,self.beam.model.shape[0]]),
                                                           self.beam.model.T[:-1],axis=0).T , bin=0)

        self.fl = f
        self.mwv = w

    def Fit_lwa(self, fit_Z, fit_t, fit_z, fit_d, metal_array, age_array, tau_array):
        
        lwa_grid = np.load('../data/light_weight_scaling_3.npy')
        chi = []
        good_age =[]
        good_tau =[]
        for i in range(len(tau_array)):
            for ii in range(age_array.size):
                
                lwa = lwa_grid[np.argwhere(np.round(metal_array,3) == np.round(fit_Z,3))[0][0]][ii][i]
                
                if (fit_t - 0.1) < lwa < (fit_t + 0.1):
                    self.Sim_spec(fit_Z,age_array[ii],tau_array[i],fit_z, fit_d)
                    chi.append(sum(((self.gal_fl - self.fl) / self.gal_er)**2))
                    good_age.append(age_array[ii])
                    good_tau.append(tau_array[i])

        self.bfage = np.array(good_age)[chi == min(chi)][0]
        self.bftau = np.array(good_tau)[chi == min(chi)][0]
        if self.bftau == 0.0:
            self.bftau = int(0)
        self.Sim_spec(fit_Z, self.bfage, self.bftau, fit_z, fit_d)    


    def Sim_spec_BC03(self, metal, age, tau):
        import pysynphot as S
        model = '../../../bc03_models_for_fit/bc03_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl_bc = C * adj_ifl
        

def Single_gal_fit(metal, age, tau, specz, galaxy, name, minwv = 7900, maxwv = 11300):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv)

    if galaxy == 'n21156' or galaxy == 'n38126':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4855 <= spec.gal_wv_rf[ii] <= 4880:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's47677' or galaxy == 'n14713':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4845 <= spec.gal_wv_rf[ii] <= 4863:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's39170':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4865 <= spec.gal_wv_rf[ii] <= 4885:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1 = '../chidat/%s_chidata' % name
 
    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(tau),len(spec.gal_wv_rf)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl
    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)


    ################Write chigrid file###############
    np.save(chifile1,chigrid1)


#    P, PZ, Pt = Analyze_LH_lwa(chifile1 + '.npy', specz, metal, age, tau)

#    np.save('../chidat/%s_tZ_pos' % name,P)
#    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
#    np.save('../chidat/%s_t_pos' % name,[age,Pt])

    print('Done!')
    return


def Specz_fit(galaxy, metal, age, rshift, name):
    #############initialize spectra#################
    spec = RT_spec(galaxy)

    #############Prep output file###############
    chifile = '../rshift_dat/%s_z_fit' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(rshift),len(spec.gal_wv)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(rshift)):
                spec.Sim_spec(metal[i], age[ii], 0, rshift[iii])
                mfl[i*len(age)*len(rshift)+ii*len(rshift)+iii]=spec.fl
    chigrid = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(rshift)]).\
        astype(np.float128)

    np.save(chifile,chigrid)
    ###############Write chigrid file###############
    Analyze_specz(chifile + '.npy', rshift, metal, age, name)

    print('Done!')

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


def Analyze_specz(chifits, rshift, metal, age, name):
    ####### Read in file
    dat = np.load(chifits)

    ###### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P_specz(rshift, metal, age, dat.T)).astype(np.float128)

    ###### get best fit values
    print('Best fit specz is %s' % rshift[np.argmax(prob)])

    np.save('../rshift_dat/%s_Pofz' % name,[rshift, prob])
    return


def Highest_likelihood_model_galaxy(galaxy, rshift, bfmetal, bfage, tau):
    wv, fl, er = np.load('../spec_stacks_jan24/%s_stack.npy' % galaxy)
    fp = '../../../fsps_models_for_fit/galaxy_models/'

    chi = []
    for i in range(len(tau)):
        mwv, mfl = np.load(fp + 'm%s_a%s_t%s_z%s_%s_model.npy' % (bfmetal, bfage, tau[i], rshift, galaxy))
        imfl = interp1d(mwv, mfl)(wv)
        C = Scale_model(fl, er, imfl)
        chi.append(Identify_stack(fl, er, C * imfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def Sim_fit(galaxy, metal, age, tau, sim_m, sim_a, sim_t, specz, name, minwv=7900, maxwv=11400,
           age_conv='../data/tau_scale_ntau.dat'):
    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    spec = Gen_sim(galaxy, specz, sim_m, sim_a, sim_t,minwv=minwv,maxwv=maxwv)

    ###############Get model list
    mfl = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    mfl_nc = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl
                spec.RM_sim_spec_cont()
                mfl_nc[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.nc_mfl


    convtau = np.array([0, 8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2,
                        9.23, 9.26, 9.28, 9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48])
    convage = np.arange(.5, 6.1, .1)

    mt = [U for U in range(len(convtau)) if convtau[U] in tau]
    ma = [U for U in range(len(convage)) if np.round(convage[U], 1) in np.round(age, 1)]

    convtable = Readfile(age_conv)
    scale = convtable[mt[0]:mt[-1] + 1, ma[0]:ma[-1] + 1]

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    spec.Perturb_both()
    chi = np.sum(((spec.flx_err - mfl) / spec.gal_er) ** 2, axis=1).reshape(
        [len(metal), len(age), len(tau)]).astype(
        np.float128).T
    NCchi = np.sum(((spec.nc_flx_err - mfl_nc) / spec.nc_er) ** 2, axis=1).reshape(
        [len(metal), len(age), len(tau)]).astype(
        np.float128).T

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)
    newNCchi = np.zeros(NCchi.shape)

    for i in range(len(chi)):
        if i == 0:
            newchi[i] = chi[i]
            newNCchi[i] = NCchi[i]
        else:
            frame = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

            ncframe = interp2d(metal, scale[i], NCchi[i])(metal, age[:-overhead[i]])
            newNCchi[i] = np.append(ncframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau
    prob = np.exp(-newchi.T.astype(np.float128) / 2)
    ncprob = np.exp(-newNCchi.T.astype(np.float128) / 2)

    P = np.trapz(prob, ultau, axis=2)
    C = np.trapz(np.trapz(P, age, axis=1), metal)

    Pnc = np.trapz(ncprob, ultau, axis=2)
    Cnc = np.trapz(np.trapz(Pnc, age, axis=1), metal)

    #### Get Z and t posteriors
    PZ = np.trapz(P / C, age, axis=1)
    Pt = np.trapz(P.T / C, metal, axis=1)

    PZnc = np.trapz(Pnc / Cnc, age, axis=1)
    Ptnc = np.trapz(Pnc.T / Cnc, metal, axis=1)


    np.save('../mcerr/' + name, P/C)
    np.save('../mcerr/' + name + '_NC', Pnc/Cnc)
    np.save('../mcerr/' + name + '_Z', [metal, PZ])
    np.save('../mcerr/' + name + '_t', [age, Pt])
    np.save('../mcerr/' + name + '_ncZ', [metal, PZnc])
    np.save('../mcerr/' + name + '_nct', [age, Ptnc])
    np.save('../mcerr/' + name + '_sim', [spec.gal_wv_rf, spec.flx_err])
    np.save('../mcerr/' + name + '_ncsim', [spec.gal_wv_rf, spec.nc_flx_err])

    return

def Analyze_LH_lwa(chifits, specz, metal, age, tau, age_conv='../data/light_weight_scaling_3.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        # if i == 0:
        #     newchi[i] = chi[i]
        # else:
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii],chi[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(1E5, overhead[i][ii]))
        newchi[i] = frame.T

    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt


class Galaxy_set(object):
    def __init__(self, galaxy_id):
        self.galaxy_id = galaxy_id
        if os.path.isdir('../../../../vestrada'):
            gal_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s/' % self.galaxy_id
        else:
            gal_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s/' % self.galaxy_id

        # test
        # gal_dir = '/Users/Vince.ec/Clear_data/test_data/%s/' % self.galaxy_id
        one_d = glob(gal_dir + '*1D.fits')
        self.two_d = glob(gal_dir + '*png')
        one_d_l = [len(U) for U in one_d]
        self.one_d_stack = one_d[np.argmin(one_d_l)]
        self.one_d_list = np.delete(one_d, [np.argmin(one_d_l)])

    def Get_flux(self, FILE):
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
            if w[i] < 11900:
                INDEX.append(i)

        w = w[INDEX]
        f = f[INDEX]
        e = e[INDEX]

        # for i in range(len(f)):
        #     if f[i] < 0:
        #         f[i] = 0

        return w, f, e

    def Display_spec(self, override_quality=False):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            if override_quality == True:

                if len(self.two_d) > 0:
                    for i in range(len(self.two_d)):
                        os.system("open " + self.two_d[i])

                if len(self.one_d_list) > 0:
                    if len(self.one_d_list) < 10:
                        plt.figure(figsize=[15, 10])
                        for i in range(len(self.one_d_list)):
                            wv, fl, er = Get_flux(self.one_d_list[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(self.one_d_list) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                    if len(self.one_d_list) > 10:

                        smlist1 = self.one_d_list[:9]
                        smlist2 = self.one_d_list[9:]

                        plt.figure(figsize=[15, 10])
                        for i in range(len(smlist1)):
                            wv, fl, er = Get_flux(smlist1[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(smlist1) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                        plt.figure(figsize=[15, 10])
                        for i in range(len(smlist2)):
                            wv, fl, er = Get_flux(smlist2[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(smlist2) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                self.quality = np.repeat(1, len(self.one_d_list)).astype(int)
                self.Mask = np.zeros([len(self.one_d_list), 2])
                self.pa_names = []

                for i in range(len(self.one_d_list)):
                    self.pa_names.append(self.one_d_list[i].replace(n_dir, ''))
                    wv, fl, er = Get_flux(self.one_d_list[i])
                    IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                    plt.figure(figsize=[15, 5])
                    plt.plot(wv[IDX], fl[IDX])
                    plt.plot(wv[IDX], er[IDX])
                    plt.ylim(min(fl[IDX]), max(fl[IDX]))
                    plt.xlim(7800, 11500)
                    plt.title(self.pa_names[i])
                    plt.show()
                    self.quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
                    if self.quality[i] == 1:
                        minput = int(input('Mask region: (0 if no mask needed)'))
                        if minput != 0:
                            rinput = int(input('Lower bounds'))
                            linput = int(input('Upper bounds'))
                            self.Mask[i] = [rinput, linput]
                ### save quality file
                l_mask = self.Mask.T[0]
                h_mask = self.Mask.T[1]

                qual_dat = Table([self.pa_names, self.quality, l_mask, h_mask],
                                 names=['id', 'good_spec', 'mask_low', 'mask_high'])
                fn = n_dir + '/%s_quality.txt' % self.galaxy_id
                ascii.write(qual_dat, fn, overwrite=True)

        else:

            if len(self.two_d) > 0:
                for i in range(len(self.two_d)):
                    os.system("open " + self.two_d[i])

            if len(self.one_d_list) > 0:
                if len(self.one_d_list) < 10:
                    plt.figure(figsize=[15, 10])
                    for i in range(len(self.one_d_list)):
                        wv, fl, er = Get_flux(self.one_d_list[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11750]
                        plt.subplot(11 + i + len(self.one_d_list) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11750)
                    plt.show()

                if len(self.one_d_list) > 10:

                    smlist1 = self.one_d_list[:9]
                    smlist2 = self.one_d_list[9:]

                    plt.figure(figsize=[15, 10])
                    for i in range(len(smlist1)):
                        wv, fl, er = Get_flux(smlist1[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                        plt.subplot(11 + i + len(smlist1) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11500)
                    plt.show()

                    plt.figure(figsize=[15, 10])
                    for i in range(len(smlist2)):
                        wv, fl, er = Get_flux(smlist2[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                        plt.subplot(11 + i + len(smlist2) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11500)
                    plt.show()

            self.quality = np.repeat(1, len(self.one_d_list)).astype(int)
            self.Mask = np.zeros([len(self.one_d_list), 2])
            self.pa_names = []

            for i in range(len(self.one_d_list)):
                self.pa_names.append(self.one_d_list[i].replace(n_dir, ''))
                wv, fl, er = Get_flux(self.one_d_list[i])
                IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                plt.figure(figsize=[15, 5])
                plt.plot(wv[IDX], fl[IDX])
                plt.plot(wv[IDX], er[IDX])
                plt.ylim(min(fl[IDX]), max(fl[IDX]))
                plt.xlim(7800, 11500)
                plt.title(self.pa_names[i])
                plt.show()
                self.quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
                if self.quality[i] == 1:
                    minput = int(input('Mask region: (0 if no mask needed)'))
                    if minput != 0:
                        rinput = int(input('Lower bounds'))
                        linput = int(input('Upper bounds'))
                        self.Mask[i] = [rinput, linput]
            ### save quality file
            l_mask = self.Mask.T[0]
            h_mask = self.Mask.T[1]

            qual_dat = Table([self.pa_names, self.quality, l_mask, h_mask],
                             names=['id', 'good_spec', 'mask_low', 'mask_high'])
            fn = n_dir + '/%s_quality.txt' % self.galaxy_id
            ascii.write(qual_dat, fn, overwrite=True)

    def Get_wv_list(self):
        W = []
        lW = []

        for i in range(len(self.one_d_list)):
            wv, fl, er = self.Get_flux(self.one_d_list[i])
            W.append(wv)
            lW.append(len(wv))

        W = np.array(W)
        self.wv = W[np.argmax(lW)]

    def Mean_stack_galaxy(self):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        ### select good galaxies
        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            self.pa_names, self.quality, l_mask, h_mask = Readfile(n_dir + '/%s_quality.txt' % self.galaxy_id,
                                                                   is_float=False)
            self.quality = self.quality.astype(float)
            l_mask = l_mask.astype(float)
            h_mask = h_mask.astype(float)
            self.Mask = np.vstack([l_mask, h_mask]).T

        new_speclist = []
        new_mask = []
        for i in range(len(self.quality)):
            if self.quality[i] == 1:
                new_speclist.append(self.one_d_list[i])
                new_mask.append(self.Mask[i])

        self.Get_wv_list()
        self.good_specs = new_speclist
        self.good_Mask = new_mask

        # Define grids used for stacking
        flgrid = np.zeros([len(self.good_specs), len(self.wv)])
        errgrid = np.zeros([len(self.good_specs), len(self.wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(self.good_specs)):
            wave, flux, error = self.Get_flux(self.good_specs[i])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv])
            ifl = interp1d(wave, flux)(self.wv[mask])
            ier = interp1d(wave, error)(self.wv[mask])

            if sum(self.good_Mask[i]) > 0:
                for ii in range(len(self.wv[mask])):
                    if self.good_Mask[i][0] < self.wv[mask][ii] < self.good_Mask[i][1]:
                        ifl[ii] = 0
                        ier[ii] = 0

            flgrid[i][mask] = ifl
            errgrid[i][mask] = ier

        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv)])
        for i in range(len(self.wv)):
            # fl_filter = np.ones(len(flgrid[i]))
            # for ii in range(len(flgrid[i])):
            #     if flgrid[i][ii] == 0:
            #         fl_filter[ii] = 0
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.fl = np.array(stack)
        self.er = np.array(err)

    def Bootstrap(self, galaxy_list, repeats=1000):
        gal_index = np.arange(len(galaxy_list))

    def Median_w_bootstrap_stack_galaxy(self):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        ### select good galaxies
        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            self.pa_names, self.quality, l_mask, h_mask = Readfile(n_dir + '/%s_quality.txt' % self.galaxy_id,
                                                                   is_float=False)
            self.quality = self.quality.astype(float)
            l_mask = l_mask.astype(float)
            h_mask = h_mask.astype(float)
            self.Mask = np.vstack([l_mask, h_mask]).T

        new_speclist = []
        new_mask = []
        for i in range(len(self.quality)):
            if self.quality[i] == 1:
                new_speclist.append(self.one_d_list[i])
                new_mask.append(self.Mask[i])

        self.Get_wv_list()
        self.good_specs = new_speclist
        self.good_Mask = new_mask

        # Define grids used for stacking
        flgrid = np.zeros([len(self.good_specs), len(self.wv)])
        errgrid = np.zeros([len(self.good_specs), len(self.wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(self.good_specs)):
            wave, flux, error = self.Get_flux(self.good_specs[i])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv])
            ifl = interp1d(wave, flux)(self.wv[mask])
            ier = interp1d(wave, error)(self.wv[mask])

            if sum(self.good_Mask[i]) > 0:
                for ii in range(len(self.wv[mask])):
                    if self.good_Mask[i][0] < self.wv[mask][ii] < self.good_Mask[i][1]:
                        ifl[ii] = 0
                        ier[ii] = 0

            flgrid[i][mask] = ifl
            errgrid[i][mask] = ier

        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv)])
        for i in range(len(self.wv)):
            # fl_filter = np.ones(len(flgrid[i]))
            # for ii in range(len(flgrid[i])):
            #     if flgrid[i][ii] == 0:
            #         fl_filter[ii] = 0
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.fl = np.array(stack)
        self.er = np.array(err)

    def Get_stack_info(self):
        wv, fl, er = Get_flux(self.one_d_stack)
        self.s_wv = wv
        self.s_fl = fl
        self.s_er = er


"""MC fits"""


class Gen_sim(object):
    def __init__(self, galaxy_id, redshift, metal, age, tau, minwv=7900, maxwv=11400, pad=100):
        import pysynphot as S
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.metal = metal
        self.age = age
        self.tau = tau
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.gal_wv - output wavelength array of galaxy
        **
        self.gal_wv_rf - output wavelength array in restframe
        **
        self.gal_fl - output flux array of galaxy
        **
        self.gal_er - output error array of galaxy
        **
        self.fl - output flux array of model used for simulation
        **
        self.flx_err - output flux array of model perturb by the galaxy's 1 sigma errors
        **
        self.mfl - output flux array of model generated to fit against 
        """

        gal_wv, gal_fl, gal_er = np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0]
        self.gal_wv = self.gal_wv[self.gal_fl > 0]
        self.gal_er = self.gal_er[self.gal_fl > 0]
        self.gal_fl = self.gal_fl[self.gal_fl > 0]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

        ## create basis model for sim

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (self.metal, self.age, self.tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

        m2r = [3175, 3280, 3340, 3515, 3550, 3650, 3710, 3770, 3800, 3850,
               3910, 4030, 4080, 4125, 4250, 4385, 4515, 4570, 4810, 4910, 4975, 5055, 5110, 5285]

        Mask = np.zeros(len(self.gal_wv_rf))
        for i in range(len(Mask)):
            if m2r[0] <= self.gal_wv_rf[i] <= m2r[1]:
                Mask[i] = 1
            if m2r[2] <= self.gal_wv_rf[i] <= m2r[3]:
                Mask[i] = 1
            if m2r[4] <= self.gal_wv_rf[i] <= m2r[5]:
                Mask[i] = 1
            if m2r[6] <= self.gal_wv_rf[i] <= m2r[7]:
                Mask[i] = 1
            if m2r[8] <= self.gal_wv_rf[i] <= m2r[9]:
                Mask[i] = 1
            if m2r[8] <= self.gal_wv_rf[i] <= m2r[9]:
                Mask[i] = 1
            if m2r[10] < self.gal_wv_rf[i] <= m2r[11]:
                Mask[i] = 1
            if m2r[12] <= self.gal_wv_rf[i] <= m2r[13]:
                Mask[i] = 1
            if m2r[14] <= self.gal_wv_rf[i] <= m2r[15]:
                Mask[i] = 1
            if m2r[16] <= self.gal_wv_rf[i] <= m2r[17]:
                Mask[i] = 1
            if m2r[18] <= self.gal_wv_rf[i] <= m2r[19]:
                Mask[i] = 1
            if m2r[20] <= self.gal_wv_rf[i] <= m2r[21]:
                Mask[i] = 1
            if m2r[22] <= self.gal_wv_rf[i] <= m2r[23]:
                Mask[i] = 1

        self.maskw = np.ma.masked_array(self.gal_wv_rf, Mask)

        params = np.ma.polyfit(self.maskw, self.fl, 3, w=1 / self.gal_er ** 2)
        C0 = np.polyval(params,self.gal_wv_rf)

        self.nc_fl = self.fl / C0
        self.nc_er = self.gal_er / C0
        self.C = C0


    def Perturb_flux(self):
        self.flx_err = np.abs(self.fl + np.random.normal(0, self.gal_er))


    def Perturb_flux_nc(self):
        self.nc_flx_err = np.abs(self.nc_fl + np.random.normal(0, self.nc_er))


    def Perturb_both(self):
        one_sig_pert = np.random.normal(0, np.ones(len(self.gal_er)))
        self.flx_err = np.abs(self.fl + one_sig_pert * self.gal_er)
        self.nc_flx_err = np.abs(self.nc_fl + one_sig_pert * self.nc_er)


    def Sim_spec(self, metal, age, tau):
        import pysynphot as S

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.mfl = C * adj_ifl


    def RM_sim_spec_cont(self):
        params = np.ma.polyfit(self.maskw, self.mfl, 3)
        C0 = np.polyval(params,self.gal_wv_rf)

        self.nc_mfl = self.mfl / C0



def MC_fit_methods(galaxy, metal, age, tau, sim_m, sim_a, sim_t, specz, name, minwv=7900, maxwv=11300, repeats=100,
           age_conv='../data/tau_scale_ntau.dat'):
    mlist = []
    alist = []

    mlistnc = []
    alistnc = []

    mlistdf = []
    alistdf = []

    mlist_mn = []
    alist_mn = []

    mlistnc_mn = []
    alistnc_mn = []

    mlistdf_mn = []
    alistdf_mn = []

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    iZ = np.linspace(metal[0], metal[-1], 100)
    it = np.linspace(age[0], age[-1], 100)
    spec = Gen_sim(galaxy, specz, sim_m, sim_a, sim_t,minwv=minwv,maxwv=maxwv)

    ###############Get indicies
    IDF = []
    for i in range(len(spec.gal_wv_rf)):
        if 3800 <= spec.gal_wv_rf[i] <= 3850 or 3910 <= spec.gal_wv_rf[i] <= 4030 or 4080 <= spec.gal_wv_rf[i] <= 4125 \
                or 4250 <= spec.gal_wv_rf[i] <= 4385 or 4515 <= spec.gal_wv_rf[i] <= 4570 or 4810 <= spec.gal_wv_rf[i] \
                <= 4910 or 4975 <= spec.gal_wv_rf[i] <= 5055 or 5110 <= spec.gal_wv_rf[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(spec.gal_wv_rf)):
        if spec.gal_wv_rf[0] <= spec.gal_wv_rf[i] <= 3800 or 3850 <= spec.gal_wv_rf[i] <= 3910 or 4030 <= \
                spec.gal_wv_rf[i] <= 4080 or 4125 <= spec.gal_wv_rf[i] <= 4250 or 4385 <= spec.gal_wv_rf[i] <= 4515 or \
                                4570 <= spec.gal_wv_rf[i] <= 4810 or 4910 <= spec.gal_wv_rf[i] <= 4975 or 5055 <= \
                spec.gal_wv_rf[i] <= \
                5110 or 5285 <= spec.gal_wv_rf[i] <= spec.gal_wv_rf[-1]:
            IDC.append(i)

    ###############Get model list
    mfl = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    mfl_nc = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    mfl_f = np.zeros([len(metal) * len(age) * len(tau), len(IDF)])
    mfl_c = np.zeros([len(metal) * len(age) * len(tau), len(IDC)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl
                mfl_f[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDF]
                mfl_c[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDC]
                spec.RM_sim_spec_cont()
                mfl_nc[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.nc_mfl


    convtau = np.array([0, 8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2,
                        9.23, 9.26, 9.28, 9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48])
    convage = np.arange(.5, 6.1, .1)

    mt = [U for U in range(len(convtau)) if convtau[U] in tau]
    ma = [U for U in range(len(convage)) if np.round(convage[U], 1) in np.round(age, 1)]

    convtable = Readfile(age_conv)
    scale = convtable[mt[0]:mt[-1] + 1, ma[0]:ma[-1] + 1]

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for xx in range(repeats):
        spec.Perturb_flux()
        spec.Perturb_flux_nc()
        chi = np.sum(((spec.flx_err - mfl) / spec.gal_er) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        NCchi = np.sum(((spec.nc_flx_err - mfl_nc) / spec.nc_er) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        Fchi = np.sum(((spec.flx_err[IDF] - mfl_f) / spec.gal_er[IDF]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        Cchi = np.sum(((spec.flx_err[IDC] - mfl_c) / spec.gal_er[IDC]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T

        ######## Reshape likelihood to get average age instead of age when marginalized
        newchi = np.zeros(chi.shape)
        newNCchi = np.zeros(NCchi.shape)
        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newchi[i] = chi[i]
                newNCchi[i] = NCchi[i]
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                frame = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
                newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                ncframe = interp2d(metal, scale[i], NCchi[i])(metal, age[:-overhead[i]])
                newNCchi[i] = np.append(ncframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        prob = np.exp(-newchi.T.astype(np.float128) / 2)
        ncprob = np.exp(-newNCchi.T.astype(np.float128) / 2)
        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)
        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        P = np.trapz(prob, ultau, axis=2)
        C = np.trapz(np.trapz(P, age, axis=1), metal)

        Pnc = np.trapz(ncprob, ultau, axis=2)
        Cnc = np.trapz(np.trapz(Pnc, age, axis=1), metal)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

        ########

        comb_prob = cprob / Cc * fprob / Cf

        df_post = np.trapz(comb_prob, ultau, axis=2)
        C0 = np.trapz(np.trapz(df_post, age, axis=1), metal)
        df_post /= C0

        #### Get Z and t posteriors
        PZ = np.trapz(P / C, age, axis=1)
        Pt = np.trapz(P.T / C, metal, axis=1)

        PZnc = np.trapz(Pnc / Cnc, age, axis=1)
        Ptnc = np.trapz(Pnc.T / Cnc, metal, axis=1)

        PZdf = np.trapz(df_post, age, axis=1)
        Ptdf = np.trapz(df_post.T, metal, axis=1)

        iPZ = interp1d(metal, PZ)(iZ)
        iPt = interp1d(age, Pt)(it)

        iPZnc = interp1d(metal, PZnc)(iZ)
        iPtnc = interp1d(age, Ptnc)(it)

        iPZdf = interp1d(metal, PZdf)(iZ)
        iPtdf = interp1d(age, Ptdf)(it)

        med = 0
        mednc = 0
        meddf = 0
        for i in range(len(iZ)):
            e = np.trapz(iPZ[0:i + 1], iZ[0:i + 1])
            enc = np.trapz(iPZnc[0:i + 1], iZ[0:i + 1])
            edf = np.trapz(iPZdf[0:i + 1], iZ[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = iZ[i]
            if mednc == 0:
                if enc >= .5:
                    mednc = iZ[i]
            if meddf == 0:
                if edf >= .5:
                    meddf = iZ[i]

        mlist.append(med)
        mlistnc.append(mednc)
        mlistdf.append(meddf)

        med = 0
        mednc = 0
        meddf = 0
        for i in range(len(it)):
            e = np.trapz(iPt[0:i + 1], it[0:i + 1])
            enc = np.trapz(iPtnc[0:i + 1], it[0:i + 1])
            edf = np.trapz(iPtdf[0:i + 1], it[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = it[i]
            if mednc == 0:
                if enc >= .5:
                    mednc = it[i]
            if meddf == 0:
                if edf >= .5:
                    meddf = it[i]

        alist.append(med)
        alistnc.append(mednc)
        alistdf.append(meddf)

        mlist_mn.append(np.trapz(PZ*metal,metal))
        mlistnc_mn.append(np.trapz(PZnc*metal,metal))
        mlistdf_mn.append(np.trapz(PZdf*metal,metal))

        alist_mn.append(np.trapz(Pt*age,age))
        alistnc_mn.append(np.trapz(Ptnc*age,age))
        alistdf_mn.append(np.trapz(Ptdf*age,age))


    np.save('../mcerr/' + name, [mlist, alist])
    np.save('../mcerr/' + name + 'NC', [mlistnc, alistnc])
    np.save('../mcerr/' + name + 'DF', [mlistdf, alistdf])
    np.save('../mcerr/' + name + 'mean', [mlist_mn, alist_mn])
    np.save('../mcerr/' + name + 'NCmean', [mlistnc_mn, alistnc_mn])
    np.save('../mcerr/' + name + 'DFmean', [mlistdf_mn, alistdf_mn])

    return


def MC_fit(galaxy, metal, age, tau, sim_m, sim_a, sim_t, specz, name, repeats=100,
           age_conv='../data/tau_scale_ntau.dat'):
    mlist = []
    alist = []

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    iZ = np.linspace(metal[0], metal[-1], 100)
    it = np.linspace(age[0], age[-1], 100)
    spec = Gen_sim(galaxy, specz, sim_m, sim_a, sim_t)

    ###############Get indicies
    IDF = []
    for i in range(len(spec.gal_wv_rf)):
        if 3800 <= spec.gal_wv_rf[i] <= 3850 or 3910 <= spec.gal_wv_rf[i] <= 4030 or 4080 <= spec.gal_wv_rf[i] <= 4125 \
                or 4250 <= spec.gal_wv_rf[i] <= 4385 or 4515 <= spec.gal_wv_rf[i] <= 4570 or 4810 <= spec.gal_wv_rf[i] \
                <= 4910 or 4975 <= spec.gal_wv_rf[i] <= 5055 or 5110 <= spec.gal_wv_rf[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(spec.gal_wv_rf)):
        if spec.gal_wv_rf[0] <= spec.gal_wv_rf[i] <= 3800 or 3850 <= spec.gal_wv_rf[i] <= 3910 or 4030 <= \
                spec.gal_wv_rf[i] <= 4080 or 4125 <= spec.gal_wv_rf[i] <= 4250 or 4385 <= spec.gal_wv_rf[i] <= 4515 or \
                                4570 <= spec.gal_wv_rf[i] <= 4810 or 4910 <= spec.gal_wv_rf[i] <= 4975 or 5055 <= \
                spec.gal_wv_rf[i] <= \
                5110 or 5285 <= spec.gal_wv_rf[i] <= spec.gal_wv_rf[-1]:
            IDC.append(i)

    ###############Get model list
    mfl_f = np.zeros([len(metal) * len(age) * len(tau), len(IDF)])
    mfl_c = np.zeros([len(metal) * len(age) * len(tau), len(IDC)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl_f[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDF]
                mfl_c[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDC]

    convtau = np.array([0, 8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2,
                        9.23, 9.26, 9.28, 9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48])
    convage = np.arange(.5, 6.1, .1)

    mt = [U for U in range(len(convtau)) if convtau[U] in tau]
    ma = [U for U in range(len(convage)) if np.round(convage[U], 1) in np.round(age, 1)]

    convtable = Readfile(age_conv)
    scale = convtable[mt[0]:mt[-1] + 1, ma[0]:ma[-1] + 1]

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for xx in range(repeats):
        spec.Perturb_flux()

        Fchi = np.sum(((spec.flx_err[IDF] - mfl_f) / spec.gal_er[IDF]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        Cchi = np.sum(((spec.flx_err[IDC] - mfl_c) / spec.gal_er[IDC]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T

        ######## Reshape likelihood to get average age instead of age when marginalized
        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                #print scale[i]
                cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
                #print cframe.shape
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
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

        #### Get Z and t posteriors
        PZ = np.trapz(prob, age, axis=1)
        Pt = np.trapz(prob.T, metal, axis=1)

        iPZ = interp1d(metal, PZ)(iZ)
        iPt = interp1d(age, Pt)(it)

        med = 0
        for i in range(len(iZ)):
            e = np.trapz(iPZ[0:i + 1], iZ[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = iZ[i]
                    break
        mlist.append(med)

        med = 0
        for i in range(len(it)):
            e = np.trapz(iPt[0:i + 1], it[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = it[i]
                    break
        alist.append(med)

    np.save('../mcerr/' + name, [mlist, alist])

    return



"""Test Functions"""


def Best_fit_model(input_file, metal, age, tau):
    dat = fits.open(input_file)

    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    x = np.argwhere(chi == np.min(chi))
    print(metal[x[0][0]], age[x[0][1]], tau[x[0][2]])
    return metal[x[0][0]], age[x[0][1]], tau[x[0][2]]


def Stack_gal_spec(spec, wv, mregion):
    flgrid = np.zeros([len(spec), len(wv)])
    errgrid = np.zeros([len(spec), len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Get_flux(spec[i]))
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl = interp1d(wave, flux)(wv[mask])
        ier = interp1d(wave, error)(wv[mask])
        if sum(mregion[i]) > 0:
            # flmask=np.array([mregion[i][0] < U < mregion[i][1] for U in wv[mask]])
            for ii in range(len(wv[mask])):
                if mregion[i][0] < wv[mask][ii] < mregion[i][1]:
                    ifl[ii] = 0
                    ier[ii] = 0
                    # flgrid[i][mask]=ifl[flmask]
                    # errgrid[i][mask]=ier[flmask]
        # else:
        flgrid[i][mask] = ifl
        errgrid[i][mask] = ier
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

    return wv, stack, err


def B_factor(input_chi_file, tau, metal, age):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    dat = fits.open(input_chi_file)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

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

    return C


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
    def __init__(self, speclist, redshifts, wv_range, norm_range):
        self.speclist = speclist
        self.redshifts = redshifts
        self.wv_range = wv_range
        self.norm_range = norm_range

    def Stack_normwmean(self):
        flgrid = np.zeros([len(self.speclist), len(self.wv_range)])
        errgrid = np.zeros([len(self.speclist), len(self.wv_range)])
        for i in range(len(self.speclist)):
            spec = Gen_spec(self.speclist[i], self.redshifts[i])

            if self.speclist[i] == 'n21156' or self.speclist[i] == 's39170' or self.speclist[i] == 'n34694':
                IDer = []
                for ii in range(len(spec.gal_wv_rf)):
                    if 4855 <= spec.gal_wv_rf[ii] <= 4880 :
                        IDer.append(ii)
                spec.gal_er[IDer] = 1E8
                spec.gal_fl[IDer] = 0

            mask = np.array([spec.gal_wv_rf[0] < U < spec.gal_wv_rf[-1] for U in self.wv_range])
            ifl = interp1d(spec.gal_wv_rf, spec.gal_fl)
            ier = interp1d(spec.gal_wv_rf, spec.gal_er)
            Cr = np.trapz(ifl(self.norm_range), self.norm_range)
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

        self.wv = self.wv_range[IDX]
        self.fl = stack[IDX]
        self.er = err[IDX]

    def Stack_normwmean_model(self, bfmetal, bfage, tau, bftau = None):
        self.bfmetal = bfmetal
        self.bfage = bfage

        if bftau == None:
            self.Highest_likelihood_model_mlist(self.bfmetal,bfage,tau)
        else:
            self.bftau = bftau

        mwv, mfl = Stack_model_normwmean(self.speclist, self.redshifts, self.bfmetal, self.bfage,
                                         self.bftau, self.wv, self.norm_range)

        self.mwv = mwv
        self.mfl = mfl

    def Highest_likelihood_model_mlist(self, bfmetal, bfage, tau):

        chi = []
        for i in range(len(tau)):
            mwv, mfl = Stack_model_normwmean(self.speclist, self.redshifts, bfmetal, bfage,
                                             tau[i], self.wv, self.norm_range)
            chi.append(Identify_stack(self.fl, self.er, mfl))

        print([bfmetal, bfage, tau[np.argmin(chi)]])

        self.bftau = tau[np.argmin(chi)]


def MC_fit_methods_test_2(galaxy, metal, age, tau, sim_m, sim_a, sim_t, specz, minwv=7900, maxwv=11400, repeats=1,
           age_conv='../data/tau_scale_ntau.dat'):

    bfm=np.zeros(repeats)
    bfmnc=np.zeros(repeats)
    bfmdf=np.zeros(repeats)
    bfmdf2d=np.zeros(repeats)
    bfmdf1d=np.zeros(repeats)
    bfa=np.zeros(repeats)
    bfanc=np.zeros(repeats)
    bfadf=np.zeros(repeats)
    bfadf2d=np.zeros(repeats)
    bfadf1d=np.zeros(repeats)


    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    spec = Gen_sim(galaxy, specz, sim_m, sim_a, sim_t,minwv=minwv,maxwv=maxwv)

    ###############Get indicies
    IDF = []
    for i in range(len(spec.gal_wv_rf)):
        if 3800 <= spec.gal_wv_rf[i] <= 3850 or 3910 <= spec.gal_wv_rf[i] <= 4030 or 4080 <= spec.gal_wv_rf[i] <= 4125 \
                or 4250 <= spec.gal_wv_rf[i] <= 4385 or 4515 <= spec.gal_wv_rf[i] <= 4570 or 4810 <= spec.gal_wv_rf[i]\
                <= 4910 or 4975 <= spec.gal_wv_rf[i] <= 5055 or 5110 <= spec.gal_wv_rf[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(spec.gal_wv_rf)):
        if spec.gal_wv_rf[0] <= spec.gal_wv_rf[i] <= 3800 or 3850 <= spec.gal_wv_rf[i] <= 3910 or 4030 <= \
                spec.gal_wv_rf[i] <= 4080 or 4125 <= spec.gal_wv_rf[i] <= 4250 or 4385 <= spec.gal_wv_rf[i] <= 4515 or \
                4570 <= spec.gal_wv_rf[i] <= 4810 or 4910 <= spec.gal_wv_rf[i] <= 4975 or 5055 <= spec.gal_wv_rf[i] <= \
                5110 or 5285 <= spec.gal_wv_rf[i] <= spec.gal_wv_rf[-1]:
            IDC.append(i)

    ###############Get model list
    mfl = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    mfl_nc = np.zeros([len(metal) * len(age) * len(tau), len(spec.gal_wv_rf)])
    mfl_f = np.zeros([len(metal) * len(age) * len(tau), len(IDF)])
    mfl_c = np.zeros([len(metal) * len(age) * len(tau), len(IDC)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl
                mfl_f[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDF]
                mfl_c[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.mfl[IDC]
                spec.RM_sim_spec_cont()
                mfl_nc[i * len(age) * len(tau) + ii * len(tau) + iii] = spec.nc_mfl


    convtau = np.array([0, 8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2,
                        9.23, 9.26, 9.28, 9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48])
    convage = np.arange(.5, 6.1, .1)

    mt = [U for U in range(len(convtau)) if convtau[U] in tau]
    ma = [U for U in range(len(convage)) if np.round(convage[U], 1) in np.round(age, 1)]

    convtable = Readfile(age_conv)
    scale = convtable[mt[0]:mt[-1] + 1, ma[0]:ma[-1] + 1]

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for xx in range(repeats):
        spec.Perturb_flux()
        spec.Perturb_flux_nc()
        chi = np.sum(((spec.flx_err - mfl) / spec.gal_er) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        NCchi = np.sum(((spec.nc_flx_err - mfl_nc) / spec.nc_er) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        Fchi = np.sum(((spec.flx_err[IDF] - mfl_f) / spec.gal_er[IDF]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T
        Cchi = np.sum(((spec.flx_err[IDC] - mfl_c) / spec.gal_er[IDC]) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau)]).astype(
            np.float128).T

        ######## Reshape likelihood to get average age instead of age when marginalized
        newchi = np.zeros(chi.shape)
        newNCchi = np.zeros(NCchi.shape)
        newCchi = np.zeros(Cchi.shape)
        newFchi = np.zeros(Fchi.shape)

        for i in range(len(Cchi)):
            if i == 0:
                newchi[i] = chi[i]
                newNCchi[i] = NCchi[i]
                newCchi[i] = Cchi[i]
                newFchi[i] = Fchi[i]
            else:
                frame = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
                newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                ncframe = interp2d(metal, scale[i], NCchi[i])(metal, age[:-overhead[i]])
                newNCchi[i] = np.append(ncframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
                newCchi[i] = np.append(cframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

                fframe = interp2d(metal, scale[i], Fchi[i])(metal, age[:-overhead[i]])
                newFchi[i] = np.append(fframe, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        prob = np.exp(-newchi.T.astype(np.float128) / 2)
        ncprob = np.exp(-newNCchi.T.astype(np.float128) / 2)
        cprob = np.exp(-newCchi.T.astype(np.float128) / 2)
        fprob = np.exp(-newFchi.T.astype(np.float128) / 2)

        P = np.trapz(prob, ultau, axis=2)
        C = np.trapz(np.trapz(P, age, axis=1), metal)

        Pnc = np.trapz(ncprob, ultau, axis=2)
        Cnc = np.trapz(np.trapz(Pnc, age, axis=1), metal)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)
        Pcm = np.trapz(Pc, age, axis=1) / Cc
        Pca = np.trapz(Pc.T, metal, axis=1)/ Cc

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)
        Pfm = np.trapz(Pf, age, axis=1)/ Cf
        Pfa = np.trapz(Pf.T, metal, axis=1)/ Cf
        ########

        comb_prob = cprob / Cc * fprob / Cf

        df_post = np.trapz(comb_prob, ultau, axis=2)
        C0 = np.trapz(np.trapz(df_post, age, axis=1), metal)
        df_post /= C0

        comb_prob2d = Pc / Cc * Pf / Cf
        C2d = np.trapz(np.trapz(comb_prob2d, age, axis=1), metal)
        comb_prob2d /= C2d

        #### Get Z and t posteriors
        PZ = np.trapz(P / C, age, axis=1)
        Pt = np.trapz(P.T / C, metal, axis=1)

        PZnc = np.trapz(Pnc / Cnc, age, axis=1)
        Ptnc = np.trapz(Pnc.T / Cnc, metal, axis=1)

        PZdf = np.trapz(df_post, age, axis=1)
        Ptdf = np.trapz(df_post.T, metal, axis=1)

        PZdf2d = np.trapz(comb_prob2d, age, axis=1)
        Ptdf2d = np.trapz(comb_prob2d.T, metal, axis=1)

        PZdf1d = (Pfm * Pcm) / np.trapz((Pfm * Pcm),metal)
        Ptdf1d = (Pfa * Pca) / np.trapz((Pfa * Pca),age)

        bfm[xx],ml,mh = Median_w_Error_cont(PZ,metal)
        bfmnc[xx],ml,mh = Median_w_Error_cont(PZnc,metal)
        bfmdf[xx],ml,mh = Median_w_Error_cont(PZdf,metal)
        bfmdf2d[xx],ml,mh = Median_w_Error_cont(PZdf2d,metal)
        bfmdf1d[xx],ml,mh = Median_w_Error_cont(PZdf1d,metal)
        bfa[xx],ml,mh = Median_w_Error_cont(Pt,age)
        bfanc[xx],ml,mh = Median_w_Error_cont(Ptnc,age)
        bfadf[xx],ml,mh = Median_w_Error_cont(Ptdf,age)
        bfadf2d[xx],ml,mh = Median_w_Error_cont(Ptdf2d,age)
        bfadf1d[xx],ml,mh = Median_w_Error_cont(Ptdf1d,age)

    print(np.median(bfm))
    print(np.median(bfmdf))
    print(np.median(bfmdf2d))
    print(np.median(bfmdf1d))
    print(np.median(bfa))
    print(np.median(bfadf))
    print(np.median(bfadf2d))
    print(np.median(bfadf1d))

    plt.figure()
    plt.subplot(121)
    sea.distplot(bfm)
    sea.distplot(bfmnc)
    sea.distplot(bfmdf)
    sea.distplot(bfmdf2d)
    sea.distplot(bfmdf1d)
    plt.axvline(sim_m, linestyle = '--')

    plt.subplot(122)
    sea.distplot(bfa)
    sea.distplot(bfanc)
    sea.distplot(bfadf)
    sea.distplot(bfadf2d)
    sea.distplot(bfadf1d)
    plt.axvline(sim_a, linestyle = '--')

    plt.show()

    return

#####JWST FIT

def Analyze_JWST_LH(chifits, specz, metal, age, tau, age_conv='../data/tau_scale_nirspec.dat'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    scale = Readfile(age_conv)

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        if i == 0:
            newchi[i] = chi[i]
        else:
            frame = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt


def Nirspec_fit(sim_spec, filters, metal, age, tau, name):
    #############Read in spectra#################
    wv, fl, er = np.load(sim_spec)

    flx = np.zeros(len(wv))

    for i in range(len(wv)):
        if er[i] > 0:
            flx[i] = fl[i] + np.random.normal(0, er[i])
    #############Prep output files###############
    chifile = '../chidat/%s_JWST_chidata' % name

    ##############Create chigrid and add to file#################
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(wv)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../JWST/m%s_a%s_t%s_%s.npy' %
                                   (metal[i], age[ii], tau[iii],filters))
                mfl *=(mwv)**2 / 3E14
                C = Scale_model(flx,er,mfl)
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=mfl*C
    chigrid = np.sum(((flx - mflx) / er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile, chigrid)

    P, PZ, Pt = Analyze_JWST_LH(chifile + '.npy', 3.717, metal, age, tau)

    np.save('../chidat/%s_tZ_pos' % name,P)
    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('../chidat/%s_t_pos' % name,[age,Pt])
    np.save('../data/nirspec_sim_data_%s' % filters,[wv,fl,flx,er])

    print('Done!')
    return


def Nirspec_feat_fit(sim_spec, z, filters, metal, age, tau, name, prism = True):
    #############Read in spectra#################
    wv, fl, er = np.load(sim_spec)
    IDF = []

    if prism == True:
        for i in range(len(wv)):
            if 0.3910 <= wv[i] / (1 + z) <= 0.4030 or 0.4050 <= wv[i] / (1 + z) <= 0.4180 or 0.4220 <= wv[i] / (1 + z)\
                <= 0.4480 or 0.4750 <= wv[i] / (1 + z) <= 0.5000 or 0.5090 <= wv[i] / (1 + z) <= 0.5280 or \
                0.5800 <= wv[i] / (1 + z) <= 0.5970 or 0.6460 <= wv[i] / (1 + z) <= 0.6670 or 0.8380 <= wv[i] / (1 + z)\
                <= 0.8750:
                IDF.append(i)

    else:
        for i in range(len(wv)):
            if 0.4845 <= wv[i] / (1 + z) <= 0.4885 or 0.5160 <= wv[i] / (1 + z) <= 0.5180 or 0.5255 <= wv[i] / (1 + z) \
                    <= 0.5280 or 0.5695 <= wv[i] / (1 + z) <= 0.5725 or 0.5875 <= wv[i] / (1 + z) <= 0.5910 or \
                0.6540 <= wv[i] / (1 + z)<= 0.6590 or 0.8480 <= wv[i] / (1 + z) <= 0.8580 or 0.8630 <= wv[i] / (1 + z) \
                    <= 0.8710:
                IDF.append(i)

    flx = np.zeros(len(wv))

    for i in range(len(wv)):
        if er[i] > 0:
            flx[i] = fl[i] + np.random.normal(0, er[i])
    #############Prep output files###############
    chifile = '../chidat/%s_JWST_chidata' % name

    ##############Create chigrid and add to file#################
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(IDF)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../JWST/m%s_a%s_t%s_%s.npy' %
                                   (metal[i], age[ii], tau[iii],filters))
                mfl *=(mwv)**2 / 3E14
                C = Scale_model(flx[IDF],er[IDF],mfl[IDF])
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=mfl[IDF]*C
    chigrid = np.sum(((flx[IDF] - mflx) / er[IDF]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile, chigrid)

    P, PZ, Pt = Analyze_JWST_LH(chifile + '.npy', 3.717, metal, age, tau)

    np.save('../chidat/%s_tZ_pos' % name,P)
    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('../chidat/%s_t_pos' % name,[age,Pt])
    np.save('../data/nirspec_sim_data_%s_ff' % filters,[wv,fl,flx,er])

    print('Done!')
    return


def Nirspec_feat_fit_freescale(sim_spec, z, filters, metal, age, tau, name, prism = True):
    #############Read in spectra#################
    wv, fl, er = np.load(sim_spec)
    IDF = []

    if prism == True:
        for i in range(len(wv)):
            if 0.3910 <= wv[i] / (1 + z) <= 0.4030 or 0.4050 <= wv[i] / (1 + z) <= 0.4180 or 0.4220 <= wv[i] / (1 + z)\
                <= 0.4480 or 0.4750 <= wv[i] / (1 + z) <= 0.5000 or 0.5090 <= wv[i] / (1 + z) <= 0.5280 or \
                0.5800 <= wv[i] / (1 + z) <= 0.5970 or 0.6460 <= wv[i] / (1 + z) <= 0.6670 or 0.8380 <= wv[i] / (1 + z)\
                <= 0.8750:
                IDF.append(i)

    else:
        IDhb = []
        IDmg1 = []
        IDmg2 = []
        IDmg3 = []
        IDna = []
        IDha = []
        IDca1 = []
        IDca2 = []
        IDf =[]
        for i in range(len(wv)):
            if 0.4845 <= wv[i] / (1 + z) <= 0.4885:
                IDhb.append(i)
            if 0.5160 <= wv[i] / (1 + z) <= 0.5180:
                IDmg1.append(i)
            if 0.5255 <= wv[i] / (1 + z) <= 0.5280:
                IDmg2.append(i)
            if 0.5695 <= wv[i] / (1 + z) <= 0.5725:
                IDmg3.append(i)
            if 0.5875 <= wv[i] / (1 + z) <= 0.5910:
                IDna.append(i)
            if 0.6540 <= wv[i] / (1 + z) <= 0.6590:
                IDha.append(i)
            if 0.8480 <= wv[i] / (1 + z) <= 0.8580:
                IDca1.append(i)
            if 0.8630 <= wv[i] / (1 + z) <= 0.8710:
                IDca2.append(i)
        IDF=np.array([IDhb, IDmg1, IDmg2, IDmg3, IDna, IDha, IDca1, IDca2])

        for i in range(len(wv)):
            if 0.4845 <= wv[i] / (1 + z) <= 0.4885 or 0.5160 <= wv[i] / (1 + z) <= 0.5180 or 0.5255 <= wv[i] / (1 + z) \
                    <= 0.5280 or 0.5695 <= wv[i] / (1 + z) <= 0.5725 or 0.5875 <= wv[i] / (1 + z) <= 0.5910 or \
                0.6540 <= wv[i] / (1 + z)<= 0.6590 or 0.8480 <= wv[i] / (1 + z) <= 0.8580 or 0.8630 <= wv[i] / (1 + z) \
                    <= 0.8710:
                IDf.append(i)


    flx = np.zeros(len(wv))

    for i in range(len(wv)):
        if er[i] > 0:
            flx[i] = fl[i] + np.random.normal(0, er[i])
    #############Prep output files###############
    chifile = '../chidat/%s_JWST_chidata' % name

    ##############Create chigrid and add to file#################
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(IDf)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../JWST/m%s_a%s_t%s_%s.npy' %
                                   (metal[i], age[ii], tau[iii],filters))
                mfl *=(mwv)**2 / 3E14
                Fmfl = np.array([])
                for iv in range(len(IDF)):
                    C = Scale_model(flx[IDF[iv]],er[IDF[iv]],mfl[IDF[iv]])
                    Fmfl = np.append(Fmfl,C*mfl[IDF[iv]])
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=Fmfl
    chigrid = np.sum(((flx[IDf] - mflx) / er[IDf]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile, chigrid)

    P, PZ, Pt = Analyze_JWST_LH(chifile + '.npy', 3.717, metal, age, tau)

    np.save('../chidat/%s_tZ_pos' % name,P)
    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('../chidat/%s_t_pos' % name,[age,Pt])
    np.save('../data/nirspec_sim_data_%s' % name,[wv,fl,flx,er])

    print('Done!')
    return


def Highest_likelihood_model_JWST(spec, filters, bfmetal, bfage, tau):
    wv, fl, flx, er = np.load(spec)
    fp = '../JWST/'

    chi = []
    for i in range(len(tau)):
        mwv, mfl = np.load(fp + 'm%s_a%s_t%s_%s.npy' % (bfmetal, bfage, tau[i], filters))
        mfl *= (mwv) ** 2 / 3E14
        C = Scale_model(flx, er, mfl)
        chi.append(Identify_stack(fl, er, C * mfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def MC_fit_jwst(sim_spec, metal, age, tau, name, repeats=100, age_conv='../data/tau_scale_nirspec.dat'):
    ####### Get maximum age
    max_age = Oldest_galaxy(3.717)

    mlist = []
    alist = []

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    iZ = np.linspace(metal[0], metal[-1], 100)
    it = np.linspace(age[0], age[-1], 100)

    wv, fl, er = np.load(sim_spec)
    fl = fl [wv<4.9]
    er = er [wv<4.9]

    ###############Get model list
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(wv[wv<4.9])])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../JWST/m%s_a%s_t%s_nirspec.npy' %
                                   (metal[i], age[ii], tau[iii]))
                C = Scale_model(fl,er,mfl[wv<4.9])
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=mfl[wv<4.9]*C

    scale = Readfile(age_conv)

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for xx in range(repeats):
        flx = fl + np.random.normal(0, er)

        chi = np.sum(((flx - mflx) / er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

        chi[:, len(age[age <= max_age]):, :] = 1E5
        ######## Reshape likelihood to get average age instead of age when marginalized
        newchi = np.zeros(chi.T.shape)

        for i in range(len(newchi)):
            if i == 0:
                newchi[i] = chi.T[i]
            else:
                frame = interp2d(metal, scale[i], chi.T[i])(metal, age[:-overhead[i]])
                newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        prob = np.exp(-newchi.T.astype(np.float128) / 2)

        P = np.trapz(prob, ultau, axis=2)
        C = np.trapz(np.trapz(P, age, axis=1), metal)

        #### Get Z and t posteriors
        PZ = np.trapz(P / C, age, axis=1)
        Pt = np.trapz(P.T / C, metal, axis=1)

        iPZ = interp1d(metal, PZ)(iZ)
        iPt = interp1d(age, Pt)(it)

        med = 0
        for i in range(len(iZ)):
            e = np.trapz(iPZ[0:i + 1], iZ[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = iZ[i]

        mlist.append(med)

        med = 0
        for i in range(len(it)):
            e = np.trapz(iPt[0:i + 1], it[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = it[i]

        alist.append(med)

    np.save('../mcerr/' + name, [mlist, alist])

    return


#####STATS

def Leave_one_out(dist, x):
    Y = np.zeros(x.size)
    for i in range(len(dist)):
        Y += dist[i]
    Y /= np.trapz(Y, x)

    w = np.arange(.01, 2.01, .01)
    weights = np.zeros(len(dist))
    for i in range(len(dist)):
        Ybar = np.zeros(x.size)
        for ii in range(len(dist)):
            if i != ii:
                Ybar += dist[ii]
        Ybar /= np.trapz(Ybar, x)
        weights[i] = np.sum((Ybar - Y) ** 2) ** -1
    return weights

def Stack_posteriors(P_grid, x):
    P_grid = np.array(P_grid)
    W = Leave_one_out(P_grid,x)
    top = np.zeros(P_grid.shape)
    for i in range(W.size):
        top[i] = W[i] * P_grid[i]
    P =sum(top)/sum(W)
    return P / np.trapz(P,x)

def Iterative_stacking(grid_o,x_o,iterations = 20,resampling = 250):
    ksmooth = importr('KernSmooth')
    del_x = x_o[1] - x_o[0]

    ### resample
    x = np.linspace(x_o[0],x_o[-1],resampling)
    grid = np.zeros([len(grid_o),x.size])    
    for i in range(len(grid_o)):
        grid[i] = interp1d(x_o,grid_o[i])(x)
   
    ### select bandwidth
    H = ksmooth.dpik(x)
    ### stack posteriors w/ weights
    stkpos = Stack_posteriors(grid,x)
    ### initialize prior as flat
    Fx = np.ones(stkpos.size)
    
    for i in range(iterations):
        fnew = Fx * stkpos / np.trapz(Fx * stkpos,x)
        fx = ksmooth.locpoly(x,fnew,bandwidth = H)
        X = np.array(fx[0])
        iFX = np.array(fx[1])
        Fx = interp1d(X,iFX)(x)

    Fx[Fx<0]=0
    Fx = Fx/np.trapz(Fx,x)
    return Fx,x

def Linear_fit(x,Y,sig,new_x,return_cov = False):
    A=np.array([np.ones(len(x)),x]).T
    C =np.diag(sig**2)
    iC=inv(C)
    b,m = np.dot(inv(np.dot(np.dot(A.T,iC),A)),np.dot(np.dot(A.T,iC),Y))
    cov = inv(np.dot(np.dot(A.T,iC),A))
    var_b = cov[0][0]
    var_m = cov[1][1]
    sig_mb = cov[0][1]
    sig_y = np.sqrt(var_b + new_x**2*var_m + 2*new_x*sig_mb)
    if return_cov == True:
        return m*new_x+b , sig_y, cov
    else:
        return m*new_x+b , sig_y
    
def Bootstrap_errors_lfit(masses,metals,ers,sampling=np.arange(10,11.75,.01),its=1000):
    l_grid = np.zeros([its,len(sampling)])
    IDs = np.arange(len(masses))
    for i in range(its):
        IDn = np.random.choice(IDs,len(IDs),replace=True)
        lvals = np.polyfit(masses[IDn],np.log10(metals[IDn]/.019),1,w = 1/ers[IDn]**2)
        lfit = np.polyval(lvals,sampling)
        l_grid[i] = lfit
        
    m_fit = np.mean(l_grid,axis=0)
    low_ers = np.zeros(len(samp))
    hi_ers = np.zeros(len(samp))
    
    for i in range(len(l_grid.T)):
        low_ers[i] = np.sort(l_grid.T[i])[150]
        hi_ers[i] = np.sort(l_grid.T[i])[830]
    return low_ers,hi_ers, m_fit

def Gen_grid(DB,param):
    grid=[]
    for i in DB.index:
        x,Px = np.load('../chidat/%s_dtau_%s_pos_lwa_3.npy' % (DB['gids'][i],param))
        grid.append(Px)
    return np.array(grid)

##test

def Mean_stack(spec_list):
    wv,fl,er = Get_flux(spec_list[0])
    wv = wv[2:-3]
    # Define grids used for stacking
    flgrid = np.zeros([len(spec_list), len(wv)])
    errgrid = np.zeros([len(spec_list), len(wv)])

    # Get wv,fl,er for each spectra
    for i in range(len(spec_list)):
        wave, flux, error = Get_flux(spec_list[i])
        flgrid[i] = interp1d(wave, flux)(wv)
        errgrid[i] = interp1d(wave, error)(wv)

    ################
    flgrid = np.transpose(flgrid)
    errgrid = np.transpose(errgrid)
    weigrid = errgrid ** (-2)
    infmask = np.isinf(weigrid)
    ################
    stack, err = np.zeros([2, len(wv)])
    for i in range(len(wv)):
        stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
        err[i] = 1 / np.sqrt(np.sum(weigrid[i]))     
    return wv, stack, err

def Median_stack(spec_list):
    wv,fl,er = Get_flux(spec_list[0])
    wv = wv[2:-3]
    # Define grids used for stacking
    flgrid = np.zeros([len(spec_list), len(wv)])

    # Get wv,fl,er for each spectra
    for i in range(len(spec_list)):
        wave, flux, error = Get_flux(spec_list[i])
        flgrid[i] = interp1d(wave, flux)(wv)
    ################
    flgrid = np.transpose(flgrid)
    ################
    stack = np.zeros(len(wv))
    for i in range(len(wv)):
        stack[i] = np.median(flgrid[i])
        
    l_grid = np.zeros([1000,len(wv)])
    IDs = np.arange(len(spec_list))
    for x in range(1000):
        IDn = np.random.choice(IDs,len(IDs),replace=True)
        bs_flgrid = np.zeros([len(spec_list), len(wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(spec_list)):
            bs_wave, bs_flux, bs_error = Get_flux(spec_list[IDn[i]])
            bs_flgrid[i] = interp1d(bs_wave, bs_flux)(wv)
        ################
        bs_flgrid = np.transpose(bs_flgrid)
        ################
        bs_stack = np.zeros(len(wv))
        for i in range(len(wv)):
            bs_stack[i] = np.median(bs_flgrid[i])
        l_grid[x] = bs_stack
    
    err = np.zeros(len(wv))
    for i in range(len(wv)):
        err[i] = np.std(l_grid.T[i])
    
    return wv, stack, err

"""Proposal fit 2D"""

class Gen_spec_2d(object):
    def __init__(self, stack_2d, stack_2d_error, grism_flt, direct_flt, redshift):
        self.stack_2d = stack_2d
        self.stack_2d_error = stack_2d_error
        self.grism = grism_flt
        self.direct = direct_flt
        self.redshift = redshift

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        self.gal = np.load(self.stack_2d)
        self.err = np.load(self.stack_2d_error)
        
        flt = grizli.model.GrismFLT(grism_file= self.grism, 
                                direct_file= self.direct,
                                pad=200, ref_file=None, ref_ext=0, 
                                seg_file='../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits',
                                shrink_segimage=False)

        ref_cat = Table.read('../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat', format='ascii')
        sim_cat = flt.blot_catalog(ref_cat, sextractor=False)

        id = 39170

        x0 = ref_cat['x'][39169]+1
        y0 = ref_cat['y'][39169]+1

        mag =-2.5*np.log10(ref_cat['f_F850LP']) + 25
        keep = mag < 22

        flt.compute_full_model(ids=ref_cat['id'][keep],verbose=False, 
                               mags=mag[keep])

        ### Get the beams/orders
        beam = flt.object_dispersers[id]['A'] # can choose other orders if available
        beam.compute_model()

        ### BeamCutout object
        self.co = grizli.model.BeamCutout(flt, beam, conf=flt.conf)

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)
   
        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
  
        self.model = self.co.beam.compute_model(spectrum_1d=[spec.wave, spec.flux], 
                                           in_place=False).reshape(self.co.beam.sh_beam)

        adjmodel = np.append(np.zeros([4,len(self.model)]),self.model.T[:-4], axis=0).T
        
        rs = self.gal.shape[0]*self.gal.shape[1]
        C = Scale_model(self.gal.reshape(rs),self.err.reshape(rs),adjmodel.reshape(rs))
    
        self.sim = adjmodel*C
        
def Analyze_2D(chifits, specz, metal, age, tau, age_conv='../data/light_weight_scaling.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E1

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        frame = np.zeros([metal.size, age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii], chi[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist, np.repeat(1E5, overhead[i][ii]))
        newchi[i] = frame.T


    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt
        
def Single_gal_fit_full_2d(metal, age, tau, specz,stack_2d, stack_2d_error, grism_flt, direct_flt , name):
    #############Read in spectra#################
    spec = Gen_spec_2d(stack_2d, stack_2d_error, grism_flt, direct_flt, specz)

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1 = '../chidat/%s_chidata' % name


    ##############Create chigrid and add to file#################
    chigrid1 = np.zeros([len(metal),len(age),len(tau)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                chigrid1[i][ii][iii] = np.sum(((spec.gal - spec.sim)/spec.err)**2)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)

    print('Done!')
    return

"""Proposal spec z"""

class Gen_spec_z(object):
    def __init__(self, spec_file, pad=100, minwv = 7900, maxwv = 11400):
        self.galaxy = spec_file
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """


        gal_wv, gal_fl, gal_er = np.load(self.galaxy)
        self.flt_input = '../data/galaxy_flts/n21156_flt.fits'

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + 1.251)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

    def Sim_spec(self, metal, age, redshift):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt8.0_spec.npy' % (metal, age)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl


def Specz_fit_2(spec_file, metal, age, rshift, name):
    #############initialize spectra#################
    spec = Gen_spec_z(spec_file)

    #############Prep output file###############
    chifile = '../rshift_dat/%s_z_fit' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(rshift),len(spec.gal_wv)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(rshift)):
                spec.Sim_spec(metal[i], age[ii], rshift[iii])
                mfl[i*len(age)*len(rshift)+ii*len(rshift)+iii]=spec.fl
    chigrid = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(rshift)]).\
        astype(np.float128)

    np.save(chifile,chigrid)
    ###############Write chigrid file###############
    Analyze_specz(chifile + '.npy', rshift, metal, age, name)

    print('Done!')

    return