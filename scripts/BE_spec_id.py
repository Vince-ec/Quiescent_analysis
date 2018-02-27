__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import grizli
import matplotlib.image as mpimg
from astropy.io import fits
from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import cPickle
import os
from glob import glob
from time import time
import seaborn as sea
import pysynphot as S

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Scale_model(D, sig, M):
    C = np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))
    return C

def Oldest_galaxy(z):
    return cosmo.age(z).value


class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, minwv = 7900, maxwv = 11400, pad=100, bc03 = False):
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.pad = pad
        self.bc03= bc03

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

        gal_wv, gal_fl, gal_er = \
            np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        if self.galaxy_id == 's35774':
            maxwv = 11100

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
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

    def Sim_spec(self, metal, age, tau):
        if self.bc03 == False:
            model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)
        else :
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

        self.fl = C * adj_ifl


def Single_gal_fit_full(metal, age, tau, specz, galaxy, name, minwv = 7900, maxwv = 11400, bc03 = False):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv, bc03 = bc03)

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
    chifile1 = '../BE/%s_chidata' % name

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

    if bc03 == False:
        P, PZ, Pt = Analyze_LH_lwa(chifile1 + '.npy', specz, metal, age, tau)
    else:
        P, PZ, Pt = Analyze_LH_lwa(chifile1 + '.npy', specz, metal, age, tau,
                                   age_conv='../BE/BC_age_conv.npy')

    np.save('../BE/%s_tZ_pos' % name,P)
    np.save('../BE/%s_Z_pos' % name,[metal,PZ])
    np.save('../BE/%s_t_pos' % name,[age,Pt])

    print 'Done!'
    return


def Analyze_LH_lwa(chifits, specz, metal, age, tau, age_conv='../BE/FS_age_conv.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = 10**(np.array(tau) - 9)

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

def B_evidence(chifits, specz, metal, age, tau, age_conv='../BE/FS_age_conv.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = 10**(np.array(tau) - 9)

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
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii],chi[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(1E5, overhead[i][ii]))
        newchi[i] = frame.T

    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    return C

def B_factor(chi_file1,chi_file2, specz, metal_1,metal_2, age, tau):
    FS = B_evidence(chi_file1, specz,metal_1,age,tau)
    BC = B_evidence(chi_file2, specz,metal_2,age,tau,age_conv='../BE/BC_age_conv.npy')

    BF = 2*np.log(FS/BC)
    
    return BF

def B_factor_2(chi_file1,chi_file2, specz, metal_1,metal_2, age, tau_1, tau_2):
    FS = B_evidence(chi_file1, specz,metal_1,age,tau_1,age_conv='../data/light_weight_scaling_3.npy')
    BC = B_evidence(chi_file2, specz,metal_2,age,tau_2,age_conv='../BE/BC_age_conv.npy')

    BF = 2*np.log(FS/BC)
    
    return BF