import pandas as pd
from astropy.io import fits
from spec_id import Analyze_LH_lwa,Scale_model
import numpy as np
import grizli
from scipy.interpolate import interp1d,interp2d

class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, fits_file, pad=100, delayed = True,minwv = 8000, maxwv = 11400):
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.pad = pad
        self.delayed = delayed

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

        if self.galaxy_id == 's35774':
            maxwv = 11100
        
        sp = fits.open('../Grizli_extract/ForVince/{0}'.format(fits_file))

        gal_wv=sp[2].data['wave']
        gal_fl=sp[2].data['flux']/sp[2].data['flat']
        gal_er=sp[2].data['err']/sp[2].data['flat']
        IDX = [U for U in range(len(gal_wv)) if 7900 < gal_wv[U] < 11300]
                
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

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
        import pysynphot as S
        if self.delayed == True:
            model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)
        else:
            model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_t%s_spec.npy' % (metal, age, tau)

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

    def Median_spec(self, metal, age, tau_array):

        chi = []
        for i in range(len(tau_array)):
            self.Sim_spec(metal, age, tau_array[i])
            chi.append(Identify_stack(self.gal_fl, self.gal_er, self.fl))


        self.bfmetal = metal
        self.bfage = age
        self.bftau = tau_array[np.argmin(chi)]
        self.Sim_spec(metal, age, tau_array[np.argmin(chi)])

 

def Single_gal_fit_full(metal, age, tau, fits_file, specz, galaxy, name,
                        minwv = 8000, maxwv = 11400, delayed_tau = True):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, fits_file)

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

    #############Prep output files: 1-full###############
    chifile1 = '../chidat/%s_chidata' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(tau),len(spec.gal_wv_rf)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl
    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]). astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)

    P, PZ, Pt = Analyze_LH_lwa(chifile1 + '.npy', specz, metal, age, tau)

    np.save('../chidat/%s_tZ_pos' % name,P)
    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('../chidat/%s_t_pos' % name,[age,Pt])

    print 'Done!'
    return
    
    
metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

Single_gal_fit_full(metal, age, tau, 'j033239-274648_39170.1D.fits', 1.022, 's39170', 's39170_griz_ex')
Single_gal_fit_full(metal, age, tau, 'j033239-274648_40597.1D.fits', 1.221, 's40597', 's40597_griz_ex')
