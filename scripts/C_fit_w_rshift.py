from astropy.table import Table
from astropy.io import fits
from C_spec_id import Scale_model,Oldest_galaxy
from astropy.cosmology import Planck13, z_at_value
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import interpolation
from glob import glob
import grizli.model
import numpy as np
import pandas as pd

class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, pad=100, delayed = True,minwv = 7900, maxwv = 11300):
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

        gal_wv, gal_fl, gal_er = \
            np.load('/fdata/scratch/vestrada78840/spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '/fdata/scratch/vestrada78840/galaxy_flts/%s_flt.fits' % self.galaxy_id

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
        #c = sim_g102.catalog
        c = Table.read('/fdata/scratch/vestrada78840/galaxy_flts/{0}_flt.detect.cat'.format(self.galaxy_id),format='ascii')
        sim_g102.catalog = c
        
        
        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id][2]['A'], conf=sim_g102.conf)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
    def Sim_spec(self, wave, fl, model_redshift = 0):
        import pysynphot as S

        if model_redshift ==0:
            model_redshift = self.redshift
        
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(model_redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        adj_ifl = interp1d(w, f)(self.gal_wv) /self.filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

        
def Single_gal_fit_w_redshift(metal, age, tau, rshift, specz, galaxy, name, minwv = 7900, maxwv = 11300):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv)

    #### apply special mask for specific objects
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
    chifile1 = '/home/vestrada78840/chidat/%s_chidata' % name
 
    ##############Create chigrid and add to file#################
    model_fl = []
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load('/fdata/scratch/vestrada78840/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(
                    metal[i], age[ii], tau[iii]))
                model_fl.append(fl)
    
    mfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(spec.gal_wv_rf)])
    for i in range(len(model_fl)):
        for ii in range(len(rshift)):
            spec.Sim_spec(wv,model_fl[i],rshift[ii])
            mfl[i*len(rshift) + ii]=spec.fl
    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)]).\
        astype(np.float128)


    ################Write chigrid file###############
    np.save(chifile1,chigrid1)

#    P, PZ, Pt = Analyze_LH_lwa(chifile1 + '.npy', specz, metal, age, tau)

#    np.save('../chidat/%s_tZ_pos' % name,P)
#    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
#    np.save('../chidat/%s_t_pos' % name,[age,Pt])

    print('Done!')
    return