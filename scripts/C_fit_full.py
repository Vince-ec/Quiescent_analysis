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

def Calzetti(Av,lam):
    lam = lam * 1E-4
    Rv=4.05
    k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
    cal = 10**(-0.4*k*Av/Rv)    
    
    return cal

def Scale_model_mult(D, sig, M):
    C = np.sum(((D * M) / sig ** 2), axis=1) / np.sum((M ** 2 / sig ** 2), axis=1)
    return C

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
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        gal_wv, gal_fl, gal_er = np.load(glob('/fdata/scratch/vestrada78840/stack_specs/*{0}*'.format(self.gid))[0])
        self.flt_input = glob('/fdata/scratch/vestrada78840/clear_q_beams/*{0}*'.format(self.gid))[0]

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(fits_file=self.flt_input)

        ## Get sensitivity function
        
        flat = self.beam.flat_flam.reshape(self.beam.beam.sh_beam)
        fwv, ffl, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,flat.shape[0]]),flat.T[:-1],axis=0).T , bin=0)

        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
    def Sim_spec(self, metal, age, tau, model_redshift = 0, dust = 0):
        if model_redshift ==0:
            model_redshift = self.redshift
            
        model = '/fdata/scratch/vestrada78840/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

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
        
        
def Galaxy_full_fit(metal, age, tau, rshift, specz, galaxy, name, minwv = 8000, maxwv = 11200):
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

    ##############Create chigrid and add to file#################
    model_fl = []
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load('/fdata/scratch/vestrada78840/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(
                    metal[i], age[ii], tau[iii]))
                model_fl.append(fl)
    
    mfl = []
    for i in range(len(model_fl)):
        for ii in range(len(rshift)):
            spec.Sim_spec_mult(wv,model_fl[i],rshift[ii])
            mfl.append(spec.fl)

    np.array(mfl)
    fl_mask = np.ma.masked_invalid(mfl)
    fl_mask.data[fl_mask.mask] = 0
    iflgrid = interp2d(spec.mwv,range(len(fl_mask.data)),fl_mask.data)(spec.gal_wv,range(len(fl_mask.data)))
    adjflgrid = iflgrid / spec.filt
     
    Av = np.arange(0, 1.1, 0.1)
    chifiles = []
    for i in range(len(Av)):
        dust = Calzetti(Av[i],spec.gal_wv_rf)
        redflgrid = adjflgrid * dust
        SCL = Scale_model_mult(spec.gal_fl,spec.gal_er,redflgrid)
        mfl = np.array([SCL]).T*redflgrid
        chigrid = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)]).\
        astype(np.float128)
        np.save('/fdata/scratch/vestrada78840/chidat/{0}_d{1}_chidata'.format(name, i),chigrid)
        chifiles.append('/fdata/scratch/vestrada78840/chidat/{0}_d{1}_chidata.npy'.format(name, i))

    ################Write chigrid file###############
    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(chifiles, metal, age, tau, rshift)

    np.save('/home/vestrada78840/chidat/%s_tZ_pos' % name,P)
    np.save('/home/vestrada78840/chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('/home/vestrada78840/chidat/%s_t_pos' % name,[age,Pt])
    np.save('/home/vestrada78840/chidat/%s_tau_pos' % name,[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save('/home/vestrada78840/chidat/%s_z_pos' % name,[rshift,Pz])
    np.save('/home/vestrada78840/chidat/%s_d_pos' % name,[np.arange(0,1.1,0.1),Pd])

    return
