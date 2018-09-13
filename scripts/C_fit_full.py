from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck13, z_at_value
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import interpolation
from glob import glob
import grizli.model
import numpy as np
import pandas as pd
import os
from time import time

#set home
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    from C_spec_id import Scale_model, Median_w_Error_cont, Oldest_galaxy
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/clear_q_beams/'
    out_path = '/home/vestrada78840/chidat/'
    
else:
    from spec_id import Scale_model, Median_w_Error_cont, Oldest_galaxy
    data_path = '../data/'
    model_path ='../../../fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_stacks/'
    beam_path = '../beams/'
    out_path = '../chidat/' 
    
    
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
    def __init__(self, galaxy_id, redshift,minwv = 7900, maxwv = 11200, shift = 1, errf = True):
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
        if self.galaxy_id == 's35774':
            maxwv = 10800
            
        gal_wv, gal_fl, gal_er = np.load(glob(spec_path + '*{0}*'.format(self.gid))[0])
        self.flt_input = glob(beam_path + '*{0}*'.format(self.gid))[0]

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        
        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]
        
        if errf:
            self.o_er = np.array(self.gal_er)

            WV,TEF = np.load(data_path + 'template_error_function.npy')
            iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
            self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.gal_fl)**2)

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(fits_file=self.flt_input)

        ## Get sensitivity function
        
        flat = self.beam.flat_flam.reshape(self.beam.beam.sh_beam)
        fwv, ffl, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,flat.shape[0]]),flat.T[:-1],axis=0).T , bin=0)
        IDT = [U for U in range(len(fwv)) if 7800 <= fwv[U] <= 11500] 
        self.IDT = IDT
        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
    def Sim_spec(self, metal, age, tau, model_redshift = 0, dust = 0):
        if model_redshift ==0:
            model_redshift = self.redshift
            
        model = model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

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

        self.fl = f[self.IDT]
        self.mwv = w[self.IDT]
        
        
def Galaxy_full_fit(metal, age, tau, rshift, specz, galaxy, name, minwv = 8000, maxwv = 11200, errf = True):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv, errf = errf)

    #### apply special mask for specific objects

    if galaxy == 's47677':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4845 <= spec.gal_wv_rf[ii] <= 4863:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's39170' or galaxy == 'n34694':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4865 <= spec.gal_wv_rf[ii] <= 4885:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    ##############Create chigrid and add to file#################
    for i in range(len(metal)):
        mfl = np.zeros([len(age)*len(tau)*len(rshift),len(spec.IDT)])
        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
                    metal[i], age[ii], tau[iii]))
                for iv in range(len(rshift)):
                    spec.Sim_spec_mult(wv,fl,rshift[iv])
                    mfl[ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = spec.fl
        np.save(chi_path + 'spec_files/{0}_m{1}'.format(name, metal[i]),mfl)

def Galaxy_full_analyze(metal, age, tau, rshift, specz, galaxy, name, minwv = 8000, maxwv = 11200, errf = True):
    Redden_and_stich(galaxy,name,metal,age,tau, specz, rshift,minwv, maxwv,errf)
    grids = [chi_path + '{0}_d{1}_chidata.npy'.format(name,U) for U in range(11)]
    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(grids, metal, age, tau, rshift)

    np.save(out_path + '%s_tZ_pos' % name,P)
    np.save(out_path + '%s_Z_pos' % name,[metal,PZ])
    np.save(out_path + '%s_t_pos' % name,[age,Pt])
    np.save(out_path + '%s_tau_pos' % name,[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '%s_rs_pos' % name,[rshift,Pz])
    np.save(out_path + '%s_d_pos' % name,[np.arange(0,1.1,0.1),Pd])
    
def Stich_spec(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))
        
    stc = np.array(stc)
    return stc.reshape([stc.shape[0] * stc.shape[1],stc.shape[2]])

def Redden_and_stich(galaxy,name,metal,age,tau,specz, rshift,minwv, maxwv, errf):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv)

    #### apply special mask for specific objects

    if galaxy == 's47677':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4845 <= spec.gal_wv_rf[ii] <= 4863:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's39170' or galaxy == 'n34694':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4865 <= spec.gal_wv_rf[ii] <= 4885:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0
    
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    spec.Sim_spec_mult(wv,fl)
    
    files = [chi_path + 'spec_files/{0}_m{1}.npy'.format(name,U) for U in metal]
    mfl = Stich_spec(files)
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(spec.mwv,range(len(mfl.data)),mfl.data)(spec.gal_wv,range(len(mfl.data)))
    mfl = mfl / spec.filt
    
    minidust = Gen_dust_minigrid(spec.gal_wv, rshift)
    
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    chifiles = []
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(spec.gal_wv)])
        redflgrid = mfl * dustgrid
        SCL = Scale_model_mult(spec.gal_fl,spec.gal_er,redflgrid)
        redflgrid = np.array([SCL]).T*redflgrid
        chigrid = np.sum(((spec.gal_fl - redflgrid) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)])
        np.save(chi_path + '{0}_d{1}_chidata'.format(name, i),chigrid)
        chifiles.append(chi_path + '{0}_d{1}_chidata.npy'.format(name, i))

def Gen_dust_minigrid(fit_wv,rshift):
    dust_dict = {}
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        key = str(Av[i])
        minigrid = np.zeros([len(rshift),len(fit_wv)])
        for ii in range(len(rshift)):
            minigrid[ii] = Calzetti(Av[i],fit_wv / (1 + rshift[ii]))
        dust_dict[key] = minigrid
    return dust_dict

def Stich_grids(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))
    return np.array(stc)

def Analyze_full_fit(chifiles, metal, age, tau, rshift, dust = np.arange(0,1.1,0.1), age_conv=data_path + 'light_weight_scaling_3.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(max(rshift))
    
    ####### Read in file   
    chi = Stich_grids(chifiles)
    
    chi[ : , : , len(age[age <= max_age]):] = 1E5

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

    ######## get Pd and Pz

    d,Priord = np.load(data_path + 'dust_prior.npy')
    
    P_full = np.exp(- chi / 2).astype(np.float128)
    
    for i in range(len(Priord)):
        P_full[i] = P_full[i] * Priord[i]

    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1),dust)

    Pz = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1),rshift)

    P = np.trapz(P_full, rshift, axis=4)
    P = np.trapz(P.T, dust, axis=3).T
    new_P = np.zeros(P.T.shape)

    ######## Reshape likelihood to get light weighted age instead of age when marginalized
    for i in range(len(tau)):
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii],P.T[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(0, overhead[i][ii]))
        new_P[i] = frame.T

    ####### Create normalize probablity marginalized over tau
    P = new_P.T

    # test_prob = np.trapz(test_P, ultau, axis=2)
    C = np.trapz(np.trapz(np.trapz(P, ultau, axis=2), age, axis=1), metal)

    P /= C

    prob = np.trapz(P, ultau, axis=2)
    
    # #### Get Z, t, tau, and z posteriors
    PZ = np.trapz(np.trapz(P, ultau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, ultau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return prob.T, PZ, Pt, Ptau, Pz, Pd

def Redden_and_stich_hb_mask(galaxy,name,metal,age,tau,specz, rshift,minwv, maxwv, errf):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv)

    #### apply special mask for specific objects

    IDer = []
    for ii in range(len(spec.gal_wv_rf)):
        if 4837 <= spec.gal_wv_rf[ii] <= 4843 or 4873 <= spec.gal_wv_rf[ii] <= 4900:
            IDer.append(ii)
    spec.gal_er[IDer] = 1E8
    spec.gal_fl[IDer] = 0

    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    spec.Sim_spec_mult(wv,fl)
    
    files = [chi_path + 'spec_files/{0}_m{1}.npy'.format(name,U) for U in metal]
    mfl = Stich_spec(files)
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(spec.mwv,range(len(mfl.data)),mfl.data)(spec.gal_wv,range(len(mfl.data)))
    mfl = mfl / spec.filt
    
    minidust = Gen_dust_minigrid(spec.gal_wv, rshift)
    
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    chifiles = []
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(spec.gal_wv)])
        redflgrid = mfl * dustgrid
        SCL = Scale_model_mult(spec.gal_fl,spec.gal_er,redflgrid)
        redflgrid = np.array([SCL]).T*redflgrid
        chigrid = np.sum(((spec.gal_fl - redflgrid) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)])
        np.save(chi_path + '{0}_d{1}_chidata_hb_mask'.format(name, i),chigrid)
        chifiles.append(chi_path + '{0}_d{1}_chidata_hb_mask.npy'.format(name, i))

def Galaxy_full_analyze_hb_mask(metal, age, tau, rshift, specz, galaxy, name, minwv = 8000, maxwv = 11200, errf = True):
    Redden_and_stich__hb_mask(galaxy,name,metal,age,tau, specz, rshift,minwv, maxwv,errf)
    grids = [chi_path + '{0}_d{1}_chidata_nohb.npy'.format(name,U) for U in range(11)]
    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(grids, metal, age, tau, rshift)

    np.save(out_path + '%s_tZ_pos_hb_mask' % name,P)
    np.save(out_path + '%s_Z_pos_hb_mask' % name,[metal,PZ])
    np.save(out_path + '%s_t_pos_hb_mask' % name,[age,Pt])
    np.save(out_path + '%s_tau_pos_hb_mask' % name,[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '%s_rs_pos_hb_mask' % name,[rshift,Pz])
    np.save(out_path + '%s_d_pos_hb_mask' % name,[np.arange(0,1.1,0.1),Pd])