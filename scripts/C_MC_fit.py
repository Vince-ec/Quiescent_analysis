from scipy.interpolate import interp1d, interp2d
from glob import glob
import numpy as np
import pandas as pd
import grizli.model
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
    mcerr_path = '/home/vestrada78840/mcerr/'
    
else:
    from spec_id import Scale_model, Median_w_Error_cont, Oldest_galaxy
    data_path = '../data/'
    model_path ='../../../fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_stacks/'
    beam_path = '../beams/'
    mcerr_path = '../mcerr/' 
    
    
def Scale_model_mult(D, sig, M):
    C = np.sum(((D * M) / sig ** 2), axis=1) / np.sum((M ** 2 / sig ** 2), axis=1)
    return C

def Sig_int(er):
    sig = np.zeros(len(er)-1)
    
    for i in range(len(er)-1):
        sig[i] = np.sqrt(er[i]**2 + er[i+1]**2 )
    
    return np.sum((1/2)*sig)

def SNR(wv,fl,er):
    IDX = [U for U in range(len(wv)) if 7900 < wv[U] < 11200]
    return np.trapz(fl[IDX])/ Sig_int(er[IDX])

def SNR_correct(wave,flux,error,SNR_desired): 
    sno = SNR(wave,flux,error)
    return sno / SNR_desired

def Calzetti(Av,lam):
    lam = lam * 1E-4
    Rv=4.05
    k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
    cal = 10**(-0.4*k*Av/Rv)    
    return cal

def Stich_spec(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))       
    stc = np.array(stc)
    return stc.reshape([stc.shape[0] * stc.shape[1],stc.shape[2]])


def Gen_mflgrid(fit_wv, fit_flat, metal, galaxy, specz,dataset):
    ##### set model wave
    wave, fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(0.019, 2.0, 8.0))
    
    tmp_spec = Gen_sim(galaxy, 0.019,2.0,8.0,specz,0,10)
    mwv, dummy = tmp_spec.Sim_spec_mult(wave, fl, specz)
    
    #############Read in spectra#################    
    files = [chi_path + 'spec_files/{0}_m{1}.npy'.format(dataset,U) for U in metal]
    mfl = Stich_spec(files)
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(mwv, range(len(mfl.data)),mfl.data)(fit_wv,range(len(mfl.data)))
    return mfl / fit_flat

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
         
def Redden(mfl, dust, fit_fl, fit_er, metal, age, tau,rshift):
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    fullgrid=[]
    for i in range(len(Av)):
        dustgrid = np.repeat([dust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(rshift)*len(metal)*len(age)*len(tau), len(fit_fl)])
        redflgrid = mfl * dustgrid
        SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
        fullgrid.append(np.array([SCL]).T*redflgrid)
    return np.array(fullgrid)
    
def Fit_spec(fit_grid,fit_fl,fit_er,metal,age,tau,rshift):
    fullgrid=[]
    for i in range(len(fit_grid)):
        fullgrid.append(np.sum(((fit_fl - fit_grid[i]) / fit_er) ** 2, axis=1).reshape(
            [len(metal), len(age), len(tau), len(rshift)]))

    return np.array(fullgrid)

def Analyze_full_fit(P,fit_fl, fit_er, metal, age, tau, rshift, convtable, overhead,dust = np.arange(0,1.1,0.1)):
    ####### Get maximum age
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    ######## get Pd and Pz
    P = np.exp(- P / 2).astype(np.float128)
    P = np.trapz(P, rshift, axis=4)
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

    return PZ, Pt

def MC_fit(galaxy, metal, age, tau, redshift, dust, sim_m, sim_a, sim_t, sim_z, sim_d, sn, dataset, specz, name, repeats=1000,
                    age_conv= data_path + 'light_weight_scaling_3.npy'):
#    start = time()
    ######## set paramter output arrays
    PZlist = np.zeros([repeats,metal.size])
    Ptlist = np.zeros([repeats,age.size])
    mlist = np.zeros(repeats)
    alist = np.zeros(repeats)

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    
    ######## create sim

    spec = Gen_sim(galaxy, sim_m, sim_a, sim_t, sim_z, sim_d, sn)

    ####### set up lwa
    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)
    
#    end= time()
#    print(end-start)
    ####### Generate model grid
#    start=time()

    mflgrid = Gen_mflgrid(spec.gal_wv, spec.filt, metal, galaxy, specz,dataset)

#    end= time()
#    print(end-start)
    ####### Generate dust minigrid
#    start = time()
    
    dstgrid = Gen_dust_minigrid(spec.gal_wv,redshift)
#    end= time()
#    print(end-start)                

#    start = time()
    redgrid = Redden(mflgrid, dstgrid, spec.flx_err, spec.gal_er, metal, age, tau,redshift)

#    end = time()
#    print(end - start)
    
    for xx in range(repeats):
#        start = time()
        flx_err = spec.Perturb_flux(spec.fl,spec.gal_er)
        
        ###### fit grid  
       
        chigrid = Fit_spec(redgrid,flx_err, spec.gal_er,metal,age,tau,redshift)
    
        PZlist[xx], Ptlist[xx] = Analyze_full_fit(chigrid,flx_err, spec.gal_er, metal, age, tau, 
                                                  redshift,convtable, overhead)

        mlist[xx],ml,mh = Median_w_Error_cont(PZlist[xx],metal)
        alist[xx],ml,mh = Median_w_Error_cont(Ptlist[xx],age)
        
#        end = time()
#        print(end - start)
    
    np.save(mcerr_path + 'PZ_' + name, PZlist)
    np.save(mcerr_path + 'Pt_' + name, Ptlist)
    np.save(mcerr_path + name, [mlist, alist])

    return

class Gen_sim(object):
    def __init__(self, galaxy_id, sim_metal, sim_age, sim_tau, sim_z, sim_dust, sn, minwv = 7900, maxwv = 11200, shift = 1):
        self.galaxy_id = galaxy_id
        self.gid = int(self.galaxy_id[1:])
        self.sim_metal = sim_metal
        self.sim_age = sim_age
        self.sim_tau = sim_tau
        self.sn = sn
        self.sim_z = sim_z
        self.sim_dust = sim_dust
        self.shift = shift
        
        o_wv, gal_fl, gal_er = np.load(glob(spec_path + '*{0}*'.format(self.gid))[0])
        self.flt_input = glob(beam_path + '*{0}*'.format(self.gid))[0]
        
        #IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        ewv, mederr = np.load(data_path + 'med_err.npy')
        
        self.gal_wv = np.arange(7900,11200,12)
        
        self.gal_wv_rf = self.gal_wv / (1 + self.sim_z)
        self.gal_fl = interp1d(o_wv,gal_fl)(self.gal_wv)
        self.gal_er = interp1d(ewv,mederr)(self.gal_wv)

        self.gal_wv_rf = self.gal_wv_rf
        self.gal_wv = self.gal_wv
        self.gal_er = self.gal_er
        self.gal_fl = self.gal_fl
        self.o_er = np.array(self.gal_er)
             
        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(fits_file=self.flt_input)

        ## Get sensitivity function
        flat = self.beam.flat_flam.reshape(self.beam.beam.sh_beam)
        fwv, ffl, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,flat.shape[0]]),flat.T[:-1],axis=0).T , bin=0)
        IDT = [U for U in range(len(fwv)) if 7800 <= fwv[U] <= 11500] 
        self.IDT = IDT
        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
        ### set dummy spec
        wave, fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
        self.sim_metal, self.sim_age, self.sim_tau))

        cal = Calzetti(self.sim_dust,wave)
        
        w,f = self.Sim_spec_mult(wave, fl*cal, self.sim_z)
        ifl = interp1d(w,f)(self.gal_wv)
        adj_ifl = ifl /self.filt
        
        snc = SNR_correct(self.gal_wv, adj_ifl, self.o_er, self.sn)
        self.fl = adj_ifl / snc   
              
        WV,TEF = np.load(data_path + 'template_error_function.npy')
        iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
        self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.fl)**2)

        ## set mask for continuum removal
        m2r = [3175, 3280, 3340, 3515, 3550, 3650, 3710, 3770, 3800, 3850,
               3910, 4030, 4080, 4125, 4250, 4385, 4515, 4570, 4810, 4910, 4975, 5055, 5110, 5285]

        Mask = np.zeros(len(self.gal_wv_rf))
        for i in range(len(Mask)):
            for ii in range(len(m2r)//2):
                if m2r[ii * 2] <= self.gal_wv_rf[i] <= m2r[ii*2 + 1]:
                    Mask[i] = 1
        
        self.maskw = np.ma.masked_array(self.gal_wv_rf, Mask)
        params = np.ma.polyfit(self.maskw, self.gal_fl, 3)
        C0 = np.polyval(params,self.gal_wv_rf)

        self.nc_gal_fl = self.gal_fl / C0
        self.nc_gal_er = self.gal_er / C0
        self.nc_o_er = self.o_er / C0
        
        self.Set_spec()    

        
    def Perturb_flux(self,fl,err):
        return np.abs(fl + np.random.normal(0, err))

    def Interp_and_scale(self, w, f, influx):
        
        ifl = interp1d(w,f)(self.gal_wv)
        adj_ifl = ifl /self.filt
        
        C = Scale_model(influx, self.gal_er, adj_ifl)

        return C * adj_ifl
        
    def Sim_spec_mult(self, wave, fl, model_redshift):
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[wave*(1+model_redshift), fl])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(np.append(np.zeros([self.shift,self.beam.model.shape[0]]),
                                                           self.beam.model.T[:-1],axis=0).T , bin=0)
        
        return w[self.IDT], f[self.IDT]

    def Rm_cont(self, fl):
        params = np.ma.polyfit(self.maskw, fl, 3)
        C0 = np.polyval(params,self.gal_wv_rf)  
        return fl / C0
    
    def Set_spec(self):
        self.flx_err = self.Perturb_flux(self.fl, self.gal_er)

        self.nc_fl = self.Rm_cont(self.fl)
        self.nc_flx_err = self.Perturb_flux(self.nc_fl, self.nc_gal_er)
     
    def Sim_spec(self, metal, age, tau, model_redshift = 0, dust = 0, no_cont = False):
        if model_redshift ==0:
            model_redshift = self.sim_z
        
        wave, fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))

        cal = Calzetti(dust,wave)
        
        w,f = self.Sim_spec_mult(wave, fl*cal, model_redshift)

        self.fl = self.Interp_and_scale(w, f, self.flx_err)
        if no_cont:
            self.nc_fl = self.Rm_cont(self.fl)

def Galaxy_gen_spec(metal, age, tau, rshift, specz, galaxy, name):
    #############Read in spectra#################
    spec = Gen_sim(galaxy, 0.019, 2.0, 0, specz, 0, 10)

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