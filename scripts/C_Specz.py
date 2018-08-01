from astropy.io import fits
from scipy.interpolate import interp1d, interp2d
from C_spec_id import Scale_model
from glob import glob
import grizli.model
import numpy as np

def Scale_model_mult(D, sig, M):
    C = np.sum(((D * M) / sig ** 2), axis=1) / np.sum((M ** 2 / sig ** 2), axis=1)
    return C

class Gen_spec(object):
    def __init__(self, galaxy_id, redshift,minwv = 7900, maxwv = 11300):
        self.galaxy_id = galaxy_id
        self.gid = int(self.galaxy_id[1:])
        self.redshift = redshift

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

        if self.galaxy_id == 's35774':
            maxwv = 11100

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
        if self.gid in [17070,19148,45775,19442]:
            fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        
        else:
            flat = beam.flat_flam.reshape(beam.beam.sh_beam)
            fwv,ffl,ferr = beam.beam.optimal_extract(flat, bin=0, ivar=beam.ivar)
        
        self.filt = interp1d(fwv, ffl)(self.gal_wv)
        
    def Sim_spec(self, metal, age, tau, model_redshift = 0):
        if model_redshift ==0:
            model_redshift = self.redshift
        
        model = '/fdata/scratch/vestrada78840/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

        wave, fl = np.load(model)

        ## Compute the models
        self.beam.compute_model(spectrum_1d=[wave*(1+model_redshift),fl])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

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
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        self.fl = f
        self.mwv = w
        
def Specz_fit(metal, age, rshift, galaxy, name, minwv = 7900, maxwv = 11400):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, 1.0, minwv = minwv, maxwv = maxwv)

    #############Prep output files: 1-full###############
    chifile1 = '/home/vestrada78840/rshift_dat/{0}_chidata'.format(name)
 
    ##############Create chigrid and add to file#################
    model_fl = []
    for i in range(len(metal)):
        for ii in range(len(age)):
            wv,fl = np.load('/fdata/scratch/vestrada78840/fsps_spec/m{0}_a{1}_dt8.0_spec.npy'.format(
                metal[i], age[ii]))
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
    C = Scale_model_mult(spec.gal_fl,spec.gal_er,adjflgrid)
    mfl = np.array([C]).T*adjflgrid

    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(rshift)]).\
        astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)

    Pz = Analyze_LH_specz(chifile1 + '.npy', metal, age, rshift)

    np.save('/home/vestrada78840/rshift_dat/{0}_Pz'.format(name),[rshift,Pz])

def Analyze_LH_specz(chifits, metal, age, rshift):
    ####### Read in file
    chi = np.load(chifits)

    P = np.exp(-chi.astype(np.float128) / 2)
    
    Pz = np.trapz(np.trapz(P, metal, axis=2), age, axis=1) /\
        np.trapz(np.trapz(np.trapz(P, metal, axis=2), age, axis=1),rshift)
    
    return Pz