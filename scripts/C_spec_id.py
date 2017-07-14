import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import grizli
import pysynphot as S
from vtl.Readfile import Readfile


def Scale_model(D, sig, M):
    C = np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))
    return C


def Oldest_galaxy(z):
    return cosmo.age(z).value


#####SPECZ FIT

class RT_spec(object):
    def __init__(self, galaxy_id, pad=100):
        self.galaxy_id = galaxy_id
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

        gal_wv, gal_fl, gal_er = \
            np.load('../../../../fdata/scratch/vestrada78840/spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../../../../fdata/scratch/vestrada78840/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if 7900 <= gal_wv[U] <= 11300]

        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

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

    def Sim_spec(self, metal, age, tau, redshift):
        model = '../../../../fdata/scratch/vestrada78840/fsps_spec/m%s_a%s_t%s_spec.npy' % (metal, age, tau)

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
    print 'Best fit specz is %s' % rshift[np.argmax(prob)]

    np.save('/home/vestrada78840/rshift_dat/%s_Pofz' % name,[rshift, prob])
    return

def Specz_fit(galaxy, metal, age, rshift, name):
    #############initialize spectra#################
    spec = RT_spec(galaxy)

    #############Prep output file###############
    chifile = '/home/vestrada78840/rshift_dat/%s_z_fit' % name

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

    print 'Done!'

    return

#####GALAXY FIT

class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, minwv = 7900, maxwv = 11300, pad=100):
        self.galaxy_id = galaxy_id
        self.redshift = redshift
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

        gal_wv, gal_fl, gal_er = \
            np.load('../../../../fdata/scratch/vestrada78840/spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../../../../fdata/scratch/vestrada78840/galaxy_flts/%s_flt.fits' % self.galaxy_id

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
        model = '../../../../fdata/scratch/vestrada78840/fsps_spec/m%s_a%s_t%s_spec.npy' % (metal, age, tau)

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


def Single_gal_fit_full(metal, age, tau, specz, galaxy, name, minwv = 7900, maxwv = 11300):
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

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1 = '/home/vestrada78840/chidat/%s_chidata' % name
    chifile2 = '/home/vestrada78840/chidat/%s_cont_chidata' % name
    chifile3 = '/home/vestrada78840/chidat/%s_feat_chidata' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(tau),len(spec.gal_wv_rf)])
    mfl_f = np.zeros([len(metal)*len(age)*len(tau),len(IDF)])
    mfl_c = np.zeros([len(metal)*len(age)*len(tau),len(IDC)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl
                mfl_f[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl[IDF]
                mfl_c[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl[IDC]
    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)
    chigrid2 = np.sum(((spec.gal_fl[IDF] - mfl_f) / spec.gal_er[IDF]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)
    chigrid3 = np.sum(((spec.gal_fl[IDC] - mfl_c) / spec.gal_er[IDC]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)
    np.save(chifile2,chigrid2)
    np.save(chifile3,chigrid3)

    P, PZ, Pt = Analyze_LH_cont_feat(chifile2 + '.npy', chifile3 + '.npy', specz, metal, age, tau)

    np.save('/home/vestrada78840/chidat/%s_tZ_pos' % name,P)
    np.save('/home/vestrada78840/chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('/home/vestrada78840/chidat/%s_t_pos' % name,[age,Pt])

    print 'Done!'
    return


def Analyze_LH_cont_feat(contfits, featfits, specz, metal, age, tau,
                         age_conv='./../../../fdata/scratch/vestrada78840/data/tau_scale_ntau.dat'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    Cchi = np.load(contfits).T
    Fchi = np.load(featfits).T

    Fchi[:, len(age[age <= max_age]):, :] = 1E5
    Cchi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    convtau = np.array([0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2,
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
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt


#####MC FIT

class Gen_sim(object):
    def __init__(self, galaxy_id, redshift, metal, age, tau, minwv=7900, maxwv=11300, pad=100):
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

        gal_wv, gal_fl, gal_er = \
            np.load('../../../../fdata/scratch/vestrada78840/spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../../../../fdata/scratch/vestrada78840/galaxy_flts/%s_flt.fits' % self.galaxy_id

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

        model = '../../../../fdata/scratch/vestrada78840/fsps_spec/m%s_a%s_t%s_spec.npy' \
                % (self.metal, self.age, self.tau)

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


    def Perturb_flux(self):
        self.flx_err = np.abs(self.fl + np.random.normal(0, self.gal_er))


    def Perturb_flux_nc(self):
        self.nc_flx_err = np.abs(self.nc_fl + np.random.normal(0, self.nc_er))


    def Sim_spec(self, metal, age, tau):
        import pysynphot as S

        model = '../../../../fdata/scratch/vestrada78840/fsps_spec/m%s_a%s_t%s_spec.npy' % (metal, age, tau)

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
           age_conv='./../../../fdata/scratch/vestrada78840/data/tau_scale_ntau.dat'):
    bfm=[]
    bfmnc=[]
    bfmdf=[]
    bfa=[]
    bfanc=[]
    bfadf=[]


    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    spec = Gen_sim(galaxy, specz, sim_m, sim_a, sim_t,minwv=minwv,maxwv=maxwv)

    ###############Get indicies
    IDF = []
    for i in range(len(spec.gal_wv_rf)):
        if 3800 <= spec.gal_wv_rf[i] <= 3850 or 4080 <= spec.gal_wv_rf[i] <= 4125 or 4250 <= spec.gal_wv_rf[i] <= 4385 \
                or 4810 <= spec.gal_wv_rf[i] <= 4910 or 5110 <= spec.gal_wv_rf[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(spec.gal_wv_rf)):
        if spec.gal_wv_rf[0] <= spec.gal_wv_rf[i] <= 3800 or 3850 <= spec.gal_wv_rf[i] <= 3910 or 4030 <= \
                spec.gal_wv_rf[i] <= 4080 or 4125 <= spec.gal_wv_rf[i] <= 4250 or 4385 <= spec.gal_wv_rf[i] <= 4515 or \
                4570 <= spec.gal_wv_rf[i] <= 4810 or 4910 <= spec.gal_wv_rf[i] <= 4975 or 5055 <= \
                spec.gal_wv_rf[i] <= 5110 or 5285 <= spec.gal_wv_rf[i] <= spec.gal_wv_rf[-1]:
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
        # C = np.trapz(np.trapz(P, age, axis=1), metal)

        Pnc = np.trapz(ncprob, ultau, axis=2)
        # Cnc = np.trapz(np.trapz(Pnc, age, axis=1), metal)

        Pc = np.trapz(cprob, ultau, axis=2)
        Cc = np.trapz(np.trapz(Pc, age, axis=1), metal)

        Pf = np.trapz(fprob, ultau, axis=2)
        Cf = np.trapz(np.trapz(Pf, age, axis=1), metal)

        ########

        comb_prob = cprob / Cc * fprob / Cf

        df_post = np.trapz(comb_prob, ultau, axis=2)
        # C0 = np.trapz(np.trapz(df_post, age, axis=1), metal)
        # df_post /= C0

        ids = np.argwhere(P == np.max(P))
        bfm.append(metal[ids[0][0]])
        bfa.append(age[ids[0][i]])

        ids = np.argwhere(Pnc == np.max(Pnc))
        bfmnc.append(metal[ids[0][0]])
        bfanc.append(age[ids[0][i]])

        ids = np.argwhere(df_post == np.max(df_post))
        bfmdf.append(metal[ids[0][0]])
        bfadf.append(age[ids[0][i]])


    np.save('/home/vestrada78840/mcerr/' + name, [bfm, bfa])
    np.save('/home/vestrada78840/mcerr/' + name + 'NC', [bfmnc, bfanc])
    np.save('/home/vestrada78840/mcerr/' + name + 'DF', [bfmdf, bfadf])

    return


def MC_fit(galaxy, metal, age, tau, sim_m, sim_a, sim_t, specz, name, repeats=100,
           age_conv='./../../../fdata/scratch/vestrada78840/data/tau_scale_ntau.dat'):
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
                print scale[i]
                cframe = interp2d(metal, scale[i], Cchi[i])(metal, age[:-overhead[i]])
                print cframe.shape
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

    np.save('/home/vestrada78840/mcerr/' + name, [mlist, alist])

    return
