import numpy as np
from spec_id import Scale_model, Stack_spec_normwmean, Stack_model_normwmean, Likelihood_contours,\
    Analyze_Stack_avgage,Identify_stack
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

def Build_Flerr(wv,err_list):
    bins=[[2000, 3910], [3910, 3980], [3980, 4030], [4030, 4080], [4080, 4125], [4125, 4250], [4250, 4400],
          [4400, 4830], [4830, 4930], [4930, 4990], [4990, 5030], [5030, 5110], [5110, 5250], [5250, 6000]]
    repeats=[]
    for i in range(len(bins)):
        num=[]
        for ii in range(len(wv)):
            if bins[i][0]<=wv[ii]<bins[i][1]:
                num.append(1)
        repeats.append(sum(num))
        # print repeats[i]


    flerr=np.array([])
    for i in range(len(repeats)):
        r=np.repeat(err_list[i],repeats[i])
        flerr=np.append(flerr,r)
    return flerr

def Model_fit_stack_normwmean_ereg(speclist, tau, metal, A, specz, wv_range,name, pkl_name, erange, res=5, fsps=False):

    #############Get redshift info###############

    zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
    speczs = np.round(specz, 2)

    for i in range(len(speczs)):
        zinput = int(speczs[i] * 100) / 5 / 20.
        if zinput < 1:
            zinput = 1.0
        if zinput >1.8:
            zinput = 1.8
        zlist.append(zinput)
    for i in range(len(bins)):
        b = []
        for ii in range(len(zlist)):
            if bins[i] == zlist[ii]:
                b.append(ii)
        if len(b) > 0:
            zcount.append(len(b))
    zbin = sorted(set(zlist))

    ##############Stack spectra################

    wv,fl,er=Stack_spec_normwmean(speclist,specz,wv_range)


    err=np.sqrt(er**2+(erange*fl)**2)

    #############Prep output file###############

    chifile='chidat/%s_chidata.fits' % name
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList(prihdu)

    #############Get list of models to fit againts##############

    if fsps==False:

        filepath = '../../../bc03_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m=[]
            for ii in range(len(A)):
                a = []
                for iii in range(len(tau)):
                    t = []
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
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
                    for iv in range(len(zlist)):
                        t.append(filepath + 'm%s_a%s_t%s_z%s_model.dat' % (metal[i], A[ii], tau[iii], zlist[iv]))
                    a.append(t)
                m.append(a)
            modellist.append(m)

    ###############Pickle spectra##################

    pklname='pickled_mstacks/%s.pkl' % pkl_name

    if os.path.isfile(pklname)==False:

        pklspec = open(pklname, 'wb')

        for i in range(len(metal)):
            for ii in range(len(A)):
                for iii in range(len(tau)):
                    mw, mf, me = Stack_model_normwmean(speclist,modellist[i][ii][iii], specz, zlist, np.arange(wv[0],wv[-1]+res,res))
                    cPickle.dump(mf, pklspec, protocol=-1)

        pklspec.close()

        print 'pickle done'

    ##############Create chigrid and add to file#################

    outspec = open(pklname, 'rb')

    chigrid=np.zeros([len(metal),len(A),len(tau)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            for iii in range(len(tau)):
                mf = np.array(cPickle.load(outspec))
                chigrid[i][ii][iii]=Identify_stack(fl,err,mf)
        inputgrid = np.array(chigrid[i])
        spc ='metal_%s' % metal[i]
        mchi = fits.ImageHDU(data=inputgrid, name=spc)
        hdulist.append(mchi)

    outspec.close()

    ################Write chigrid file###############

    hdulist.writeto(chifile)
    return

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871<=lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
     1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""Make sim spec"""
# filepath = '../../../fsps_models_for_fit/models/'
# modellist = []
# for i in range(len(rshift[IDS])):
#     modellist.append(filepath + 'm0.012_a2.4_t8.0_z%s_model.dat' % rshift[IDS][i])
# #
# for i in range(len(speclist[IDS])):
#     wave,flux,error=np.array(Readfile(speclist[IDS][i],1))
#
#     #######read in corresponding model, and interpolate flux
#     W,F,E=np.array(Readfile(modellist[i],1))
#     iF=interp1d(W,F)(wave)
#
#     #######scale the model
#     C=Scale_model(flux,error,iF)
#
#     Fl = C*iF
#     er = error
#
#     ########add in noise
#     fl=Fl+np.random.normal(0,er)
#     fl[fl<0]=0
#
#     dat=Table([wave,fl,er],names=['wv','fl','err'])
#     ascii.write(dat,'sim_specs/%s_sim.dat' % ids[IDS][i])

"""create sim stack"""
mlist=glob('sim_specs/*')

simlist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(mlist)):
        if ids[IDS][i]==mlist[ii][10:16]:
            simlist.append(mlist[ii])

# w_rng=np.arange(3600,5250,10)
#
#
# wv,fl,er=Stack_spec_normwmean(speclist,rshift[IDS],w_rng)
#
# # erlist=[.0,.0,.0,.0,.0,.0,.02,.025,.0,.0,.0,.0,.0,.01]
# # erlist=[.0,.0,.0,.0,.0,.01,.03,.01,.0,.01,.0,.0,.0,.01]
# # erlist=[.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]
# # erlist=[0.085, 0., 0., 0., 0., 0.03, 0.035, 0.005, 0., 0., 0., 0., 0.,0]
# # erlist=[0.085, 0, 0, 0, 0, 0, 0.035, 0.01, 0, 0, 0, 0, 0, 0]##3250,5250
# # erlist=[0.055, 0, 0, 0, 0, 0, 0.085, 0.005, 0, 0, 0, 0, 0, 0] ##3400,5250
# # erlist=[0.04, 0, 0, 0, 0, 0, 0.1, 0.01, 0, 0, 0, 0, 0, 0] ##3500,5250
# erlist=[0.085, 0, 0, 0, 0, 0, 0.1, 0.005, 0, 0, 0, 0, 0, 0] ##3600,5250
# # plt.plot(wv,fl)
# # plt.plot(wv,er)
# # plt.show()
#
# flxerr=Build_Flerr(wv,erlist)
#
# M,A=np.meshgrid(metal,age)
#
# Model_fit_stack_normwmean_ereg(speclist,tau,metal,age,rshift[IDS],w_rng,
#                          'spec_3600_5250_ne_stackfit','sim_3600_5250_spec',flxerr ,res=10,fsps=True)
#
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/spec_3600_5250_ne_stackfit_chidata.fits', np.array(tau),metal,age)
# onesig,twosig=Likelihood_contours(age,metal,Pr)
# levels=np.array([twosig,onesig])
# print levels
# plt.contour(M,A,Pr,levels,colors='k',linewidths=2)
# plt.contourf(M,A,Pr,40,cmap=colmap)
# plt.plot(bfmetal,bfage,'cp',label='\nBest fit\nt=%s Gyrs\nZ=%s Z$_\odot$' % (bfage,np.round(bfmetal/0.019,2)))
# plt.xticks([0,.005,.01,.015,.02,.025,.03],np.round(np.array([0,.005,.01,.015,.02,.025,.03])/0.02,2))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.minorticks_on()
# plt.xlabel('Metallicity (Z$_\odot$)')
# plt.ylabel('Age (Gyrs)')
# plt.legend()
# # plt.show()
# plt.savefig('../research_plots/spec_3600_5250_ne_lh.png')