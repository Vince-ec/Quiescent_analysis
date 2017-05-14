from scipy.interpolate import interp1d
from spec_id import Scale_model
from vtl.Readfile import Readfile
import vtl.Constants as C
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
metal = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
# wv,fl=np.array(Readfile('../../../bc03_models_for_fit/bc03_spec/m0.02_a1.85_t8.72_z1.65_spec.dat',1))
# wv/=2.65

def Mgb_Fe(wv, fl):
    ######Interpolate spec######
    waves=np.linspace(5000,5500,1000)
    sed=interp1d(wv,fl)(waves)


    ######Define Lick indices#####
    mgb = np.array([5160.125, 5192.625])
    mgcont=np.array([[5142.625, 5161.375],[5191.375,5206.375]])
    fe5270 = np.array([5245.650, 5285.650])
    fe5270cont=np.array([[5233.150, 5248.150],[5285.65,5318.15]])
    fe5335 = np.array([5312.125, 5352.125])
    fe5335cont=np.array([[5304.625, 5315.875],[5353.375,5363.375]])

    #######setup arrays for regions#####
    contspecmg = [[],[]]
    contwavemg = [[],[]]
    contspecfe5270  = [[],[]]
    contwavefe5270  = [[],[]]
    contspecfe5335 = [[],[]]
    contwavefe5335 = [[],[]]
    mgwave,mgspec = [[],[]]
    fe5270wave,fe5270spec = [[],[]]
    fe5335wave,fe5335spec = [[],[]]

    ########Define regions and find continua#######
    for i in range(len(sed)):
        if mgb[0] <= waves[i] <= mgb[1]:
            mgwave.append(waves[i])
            mgspec.append(sed[i])
        if fe5270[0] <= waves[i] <= fe5270[1]:
            fe5270wave.append(waves[i])
            fe5270spec.append(sed[i])
        if fe5335[0] <= waves[i] <= fe5335[1]:
            fe5335wave.append(waves[i])
            fe5335spec.append(sed[i])

        if mgcont[0][0] <= waves[i] <= mgcont[0][1]:
            contspecmg[0].append(sed[i])
            contwavemg[0].append(waves[i])
        if mgcont[1][0] <= waves[i] <= mgcont[1][1]:
            contspecmg[1].append(sed[i])
            contwavemg[1].append(waves[i])

        if fe5270cont[0][0] <= waves[i] <= fe5270cont[0][1]:
            contspecfe5270[0].append(sed[i])
            contwavefe5270[0].append(waves[i])
        if fe5270cont[1][0] <= waves[i] <= fe5270cont[1][1]:
            contspecfe5270[1].append(sed[i])
            contwavefe5270[1].append(waves[i])

        if fe5335cont[0][0] <= waves[i] <= fe5335cont[0][1]:
            contspecfe5335[0].append(sed[i])
            contwavefe5335[0].append(waves[i])
        if fe5335cont[1][0] <= waves[i] <= fe5335cont[1][1]:
            contspecfe5335[1].append(sed[i])
            contwavefe5335[1].append(waves[i])

    ######Find the mean of the continua########
    cont_linemg = [1/(contwavemg[0][-1]-contwavemg[0][0])*np.trapz(contspecmg[0],contwavemg[0])
        ,1/(contwavemg[1][-1]-contwavemg[1][0])*np.trapz(contspecmg[1],contwavemg[1])]
    cont_linefe5270 = [1/(contwavefe5270[0][-1]-contwavefe5270[0][0])*np.trapz(contspecfe5270[0],contwavefe5270[0])
        ,1/(contwavefe5270[1][-1]-contwavefe5270[1][0])*np.trapz(contspecfe5270[1],contwavefe5270[1])]
    cont_linefe5335 = [1/(contwavefe5335[0][-1]-contwavefe5335[0][0])*np.trapz(contspecfe5335[0],contwavefe5335[0])
        ,1/(contwavefe5335[1][-1]-contwavefe5335[1][0])*np.trapz(contspecfe5335[1],contwavefe5335[1])]
    cont_wavemg=[np.mean(contwavemg[0]),np.mean(contwavemg[1])]
    cont_wavefe5270=[np.mean(contwavefe5270[0]),np.mean(contwavefe5270[1])]
    cont_wavefe5335=[np.mean(contwavefe5335[0]),np.mean(contwavefe5335[1])]

    #######interpolate regions for final analysis########
    icontmg=interp1d(cont_wavemg,cont_linemg)
    icontfe5270=interp1d(cont_wavefe5270,cont_linefe5270)
    icontfe5335=interp1d(cont_wavefe5335,cont_linefe5335)

    #####plot check######
    # plt.plot(waves, sed)
    # plt.plot(mgwave,mgspec)
    # plt.plot(cont_wavemg,cont_linemg)
    #
    # plt.plot(fe5270wave, fe5270spec)
    # plt.plot(cont_wavefe5270, cont_linefe5270)
    #
    # plt.plot(fe5335wave, fe5335spec)
    # plt.plot(cont_wavefe5335, cont_linefe5335)
    #
    # plt.xlim(4000, 5500)
    # plt.show()

    ########Fing EWs##########
    Mgb=np.trapz((1-mgspec/icontmg(mgwave)),mgwave)
    Fe5270=np.trapz((1-fe5270spec/icontfe5270(fe5270wave)),fe5270wave)
    Fe5335=np.trapz((1-fe5335spec/icontfe5335(fe5335wave)),fe5335wave)

    print Mgb
    print Fe5270
    print Fe5335

    ########Calculate [Mgb/Fe]##########
    mgfe=np.sqrt(Mgb*(0.72*Fe5270+0.28*Fe5335))

    return mgfe

def D4000(wv, flam):
    c=C.c
    h = C.h
    atocm = C.angtocm

    wave = np.multiply(wv, atocm)
    nu = np.divide(c, wave)
    fnu = np.multiply(np.divide(c, np.square(nu)), flam)
    Fnu = interp1d(nu, fnu)

    Dblue = np.divide(c,np.multiply([3850, 3950],atocm))
    Dred = np.divide(c,np.multiply([4000, 4100],atocm))

    dbluenu=np.linspace(Dblue[0],Dblue[1],100)
    drednu=np.linspace(Dred[0],Dred[1],100)

    energy = np.divide(1 / h, dbluenu)
    top1 = np.multiply(Fnu(dbluenu), energy)
    top = np.trapz(top1, dbluenu)
    bottom = np.trapz(energy, dbluenu)
    blue = top / bottom

    energy = np.divide(1 / h, drednu)
    top1 = np.multiply(Fnu(drednu), energy)
    top = np.trapz(top1, drednu)
    bottom = np.trapz(energy, drednu)
    red = top / bottom

    D4=red / blue

    return D4

def H_d(waves, sed):

    h_d = np.array([4082, 4122])
    continuum = np.array([4030, 4170])

    contwave = []
    contspec = []

    hdwave = []
    hdspec = []

    for i in range(len(sed)):
        if h_d[0] <= waves[i] <= h_d[1]:
            hdwave.append(waves[i])
            hdspec.append(sed[i])
        if continuum[0] <= waves[i] <= continuum[1]:
            contwave.append(waves[i])
            contspec.append(sed[i])

    cont_line = interp1d([contwave[0], contwave[-1]], [contspec[0], contspec[-1]])

    Fc_Flam = []
    Fc = []
    for i in range(len(hdspec)):
        Fc_Flam.append(cont_line(hdwave[i]) - hdspec[i])
        Fc.append(cont_line(hdwave[i]))

    ratio = np.divide(Fc_Flam, Fc)
    W = np.trapz(ratio, hdwave)

    return W

def Stack_spec_normwmean(spec,redshifts, wv):

    flgrid=np.zeros([len(spec),len(wv)])
    errgrid=np.zeros([len(spec),len(wv)])
    for i in range(len(spec)):
        wave, flux, error = np.array(Readfile(spec[i], 1))
        wave /= (1 + redshifts[i])
        mask = np.array([wave[0] < U < wave[-1] for U in wv])
        ifl=interp1d(wave,flux)
        ier=interp1d(wave,error)
        reg = np.arange(4000, 4210, 1)
        Cr = np.trapz(ifl(reg), reg)
        flgrid[i][mask] = ifl(wv[mask]) / Cr
        errgrid[i][mask] = ier(wv[mask]) / Cr
    ################

    flgrid=np.transpose(flgrid)
    errgrid=np.transpose(errgrid)
    weigrid=errgrid**(-2)
    infmask=np.isinf(weigrid)
    weigrid[infmask]=0
    ################

    stack,err=np.zeros([2,len(wv)])
    for i in range(len(wv)):
        stack[i]=np.sum(flgrid[i]*weigrid[[i]])/np.sum(weigrid[i])
        err[i]=1/np.sqrt(np.sum(weigrid[i]))
    ################
    ###take out nans

    IDX=[U for U in range(len(wv)) if stack[U] > 0]

    return wv[IDX], stack[IDX], err[IDX]

ids,lmass,rshift=np.array(Readfile('masslist_sep28.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)
nlist=glob('spec_stacks/*')

IDS=[]

for i in range(len(ids)):
    if 10.87<lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

speclist=[]
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i]==nlist[ii][12:18]:
            speclist.append(nlist[ii])

speczs = np.round(rshift[IDS], 2)

swv, sfl, serr = Stack_spec_normwmean(speclist, speczs, np.arange(3000,5550,5))

"""mgbfe"""
# ew=np.zeros(len(age))
# ewgr=np.zeros(len(age))
# for i in range(len(age)):
#     wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m0.015_a%s_t8.72_spec.dat' % age[i], 1))
#     ew[i]= Mgb_Fe(wv,fl)
#     wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m0.015_a%s_t8.72_z1.0_model.dat' % age[i], 1))
#     wvgr/=2
#     ewgr[i]=Mgb_Fe(wvgr,flgr)
#
# M=Mgb_Fe(swv,sfl)
# print M

# plt.plot(age,ew,label='Normal resolution')
# plt.plot(age,ewgr,label='Grism resolution')
# plt.hlines(M,min(metal),max(metal))
# plt.ylabel("[Mg/Fe]' ($\AA$)",size=20)
# plt.xlabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# plt.show()
# plt.savefig('../research_plots/lick_mgbfe_age.png')
# plt.close()
# #
# ew=np.zeros(len(metal))
# ewgr=np.zeros(len(metal))
# for i in range(len(metal)):
#     wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m%s_a1.62_t8.72_spec.dat' % metal[i], 1))
#     ew[i]= Mgb_Fe(wv,fl)
#     wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m%s_a1.62_t8.72_z1.0_model.dat' % metal[i], 1))
#     wvgr/=2
#     ewgr[i]=Mgb_Fe(wvgr,flgr)
#
# plt.plot(metal/.02,ew,label='Normal resolution')
# plt.plot(metal/.02,ewgr,label='Grism resolution')
# plt.hlines(M,min(metal),max(metal))
# plt.ylabel("[Mg/Fe]' ($\AA$)",size=20)
# plt.xlabel('Metallicity (Z/Z$_\odot$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# # plt.savefig('../research_plots/lick_mgbfe_metal.png')
# plt.show()

"""d4000"""
ew=np.zeros(len(age))
ewgr=np.zeros(len(age))
for i in range(len(age)):
    wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m0.015_a%s_t8.72_spec.dat' % age[i], 1))
    ew[i]= D4000(wv,fl)
    wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m0.015_a%s_t8.72_z1.5_model.dat' % age[i], 1))
    wvgr/=2.5
    ewgr[i]=D4000(wvgr,flgr)

M=D4000(swv,sfl)
print M

print wvgr[0],wvgr[-1]
print swv[0],swv[-1]

C1=Scale_model(sfl,serr,interp1d(wv,fl)(swv))
# C2=Scale_model(sfl,serr,interp1d(wvgr,flgr)(swv))

plt.plot(swv,sfl)
plt.plot(wv,fl*C1)
# plt.plot(wvgr,flgr*C2)
plt.xlim(3000,5500)
plt.show()
# plt.plot(age,ew,label='Normal resolution')
# plt.plot(age,ewgr,label='Grism resolution')
# plt.hlines(M,min(age),max(age))
# plt.ylabel("D4000",size=20)
# plt.xlabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# plt.show()
# # # plt.savefig('../research_plots/lick_mgbfe_age.png')
# # # plt.close()
# # #
# ew=np.zeros(len(metal))
# ewgr=np.zeros(len(metal))
# for i in range(len(metal)):
#     wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m%s_a1.62_t8.72_spec.dat' % metal[i], 1))
#     ew[i]= D4000(wv,fl)
#     wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m%s_a1.62_t8.72_z1.5_model.dat' % metal[i], 1))
#     wvgr/=2.5
#     ewgr[i]=D4000(wvgr,flgr)
# #
# plt.plot(metal/.019,ew,label='Normal resolution')
# plt.plot(metal/.019,ewgr,label='Grism resolution')
# plt.hlines(M,min(metal)/.019,max(metal)/.019)
# plt.ylabel("D4000",size=20)
# plt.xlabel('Metallicity (Z/Z$_\odot$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# # plt.savefig('../research_plots/lick_mgbfe_metal.png')
# plt.show()

"""H_delta"""
# ew=np.zeros(len(age))
# ewgr=np.zeros(len(age))
# for i in range(len(age)):
#     wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m0.015_a%s_t8.72_spec.dat' % age[i], 1))
#     ew[i]= H_d(wv,fl)
#     wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m0.015_a%s_t8.72_z1.5_model.dat' % age[i], 1))
#     wvgr/=2.5
#     ewgr[i]=H_d(wvgr,flgr)
#
# M=H_d(swv,sfl)
# print M
#
# plt.plot(age,ew,label='Normal resolution')
# plt.plot(age,ewgr,label='Grism resolution')
# plt.hlines(M,min(age),max(age))
# plt.ylabel("H$_\delta$ ($\AA$)",size=20)
# plt.xlabel('Age (Gyrs)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# plt.show()
# # # plt.savefig('../research_plots/lick_mgbfe_age.png')
# # # plt.close()
# # #
# ew=np.zeros(len(metal))
# ewgr=np.zeros(len(metal))
# for i in range(len(metal)):
#     wv, fl = np.array(Readfile('../../../fsps_models_for_fit/fsps_spec/m%s_a1.62_t8.72_spec.dat' % metal[i], 1))
#     ew[i]= H_d(wv,fl)
#     wvgr,flgr,ergr= np.array(Readfile('../../../fsps_models_for_fit/models/m%s_a1.62_t8.72_z1.5_model.dat' % metal[i], 1))
#     wvgr/=2.5
#     ewgr[i]=H_d(wvgr,flgr)
# #
# plt.plot(metal/.019,ew,label='Normal resolution')
# plt.plot(metal/.019,ewgr,label='Grism resolution')
# plt.hlines(M,min(metal)/.019,max(metal)/.019)
# plt.ylabel("H$_\delta$ ($\AA$)",size=20)
# plt.xlabel('Metallicity (Z/Z$_\odot$)',size=20)
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=2,fontsize=12)
# # plt.savefig('../research_plots/lick_mgbfe_metal.png')
# plt.show()