from spec_id  import Scale_model,Identify_stack,Analyze_stack, Stack_model
from astropy.io import ascii
from astropy.table import Table
from scipy.interpolate import interp1d
from vtl.Readfile import Readfile
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

def Stack_sim_spec(wv,fl,err,redshifts, wv_range):

    Data=[]

    for i in range(len(wv)):
        w = wv[i]/ (1 + redshifts[i])
        for ii in range(len(fl[i])):
            if fl[i][ii] < 0:
                fl[i][ii] = 0

        mr = [[3910, 3990], [4082, 4122], [4250, 4330], [4830, 4890], [4990, 5030]]

        Mask = np.zeros(len(w))

        for ii in range(len(mr)):
            for iii in range(len(w)):
                if mr[ii][0] <= w[iii] <= mr[ii][1]:
                    Mask[iii] = 1

        maskw = np.ma.masked_array(w, Mask)
        x3, x2, x1, x0 = np.ma.polyfit(maskw, fl[i], 3, w=1 / err[i] ** 2)
        C0 = x3 * w ** 3 + x2 * w ** 2 + x1 * w + x0
        f = fl[i]/C0
        er = err[i]/C0
        Data.append(np.array([w,f,er]))

    wv=np.array(wv_range)

    interpdata=[]
    interpweight=[]
    stack=np.zeros(len(wv))
    err=np.zeros(len(wv))

    for i in range(len(Data)):
        wt=(Data[i][2])**(-2)
        interpdata.append(interp1d(Data[i][0],Data[i][1]))
        interpweight.append(interp1d(Data[i][0],wt))

    for i in range(len(wv)):
        flu=np.zeros(len(interpdata))
        wei=np.zeros(len(interpdata))
        for ii in range(len(interpdata)):
            if Data[ii][0][0]<=wv[i]<=Data[ii][0][-1]:
                flu[ii]=interpdata[ii](wv[i])
                wei[ii]=interpweight[ii](wv[i])
        stack[i] = np.sum(wei*flu)/np.sum(wei)
        err[i]=1/np.sqrt(np.sum(wei))

    return wv, stack, err;

def Stack_model_spec(modellist, wvlist,fllist, errlist, zlist, redshifts, redshift_counts, wv_range):
    Data = []

    for i in range(len(modellist)):
        W, F, E = np.array(Readfile(modellist[i], 1))
        W= W/(1+redshifts[i])
        ifl = interp1d(W, F)

        for ii in range(len(redshift_counts[i])):
            zwv = wvlist[redshift_counts[i][ii]]/(1+zlist[redshift_counts[i][ii]])
            zfl = ifl(zwv)
            zer = errlist[redshift_counts[i][ii]]
            C = Scale_model(fllist[redshift_counts[i][ii]], zer, zfl)
            zfl *= C

            mr = [[3910, 3990], [4082, 4122], [4250, 4330], [4830, 4890], [4990, 5030]]

            Mask = np.zeros(len(zwv))

            for iii in range(len(mr)):
                for iv in range(len(zwv)):
                    if mr[iii][0] <= zwv[iv] <= mr[iii][1]:
                        Mask[iv] = 1

            maskw = np.ma.masked_array(zwv, Mask)

            x3, x2, x1, x0 = np.ma.polyfit(maskw, zfl, 3, w=1 / zer ** 2)
            C0 = x3 * zwv ** 3 + x2 * zwv ** 2 + x1 * zwv + x0

            zfl =zfl/ C0
            zer =zer/ C0

            # plt.plot(zwv,zer)

            Data.append([zwv, zfl, zer])
    # plt.show()
    wv = np.array(wv_range)

    interpdata = []
    interpweight = []
    stack = np.zeros(len(wv))
    err = np.zeros(len(wv))

    for i in range(len(Data)):
        wt = (Data[i][2]) ** (-2)
        interpdata.append(interp1d(Data[i][0], Data[i][1]))
        interpweight.append(interp1d(Data[i][0], wt))

    for i in range(len(wv)):
        flu = np.zeros(len(interpdata))
        wei = np.zeros(len(interpdata))
        for ii in range(len(interpdata)):
            if Data[ii][0][0] <= wv[i] <= Data[ii][0][-1]:
                flu[ii] = interpdata[ii](wv[i])
                wei[ii] = interpweight[ii](wv[i])
        stack[i] = np.sum(wei * flu) / np.sum(wei)
        err[i] = 1 / np.sqrt(np.sum(wei))

    return wv, stack, err

def Model_fit_stack(w,f,er, metal, A, speczs, wv_range,name, fsps=False):

    #############Get redshift info###############

    zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]

    for i in range(len(speczs)):
        zlist.append(int(speczs[i] * 100) / 5 / 20.)
    for i in range(len(bins)):
        b = []
        for ii in range(len(zlist)):
            if bins[i] == zlist[ii]:
                b.append(ii)
        if len(b) > 0:
            zcount.append(b)
    zbin = sorted(set(zlist))

    ##############Stack spectra################

    wv,fl,err=Stack_sim_spec(w,f,er,speczs,wv_range)

    #############Name of output file###############

    chifile='chidat/%s_chidata.dat' % name

    #############Get list of models to fit againts##############

    if fsps==False:
        filepath = '../../../Models_for_fitting/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(zbin)):
                    a.append(filepath + 'gal_%s_%s_%s.dat' % (metal[i], A[ii], zbin[iii]))
                m.append(a)
            modellist.append(m)

    else:
        filepath = '../../../fsps_models_for_fit/models/'
        modellist = []
        for i in range(len(metal)):
            m = []
            for ii in range(len(A)):
                a = []
                for iii in range(len(zbin)):
                    if zbin[iii]==1:
                        zbin[iii]=int(1)
                    a.append(filepath + 'm%s_a%s_z%s_model.dat' % (metal[i], A[ii], zbin[iii]))
                m.append(a)
            modellist.append(m)

    ##############Create chigrid#################

    chigrid=np.zeros([len(metal),len(A)])
    for i in range(len(metal)):
        for ii in range(len(A)):
            mw,mf,me=Stack_model_spec(modellist[i][ii],w,f,er,
                                 speczs,zbin,zcount,wv_range)
            chigrid[i][ii]=Identify_stack(wv,fl,err,mf)

    chigrid=np.array(chigrid)

    ################Write chigrid file###############

    dat=Table(chigrid)
    ascii.write(dat,chifile)

    ################Find best fit##################

    MM,AA,mm,aa=Analyze_stack(chifile,metal,A,fsps=fsps)

    return MM,AA,mm,aa

#########################################################################

speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_zps2.dat',1,is_float=False))

zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

zlist,zbin,binlist,bins=[[],[],[],np.linspace(1,1.8,17)]

for i in range(len(zsmax)):
    zlist.append(int(zsmax[i]*100)/5/20.)
for i in range(len(bins)):
    b=[]
    for ii in range(len(zlist)):
        if bins[i]==zlist[ii]:
            b.append(ii)
    if len(b)>0:
        binlist.append(b)
zbin=sorted(set(zlist))

modeldat=[]
z = zsmax
filepath = '../../../fsps_models_for_fit/models/'
for i in range(len(z)):
    modeldat.append(filepath + 'm0.019_a2.11_z%s_model.dat' % z[i])

modeldat2=[]
z = zbin
filepath = '../../../fsps_models_for_fit/models/'
for i in range(len(z)):
    if z[i]==1:
        z[i]=int(1)
    modeldat2.append(filepath + 'm0.019_a2.11_z%s_model.dat' % z[i])

smwv,smfl,smer=[[],[],[]]
cwv,cfl,cer=[[],[],[]]

for i in range(len(speclist)):
    swv, sfl, serr = np.array(Readfile(speclist[i], 1))
    mwv, mfl, merr = np.array(Readfile(modeldat[i], 1))

    ifl=interp1d(mwv,mfl)
    nfl=ifl(swv)
    C = Scale_model(sfl, serr, nfl)
    nfl *= C

    ier = interp1d(mwv, merr)
    ner = ier(swv)
    ner *= C
    cwv.append(np.array(swv))
    cfl.append(np.array(nfl))
    cer.append(np.array(serr))

    nfl+=np.random.normal(0,serr)

    smwv.append(swv)
    smfl.append(nfl)
    smer.append(serr)

twv=np.array(cwv)
tfl=np.array(cfl)
ter=np.array(cer)

sw,ss,se=Stack_sim_spec(smwv,smfl,smer,zsmax,np.arange(3250,5550,10))
cw,cs,ce=Stack_sim_spec(cwv,cfl,cer,zsmax,np.arange(3250,5550,10))

age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
#
metal = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]
#
# print Model_fit_stack(smwv,smfl,smer,metal,age,zsmax,np.arange(3250,5550,10),'simstack_fit_fsps_werr',fsps=True)
# print Model_fit_stack(cwv,cfl,cer,metal,age,zsmax,np.arange(3250,5550,10),'simstack_fit_fsps',fsps=True)

mw,ms,me=Stack_model_spec(modeldat2,twv,tfl,ter,zsmax,zbin,binlist,np.arange(3250,5550,10))

# x = ((cs -  ms) / ce) ** 2
# print sum(x)
# plt.plot(sw,ss)
# plt.plot(sw,se)
# plt.plot(mw,ms)
# plt.plot(mw,me)
plt.plot(cw,cs)
plt.plot(cw,ce)
plt.show()