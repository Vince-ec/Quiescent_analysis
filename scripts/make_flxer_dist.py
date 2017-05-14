import numpy as np
from spec_id import Stack_spec_normwmean,Stack_model_normwmean,Likelihood_contours,\
    Analyze_Stack_avgage,Identify_stack, Gauss_dist
import matplotlib.pyplot as plt
from vtl.Readfile import Readfile
from glob import glob
from astropy.io import fits,ascii
from scipy.interpolate import interp1d
import os
import cPickle
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDS=[]

for i in range(len(ids)):
    if 10.871>=lmass[i] and 1<rshift[i]<1.75:
        IDS.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
     2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]

"""models"""
# Pr,bfage,bfmetal=Analyze_Stack_avgage('chidat/gt10.87_fsps_nage_noer_stackfit_chidata.fits', np.array(tau),metal,age)
# for i in range(len(metal)):
#     print age[np.argmax(Pr.T[i])], metal[i]

wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3380,5600,1))

flx_list=np.zeros([11,len(wv)])

flist6=[]
flist7=[]
flist8=[]
flist9=[]
flist10=[]
flist11=[]
flist12=[]
flist13=[]
flist14=[]
flist15=[]
flist16=[]
for i in range(len(rshift[IDS])):
    flist6.append('../../../fsps_models_for_fit/models/m0.0068_a2.95_t0_z%s_model.dat' % rshift[IDS][i])
    flist7.append('../../../fsps_models_for_fit/models/m0.0077_a2.68_t0_z%s_model.dat' % rshift[IDS][i])
    flist8.append('../../../fsps_models_for_fit/models/m0.0085_a2.68_t0_z%s_model.dat' % rshift[IDS][i])
    flist9.append('../../../fsps_models_for_fit/models/m0.0096_a2.64_t0_z%s_model.dat' % rshift[IDS][i])
    flist10.append('../../../fsps_models_for_fit/models/m0.0106_a2.56_t0_z%s_model.dat' % rshift[IDS][i])
    flist11.append('../../../fsps_models_for_fit/models/m0.012_a2.44_t0_z%s_model.dat' % rshift[IDS][i])
    flist12.append('../../../fsps_models_for_fit/models/m0.0132_a2.44_t0_z%s_model.dat' % rshift[IDS][i])
    flist13.append('../../../fsps_models_for_fit/models/m0.014_a2.3_t0_z%s_model.dat' % rshift[IDS][i])
    flist14.append('../../../fsps_models_for_fit/models/m0.015_a2.2_t0_z%s_model.dat' % rshift[IDS][i])
    flist15.append('../../../fsps_models_for_fit/models/m0.0164_a2.11_t0_z%s_model.dat' % rshift[IDS][i])
    flist16.append('../../../fsps_models_for_fit/models/m0.018_a2.11_t0_z%s_model.dat' % rshift[IDS][i])
# #
fwv6,flx_list[0],fe6=Stack_model_normwmean(speclist[IDS],flist6, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv7,flx_list[1],fe7=Stack_model_normwmean(speclist[IDS],flist7, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv8,flx_list[2],fe8=Stack_model_normwmean(speclist[IDS],flist8, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv9,flx_list[3],fe9=Stack_model_normwmean(speclist[IDS],flist9, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv10,flx_list[4],fe10=Stack_model_normwmean(speclist[IDS],flist10, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv11,flx_list[5],fe11=Stack_model_normwmean(speclist[IDS],flist11, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv12,flx_list[6],fe12=Stack_model_normwmean(speclist[IDS],flist12, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv13,flx_list[7],fe13=Stack_model_normwmean(speclist[IDS],flist13, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv14,flx_list[8],fe14=Stack_model_normwmean(speclist[IDS],flist14, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv15,flx_list[9],fe15=Stack_model_normwmean(speclist[IDS],flist15, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))
fwv16,flx_list[10],fe16=Stack_model_normwmean(speclist[IDS],flist16, rshift[IDS], np.arange(wv[0],wv[-1]+1,1))

"""show neighbors"""
# fs6=flx_list[0]
# fs7=flx_list[1]
# fs8=flx_list[2]
# fs9=flx_list[3]
# fs10=flx_list[4]
# fs11=flx_list[5]
# fs12=flx_list[6]
# fs13=flx_list[7]
# fs14=flx_list[8]
# fs15=flx_list[9]
# fs16=flx_list[10]
#
# chi=np.zeros(11)
# chi[0]=np.round(sum(((fl - fs6) / er) ** 2),2)
# chi[1]=np.round(sum(((fl - fs7) / er) ** 2),2)
# chi[2]=np.round(sum(((fl - fs8) / er) ** 2),2)
# chi[3]=np.round(sum(((fl - fs9) / er) ** 2),2)
# chi[4]=np.round(sum(((fl - fs10) / er) ** 2),2)
# chi[5]=np.round(sum(((fl - fs11) / er) ** 2),2)
# chi[6]=np.round(sum(((fl - fs12) / er) ** 2),2)
# chi[7]=np.round(sum(((fl - fs13) / er) ** 2),2)
# chi[8]=np.round(sum(((fl - fs14) / er) ** 2),2)
# chi[9]=np.round(sum(((fl - fs15) / er) ** 2),2)
# chi[10]=np.round(sum(((fl - fs16) / er) ** 2),2)
#
# fs6*=1000
# fs7*=1000
# fs8*=1000
# fs9*=1000
# fs10*=1000
# fs11*=1000
# fs12*=1000
# fs13*=1000
# fs14*=1000
# fs15*=1000
# fs16*=1000
# fl*=1000
# er*=1000
#
# plt.plot(wv,fl,'k',alpha=.7,linewidth=1,label='>10.87 Stack')
# plt.fill_between(wv,fl-er,fl+er,color='k',alpha=.3)
# plt.plot(fwv6,fs6, color='#8E2E4E',alpha=.33,label='Z=%s, $\chi^2$=%s' % (np.round((.0068/.019),2), chi[0]))
# plt.plot(fwv7,fs7, color='#8E2E4E',alpha=.66,label='Z=%s, $\chi^2$=%s' % (np.round((.0077/.019),2), chi[1]))
# plt.plot(fwv8,fs8, color='#8E2E4E',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.0085/.019),2), chi[2]))
# plt.plot(fwv9,fs9, color='#264C67',alpha=.25,label='Z=%s, $\chi^2$=%s' % (np.round((.0096/.019),2), chi[3]))
# plt.plot(fwv10,fs10, color='#264C67',alpha=.5,label='Z=%s, $\chi^2$=%s' % (np.round((.0106/.019),2), chi[4]))
# plt.plot(fwv11,fs11, color='#264C67',alpha=.75,label='Z=%s, $\chi^2$=%s' % (np.round((.012/.019),2), chi[5]))
# plt.plot(fwv12,fs12, color='#264C67',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.0132/.019),2), chi[6]))
# plt.plot(fwv13,fs13, color='#5F912F',alpha=.33,label='Z=%s, $\chi^2$=%s' % (np.round((.014/.019),2), chi[7]))
# plt.plot(fwv14,fs14, 'k', label='Z=%s, $\chi^2$=%s' % (np.round((.015/.019),2), chi[8]))
# plt.plot(fwv15,fs15, color='#5F912F',alpha=.66,label='Z=%s, $\chi^2$=%s' % (np.round((.0164/.019),2), chi[9]))
# plt.plot(fwv16,fs16, color='#5F912F',alpha=1,label='Z=%s, $\chi^2$=%s' % (np.round((.018/.019),2), chi[10]))
# plt.axvspan(3910, 3979, alpha=.2)
# plt.axvspan(3981, 4030, alpha=.2)
# plt.axvspan(4082, 4122, alpha=.2)
# plt.axvspan(4250, 4400, alpha=.2)
# plt.axvspan(4830, 4930, alpha=.2)
# plt.axvspan(4990, 5030, alpha=.2)
# plt.axvspan(5109, 5250, alpha=.2)
# plt.xlim(min(wv),max(wv))
# plt.ylim(.5,6.5)
# plt.xlabel('Restframe Wavelength ($\AA$)',size=15)
# plt.ylabel('Relative Flux',size=15)
# plt.title('N=%s' % len(wv))
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.legend(loc=4)
# plt.show()

"""get residuals"""

res_list=np.zeros([11,len(wv)])
for i in range(len(flx_list)):
    res_list[i]=np.abs((fl-flx_list[i])/er)

res_mean=[np.mean(U) for U in res_list.T]

u=0
res_smooth=[]
wv_smooth=[]
while u < len(wv):
    res_smooth.append(np.trapz(res_mean[u:u+20],wv[u:u+20]))
    wv_smooth.append(np.mean(wv[u:u+20]))
    u+=20

wv,fl,er=Stack_spec_normwmean(speclist[IDS],rshift[IDS],np.arange(3600,5350,10))
mwv,mfl,mer=Stack_model_normwmean(speclist[IDS],flist11, rshift[IDS], np.arange(wv[0],wv[-1]+10,10))

flxerr=interp1d(wv_smooth,res_smooth)(wv)/20000
# # print flxerr
# ascii.write([wv,flxerr],'flx_err/LM_10_3600-5350.dat')
#
error=np.sqrt(er**2+(flxerr*fl)**2)
# #
#
nsr=(fl-mfl)/error

print 'mu=%0.3f' % np.mean(nsr)
print 'sigma=%0.3f' % np.std(nsr)

# rng=np.linspace(-3,3,100)
# # #
# plt.hist(nsr,15,normed=True)
# plt.plot(rng,Gauss_dist(rng,0,1),label='$\mu$=0, $\sigma$=1')
# plt.axvline(np.mean(nsr),color='k',alpha=.5,linestyle='-.',label='$\mu$=%0.3f' % np.mean(nsr))
# plt.axvline(np.std(nsr),color='k',alpha=.5,linestyle='--',label='$\sigma$=%0.3f' % np.std(nsr))
# plt.axvline(-np.std(nsr),color='k',alpha=.5,linestyle='--')
# plt.xlabel('Sigma',size=15)
# plt.ylabel('Normalized units',size=15)
# plt.legend()
# plt.minorticks_on()
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
# #
# plt.plot(wv,fl)
# plt.plot(wv,er)
# plt.plot(wv,error)
# plt.show()
