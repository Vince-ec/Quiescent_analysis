import numpy as np
from spec_id import Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat
from vtl.Readfile import Readfile
import seaborn as sea

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

ids,speclist,lmass,rshift,rad,sig,comp=np.array(Readfile('masslist_mar22.dat',is_float=False))
lmass,rshift,rad,sig,comp=np.array([lmass,rshift,rad,sig,comp]).astype(float)

IDc=[]  # compact sample
IDd=[]  # diffuse sample

IDmL=[]  # low mass sample
IDmH=[]  # high mass sample

for i in range(len(ids)):
    if 0.11 < comp[i]:
        IDd.append(i)
    if 0.11 > comp[i]:
        IDc.append(i)
    if 10.931 > lmass[i]:
        IDmL.append(i)
    if 10.931 < lmass[i]:
        IDmH.append(i)

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

mp = [0.005,0.005,0.005,0.012,0.012,0.012,0.019,0.019,0.019,0.024,0.024,0.024]
ap = [1.5,3.0,4.5,1.5,3.0,4.5,1.5,3.0,4.5,1.5,3.0,4.5]

###Compact
for i in range(len(mp)):
    Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat(speclist[IDc], tau, metal, age, mp[i], ap[i], 0, rshift[IDc],
                                                          ids[IDc], np.arange(3100, 5500, 10),
                                                          'com_mcerr_mar23_%s_%s' % (mp[i], ap[i]),
                                                          'com_mar22_spec', repeats=1000,
                                                          tau_scale='tau_scale_ntau.dat')

###Diffuse
for i in range(len(mp)):
    Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat(speclist[IDd], tau, metal, age, mp[i], ap[i], 0, rshift[IDd],
                                                          ids[IDd], np.arange(3450, 5500, 10),
                                                          'ext_mcerr_mar23_%s_%s' % (mp[i], ap[i]),
                                                          'ext_mar22_spec', repeats=1000,
                                                          tau_scale='tau_scale_ntau.dat')

###gt
for i in range(len(mp)):
    Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat(speclist[IDmH], tau, metal, age, mp[i], ap[i], 0,
                                                          rshift[IDmH],
                                                          ids[IDmH], np.arange(3100, 5500, 10),
                                                          'gt10.93_mcerr_mar23_%s_%s' % (mp[i], ap[i]),
                                                          'gt10.93_mar22_spec', repeats=1000,
                                                          tau_scale='tau_scale_ntau.dat')

###lt
for i in range(len(mp)):
    Model_fit_sim_stack_MCerr_bestfit_normwmean_cont_feat(speclist[IDmL], tau, metal, age, mp[i], ap[i], 0,
                                                          rshift[IDmL],
                                                          ids[IDmL], np.arange(3100, 5500, 10),
                                                          'lt10.93_mcerr_mar23_%s_%s' % (mp[i], ap[i]),
                                                          'lt10.93_mar22_spec', repeats=1000,
                                                          tau_scale='tau_scale_ntau.dat')
