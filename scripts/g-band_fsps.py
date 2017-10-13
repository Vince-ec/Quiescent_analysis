#!/home/vestrada78840/miniconda2/envs/astroconda/bin/python

import fsps
import numpy as np
import sys

v1= float(sys.argv[1])

metal = v1
age = np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

gband_flux = np.zeros([len(tau),len(age)])
for i in range(len(tau)):
    if tau[i]==0:
        sp = fsps.StellarPopulation(imf_type=0,tpagb_norm_type=0,zcontinuous=1,logzsol=np.log10(metal/0.019), sfh=0)
    else:
        ultau=np.round(np.power(10,np.array(tau[i])-9),2)
        sp = fsps.StellarPopulation(imf_type=0,tpagb_norm_type=0,zcontinuous=1,logzsol=np.log10(metal/0.019), sfh=4,tau=ultau)
    for ii in range(len(age)):
        gband=sp.get_mags(tage=age[ii],bands=['sdss_g'])[0]
        gband_flux[i][ii]= 10**(-gband/2.5)

fn = '../../../../fdata/scratch/vestrada78840/data/m%s_gbf' % (metal)
np.save(fn, gband_flux)
