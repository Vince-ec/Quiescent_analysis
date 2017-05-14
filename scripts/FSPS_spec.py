import fsps
import numpy as np


def FSPS_spec(metallcity,age,redshift):
    L=3.828E33
    sp = fsps.StellarPopulation(imf_type=0,tpagb_norm_type=0,zred=redshift)
    wv,fl=np.array(sp.get_spectrum(zmet=metallcity,tage=age,peraa=True))
    fl/=L
    return wv,fl


# age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
#        1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
mval = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025,
         0.0031, 0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300])
metal=np.arange(len(mval))+1
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]
# z=np.linspace(1,1.8,17)

for i in range(len(metal)):
    for ii in range(len(age)):
        for iii in range(len(tau)):
            if tau[iii] == 0:
                sp = fsps.StellarPopulation(imf_type=0, tpagb_norm_type=0, sfh=1,tau=np.power(10, (tau[iii] - 9)))
            else:
                sp = fsps.StellarPopulation(imf_type=0, tpagb_norm_type=0)
            wv, fl = np.array(sp.get_spectrum(zmet=metal[i], tage=age[ii], peraa=True))
            ascii.write([wv,fl],'m%s_a%s_t%s_spec.dat' % (mval[i],age[ii],tau[iii]))