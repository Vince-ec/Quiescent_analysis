import numpy as np
from vtl.Readfile import Readfile
from scipy.interpolate import interp1d
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sea
from spec_id import Scale_model
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

"""read in data"""
###read in data
speclist,zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,\
zps,zpsl,zpsh=np.array(Readfile('stack_redshifts_10-6.dat',is_float=False))
zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh=np.array(
    [zsmax,zschi,zs,zsl,zsh,zpsmax,zpschi,zps,zpsl,zpsh]).astype(float)

# print speclist[0][18:24]
### set index and file path to models
IDS=45
fp='/Users/Vince.ec/fsps_models_for_fit/fsps_spec/'
print speclist[IDS][18:24]

### read data and models
mwv,mfl=np.array(Readfile(fp+'m0.015_a1.5_t8.0_spec.dat'))
wv,fl,er=np.array(Readfile(speclist[IDS]))

"""plotting"""
### zps
w=wv/(1+zsmax[IDS])
Mfl=interp1d(mwv,mfl)(w)
C=Scale_model(fl,er,Mfl)

plt.plot(w,fl,label='%s' % zsmax[IDS])
plt.plot(w,C*Mfl)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.title('%s' % speclist[IDS][18:24])
plt.xlim(3600,5500)
plt.legend()
# plt.show()
plt.savefig('../research_plots/zsmax.png')
plt.close()

### zps
w=wv/(1+zs[IDS])
Mfl=interp1d(mwv,mfl)(w)
C=Scale_model(fl,er,Mfl)

plt.plot(w,fl,label='%s' % zs[IDS])
plt.plot(w,C*Mfl)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.title('%s' % speclist[IDS][18:24])
plt.xlim(3600,5500)
plt.legend()
# plt.show()
plt.savefig('../research_plots/zs.png')
plt.close()

### zps
w=wv/(1+zpsmax[IDS])
Mfl=interp1d(mwv,mfl)(w)
C=Scale_model(fl,er,Mfl)

plt.plot(w,fl,label='%s' % zpsmax[IDS])
plt.plot(w,C*Mfl)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.title('%s' % speclist[IDS][18:24])
plt.xlim(3600,5500)
plt.legend()
# plt.show()
plt.savefig('../research_plots/zpsmax.png')
plt.close()

### zps
w=wv/(1+zps[IDS])
Mfl=interp1d(mwv,mfl)(w)
C=Scale_model(fl,er,Mfl)

plt.plot(w,fl,label='%s' % zps[IDS])
plt.plot(w,C*Mfl)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.title('%s' % speclist[IDS][18:24])
plt.xlim(3600,5500)
plt.legend()
# plt.show()
plt.savefig('../research_plots/zps.png')
plt.close()