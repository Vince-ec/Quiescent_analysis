from spec_id import Get_flux
from spec_id import Get_flux2
from spec_tools import Normalize
from spec_tools import Simspec
from vtl.Readfile import Readfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

info=Readfile('specfits2.dat',1,is_float=False)
speclist=np.array(info[0])
allz=np.array(info[7]).astype(float)
allzmax=np.array(info[12]).astype(float)
info=Readfile('specfits3.dat',1,is_float=False)
z=np.array(info[1]).astype(float)
zmax=np.array(info[2]).astype(float)
wv,fl=np.array(Readfile('model_spec/spec_62_2_1.0.dat',0))
wv/=2

idx=7

fn=speclist[idx].replace('../extractions_quiescent_mar17/','')
w,f,e=Get_flux(fn)
# f*=3E7

plt.plot(w,f)
plt.fill_between(w,f-e,f+e,color=sea.color_palette('muted')[5],alpha=.9)
# plt.plot(wv*(1+allz[idx]),fl)
# plt.plot(wv*(1+allzmax[idx]),fl)
# plt.plot(wv*(1+z[idx]),fl)
# plt.plot(wv*(1+zmax[idx]),fl)
# plt.xlim(8000,11500)
plt.show()