from vtl.Readfile import Readfile
from spec_id import Analyze_Stack_avgage, Stack_spec_normwmean,Stack_model_normwmean,Model_fit_stack_normwmean
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from glob import glob
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colors = [(0,i,i,i) for i in np.linspace(0,1,3)]
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

"""galaxy selection"""
ids,speclist,lmass,rshift=np.array(Readfile('masslist_dec8.dat',1,is_float=False))
lmass,rshift=np.array([lmass,rshift]).astype(float)

IDA=[]  # all masses in sample
IDL=[]  # low mass sample
IDH=[]  # high mass sample

for i in range(len(ids)):
    if 10.0<=lmass[i] and 1<rshift[i]<1.75:
        IDA.append(i)
    if 10.871>lmass[i] and 1<rshift[i]<1.75:
        IDL.append(i)
    if 10.871<lmass[i] and 1<rshift[i]<1.75:
        IDH.append(i)

metal=np.array([ 0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061,  0.0068,  0.0077,  0.0085,  0.0096,  0.0106,
                  0.012, 0.0132, 0.014,  0.0150,  0.0164, 0.018,  0.019,  0.021,  0.024, 0.027, 0.03])
age=[0.5, 0.65, 0.84, 1.1, 1.62, 2.11, 2.2, 2.26, 2.3, 2.35, 2.38, 2.44, 2.56, 2.64, 2.68,
     2.7, 2.75, 2.79, 2.81, 2.95, 3.12, 3.35, 3.45, 3.56, 4.62, 6.0]
tau=[0,8.0,8.15,8.28,8.43,8.57,8.72,8.86,9.0,9.14,9.29,9.43,9.57,9.71,9.86,10.0]


"""High mass pickles"""

R=[5,10]

for i in range(len(R)):
    Model_fit_stack_normwmean(speclist[IDH],tau,metal,age,rshift[IDH],np.arange(3250,5500,R[i]),
                         'gt10.87_fsps_10-13_%s_stackfit' % R[i],'gt10.87_fsps_10-13_%s_spec' % R[i],
                              res=R[i],fsps=True)

"""Low mass pickles"""
for i in range(len(R)):
    Model_fit_stack_normwmean(speclist[IDL],tau,metal,age,rshift[IDL],np.arange(3400,5500,R[i]),
                         'ls10.87_fsps_10-13_%s_stackfit' % R[i],'ls10.87_fsps_10-13_%s_spec' % R[i],
                              res=R[i],fsps=True)