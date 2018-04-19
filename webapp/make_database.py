from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import interp1d,interp2d
from yattag import Doc
import re
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sea
import numpy as np
import pandas as pd
from shutil import copyfile
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, 
                             light=1.2, as_cmap=True)

SIDF = pd.read_pickle('south_img_DF_vr.pkl')
NIDF = pd.read_pickle('north_img_DF_vr.pkl')

spth2a = '/Volumes/Vince_research/Extractions/Plots/South/2D/stack/'
spth2i = '/Volumes/Vince_research/Extractions/Plots/South/2D/spec/'
spth1a = '/Volumes/Vince_research/Extractions/Plots/South/1D/stack/'
spth1i = '/Volumes/Vince_research/Extractions/Plots/South/1D/spec/'

npth2a = '/Volumes/Vince_research/Extractions/Plots/North/2D/stack/'
npth2i = '/Volumes/Vince_research/Extractions/Plots/North/2D/spec/'
npth1a = '/Volumes/Vince_research/Extractions/Plots/North/1D/stack/'
npth1i = '/Volumes/Vince_research/Extractions/Plots/North/1D/spec/'

for i in range(len(SIDF.index)):
    if len(glob(spth2a + '*' + str(SIDF['id'][i]) + '*')) > 0:
        twodnames = glob(spth2a + '*' + str(SIDF['id'][i]) + '*')
        SIDF['2d_all'][i] = twodnames
        fids = []
        
        for ii in range(len(twodnames)):
            fids.append(twodnames[ii].split('/')[-1].split('-')[0])
    
        SIDF['field'][i] = fids
    
    else:
        SIDF['2d_all'][i] = 'none'
        SIDF['field'][i] = 'none'
        
    if len(glob(spth2i + '*' + str(SIDF['id'][i]) + '*')) > 0:
        SIDF['2d_ind'][i] = glob(spth2i + '*' + str(SIDF['id'][i]) + '*')

    else:
        SIDF['2d_ind'][i] = 'none'
 
    stack_names = glob(spth1a + '*' + str(SIDF['id'][i]) + '*')
    if len(stack_names) > 0:
        SIDF['stack'][i] = stack_names
    
    else:
        SIDF['stack'][i] = 'none'
    
    ind_names = glob(spth1i + '*' + str(SIDF['id'][i]) + '*')
    if len(ind_names) > 0:
        SIDF['ind_spec'][i] = ind_names
    
    else:
        SIDF['ind_spec'][i] = 'none'
  
###################################

for i in range(len(NIDF.index)):
    if len(glob(npth2a + '*' + str(NIDF['id'][i]) + '*')) > 0:
        twodnames = glob(npth2a + '*' + str(NIDF['id'][i]) + '*')
        NIDF['2d_all'][i] = twodnames
        fids = []
        
        for ii in range(len(twodnames)):
            fids.append(twodnames[ii].split('/')[-1].split('-')[0])
    
        NIDF['field'][i] = fids
    
    else:
        NIDF['2d_all'][i] = 'none'
        NIDF['field'][i] = 'none'
        
    if len(glob(npth2i + '*' + str(NIDF['id'][i]) + '*')) > 0:
        NIDF['2d_ind'][i] = glob(npth2i + '*' + str(NIDF['id'][i]) + '*')

    else:
        NIDF['2d_ind'][i] = 'none'
       
    stack_names = glob(npth1a + '*' + str(NIDF['id'][i]) + '*')
    if len(stack_names) > 0:
        NIDF['stack'][i] =stack_names
    
    else:
        NIDF['stack'][i] = 'none'
    
    ind_names = glob(npth1i + '*' + str(NIDF['id'][i]) + '*')
    if len(ind_names) > 0:
        NIDF['ind_spec'][i] = ind_names
    
    else:
        NIDF['ind_spec'][i] = 'none'
        
SIDF.to_pickle('south_img_DF_vr.pkl')
NIDF.to_pickle('north_img_DF_vr.pkl')
