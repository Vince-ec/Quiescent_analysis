import galaxy_extract as ge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from glob import glob
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

qgDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
gsDB = qgDB[qgDB['spec'] == True]

hd_path='../../../../../Volumes/Vince_research/goods_4.4/'

gal_img = ge.Image_pull(gsDB['flt_files'][220],hd_path + 'goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits',
              hd_path + 'goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_seg.fits',
              '../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat', gsDB['ids'][220])

plt.imshow(gal_img.cutout)
plt.show()