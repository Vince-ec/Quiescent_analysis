from astropy.io import fits
from shutil import copyfile
import galaxy_extract as ge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)


qgDB = pd.read_pickle('../data/quiescent_gal_DB.pkl')
gsDB = qgDB[qgDB['spec'] == True]

hd_path='../../../../../Volumes/Vince_research/goods_4.4/'

for i in gsDB.index:
    field = gsDB['gids'][i][0]
    if field == 's':
        mosaic = hd_path + 'goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
        seg = hd_path + 'goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_seg.fits'
        cat = '../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat'
    else:
        mosaic = hd_path + 'goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
        seg = hd_path + 'goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_seg.fits'
        cat = '../../../Clear_data/goodsn_mosaic/goodsn_3dhstP.cat'

    gal_img = ge.Image_pull(gsDB['flt_files'][i], mosaic, seg,cat , gsDB['ids'][i])

    cln_cutout = np.array(gal_img.cutout)
    cln_cutout[gal_img.cutout < 0.1*np.max(gal_img.cutout)]= 0
    cln_cutout=cln_cutout*20E19

    copyfile(gsDB['flt_files'][i], '../data/galaxy_flts/%s_flt.fits' % gsDB['gids'][i])
    dt = fits.open('../data/galaxy_flts/%s_flt.fits' % gsDB['gids'][i])
    orig = np.array(dt[1].data)
    #
    new_src=np.zeros(orig.shape)
    new_src[len(orig)/2-len(cln_cutout)/2:len(orig)/2+len(cln_cutout)/2,
              len(orig)/2-len(cln_cutout)/2:len(orig)/2+len(cln_cutout)/2]=cln_cutout

    fits.update('../data/galaxy_flts/%s_flt.fits' % gsDB['gids'][i], new_src, dt[1].header, 1)
