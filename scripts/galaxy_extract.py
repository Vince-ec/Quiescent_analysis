import grizli
import numpy as np
from vtl.Readfile import Readfile
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import pysynphot as S
from glob import glob



def Source_present(fn, ra, dec):
    flt = fits.open(fn)
    w = wcs.WCS(flt[1].header)
    # hdr = w.to_header(relax=True)
    present = False

    xpixlim = len(flt[1].data[0])
    ypixlim = len(flt[1].data)

    [pos] = w.wcs_world2pix([[ra, dec]], 1)

    if -100 < pos[0] < xpixlim + 100 and -100 < pos[1] < ypixlim + 100 and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
        present = True

    return present, pos

def FLT_search(ra, dec):
    fn=glob('../../../Clear_data/flt_files/i*flt.fits')

    in_flts = []
    flt_pos = []

    for i in range(len(fn)):
        pres,pos=Source_present(fn[i],ra,dec)
        if pres==True:
             in_flts.append(fn[i])
             flt_pos.append(pos)

    return in_flts,flt_pos



class Image_pull(object):
    def __init__(self, flt_input, mosaic, segment_map, reference_catalog, galaxy_id, pad=100):
        self.flt_input = flt_input
        self.mosaic = mosaic
        self.segment_map = segment_map
        self.reference_catalog = reference_catalog
        self.galaxy_id = galaxy_id
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **                 
        self.mosaic - mosaic image 
        **
        self.segment_map - segmentation map for the corresponding mosaic
        **
        self.reference_catalog - reference catalog which matches the segmentation map, currently this must be an 
                                 ascii file, this can be changed later to include fits files
        **
        self.galaxy_id - ID of the object you want to model
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.cutout - cutout of galaxy
        """

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file=self.flt_input, verbose=False,
                                         ref_file=self.mosaic, seg_file=self.segment_map,
                                         force_grism='G102', pad=self.pad)

        ref_cat = Table.read(self.reference_catalog, format='ascii')

        sim_cat = sim_g102.blot_catalog(ref_cat, sextractor=False)

        sim_g102.compute_full_model(ids=sim_cat['id'], mags=Mag(sim_cat['f_F125W']))

        ## Get Cutout
        beam_g102 = grizli.model.BeamCutout(sim_g102, sim_g102.object_dispersers[self.galaxy_id]['A'])

        self.cutout = beam_g102