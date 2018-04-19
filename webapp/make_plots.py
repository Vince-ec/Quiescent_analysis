from astropy.io import fits
from scipy.interpolate import interp1d,interp2d
from yattag import Doc
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import seaborn as sea
import os
from time import time
import numpy as np
import pandas as pd
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, 
                             light=1.2, as_cmap=True)


class Stack_galaxy(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def Get_spec(self, FILE):
        observ = fits.open(FILE)
        w = np.array(observ[1].data.field('wave'))
        f = np.array(observ[1].data.field('flux')) * 1E-17
        sens = np.array(observ[1].data.field('sensitivity'))
        contam = np.array(observ[1].data.field('contam')) * 1E-17
        e = np.array(observ[1].data.field('error')) * 1E-17
        f -= contam
        f /= sens
        e /= sens

        INDEX = []
        for i in range(len(w)):
            if 8000 < w[i] < 11350:
                INDEX.append(i)

        w = w[INDEX]
        f = f[INDEX]
        e = e[INDEX]

        return w, f, e

    def Get_wv_list(self):
        W = []
        lW = []

        for i in range(len(self.file_list)):
            wv, fl, er = self.Get_spec(self.file_list[i])
            W.append(wv)
            lW.append(len(wv))

        W = np.array(W)
        self.wv = W[np.argmax(lW)]

    def Mean_stack_galaxy(self):

        self.Get_wv_list()
        
        # Define grids used for stacking
        flgrid = np.zeros([len(self.file_list), len(self.wv)])
        errgrid = np.zeros([len(self.file_list), len(self.wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(self.file_list)):
            wave, flux, error = self.Get_spec(self.file_list[i])
            
            if len(wave) > 10:
            
                mask = np.array([wave[0] < U < wave[-1] for U in self.wv])
                ifl = interp1d(wave, flux)(self.wv[mask])
                ier = interp1d(wave, error)(self.wv[mask])


                flgrid[i][mask] = ifl
                errgrid[i][mask] = ier

        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv)])
        for i in range(len(self.wv)):
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.fl = np.array(stack)
        self.er = np.array(err)
        
        self.wv = self.wv[self.fl>0]
        self.er = self.er[self.fl>0]
        self.fl = self.fl[self.fl>0]
        
def Get_flux(FILE):
    observ = fits.open(FILE)
    w = np.array(observ[1].data.field('wave'))
    f = np.array(observ[1].data.field('flux')) * 1E-17
    sens = np.array(observ[1].data.field('sensitivity'))
    contam = np.array(observ[1].data.field('contam')) * 1E-17
    e = np.array(observ[1].data.field('error')) * 1E-17
    f -= contam
    f /= sens
    e /= sens

    INDEX = []
    for i in range(len(w)):
        if 8000 < w[i] < 11350:# and e[i] < np.abs(f[i]):
            INDEX.append(i)

            
            
    w = w[INDEX]
    f = f[INDEX]
    e = e[INDEX]

#     return w[f > 0], f[f > 0], e[f > 0]
    return w, f, e


def Plot_1d(DF,idx, position):
    if os.path.isfile('/Volumes/Vince_research/Extractions/Plots/%s/1D/stack/%s_stack.png' % (position, DF['id'][idx])):
        dummy = 1
    else:
        if len(DF['1D'][idx]) > 0:
            file_list = DF['1D'][idx].split(',')[:-1]
            viable = False
            
            for u in range(len(file_list)):
                name = file_list[u].split('/')[-1].split('_')[0]

                wv,fl,er = Get_flux(file_list[u])
                
                if len(wv) > 10:
                    viable = True
                
                IDX = [U for U in range(len(wv)) if er[U] < 2*np.median(fl)]

                plt.figure(figsize=[12,6])
                plt.errorbar(wv[IDX]*1E-4,fl[IDX]*1E18,er[IDX]*1E18,fmt='o',lw=2,ms=4)
                plt.xlabel('$\lambda$ $(\mu m)$',fontsize=20)
                plt.ylabel('F$_\lambda$ $(10^{-18} erg/ s/ cm^{2}/ \\rm \AA)$',fontsize=20)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.title('%s_%s' % (name,SDF['id'][idx]),fontsize=20)
                plt.xlim(.8,1.14)
                plt.savefig('/Volumes/Vince_research/Extractions/Plots/%s/1D/spec/%s_%s_spec.png' % (position, name, DF['id'][idx]))    
                plt.close()
            
            if viable == True:
            
                stack = Stack_galaxy(file_list)
                stack.Mean_stack_galaxy()

                IDX = [U for U in range(len(stack.wv)) if stack.er[U] < 2*np.median(stack.fl)]

                plt.figure(figsize=[12,6])
                plt.errorbar(stack.wv[IDX]*1E-4,stack.fl[IDX]*1E18,stack.er[IDX]*1E18,fmt='o',lw=2,ms=4)
                plt.xlabel('$\lambda$ $(\mu m)$',fontsize=20)
                plt.ylabel('F$_\lambda$ $(10^{-18} erg/ s/ cm^{2}/ \\rm \AA)$',fontsize=20)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.xlim(.8,1.14)
                plt.savefig('/Volumes/Vince_research/Extractions/Plots/%s/1D/stack/%s_stack.png' % (position, DF['id'][idx]))
                plt.close()
                

SDF = pd.read_pickle('south_DF_vr.pkl')
NDF = pd.read_pickle('north_DF_vr.pkl')

for i in SDF.index:
    Plot_1d(SDF,i,'South')

for i in NDF.index:
    Plot_1d(NDF,i,'North')