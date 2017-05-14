from spec_id import Analyze_Stack_avgage, Stack_spec_normwmean, Stack_model_normwmean
import seaborn as sea
from glob import glob
import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from vtl.Readfile import Readfile

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

""">10.87 Galaxies"""
###get list of spectra
ids, lmass, rshift = np.array(Readfile('masslist_sep28.dat', 1, is_float=False))
lmass, rshift = np.array([lmass, rshift]).astype(float)
nlist = glob('spec_stacks/*')

IDS = []

for i in range(len(ids)):
    if 10.87 < lmass[i] and 1 < rshift[i] < 1.75:
        IDS.append(i)

speclist = []
for i in range(len(ids[IDS])):
    for ii in range(len(nlist)):
        if ids[IDS][i] == nlist[ii][12:18]:
            speclist.append(nlist[ii])

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDS], 2)
for i in range(len(speczs)):
    zinput = int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

metal = np.array([0.0020, 0.0025, 0.0031, 0.0039, 0.0049, 0.0061, 0.0068, 0.0077, 0.0085, 0.0096, 0.0106,
                  0.012, 0.0132, 0.014, 0.0150, 0.0164, 0.018, 0.019, 0.021, 0.024, 0.027, 0.03])
# metal = np.array([.0001, .0004, .004, .008, .02, ])
age = [0.5, 0.57, 0.65, 0.74, 0.84, 0.96, 1.1, 1.25, 1.42,
       1.62, 1.85, 2.11, 2.4, 2.74, 3.12, 3.56, 4.05, 4.62, 5.26, 6.0]
tau = [0, 8.0, 8.15, 8.28, 8.43, 8.57, 8.72, 8.86, 9.0, 9.14, 9.29, 9.43, 9.57, 9.71, 9.86, 10.0]

"""FSPS Best fit >10.87"""
###make stacks
flist = []
blistl = []
blisth = []
for i in range(len(zlist)):
    flist.append('../../../fsps_models_for_fit/models/m0.015_a1.42_t0_z%s_model.dat' % zlist[i])
    blistl.append('../../../bc03_models_for_fit/models/m0.008_a1.42_t0_z%s_model.dat' % zlist[i])
    blisth.append('../../../bc03_models_for_fit/models/m0.02_a1.42_t0_z%s_model.dat' % zlist[i])

wv, fl, er = Stack_spec_normwmean(speclist, rshift[IDS], np.arange(3250, 5500, 5))
fwv, fs, fe = Stack_model_normwmean(speclist, flist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
blwv, bls, ble = Stack_model_normwmean(speclist, blistl, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
bhwv, bhs, bhe = Stack_model_normwmean(speclist, blisth, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))

###get chi square values
chi = np.zeros(3)
chi[0] = np.round(sum(((fl - fs) / er) ** 2), 2)
chi[1] = np.round(sum(((fl - bls) / er) ** 2), 2)
chi[2] = np.round(sum(((fl - bhs) / er) ** 2), 2)

fs *= 1000
bls *= 1000
bhs *= 1000
fl *= 1000
er *= 1000
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

plt.figure()
plt.subplot(gs[0])
plt.plot(wv, fl, label='>10.87 Stack')
plt.fill_between(wv, fl - er, fl + er, color='k', alpha=.3)
plt.plot(fwv, fs, color='k', label='FSPS >10.87 Best Fit $\chi^2$=%s' % chi[0])
plt.plot(blwv, bls, color='#4b9e4b', label='BC03 Z=%s $\chi^2$=%s' % (np.round((.008 / .02), 2), chi[1]))
plt.plot(bhwv, bhs, color='#A13535', label='BC03 Z=%s $\chi^2$=%s' % (np.round((.02 / .02), 2), chi[2]))
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.ylim(.5, 6.5)
plt.ylabel('Relative Flux', size=20)
plt.xticks([])
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4, fontsize=15)

plt.subplot(gs[1])
plt.plot(wv, np.zeros(len(wv)), 'k--', alpha=.8)
plt.plot(wv, fl - fs, color='k', label='BC03 residuals')
plt.plot(wv, fl - bls, color='#4b9e4b', label='BC03 residuals')
plt.plot(wv, fl - bhs, color='#a13535', label='BC03 residuals')
plt.fill_between(wv, -er, er, color='k', alpha=.3)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.xlabel('Wavelength ($\AA$)', size=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(-1.1, 1.1)
plt.yticks([-1, 0, 1, ])
plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
plt.savefig('../research_plots/gtfsps_cross.png')
plt.close()

"""BC03 Best fit >10.87"""
###make stacks
blist = []
flist = []
for i in range(len(zlist)):
    blist.append('../../../bc03_models_for_fit/models/m0.008_a2.11_t0_z%s_model.dat' % zlist[i])
    flist.append('../../../fsps_models_for_fit/models/m0.0077_a2.11_t0_z%s_model.dat' % zlist[i])

wv, fl, er = Stack_spec_normwmean(speclist, rshift[IDS], np.arange(3250, 5500, 5))
bwv, bs, be = Stack_model_normwmean(speclist, blist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
fwv, fs, fe = Stack_model_normwmean(speclist, flist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))

###get chi square values
chi = np.zeros(3)
chi[0] = np.round(sum(((fl - bs) / er) ** 2), 2)
chi[1] = np.round(sum(((fl - fs) / er) ** 2), 2)

fs *= 1000
bs *= 1000
fl *= 1000
er *= 1000
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

plt.figure()
plt.subplot(gs[0])
plt.plot(wv, fl, label='>10.87 Stack')
plt.fill_between(wv, fl - er, fl + er, color='k', alpha=.3)
plt.plot(bwv, bs, color='k', label='BC03 >10.87 Best Fit $\chi^2$=%s' % chi[0])
plt.plot(fwv, fs, color='#4b9e4b', label='FSPS Z=%s $\chi^2$=%s' % (np.round((.0077 / .019), 2), chi[1]))
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.ylim(.5, 6.5)
plt.ylabel('Relative Flux', size=20)
plt.xticks([])
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4, fontsize=15)

plt.subplot(gs[1])
plt.plot(wv, np.zeros(len(wv)), 'k--', alpha=.8)
plt.plot(wv, fl - bs, color='k', label='BC03 residuals')
plt.plot(wv, fl - fs, color='#4b9e4b', label='BC03 residuals')
plt.fill_between(wv, -er, er, color='k', alpha=.3)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.xlabel('Wavelength ($\AA$)', size=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(-1.1, 1.1)
plt.yticks([-1, 0, 1, ])
plt.gcf().subplots_adjust(bottom=0.16)

# plt.show()
plt.savefig('../research_plots/gtbc03_cross.png')
plt.close()

"""<10.87 Galaxies"""
###get list of spectra
IDX = []

for i in range(len(ids)):
    if 10.87 > lmass[i] and 1 < rshift[i] < 1.75:
        IDX.append(i)

speclist = []
for i in range(len(ids[IDX])):
    for ii in range(len(nlist)):
        if ids[IDX][i] == nlist[ii][12:18]:
            speclist.append(nlist[ii])

zlist, zbin, zcount, bins = [[], [], [], np.linspace(1, 1.8, 17)]
speczs = np.round(rshift[IDX], 2)
for i in range(len(speczs)):
    zinput = int(speczs[i] * 100) / 5 / 20.
    if zinput < 1:
        zinput = 1.0
    if zinput > 1.8:
        zinput = 1.8
    zlist.append(zinput)

"""FSPS Best fit <10.87"""
###make stacks
flist = []
blistl = []
blisth = []
for i in range(len(zlist)):
    flist.append('../../../fsps_models_for_fit/models/m0.012_a2.11_t0_z%s_model.dat' % zlist[i])
    blistl.append('../../../bc03_models_for_fit/models/m0.008_a2.11_t0_z%s_model.dat' % zlist[i])
    blisth.append('../../../bc03_models_for_fit/models/m0.02_a2.11_t0_z%s_model.dat' % zlist[i])

wv, fl, er = Stack_spec_normwmean(speclist, rshift[IDX], np.arange(3500, 5500, 5))
fwv, fs, fe = Stack_model_normwmean(speclist, flist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
blwv, bls, ble = Stack_model_normwmean(speclist, blistl, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
bhwv, bhs, bhe = Stack_model_normwmean(speclist, blisth, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))

chi = np.zeros(3)
chi[0] = np.round(sum(((fl - fs) / er) ** 2), 2)
chi[1] = np.round(sum(((fl - bls) / er) ** 2), 2)
chi[2] = np.round(sum(((fl - bhs) / er) ** 2), 2)

fs *= 1000
bls *= 1000
bhs *= 1000
fl *= 1000
er *= 1000
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

plt.figure()
plt.subplot(gs[0])
plt.plot(wv, fl, label='<10.87 Stack')
plt.fill_between(wv, fl - er, fl + er, color='k', alpha=.3)
plt.plot(fwv, fs, color='k', label='FSPS <10.87 Best Fit $\chi^2$=%s' % chi[0])
plt.plot(blwv, bls, color='#4b9e4b', label='BC03 Z=%s $\chi^2$=%s' % (np.round((.008 / .02), 2), chi[1]))
plt.plot(bhwv, bhs, color='#A13535', label='BC03 Z=%s $\chi^2$=%s' % (np.round((.02 / .02), 2), chi[2]))
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.ylim(.5, 7)
plt.ylabel('Relative Flux', size=20)
plt.xticks([])
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4, fontsize=15)

plt.subplot(gs[1])
plt.plot(wv, np.zeros(len(wv)), 'k--', alpha=.8)
plt.plot(wv, fl - fs, color='k', label='BC03 residuals')
plt.plot(wv, fl - bls, color='#4b9e4b', label='BC03 residuals')
plt.plot(wv, fl - bhs, color='#a13535', label='BC03 residuals')
plt.fill_between(wv, -er, er, color='k', alpha=.3)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.xlabel('Wavelength ($\AA$)', size=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(-1.1, 2)
plt.yticks([-1, 0, 1, ])
plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
plt.savefig('../research_plots/ltfsps_cross.png')
plt.close()


"""BC03 Best fit <10.87"""
###make stacks
blist = []
flist = []
for i in range(len(zlist)):
    blist.append('../../../bc03_models_for_fit/models/m0.004_a4.62_t0_z%s_model.dat' % zlist[i])
    flist.append('../../../fsps_models_for_fit/models/m0.0039_a4.62_t0_z%s_model.dat' % zlist[i])

wv, fl, er = Stack_spec_normwmean(speclist, rshift[IDX], np.arange(3500, 5500, 5))
bwv, bs, be = Stack_model_normwmean(speclist, blist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))
fwv, fs, fe = Stack_model_normwmean(speclist, flist, speczs, zlist, np.arange(wv[0], wv[-1] + 5, 5))

###get chi square values
chi = np.zeros(3)
chi[0] = np.round(sum(((fl - bs) / er) ** 2), 2)
chi[1] = np.round(sum(((fl - fs) / er) ** 2), 2)

fs *= 1000
bs *= 1000
fl *= 1000
er *= 1000
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

plt.figure()
plt.subplot(gs[0])
plt.plot(wv, fl, label='>10.87 Stack')
plt.fill_between(wv, fl - er, fl + er, color='k', alpha=.3)
plt.plot(bwv, bs, color='k', label='BC03 >10.87 Best Fit $\chi^2$=%s' % chi[0])
plt.plot(fwv, fs, color='#4b9e4b', label='FSPS Z=%s $\chi^2$=%s' % (np.round((.0039 / .019), 2), chi[1]))
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.ylim(.5, 7)
plt.xticks([])
plt.xlabel('Restframe Wavelength ($\AA$)', size=15)
plt.ylabel('Relative Flux', size=15)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.gcf().subplots_adjust(bottom=0.16)
plt.legend(loc=4, fontsize=15)

plt.subplot(gs[1])
plt.plot(wv, np.zeros(len(wv)), 'k--', alpha=.8)
plt.plot(wv, fl - bs, color='k', label='BC03 residuals')
plt.plot(wv, fl - fs, color='#4b9e4b', label='BC03 residuals')
plt.fill_between(wv, -er, er, color='k', alpha=.3)
plt.axvspan(3910, 3979, alpha=.2)
plt.axvspan(3981, 4030, alpha=.2)
plt.axvspan(4082, 4122, alpha=.2)
plt.axvspan(4250, 4400, alpha=.2)
plt.axvspan(4830, 4930, alpha=.2)
plt.axvspan(4990, 5030, alpha=.2)
plt.axvspan(5109, 5250, alpha=.2)
plt.xlim(min(wv), max(wv))
plt.xlabel('Wavelength ($\AA$)', size=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(-1.1, 2)
plt.yticks([-1, 0, 1])
plt.gcf().subplots_adjust(bottom=0.16)
# plt.show()
plt.savefig('../research_plots/ltbc03_cross.png')
plt.close()
