import pandas as pd
from spec_id import Specz_fit,RT_spec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)

gsDB = pd.read_pickle('../data/good_spec_gal_DB.pkl')

metal=np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
age=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

def Best_fit_model(input_file, metal, age, redshift):
    chi = np.load(input_file)

    x = np.argwhere(chi == np.min(chi))
    print np.min(chi)
    print metal[x[0][0]], age[x[0][1]], redshift[x[0][2]]
    return metal[x[0][0]], age[x[0][1]], redshift[x[0][2]]

for i in gsDB.index:
    z=np.arange(gsDB['low_res_specz'][i] - 0.3 ,gsDB['low_res_specz'][i] + 0.3,.001)
    gal_spec = RT_spec(gsDB['gids'][i])
    gal_spec.Sim_spec(0.015,3.5,0,gsDB['hi_res_specz'][i])

    plt.figure(figsize=[8,8])
    plt.plot(gal_spec.gal_wv,gal_spec.gal_fl)
    plt.plot(gal_spec.gal_wv,gal_spec.fl,label = 'z=%s'% gsDB['hi_res_specz'][i] )

    bfm,bfa,bfz = Best_fit_model('../rshift_dat/%s_hires_z_fit.npy'% gsDB['gids'][i],metal,age,z)
    gal_spec.Sim_spec(bfm,bfa,0,bfz)
    plt.plot(gal_spec.gal_wv,gal_spec.fl,label = 'z=%s' % bfz)
    plt.legend()
    plt.savefig('../plots/z_inspect_%s.png' % gsDB['gids'][i])

