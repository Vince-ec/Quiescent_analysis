from spec_id  import Gen_spec,Median_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})

galDB = pd.read_pickle('../data/sgal_param_DB.pkl')

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

for i in galDB.index:
    spec = Gen_spec(galDB['gids'][i],galDB['hi_res_specz'][i])
    bftau = Median_model(galDB['gids'][i],galDB['hi_res_specz'][i],galDB['Z'][i],galDB['t'][i],tau)
    spec.Sim_spec(galDB['Z'][i],galDB['t'][i],bftau)

    plt.figure(figsize=[12,5])
    plt.errorbar(spec.gal_wv_rf,spec.gal_fl,spec.gal_er,fmt='o',ms = 5)
    plt.plot(spec.gal_wv_rf,spec.fl,color='#EF1616',
             label = 'Z=%s, t=%s, $\\tau$=%s' % (np.round(galDB['Z'][i]/0.019,2) ,galDB['t'][i] ,bftau))
    plt.axvspan(4810, 4910, color='k', alpha=.1)
    plt.axvspan(5120, 5240, color='k', alpha=.1)
    plt.xlim(spec.gal_wv_rf[0],spec.gal_wv_rf[-1])
    plt.title(galDB['gids'][i], size=13)
    plt.xlabel('$\lambda$',size=13)
    plt.ylabel('Flux',size=13)
    plt.legend()
    plt.savefig('../plots/%s_bfit.png' % galDB['gids'][i])
    plt.close()