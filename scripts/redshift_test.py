import pandas as pd
from spec_id import Specz_fit
import numpy as np

gsDB = pd.read_pickle('../data/good_spec_gal_DB.pkl')

metal=np.array([0.002,0.005,0.01,0.015,0.02,0.025,0.03])
age=np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5])

for i in gsDB.index:
    z=np.arange(gsDB['low_res_specz'][i] - 0.5 ,gsDB['low_res_specz'][i] + 0.5,.05)

    Specz_fit(gsDB['gids'][i],metal,age,z,'%s_lowres' % gsDB['gids'][i])
