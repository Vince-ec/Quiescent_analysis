import pandas as pd
from spec_id import Specz_fit
import numpy as np

gsDB = pd.read_pickle('../data/good_spec_gal_DB.pkl')

metal=np.array([0.002,0.01,0.02,0.03])
age=np.array([1.0,2.0,3.0, 4.0, 5.0, 6.0])

for i in gsDB.index:
    z=np.arange(gsDB['low_res_specz'][i] - 0.2 ,gsDB['low_res_specz'][i] + 0.2,.001)

    Specz_fit(gsDB['gids'][i],metal,age,z,'%s_hires' % gsDB['gids'][i])
