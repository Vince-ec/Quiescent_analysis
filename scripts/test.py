import pandas as pd
from spec_id import Sim_fit
import numpy as np

sampDB = pd.read_pickle('../data/sgal_param_DB.pkl')

metal=np.arange(0.002,0.031,0.001)
age=np.arange(.5,6.1,.1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]

Sim_fit(sampDB['gids'][220],metal,age,tau,0.015,3.0,8.48,sampDB['hi_res_specz'][220],'sim_fit_s39170')
