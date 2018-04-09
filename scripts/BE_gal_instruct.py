from BE_spec_id import Single_gal_fit_full
import numpy as np
import pandas as pd

Zfs = np.array([0.004,0.008,0.019])
Zbc = np.array([0.004,0.008,0.02])
tau = [8,8.48,8.7,8.85,8.95,9.04,9.11,9.18,9.23,9.28]
age=np.arange(.5,6.1,.1)
galDB = pd.read_pickle('../data/sgal_param_DB.pkl')

Single_gal_fit_full(Zfs,age,np.array(tau),galDB['hi_res_specz'][61],galDB['gids'][61],'BE_fs_n21156')
print 'done'
Single_gal_fit_full(Zbc,age,tau,galDB['hi_res_specz'][61],galDB['gids'][61],'BE_bc_n21156',bc03=True)
print 'done'