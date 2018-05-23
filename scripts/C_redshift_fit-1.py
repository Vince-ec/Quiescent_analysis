import pandas as pd
from C_spec_id import Specz_fit
import numpy as np
import sys

if __name__ == '__main__':
   galaxy = sys.argv[1] 

gsDB = pd.read_pickle('../../../../fdata/scratch/vestrada78840/data/select_samp.pkl')

metal=np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
age=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

z=np.arange(gsDB['low_res_specz'][i] - 0.3 ,gsDB['low_res_specz'][i] + 0.3,.001)

Specz_fit(galaxy,metal,age,z,'%s_hires_refit' % galaxy)