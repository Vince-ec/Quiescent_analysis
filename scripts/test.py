from spec_id import Gen_sim,MC_fit_methods_test_2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sea

import rpy2
import rpy2.robjects as robjects
R = robjects.r