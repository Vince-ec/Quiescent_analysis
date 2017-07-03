import pandas as pd
from spec_id import Specz_fit,RT_spec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in","ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)


metal=np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
age=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

z=np.arange(1.222 - 0.1 ,1.222 + 0.1,.001)
Specz_fit('s35774',metal,age,z,'s35774_hires2')
