import numpy as np
from Constants import c

def ABMagnitude(wv,flam):
    mag=-2.5*np.log10((wv**2/c)*flam)-48.6
    return mag

def Magnitude(flam):
    mag=-2.5*np.log10(flam)
    return mag