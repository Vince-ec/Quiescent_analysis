import numpy as np
from Magnitude import Magnitude

def Get_color(flux1,flux2):
    c1=Magnitude(flux1)
    c2=Magnitude(flux2)
    color=c1-c2
    return color