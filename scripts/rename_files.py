import os
from glob import glob
import numpy as np
from vtl.Readfile import Readfile

good_gals=['n32566', 'n37686', 's38785', 's39805']


flist=glob('/Users/Vince.ec/fsps_models_for_fit/rshift_models/*spec.npy*s39805*')

print flist[0].replace('_spec.npy','')

for i in range(len(flist)):
    nname=flist[i].replace('_spec.npy','')
    os.rename(flist[i],nname)


# for i in range(len(ids)):
#     if 1 < rshift[i] < 1.75 and ids[i] != 'n14713':
#         IDA.append(i)
#
#
# fp='/Users/Vince.ec/fsps_models_for_fit/galaxy_models/'
#
# for i in range(len(IDA)):
#     for ii in range(len(metal)):
#         for iii in range(len(age)):
#             for iv in range(len(tau)):
#                 fname=fp+'m%s_a%s_t%s_z%s_%s_model.npy' % (metal[ii],age[iii],tau[iv],rshift[IDA][i],ids[IDA][i])
#                 os.remove(fname)
#
# fp='/Users/Vince.ec/fsps_models_for_fit/fsps_spec/'
# for ii in range(len(metal)):
#     for iii in range(len(age)):
#         for iv in range(len(tau)):
#             fname=fp+'m%s_a%s_t%s_spec.npy' % (metal[ii],age[iii],tau[iv])
#             os.remove(fname)
