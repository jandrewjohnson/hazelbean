import hazelbean as hb

import os, sys, time
# from hazelbean.calculation_core import cython_functions

import numpy as np

sizes  = [6, 50, 500]



for size in sizes:
    array = np.random.rand(size, size)
    nan_mask = np.zeros((size, size))
    nan_mask[1:3, 2:5] = 1

    nan_mask = None

    start = time.time()
    ranked_array, ranked_pared_keys = hb.get_rank_array_and_keys(array, nan_mask=nan_mask)

    print('time elapsed', size, size**2, time.time()-start, (time.time()-start) / size**2)



