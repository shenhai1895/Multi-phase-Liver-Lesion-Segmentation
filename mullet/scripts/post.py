import os
from collections import Counter
from glob import glob
from multiprocessing import Pool

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import opening, disk

os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1


def vote(array):
    l, n = label(array > 0, return_num=True)
    for i in range(n):
        idx = l == (i + 1)
        array[idx] = Counter(array[idx]).most_common(1)[0][0]
    return array


def max_region(array):
    l, n = label(array > 0, return_num=True)
    regs = regionprops(l)
    if len(regs) == 0:
        return array
    regs.sort(key=lambda x: x.filled_area, reverse=True)
    _max_region = regs[0]
    coords = _max_region.coords
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    array[:, :, :] = 0
    array[x, y, z] = 1
    return array