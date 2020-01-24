# cython: cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: boundscheck=False, wraparound=False
from libc.math cimport log
import time
from collections import OrderedDict
import cython
cimport cython
from cython.parallel cimport prange
import numpy as np  # NOTE, both imports are required. cimport adds extra information to the pyd while the import actually defines numppy
cimport numpy as np
from numpy cimport ndarray

import math, time

# DTYPEBYTE = np.byte
# DTYPEINT = np.int
# DTYPEINT64 = np.int64
# DTYPELONG = np.long
# DTYPEFLOAT32 = np.float32
# DTYPEFLOAT64 = np.float64
# ctypedef np.int_t DTYPEINT_t
# ctypedef np.int64_t DTYPEINT64_t
# ctypedef np.float32_t DTYPEFLOAT32_t
# ctypedef np.float64_t DTYPEFLOAT64_t

DTYPEBYTE = np.byte
DTYPEINT = np.int
DTYPEINT64 = np.int64
DTYPELONG = np.long
DTYPEFLOAT32 = np.float32
DTYPEFLOAT64 = np.float64
ctypedef np.int_t DTYPEINT_t
ctypedef np.int64_t DTYPEINT64_t
ctypedef np.float32_t DTYPEFLOAT32_t
ctypedef np.float64_t DTYPEFLOAT64_t


def cython_calc_proportion_of_coarse_res_with_valid_fine_res(np.ndarray[np.float64_t, ndim=2] coarse_res_array,
                                                      np.ndarray[np.int64_t, ndim=2] fine_res_array,
    ):
    cdef long long num_coarse_rows = coarse_res_array.shape[0]
    cdef long long num_coarse_cols = coarse_res_array.shape[1]
    cdef long long num_fine_rows = fine_res_array.shape[0]
    cdef long long num_fine_cols = fine_res_array.shape[1]
    cdef long long fr, fc, cr, cc # Different row col indices

    cdef long long aspect_ratio = <long long>(num_fine_rows / num_coarse_rows)
    cdef long long fine_res_cells_per_coarse_cell = <long long>(aspect_ratio * aspect_ratio)

    cdef np.ndarray[np.float64_t, ndim=2] coarse_res_proporition_array = np.empty([num_coarse_rows, num_coarse_cols], dtype=np.float64)
    cdef double current_proportion = 0.0

    for cr in range(num_coarse_rows):
        for cc in range(num_coarse_cols):
            current_proportion = np.sum(fine_res_array[cr * aspect_ratio: (cr + 1) * aspect_ratio, cc * aspect_ratio: (cc + 1) * aspect_ratio]) / fine_res_cells_per_coarse_cell
            coarse_res_proporition_array[cr, cc] = current_proportion

    return coarse_res_proporition_array


def naive_upsample(np.ndarray[np.float64_t, ndim=2] coarse_res_array, long long upsample_factor):
    """Return an array that makes a n by m array into a n * upsample_factor by m * upsample_factor with the n by m value put into each higher-res cell. """
    cdef long long num_coarse_rows = coarse_res_array.shape[0]
    cdef long long num_coarse_cols = coarse_res_array.shape[1]
    cdef long long num_fine_rows = num_coarse_rows * upsample_factor
    cdef long long num_fine_cols = num_coarse_cols * upsample_factor
    cdef long long cr, cc

    cdef np.ndarray[np.float64_t, ndim=2] output_array = np.empty([num_fine_rows, num_fine_cols], dtype=np.float64)

    for cr in range(num_coarse_rows):
        for cc in range(num_coarse_cols):
            output_array[cr * upsample_factor: (cr + 1) * upsample_factor, cc * upsample_factor: (cc + 1) * upsample_factor] = coarse_res_array[cr, cc]

    return output_array


def naive_upsample_byte(np.ndarray[np.int_t, ndim=2] coarse_res_array, long long upsample_factor):
    """Return an array that makes a n by m array into a n * upsample_factor by m * upsample_factor with the n by m value put into each higher-res cell. """
    cdef long long num_coarse_rows = coarse_res_array.shape[0]
    cdef long long num_coarse_cols = coarse_res_array.shape[1]
    cdef long long num_fine_rows = num_coarse_rows * upsample_factor
    cdef long long num_fine_cols = num_coarse_cols * upsample_factor
    cdef long long cr, cc

    cdef np.ndarray[np.int_t, ndim=2] output_array = np.empty([num_fine_rows, num_fine_cols], dtype=np.int)

    for cr in range(num_coarse_rows):
        for cc in range(num_coarse_cols):
            output_array[cr * upsample_factor: (cr + 1) * upsample_factor, cc * upsample_factor: (cc + 1) * upsample_factor] = coarse_res_array[cr, cc]

    return output_array






