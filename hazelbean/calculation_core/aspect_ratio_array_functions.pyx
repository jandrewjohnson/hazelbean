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

def cython_calc_proportion_of_coarse_res_with_valid_fine_res(np.ndarray[np.int_t, ndim=2] coarse_res_array,
                                                      np.ndarray[np.int_t, ndim=2] fine_res_array,
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

# def get_array_neighborhood_by_radius(np.ndarray[DTYPEFLOAT32_t, ndim=2] input_array, int point_x, int point_y, int radius, double fill_value = -255):
    # cdef int num_input_rows = input_array.shape[0]
    # cdef int num_input_cols = input_array.shape[1]
    # cdef int diameter = 2 * radius + 1
    # cdef int neighborhood_r, neighborhood_c, offset_r, offset_c
    #
    # cdef int offset_start = -1 * radius
    #
    # cdef np.ndarray[DTYPEINT_t, ndim=2] offset_r_array = np.empty([diameter, diameter], dtype=DTYPEINT)
    # cdef np.ndarray[DTYPEINT_t, ndim=2] offset_c_array = np.empty([diameter, diameter], dtype=DTYPEINT)
    # for offset_r in range(diameter):
    #     for offset_c in range(diameter):
    #         offset_r_array[offset_r, offset_c] = offset_start + offset_r
    #         offset_c_array[offset_r, offset_c] = offset_start + offset_c
    #
    # cdef np.ndarray[DTYPEFLOAT32_t, ndim=2] output_array = np.zeros([diameter, diameter], dtype=DTYPEFLOAT32)
    #
    # for neighborhood_r in range(diameter):
    #     for neighborhood_c in range(diameter):
    #         if 0 <= point_y + (offset_r_array[neighborhood_r, neighborhood_c]) < num_input_rows and 0 <= point_x + (
    #         offset_c_array[neighborhood_r, neighborhood_c]) < num_input_cols:
    #             output_array[neighborhood_r, neighborhood_c] = input_array[
    #                 point_y + offset_r_array[neighborhood_r, neighborhood_c], point_x + offset_c_array[neighborhood_r, neighborhood_c]]
    #         else:
    #             output_array[neighborhood_r, neighborhood_c] = fill_value
    # return output_array











