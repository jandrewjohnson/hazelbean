import os, sys, warnings, logging, inspect

from osgeo import gdal, osr, ogr
import numpy as np
import hazelbean as hb

L = hb.get_logger('arrayframe_numpy_functions', logging_level='warning') # hb.arrayframe.L.setLevel(logging.DEBUG)

def raster_calculator_flex(input_, op, output_path, **kwargs): #, datatype=None, ndv=None, gtiff_creation_options=None, compress=False

    # If input is a string, put it into a list
    if isinstance(input_, str):
        input_ = [input_]
    elif isinstance(input_, hb.ArrayFrame):
        input_ = input_.path

    final_input = [''] * len(input_)
    for c, i in enumerate(input_):
        if isinstance(i, hb.ArrayFrame):
            final_input[c] = i.path
        else:
            final_input[c] = i
    input_ = final_input

    # Determine size of inputs
    if isinstance(input_, str) or isinstance(input_, hb.ArrayFrame):
        input_size = 1
    elif isinstance(input_, list):
        input_size = len(input_)
    else:
        raise NameError('input_ given to raster_calculator_flex() not understood. Give a path or list of paths.')

    # Check that files exist.
    for i in input_:
        if not os.path.exists(i):
            raise FileNotFoundError(str(input_) + ' not found by raster_calculator_flex()')

     # Verify datatypes
    datatype = kwargs.get('datatype', None)
    if not datatype:
        datatypes = [hb.get_datatype_from_uri(i) for i in input_]
        if len(set(datatypes)) > 1:
            L.info('Rasters given to raster_calculator_flex() were not all of the same type. Defaulting to using first input datatype.')
        datatype = datatypes[0]

    # Check NDVs.
    ndv = kwargs.get('ndv', None)
    if not ndv:
        ndvs = [hb.get_nodata_from_uri(i) for i in input_]
        if len(set(ndvs)) > 1:
            L.info('NDVs used in rasters given to raster_calculator_flex() were not all the same. Defaulting to using first value.')
        ndv = ndvs[0]

    gtiff_creation_options = kwargs.get('gtiff_creation_options', None)
    if not gtiff_creation_options:
        gtiff_creation_options = ['TILED=YES', 'BIGTIFF=IF_SAFER'] #, 'COMPRESS=lzw']

    compress = kwargs.get('compress', None)
    if compress:
        gtiff_creation_options.append('COMPRESS=lzw')

    # Build tuples to match the required format of raster_calculator.
    if input_size == 1:
        if isinstance(input_[0], str):
            input_tuples_list = [(input_[0], 1)]
        else:
            input_tuples_list = [(input_[0].path, 1)]
    else:
        if isinstance(input_[0], str):
            input_tuples_list = [(i, 1) for i in input_]
        else:
            input_tuples_list = [(i.path, 1) for i in input_]

    # Check that the op matches the number of rasters.
    if len(inspect.signature(op).parameters) != input_size:
        raise NameError('op given to raster_calculator_flex() did not have the same number of parameters as the number of rasters given.')

    hb.raster_calculator(input_tuples_list, op, output_path,
                         datatype, ndv, gtiff_creation_options=gtiff_creation_options)

    output_af = hb.ArrayFrame(output_path)
    return output_af


def apply_op(op, output_path):
    input_ = 0
    raster_calculator_flex(input_, op, output_path)

def add(a_path, b_path, output_path):
    def op(a, b):
        return a + b
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def add_with_valid_mask(a_path, b_path, output_path, valid_mask_path, ndv):
    def op(a, b, valid_mask):
        return np.where(valid_mask==1, a + b, ndv)
    hb.raster_calculator_flex([a_path, b_path, valid_mask_path], op, output_path, ndv=ndv)
    return hb.ArrayFrame(output_path)

def add_smart(a, b, a_valid_mask, b_valid_mask, output_ndv, output_path):
    def op(a, b, a_valid_mask, b_valid_mask, output_ndv):
        return np.where((a_valid_mask==1 & b_valid_mask==1), a + b, output_ndv)
    hb.raster_calculator_flex([a, b, a.valid_mask, b.valid_mask], op, output_path, ndv=output_ndv)
    return hb.ArrayFrame(output_path)



def subtract(a_path, b_path, output_path):
    def op(a, b):
        return a - b
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def multiply(a_path, b_path, output_path):
    def op(a, b):
        return a * b
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def divide(a_path, b_path, output_path):
    def op(a, b):
        return a / b
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def greater_than(a_path, b_path, output_path):
    def op(a, b):
        return np.where(a > b, 1, 0)
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def a_greater_than_zero_b_equal_zero(a_path, b_path, output_path):
    def op(a, b):
        return np.where((a > 0) & (b==0), 1, 0)
    hb.raster_calculator_flex([a_path, b_path], op, output_path)
    return hb.ArrayFrame(output_path)

def proportion_change(after, before, output_path):
    def op(after, before):
        return (after - before) / before

    hb.raster_calculator_flex([after, before], op, output_path)
    return hb.ArrayFrame(output_path)



def tiled_sum(input_path):
    return_sum = 0
    for offsets, data in hb.iterblocks(input_path):
        return_sum += np.sum(data)

    return return_sum

def tiled_num_nonzero(input_path):
    return_sum = 0
    for offsets, data in hb.iterblocks(input_path):
        return_sum += np.count_nonzero(data)

    return return_sum
