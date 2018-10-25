"""Hazelbean relies on pygeoprocessing, but there are some cases where an optimal change requires duplicating some code. To minimize upgrading challenges
Hazelbean offers a few functions that reimplement pgp and by convention keep a similar funciton name with the change listed after the matched function name."""

import os, sys, time, math
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy as np
import numpy as numpy

import hazelbean as hb

# Key lines here: for convenience and cross-compatibility, I import pgp into the hb namespace
import pygeoprocessing as pgp
import pygeoprocessing.geoprocessing as gp
import pygeoprocessing.geoprocessing_core as gpc
from pygeoprocessing.geoprocessing import *
from pygeoprocessing.geoprocessing_core import *

L = hb.get_logger('geoprocessing')

def align_and_resize_raster_stack_ensuring_fit(
        base_raster_path_list, target_raster_path_list, resample_method_list,
        target_pixel_size, bounding_box_mode, base_vector_path_list=None,
        raster_align_index=None, ensure_fits=False, all_touched=False,
        gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Parameters:
        base_raster_path_list (list): a list of base raster paths that will
            be transformed and will be used to determine the target bounding
            box.
        target_raster_path_list (list): a list of raster paths that will be
            created to one-to-one map with `base_raster_path_list` as aligned
            versions of those original rasters.
        resample_method_list (list): a list of resampling methods which
            one to one map each path in `base_raster_path_list` during
            resizing.  Each element must be one of
            "nearest|bilinear|cubic|cubic_spline|lanczos|mode".
        target_pixel_size (tuple): the target raster's x and y pixel size
            example: [30, -30].
        bounding_box_mode (string): one of "union", "intersection", or
            a list of floats of the form [minx, miny, maxx, maxy].  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (list): a list of base vector paths whose
            bounding boxes will be used to determine the final bounding box
            of the raster stack if mode is 'union' or 'intersection'.  If mode
            is 'bb=[...]' then these vectors are not used in any calculation.
        raster_align_index (int): indicates the index of a
            raster in `base_raster_path_list` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If `None` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        gtiff_creation_options (list): list of strings that will be passed
            as GDAL "dataset" creation options to the GTIFF driver, or ignored
            if None.

    Returns:
        None
    """
    last_time = time.time()


    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list), len(target_raster_path_list),
        len(resample_method_list)]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
            not isinstance(bounding_box_mode, (list, tuple)) or
            len(bounding_box_mode) != 4):
        raise ValueError("Unknown bounding_box_mode %s" % (
            str(bounding_box_mode)))

    if ((raster_align_index is not None) and
            ((raster_align_index < 0) or
             (raster_align_index >= len(base_raster_path_list)))):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (
                raster_align_index, len(base_raster_path_list)))

    raster_info_list = [
        get_raster_info(path) for path in base_raster_path_list]
    if base_vector_path_list is not None:
        vector_info_list = [
            get_vector_info(path) for path in base_vector_path_list]
    else:
        vector_info_list = []

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union
        target_bounding_box = reduce(
            functools.partial(gp._merge_bounding_boxes, mode=bounding_box_mode),
            [info['bounding_box'] for info in
             (raster_info_list + vector_info_list)])

    if bounding_box_mode == "intersection" and (
            target_bounding_box[0] > target_bounding_box[2] or
            target_bounding_box[1] > target_bounding_box[3]):
        raise ValueError("The rasters' and vectors' intersection is empty "
                         "(not all rasters and vectors touch each other).")

    if raster_align_index is not None:
        if raster_align_index >= 0:
            # bounding box needs alignment
            align_bounding_box = (
                raster_info_list[raster_align_index]['bounding_box'])
            align_pixel_size = (
                raster_info_list[raster_align_index]['pixel_size'])
            # adjust bounding box so lower left corner aligns with a pixel in
            # raster[raster_align_index]
            target_rc = [
                math.ceil((target_bounding_box[2] - target_bounding_box[0]) / target_pixel_size[0]),
                math.floor((target_bounding_box[3] - target_bounding_box[1]) / target_pixel_size[1])
            ]

            original_bounding_box = list(target_bounding_box)

            for index in [0, 1]:
                n_pixels = int((target_bounding_box[index] - align_bounding_box[index]) / float(align_pixel_size[index]))

                target_bounding_box[index] = align_bounding_box[index] + (n_pixels * align_pixel_size[index])
                target_bounding_box[index + 2] = target_bounding_box[index] + target_rc[index] * target_pixel_size[index]

            if ensure_fits:
                # This addition to the core geoprocessing code was to fix the case where the alignment moved the target tif
                # up and to the left, but in a way that then trunkated 1 row/col on the bottom right, causing wrong-shape
                # raster_math errors.z
                if original_bounding_box[2] > target_bounding_box[2]:
                    target_bounding_box[2] += target_pixel_size[0]

                if original_bounding_box[3] > target_bounding_box[3]:
                    target_bounding_box[3] -= target_pixel_size[1]


    option_list = list(gtiff_creation_options)
    if all_touched:
        option_list.append("ALL_TOUCHED=TRUE")

    for index, (base_path, target_path, resample_method) in enumerate(zip(
            base_raster_path_list, target_raster_path_list,
            resample_method_list)):
        last_time = gp._invoke_timed_callback(
            last_time, lambda: L.info(
                "align_dataset_list aligning dataset %d of %d",
                index, len(base_raster_path_list)), hb.LOGGING_PERIOD)

        warp_raster(
            base_path, target_pixel_size,
            target_path, resample_method,
            target_bb=target_bounding_box,
            gtiff_creation_options=option_list)

