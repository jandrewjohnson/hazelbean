import os, sys, shutil, random, math, atexit, time

from osgeo import gdal, ogr, osr
import numpy as np

import hazelbean as hb



L = hb.get_logger('spatial_projection')
def get_wkt_from_epsg_code(epsg_code):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg_code))
    wkt = srs.ExportToWkt()

    return wkt

# class DatasetUnprojected(Exception):
#     """An exception in case a dataset is unprojected"""
#     pass
#
#
# class DifferentProjections(Exception):
#     """An exception in case a set of datasets are not in the same projection"""
#     pass



def get_datasource_srs_uri(dataset_uri):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(dataset_uri)
    layer = dataset.GetLayer()
    spatialRef = layer.GetSpatialRef()
    return spatialRef




def get_dataset_projection_wkt_uri(dataset_uri):
    """Get the projection of a GDAL dataset as well known text (WKT).

    Args:
        dataset_uri (string): a URI for the GDAL dataset

    Returns:
        proj_wkt (string): WKT describing the GDAL dataset project
    """
    dataset = gdal.Open(dataset_uri)
    proj_wkt = dataset.GetProjection()
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return proj_wkt



def get_linear_unit(input_uri):
    input_ds = gdal.Open(input_uri)
    geo_t = input_ds.GetGeoTransform()
    resolution = geo_t[1]

    return resolution


def get_linear_unit_from_other_projection(input_uri, projected_uri, also_return_size=False):
    input_ds = gdal.Open(input_uri)
    projected_wkt = hb.get_dataset_projection_wkt_uri(projected_uri)

    # Create a virtual raster that is projected based on the output WKT. This
    # vrt does not save to disk and is used to get the proper projected
    # bounding box and size.
    vrt = gdal.AutoCreateWarpedVRT(
        input_ds, None, projected_wkt, gdal.GRA_Bilinear)

    geo_t = vrt.GetGeoTransform()
    x_size = vrt.RasterXSize  # Raster xsize
    y_size = vrt.RasterYSize  # Raster ysize

    resolution = geo_t[1]

    if also_return_size:
        return resolution, x_size, y_size
    else:
        return resolution


def calc_cylindrical_geotransform_from_array(input_array):
    """Assume the array is a global cylindrical geotiff. Calculate the geotransform that would make it such."""

    y_size = 180.0 / float(input_array.shape[0])
    x_size = 360.0 / float(input_array.shape[1])

    if x_size != y_size:
        print('Warning, x_size not same as y_size')

    geotransform = (-180.0, x_size, 0.0, 90.0, 0.0, -y_size)
    return geotransform


def assert_datasets_in_same_projection(dataset_uri_list):
    """Assert that provided datasets are all in the same projection.

    Tests if datasets represented by their uris are projected and in
    the same projection and raises an exception if not.

    Args:
        dataset_uri_list (list): (description)

    Returns:
        is_true (boolean): True (otherwise exception raised)

    Raises:
        DatasetUnprojected: if one of the datasets is unprojected.
        DifferentProjections: if at least one of the datasets is in
            a different projection
    """
    dataset_list = [gdal.Open(dataset_uri) for dataset_uri in dataset_uri_list]
    dataset_projections = []

    unprojected_datasets = set()

    for dataset in dataset_list:
        projection_as_str = dataset.GetProjection()
        dataset_sr = osr.SpatialReference()
        dataset_sr.ImportFromWkt(projection_as_str)
        if not dataset_sr.IsProjected():
            unprojected_datasets.add(dataset.GetFileList()[0])
        dataset_projections.append((dataset_sr, dataset.GetFileList()[0]))

    if len(unprojected_datasets) > 0:
        raise DatasetUnprojected(
            "These datasets are unprojected %s" % (unprojected_datasets))

    for index in range(len(dataset_projections)-1):
        if not dataset_projections[index][0].IsSame(
                dataset_projections[index+1][0]):
            L.warn(
                "These two datasets might not be in the same projection."
                " The different projections are:\n\n'filename: %s'\n%s\n\n"
                "and:\n\n'filename:%s'\n%s\n\n",
                dataset_projections[index][1],
                dataset_projections[index][0].ExportToPrettyWkt(),
                dataset_projections[index+1][1],
                dataset_projections[index+1][0].ExportToPrettyWkt())

    for dataset in dataset_list:
        # Close and clean up dataset
        gdal.Dataset.__swig_destroy__(dataset)
    dataset_list = None
    return True

def get_bounding_box(dataset_uri, return_in_basemap_order=False):
    """Get bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates
    """
    dataset = gdal.Open(dataset_uri)

    geotransform = dataset.GetGeoTransform()
    n_cols = dataset.RasterXSize
    n_rows = dataset.RasterYSize

    bounding_box = [geotransform[0],
                    geotransform[3],
                    geotransform[0] + n_cols * geotransform[1],
                    geotransform[3] + n_rows * geotransform[5]]

    # Close and cleanup dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    if return_in_basemap_order:
        bounding_box = [
            bounding_box[3], # llcrnrlat
            bounding_box[1], # urcrnrlat
            bounding_box[0], # llcrnrlon
            bounding_box[2], # urcrnrlon
        ]
        return bounding_box

    new_bounding_box = hb.get_raster_info(dataset_uri)['bounding_box']
    if bounding_box != new_bounding_box:
        pass
        # L.warn('potential bounding box mismatch', new_bounding_box, bounding_box)
    return bounding_box

def get_datasource_bounding_box(datasource_uri):
    """
    Returns a bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates

    """

    datasource = ogr.Open(datasource_uri)
    layer = datasource.GetLayer(0)
    extent = layer.GetExtent()
    #Reindex datasource extents into the upper left/lower right coordinates
    bounding_box = [extent[0],
                    extent[3],
                    extent[1],
                    extent[2]]
    return bounding_box

def resize_and_resample_dataset_uri(
        original_dataset_uri, bounding_box, out_pixel_size, output_uri,
        resample_method, output_datatype=None):
    """
    A function to  a datsaet to larger or smaller pixel sizes

    Args:
        original_dataset_uri (string): a GDAL dataset
        bounding_box (list): [upper_left_x, upper_left_y, lower_right_x,
            lower_right_y]
        out_pixel_size (?): the pixel size in projected linear units
        output_uri (string): the location of the new resampled GDAL dataset
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"

    Returns:
        nothing

    """

    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "near": gdal.GRA_NearestNeighbour,
        "nearest_neighbor": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average
    }


    original_dataset = gdal.Open(original_dataset_uri)
    original_band = original_dataset.GetRasterBand(1)
    original_nodata = original_band.GetNoDataValue()
    #gdal python doesn't handle unsigned nodata values well and sometime returns
    #negative numbers.  this guards against that
    if original_band.DataType == gdal.GDT_Byte:
        original_nodata %= 2**8
    if original_band.DataType == gdal.GDT_UInt16:
        original_nodata %= 2**16
    if original_band.DataType == gdal.GDT_UInt32:
        original_nodata %= 2**32

    if not output_datatype:
        output_datatype = original_band.DataType

    if original_nodata is None:
        L.debug('Nodata not defined in resize_and_resample_dataset_uri on ' + str(original_dataset_uri) + '. This can be correct but is dangerous because you might have the no_data_value contribute to the resampled values.')
        original_nodata = -9999

    original_sr = osr.SpatialReference()
    original_sr.ImportFromWkt(original_dataset.GetProjection())

    output_geo_transform = [
        bounding_box[0], out_pixel_size, 0.0, bounding_box[1], 0.0,
        -out_pixel_size]
    new_x_size = abs(
        int(np.round((bounding_box[2] - bounding_box[0]) / out_pixel_size)))
    new_y_size = abs(
        int(np.round((bounding_box[3] - bounding_box[1]) / out_pixel_size)))

    #create the new x and y size
    block_size = original_band.GetBlockSize()
    #If the original band is tiled, then its x blocksize will be different than
    #the number of columns
    if block_size[0] != original_band.XSize and original_band.XSize > 256 and original_band.YSize > 256:
        #it makes sense for a wad of invest functions to use 256x256 blocks, lets do that here
        block_size[0] = 256
        block_size[1] = 256
        gtiff_creation_options = [
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=%d' % block_size[0],
                                             'BLOCKYSIZE=%d' % block_size[1]]
    else:
        #this thing is so small or strangely aligned, use the default creation options
        gtiff_creation_options = []

    hb.create_directories([os.path.dirname(output_uri)])
    gdal_driver = gdal.GetDriverByName('GTiff')

    output_dataset = gdal_driver.Create(
        output_uri, new_x_size, new_y_size, 1, output_datatype,
        options=gtiff_creation_options)
    output_band = output_dataset.GetRasterBand(1)
    if original_nodata is None:
        original_nodata = float(
            calculate_value_not_in_dataset(original_dataset))

    output_band.SetNoDataValue(original_nodata)

    # Set the geotransform
    output_dataset.SetGeoTransform(output_geo_transform)
    output_dataset.SetProjection(original_sr.ExportToWkt())

    #need to make this a closure so we get the current time and we can affect
    #state
    def reproject_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - reproject_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and reproject_callback.total_time >= 5.0)):
                # LOGGER.info(
                #     "ReprojectImage %.1f%% complete %s, psz_message %s",
                #     df_complete * 100, p_progress_arg[0], psz_message)
                print("ReprojectImage for resize_and_resample_dataset_uri " + str(df_complete * 100) + " percent complete")
                reproject_callback.last_time = current_time
                reproject_callback.total_time += current_time
        except AttributeError:
            reproject_callback.last_time = time.time()
            reproject_callback.total_time = 0.0

    # Perform the projection/resampling
    gdal.ReprojectImage(
        original_dataset, output_dataset, original_sr.ExportToWkt(),
        original_sr.ExportToWkt(), resample_dict[resample_method], 0, 0,
        reproject_callback, [output_uri])

    #Make sure the dataset is closed and cleaned up
    original_band = None
    gdal.Dataset.__swig_destroy__(original_dataset)
    original_dataset = None

    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    hb.calculate_raster_stats_uri(output_uri)



def resize_and_resample_dataset_uri_hb_old(
        original_dataset_uri, bounding_box, out_pixel_size, output_uri,
        resample_method):
    """Resize and resample the given dataset.

    Args:
        original_dataset_uri (string): a GDAL dataset
        bounding_box (list): [upper_left_x, upper_left_y, lower_right_x,
            lower_right_y]
        out_pixel_size: the pixel size in projected linear units
        output_uri (string): the location of the new resampled GDAL dataset
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"

    Returns:
        None
    """
    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "nearest_neighbor": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
    }

    original_dataset = gdal.Open(original_dataset_uri)
    original_band = original_dataset.GetRasterBand(1)
    original_nodata = original_band.GetNoDataValue()

    if original_nodata is None:
        original_nodata = -9999

    original_sr = osr.SpatialReference()
    original_sr.ImportFromWkt(original_dataset.GetProjection())

    output_geo_transform = [
        bounding_box[0], out_pixel_size, 0.0, bounding_box[1], 0.0,
        -out_pixel_size]
    new_x_size = abs(
        int(np.round((bounding_box[2] - bounding_box[0]) / out_pixel_size)))
    new_y_size = abs(
        int(np.round((bounding_box[3] - bounding_box[1]) / out_pixel_size)))

    if new_x_size == 0:
        print(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        new_x_size = 1
    if new_y_size == 0:
        print(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        new_y_size = 1

    # create the new x and y size
    block_size = original_band.GetBlockSize()
    # If the original band is tiled, then its x blocksize will be different
    # than the number of columns
    if original_band.XSize > 256 and original_band.YSize > 256:
        # it makes sense for many functions to have 256x256 blocks
        block_size[0] = 256
        block_size[1] = 256
        gtiff_creation_options = [
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=%d' % block_size[0],
                                             'BLOCKYSIZE=%d' % block_size[1]]

        metadata = original_band.GetMetadata('IMAGE_STRUCTURE')
        if 'PIXELTYPE' in metadata:
            gtiff_creation_options.append('PIXELTYPE=' + metadata['PIXELTYPE'])
    else:
        # it is so small or strangely aligned, use the default creation options
        gtiff_creation_options = []

    hb.create_directories([os.path.dirname(output_uri)])
    gdal_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gdal_driver.Create(
        output_uri, new_x_size, new_y_size, 1, original_band.DataType,
        options=gtiff_creation_options)
    output_band = output_dataset.GetRasterBand(1)

    output_band.SetNoDataValue(original_nodata)

    # Set the geotransform
    output_dataset.SetGeoTransform(output_geo_transform)
    output_dataset.SetProjection(original_sr.ExportToWkt())

    # need to make this a closure so we get the current time and we can affect
    # state
    def reproject_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - reproject_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and reproject_callback.total_time >= 5.0)):
                print(
                    "ReprojectImage %.1f%% complete %s, psz_message %s",
                    df_complete * 100, p_progress_arg[0], psz_message)
                reproject_callback.last_time = current_time
                reproject_callback.total_time += current_time
        except AttributeError:
            reproject_callback.last_time = time.time()
            reproject_callback.total_time = 0.0

    # Perform the projection/resampling
    gdal.ReprojectImage(
        original_dataset, output_dataset, original_sr.ExportToWkt(),
        original_sr.ExportToWkt(), resample_dict[resample_method], 0, 0,
        reproject_callback, [output_uri])

    # Make sure the dataset is closed and cleaned up
    original_band = None
    gdal.Dataset.__swig_destroy__(original_dataset)
    original_dataset = None

    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    hb.calculate_raster_stats_uri(output_uri)



def force_geotiff_to_match_projection_ndv_and_datatype(input_path, match_path, output_path, output_datatype=None, output_ndv=None):
    """Rather than actually projecting, just change the metadata so it matches exactly. This only will be useful
    if there was a data error and something got a projection defined when the underlying data wasnt actually transofmred
    into that shape.

    NOTE that the output will keep the same geotransform as input, and only the projection, no data and datatype will change.
    """

    if not output_datatype:

        output_datatype = hb.get_datatype_from_uri(match_path)

    if not output_ndv:
        output_ndv = hb.get_nodata_from_uri(match_path)
    match_wkt = hb.get_dataset_projection_wkt_uri(match_path)
    input_geotransform  = hb.get_geotransform_uri(input_path)

    # Load the array, but use numpy to convert it to the new datatype
    input_array = hb.as_array(input_path).astype(hb.gdal_number_to_numpy_type[output_datatype])

    if not output_ndv:
        output_ndv = -9999

    hb.save_array_as_geotiff(input_array, output_path,
                             data_type=output_datatype,
                             ndv=output_ndv,
                             geotransform_override=input_geotransform,
                             projection_override=match_wkt)


def force_global_angular_data_to_plate_carree(input_path, output_path):

    output_datatype = hb.get_datatype_from_uri(input_path)
    output_ndv = hb.get_nodata_from_uri(input_path)
    match_wkt = hb.get_dataset_projection_wkt_uri(input_path)
    match_wkt = hb.get_wkt_from_epsg_code(32662)

    input_geotransform  = hb.get_geotransform_uri(input_path)

    output_geotransform = list(hb.common_geotransforms['wec_30s'])

    output_geotransform[1] = input_geotransform[1] * hb.size_of_one_arcdegree_at_equator_in_meters
    output_geotransform[5] = input_geotransform[5] * hb.size_of_one_arcdegree_at_equator_in_meters


    # Load the array, but use numpy to convert it to the new datatype
    input_array = hb.as_array(input_path).astype(hb.gdal_number_to_numpy_type[output_datatype])

    if not output_ndv:
        output_ndv = -9999

    hb.save_array_as_geotiff(input_array, output_path,
                             data_type=output_datatype,
                             ndv=output_ndv,
                             geotransform_override=output_geotransform,
                             projection_override=match_wkt)


def force_global_angular_data_to_equal_area_earth_grid(input_path, output_path):

    output_datatype = hb.get_datatype_from_uri(input_path)
    output_ndv = hb.get_nodata_from_uri(input_path)
    match_wkt = hb.get_dataset_projection_wkt_uri(input_path)
    match_wkt = hb.get_wkt_from_epsg_code(6933)

    input_geotransform  = hb.get_geotransform_uri(input_path)

    output_geotransform = list(hb.common_geotransforms['wec_30s'])

    output_geotransform[1] = input_geotransform[1] * hb.size_of_one_arcdegree_at_equator_in_meters
    output_geotransform[5] = input_geotransform[5] * hb.size_of_one_arcdegree_at_equator_in_meters


    # Load the array, but use numpy to convert it to the new datatype
    input_array = hb.as_array(input_path).astype(hb.gdal_number_to_numpy_type[output_datatype])

    if not output_ndv:
        output_ndv = -9999

    hb.save_array_as_geotiff(input_array, output_path,
                             data_type=output_datatype,
                             ndv=output_ndv,
                             geotransform_override=output_geotransform,
                             projection_override=match_wkt)









