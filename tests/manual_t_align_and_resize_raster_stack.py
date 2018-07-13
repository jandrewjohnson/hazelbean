import os, sys

sys.path.extend(['C:/OneDrive/Projects'])

import numpy as np

import hazelbean as hb

global_random_floats_15m_32bit_path = os.path.join(hb.TEST_DATA_DIR, 'global_random_floats_15m_32bit.tif')
two_poly_eckert_iv_aoi_path = os.path.join(hb.TEST_DATA_DIR, 'two_poly_eckert_iv_aoi.shp')
two_poly_wgs84_aoi_path = os.path.join(hb.TEST_DATA_DIR, 'two_poly_wgs84_aoi.shp')




a = hb.as_array(global_random_floats_15m_32bit_path)

# Old clip method for reference
# hb.clip_dataset_uri(global_random_floats_15m_32bit_path, two_poly_wgs84_aoi_path, hb.temp('.tif', 'clip1', False, 'tests'))

base_raster_path_list = [global_random_floats_15m_32bit_path]
target_raster_path_list = [hb.temp('.tif', 'clip1', False, 'tests')]
resample_method_list = ['bilinear']
target_pixel_size = hb.get_raster_info(global_random_floats_15m_32bit_path)['pixel_size']
bounding_box_mode = 'intersection'
base_vector_path_list = [two_poly_wgs84_aoi_path]
raster_align_index = 0


hb.align_and_resize_raster_stack(
    base_raster_path_list, target_raster_path_list, resample_method_list,
    target_pixel_size, bounding_box_mode, base_vector_path_list=base_vector_path_list, all_touched=True,
    raster_align_index=raster_align_index,
    gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)


hb.clip_raster_by_vector(global_random_floats_15m_32bit_path, hb.temp('.tif', 'clip2', False, 'tests'), two_poly_wgs84_aoi_path)