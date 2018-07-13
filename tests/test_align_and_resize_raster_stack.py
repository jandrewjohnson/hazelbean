from unittest import TestCase
import gdal
import os, sys

sys.path.insert(0, '..') #
import hazelbean as hb
import numpy as np

class TestAlign_and_resize_raster_stack(TestCase):
    def setUp(self):
        self.global_ha_per_cell_5m = os.path.join(hb.TEST_DATA_DIR, 'ha_per_cell_5m.tif')
        self.two_poly_eckert_iv_aoi_path = os.path.join(hb.TEST_DATA_DIR, 'two_poly_eckert_iv_aoi.shp')
        self.two_poly_wgs84_aoi_path = os.path.join(hb.TEST_DATA_DIR, 'two_poly_wgs84_aoi.shp')

    def test_align_and_resize_raster_stack(self):
        a = hb.as_array(self.global_ha_per_cell_5m)

        # Old clip method for reference
        # hb.clip_dataset_uri(self.global_ha_per_cell_5m, self.two_poly_wgs84_aoi_path, hb.temp('.tif', 'clip1', False, 'tests'))


        base_raster_path_list = [self.global_ha_per_cell_5m]
        target_raster_path_list = [hb.temp('.tif', 'clip1', True)]
        resample_method_list = ['bilinear']
        target_pixel_size = hb.get_raster_info(self.global_ha_per_cell_5m)['pixel_size']
        bounding_box_mode = 'intersection'
        base_vector_path_list = [self.two_poly_wgs84_aoi_path]
        raster_align_index = 0


        hb.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list, resample_method_list,
            target_pixel_size, bounding_box_mode, base_vector_path_list=base_vector_path_list,
            raster_align_index=raster_align_index, all_touched=True,
            gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

        # os.remove(target_raster_path_list[0])
        # self.fail()


