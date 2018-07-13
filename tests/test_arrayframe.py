from unittest import TestCase

import os, sys
import hazelbean as hb
import pandas as pd
import numpy as np


class ArrayFrameTester(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_arrayframe_load_and_save(self):
        input_array = np.arange(0, 18, 1).reshape((3,6))
        input_uri = hb.temp('.tif', remove_at_exit=True)
        geotransform = hb.calc_cylindrical_geotransform_from_array(input_array)
        # projection = hb.get_wkt_from_epsg_code(hb.common_epsg_codes_by_name['plate_carree'])
        projection = 'wgs84'
        hb.save_array_as_geotiff(input_array, input_uri, geotransform_override=geotransform, projection_override=projection)

        hb.ArrayFrame(input_uri)

        # "C:\Anaconda363\Library\share\gdal\gcs.csv"






