# coding=utf-8
import os, sys, math, time
from osgeo import gdal, ogr, osr
from gdal import gdalconst
import numpy
import numpy as np
import logging
import warnings
import traceback
import multiprocessing
from collections import OrderedDict
import hazelbean as hb


# First check for config file specific to this computer
user_path = os.path.expanduser('~')
default_hazelbean_config_uri = os.path.join(user_path, 'documents/hazelbean/config.txt')
PRIMARY_DRIVE = 'c:/'
EXTERNAL_BULK_DATA_DRIVE = 'e:/'
if os.path.exists(default_hazelbean_config_uri):
    with open(default_hazelbean_config_uri) as f:
        for line in f:
            if '=' in line:
                line_split = line.split('=')
                if line_split[0] == 'primary_drive_letter':
                    PRIMARY_DRIVE_LETTER = line_split[1][0]
                    PRIMARY_DRIVE = PRIMARY_DRIVE_LETTER + ':/'
                if line_split[0] == 'external_bulk_data_drive':
                    EXTERNAL_BULK_DATA_DRIVE_LETTER = line_split[1][0]
                    EXTERNAL_BULK_DATA_DRIVE = EXTERNAL_BULK_DATA_DRIVE_LETTER + ':/'
                if line_split[0] == 'configured_for_cython_compilation':
                    CONFIGURED_FOR_CYTHON_COMPILATION = float(line_split[1])

# HAZELBEAN SETUP GLOBALS
TEMPORARY_DIR = os.path.join(PRIMARY_DRIVE, 'temp')
BASE_DATA_DIR = os.path.join(PRIMARY_DRIVE, 'onedrive', 'projects', 'base_data')
BULK_DATA_DIR = os.path.join(PRIMARY_DRIVE, 'bulk_data')
EXTERNAL_BULK_DATA_DIR = os.path.join(EXTERNAL_BULK_DATA_DRIVE, 'bulk_data')
HAZELBEAN_WORKING_DIRECTORY = os.path.join(PRIMARY_DRIVE, 'OneDrive\\Projects\\hazelbean\\hazelbean') # TODOO Make this based on config file?
TEST_DATA_DIR = os.path.join(HAZELBEAN_WORKING_DIRECTORY, 'tests/data')
PROJECTS_DIR = os.path.join(PRIMARY_DRIVE, 'OneDrive\\Projects')

TINY_MEMORY_ARRAY_SIZE = 1e+04
SMALL_MEMORY_ARRAY_SIZE = 1e+05
MEDIUM_MEMORY_ARRAY_SIZE = 1e+06
LARGE_MEMORY_ARRAY_SIZE = 1e+07
MAX_IN_MEMORY_ARRAY_SIZE = 1e+09

# FROM Pygeoprocessing 06
LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module
MAX_TIMEOUT = 60.0
DEFAULT_GTIFF_CREATION_OPTIONS = ['TILED=YES', 'BIGTIFF=IF_SAFER']
LARGEST_ITERBLOCK = 2**20  # largest block for iterblocks to read in cells

# A dictionary to map the resampling method input string to the gdal type
try:
    RESAMPLE_DICT = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        'mode': gdal.GRA_Mode,
        'average': gdal.GRA_Average,
        'max': gdal.GRA_Max,
        'min': gdal.GRA_Min,
        'med': gdal.GRA_Med,
        'q1': gdal.GRA_Q1,
        'q3': gdal.GRA_Q3,
    }
except:

    RESAMPLE_DICT = {
        "near": gdal.GRA_NearestNeighbour,
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
    }

resampling_methods = RESAMPLE_DICT


class CustomLogger(logging.LoggerAdapter):
    def __init__(self, logger, *args, **kwargs):
        logging.LoggerAdapter.__init__(self, logger, *args, **kwargs)
        self.L = logger
        self.DEBUG_DEEPER_1_NUM = 9
        logging.addLevelName(self.DEBUG_DEEPER_1_NUM, "DEBUG_DEEPER_1")

    def process(self, msg, kwargs):
        return msg, kwargs

    def debug_deeper_1(self, message, *args, **kws):
        # Yes, logger takes its '*args' as 'args'.
        if self.isEnabledFor(self.DEBUG_DEEPER_1_NUM):
            self._log(self.DEBUG_DEEPER_1_NUM, message, args, **kws)

    def debug(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        stack_list = traceback.format_stack()

        key_file_root = hb.file_root(sys.argv[0])

        key_stack_elements = ''
        rest_of_stack = ''
        for i in range(len(stack_list)):
            if key_file_root in stack_list[i]:
                key_stack_elements += stack_list[i].split(', in ')[0]
            rest_of_stack += ' ' + str(stack_list[i].split(', in ')[0])

        if key_stack_elements:
            msg = str(msg) + ' ' + key_stack_elements + '. Rest of stack trace: '+ rest_of_stack
        else:
            msg = str(msg) + ' Stack trace: ' + rest_of_stack
        msg = 'WARNING ' + msg
        self.logger.warning(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        stack_list = traceback.format_stack()
        warning_string = ''
        for i in range(len(stack_list)):
            warning_string += ' ' + stack_list[i].split(', in ')[0]
        msg = str(msg) + ' ' + warning_string
        msg = 'CRITICAL ' + msg
        self.logger.critical(msg, *args, **kwargs)

    def set_log_file_uri(self, uri):
        hdlr = logging.FileHandler(uri)
        self.logger.addHandler(hdlr)



FORMAT = "%(message)s              --- %(asctime)s --- %(name)s %(levelname)s"
# FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)

LOGGING_LEVEL = logging.INFO

L = logging.getLogger('hazelbean')

L.setLevel(LOGGING_LEVEL)
L.addHandler(logging.NullHandler())  # silence logging by default

L = CustomLogger(L, {'msg': 'Custom message: '})

logging_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'critical': logging.CRITICAL,
}

##Deactvated logging to file.
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)#
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# L.addHandler(handler)
# logging.Logger.debug_deeper_1 = debug_deeper_1

def get_logger(logger_name=None, logging_level='info', format='full'):
    """Used to get a custom logger specific to a file other than just susing the config defined one."""
    if not logger_name:
        try:
            logger_name = os.path.basename(main.__file__)
        except:
            logger_name = 'unnamed_logger'
    L = logging.getLogger(logger_name)
    L.setLevel(logging_levels[logging_level])
    CL = CustomLogger(L, {'msg': 'Custom message: '})
    # FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    FORMAT = "%(message)s"
    formatter = logging.Formatter(FORMAT)

    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # L.addHandler(handler)
    return CL

def critical(self, msg, *args, **kwargs):
    """
    Delegate a debug call to the underlying logger, after adding
    contextual information from this adapter instance.
    """
    msg, kwargs = self.process_critical_logger(msg, kwargs)
    L.critical(msg, *args, **kwargs)

if not os.path.exists(TEMPORARY_DIR):
    try:
        os.makedirs(TEMPORARY_DIR)
    except:
        raise Exception('Could not create temp file at ' + TEMPORARY_DIR + '. Perhaps you do not have permission? Try setting hazelbean/config.TEMPORARY_DIR to something in your user folder.')

uris_to_delete_at_exit = []
plots_to_display_at_exit = []


# -- GLOBAL CONSTANTS
start_of_numerals_ascii_int = 48
start_of_uppercase_letters_ascii_int = 65
start_of_lowercase_letters_ascii_int = 97
alphanumeric_ascii_ints = list(range(start_of_numerals_ascii_int, start_of_numerals_ascii_int + 10)) + list(range(start_of_uppercase_letters_ascii_int, start_of_uppercase_letters_ascii_int + 26)) + list(range(start_of_lowercase_letters_ascii_int, start_of_lowercase_letters_ascii_int + 26))
alphanumeric_lowercase_ascii_ints = list(range(start_of_numerals_ascii_int, start_of_numerals_ascii_int + 10)) + list(range(start_of_lowercase_letters_ascii_int, start_of_lowercase_letters_ascii_int + 26))
alphanumeric_ascii_symbols = [chr(i) for i in alphanumeric_ascii_ints]
alphanumeric_lowercase_ascii_symbols = [chr(i) for i in alphanumeric_lowercase_ascii_ints] # numbers are lowercase i assume...


def delete_path_at_exit(path):
    if not os.path.exists(path):
        raise NameError('Cannot delete path ' + path + ' that does not exist.')
    if path in uris_to_delete_at_exit:
        L.warning('Attempted to add ' + path + ' to uris_to_delete_at_exit but it was already in there.')
        return
    else:
        uris_to_delete_at_exit.append(path)

def gdal_to_numpy_type(band):
    return _gdal_to_numpy_type(band)

def _gdal_to_numpy_type(band):
    """Calculate the equivalent numpy datatype from a GDAL raster band type.

    Args:
        band (gdal.Band): GDAL band

    Returns:
        numpy_datatype (numpy.dtype): equivalent of band.DataType
    """

    gdal_type_to_numpy_lookup = {
        gdal.GDT_Int16: numpy.int16,
        gdal.GDT_Int32: numpy.int32,
        gdal.GDT_UInt16: numpy.uint16,
        gdal.GDT_UInt32: numpy.uint32,
        gdal.GDT_Float32: numpy.float32,
        gdal.GDT_Float64: numpy.float64
    }

    if band.DataType in gdal_type_to_numpy_lookup:
        return gdal_type_to_numpy_lookup[band.DataType]

    # only class not in the lookup is a Byte but double check.
    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unknown DataType: %s" % str(band.DataType))

    metadata = band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
        return numpy.int8
    return numpy.uint8



# I got confused on  the foloowing two  lists. See http://www.gdal.org/ogr__core_8h.html#a787194bea637faf12d61643124a7c9fc
gdal_number_to_ogr_field_type = {
    1: 0, # not sure if not OFSTBoolean
    2: 0, # seemed to be unimplemented as uint etc.
    3: 0,
    4: 0,
    5: 0,
    6: 2,
    7: 2, # not sure if correct
}

type_string_to_ogr_field_type = {
    'int': gdal_number_to_ogr_field_type[1],
    'uint': gdal_number_to_ogr_field_type[1],
    'uint8': gdal_number_to_ogr_field_type[1],
    'uint16': gdal_number_to_ogr_field_type[1],
    'int16': gdal_number_to_ogr_field_type[1],
    'uint32': gdal_number_to_ogr_field_type[1],
    'int32': gdal_number_to_ogr_field_type[1],
    'float': gdal_number_to_ogr_field_type[6],
    'float32': gdal_number_to_ogr_field_type[6],
    'float64': gdal_number_to_ogr_field_type[7],
    'string': 4,
}

gdal_number_to_gdal_type = {
    1: gdalconst.GDT_Byte,
    2: gdalconst.GDT_UInt16,
    3: gdalconst.GDT_Int16,
    4: gdalconst.GDT_UInt32,
    5: gdalconst.GDT_Int32,
    6: gdalconst.GDT_Float32,
    7: gdalconst.GDT_Float64,
    8: gdalconst.GDT_CInt16,
    9: gdalconst.GDT_CInt32,
    10: gdalconst.GDT_CFloat32,
    11: gdalconst.GDT_CFloat64,
}

gdal_number_to_gdal_name = {
    1: 'Byte',
    2: 'UInt16',
    3: 'Int16',
    4: 'UInt32',
    5: 'Int32',
    6: 'Float32',
    7: 'Float64',
    8: 'CInt16',
    9: 'CInt32',
    10: 'CFloat32',
    11: 'CFloat64'
}

gdal_name_to_gdal_number = {
    'Byte': 1,
    'uint8': 1,
    'Uint8': 1,
    'UInt16': 2,
    'Int16': 3,
    'UInt32': 4,
    'Int32': 5,
    'Float32': 6,
    'Float64': 7,
    'CInt16': 8,
    'CInt32': 9,
    'CFloat32': 10,
    'CFloat64': 11,
    'byte': 1,
    'uint16': 2,
    'int16': 3,
    'uint32': 4,
    'int32': 5,
    'float32': 6,
    'float64': 7,
    'cint16': 8,
    'cint32': 9,
    'cfloat32': 10,
    'cfloat64': 11,
}

gdal_number_to_numpy_type = {
    1: numpy.uint8,
    2: numpy.uint16,
    3: numpy.int16,
    4: numpy.uint32,
    5: numpy.int32,
    6: numpy.float32,
    7: numpy.float64,
    8: numpy.complex64,
    9: numpy.complex64,
    10: numpy.complex64,
    11: numpy.complex128
}

numpy_type_to_gdal_number = {
    numpy.uint8: 1,
    numpy.uint16: 2,
    numpy.int16: 3,
    numpy.uint32: 4,
    numpy.int32: 5,
    numpy.float32: 6,
    numpy.float64: 7,
    numpy.complex64: 8,  # THe omission here is from the unexplained duplication in gdal_number_to_numpy_type
    numpy.complex128: 11,
    numpy.int64: 7, # NOTE, gdal does not support 64bit ints.

    np.dtype('uint8'): 1,
    np.dtype('uint16'): 2,
    np.dtype('int16'): 3,
    np.dtype('uint32'): 4,
    np.dtype('int32'): 5,
    np.dtype('float32'): 6,
    np.dtype('float64'): 7,
    np.dtype('complex64'): 8,  # THe omission here is from the unexplained duplication in gdal_number_to_numpy_type
    np.dtype('complex128'): 11,
    np.dtype('int64'): 7,

}

numpy_name_to_gdal_number = {
    'int8': 1,
    'uint8': 1,
    'uint16': 2,
    'int16': 3,
    'uint32': 4,
    'int32': 5,
    'int64': 7, # WTF couldnt find gdal's int64 type . might not exist?
    'uint64': 7, # WTF couldnt find gdal's int64 type . might not exist?
    'float32': 6,
    'float64': 7,
    'complex64': 8,  # THe omission here is from the unexplained duplication in gdal_number_to_numpy_type
    'complex128': 11,
}

gdal_type_to_numpy_type = {
    gdalconst.GDT_Byte: numpy.uint8,
    gdalconst.GDT_UInt16: numpy.uint16,
    gdalconst.GDT_Int16: numpy.int16,
    gdalconst.GDT_UInt32: numpy.uint32,
    gdalconst.GDT_Int32: numpy.int32,
    gdalconst.GDT_Float32: numpy.float32,
    gdalconst.GDT_Float64: numpy.float64,
    gdalconst.GDT_CInt16: numpy.complex64,
    gdalconst.GDT_CInt32: numpy.complex64,
    gdalconst.GDT_CFloat32: numpy.complex64,
    gdalconst.GDT_CFloat64: numpy.complex128
}

GDAL_TO_NUMPY_TYPE = {
    gdal.GDT_Byte: numpy.uint8,
    gdal.GDT_Int16: numpy.int16,
    gdal.GDT_Int32: numpy.int32,
    gdal.GDT_UInt16: numpy.uint16,
    gdal.GDT_UInt32: numpy.uint32,
    gdal.GDT_Float32: numpy.float32,
    gdal.GDT_Float64: numpy.float64
}

numpy_type_to_gdal_type = {
    numpy.uint8: gdalconst.GDT_Byte,
    numpy.uint16: gdalconst.GDT_UInt16,
    numpy.int16: gdalconst.GDT_Int16,
    numpy.uint32: gdalconst.GDT_UInt32,
    numpy.int32: gdalconst.GDT_Int32,
    numpy.float32: gdalconst.GDT_Float32,
    numpy.float64: gdalconst.GDT_Float64,
    numpy.complex64: gdalconst.GDT_CInt16,
    numpy.complex64: gdalconst.GDT_CInt32,
    numpy.complex64: gdalconst.GDT_CFloat32,
    numpy.complex128: gdalconst.GDT_CFloat64,
    np.int64: gdalconst.GDT_Float64
}


common_epsg_codes_by_name = OrderedDict()
common_epsg_codes_by_name['wgs84'] =  4326
common_epsg_codes_by_name['wec'] =  54002
common_epsg_codes_by_name['world_eckert_iv'] =  54012
common_epsg_codes_by_name['robinson'] =  54030
# common_epsg_codes_by_name['mollweide'] =  54009
common_epsg_codes_by_name['plate_carree'] =  32662
# common_epsg_codes_by_name['mercator'] =  3857
# common_epsg_codes_by_name[]# '] = wec_old': 32663,
# common_epsg_codes_by_name[]# '] = wec_sphere': 3786,

common_projected_epsg_codes_by_name = OrderedDict()
# common_projected_epsg_codes_by_name['wgs84'] =  4326
common_projected_epsg_codes_by_name['wec'] =  54002
common_projected_epsg_codes_by_name['world_eckert_iv'] =  54012
common_projected_epsg_codes_by_name['robinson'] =  54030
# common_projected_epsg_codes_by_name['mollweide'] =  54009
common_projected_epsg_codes_by_name['plate_carree'] =  32662
# common_projected_epsg_codes_by_name['mercator'] =  3857


robinson_wkt = """PROJCS["World_Robinson",
    GEOGCS["GCS_WGS_1984",
        DATUM["WGS_1984",
            SPHEROID["WGS_1984",6378137,298.257223563]],
        PRIMEM["Greenwich",0],
        UNIT["Degree",0.017453292519943295]],
    PROJECTION["Robinson"],
    PARAMETER["False_Easting",0],
    PARAMETER["False_Northing",0],
    PARAMETER["Central_Meridian",0],
    UNIT["Meter",1],
    AUTHORITY["EPSG","54030"]]"""

wgs_84_wkt = """GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""

cylindrical_wkt = """PROJCS["WGS 84 / World Equidistant Cylindrical",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"],
        AXIS["Latitude",NORTH],
        AXIS["Longitude",EAST]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]]]"""

plate_carree_wkt = """PROJCS["WGS 84 / Plate Carree (deprecated)",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    PROJECTION["Equirectangular"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",0],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    AUTHORITY["EPSG","32662"],
    AXIS["X",EAST],
    AXIS["Y",NORTH]]"""

common_geotransforms = {
    'global_5m': (-180.0, 0.08333333333333333, 0.0, 90.0, 0.0, -0.08333333333333333),
    'global_30s': (-180.0, 0.00833333333333333, 0.0, 90.0, 0.0, -0.00833333333333333),
    # '5': (-18040094.5475796148, 1000.0, 0.0, 8825228.643963514, 0.0, -1000.0),
    'wec_30s': (-20037508.3427892, 927.66242327728, 0.0, 10018754.171394622, 0.0, -927.66242327728)

}

def get_global_geotransform_from_resolution(input_resolution):
    return (-180.0, input_resolution, 0.0, 90.0, 0.0, -input_resolution)

common_bounding_boxes_in_degrees = {
    'global': [-180., -90., 180., 90.]
}

common_projection_wkts = {
    'wgs84': "GEOGCS[\"WGS 84\", DATUM[\"WGS_1984\", SPHEROID[\"WGS 84\", 6378137, 298.257223563, AUTHORITY[\"EPSG\", \"7030\"]], AUTHORITY[\"EPSG\", \"6326\"]],PRIMEM[\"Greenwich\", 0], UNIT[\"degree\", 0.0174532925199433], AUTHORITY[\"EPSG\", \"4326\"]]"
}

default_no_data_values_by_gdal_number = {
    1: 255,
    2: 65535,
    3: -32768,
    4: 0, # NOTE MASSIVE FLAW, because QGIS/GDAL doesnt support UInt32, had to clamp it to 0
    # 5: -2147483647,
    5: -2147483648,
    6: -9999.0,
    7: -9999.0,
}

default_no_data_values_by_gdal_stringed_number = {
    '1': 255,
    '2': 65535,
    '3': -32768,
    '4': 0, # NOTE MASSIVE FLAW, because QGIS/GDAL doesnt support UInt32, had to clamp it to 0
    # '5': -2147483647,
    '5': -2147483648,
    '6': -9999.0,
    '7': -9999.0,
    ## The following didn't work beacuse they couldn't  be written via Band.SetNoDataValue() in gdal.
    # '6': -3.4028235e+38,
    # '7': -1.7976931348623157e+308
    # '6': float(np.finfo(np.float32).min),
    # '7': float(np.finfo(np.float64).min),
}

size_of_one_arcdegree_at_equator_in_meters = 111319.49079327358  # Based on (2 * math.pi * 6378.137*1000) / 360  # old 111319



esacci_standard_classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, ]

esacci_standard_class_descriptions = OrderedDict()
esacci_standard_class_descriptions[0] = 'No Data'
esacci_standard_class_descriptions[10] = 'Cropland, rainfed'
esacci_standard_class_descriptions[20] = 'Cropland, irrigated or post-flooding'
esacci_standard_class_descriptions[30] = 'Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover)(<50%)'
esacci_standard_class_descriptions[40] = 'Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland(<50%)'
esacci_standard_class_descriptions[50] = 'Tree cover, broadleaved, evergreen, closed to open (>15%)'
esacci_standard_class_descriptions[60] = 'Tree cover, broadleaved, deciduous, closed to open (>15%)'
esacci_standard_class_descriptions[70] = 'Tree cover, needleleaved, evergreen, closed to open (>15%)'
esacci_standard_class_descriptions[80] = 'Tree cover, needleleaved, deciduous, closed to open (>15%)'
esacci_standard_class_descriptions[90] = 'Tree cover, mixed leaf type (broadleaved and needleleaved)'
esacci_standard_class_descriptions[100] = 'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)'
esacci_standard_class_descriptions[110] = 'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)'
esacci_standard_class_descriptions[120] = 'Shrubland'
esacci_standard_class_descriptions[130] = 'Grassland'
esacci_standard_class_descriptions[140] = 'Lichens and mosses'
esacci_standard_class_descriptions[150] = 'Sparse vegetation (tree, shrub, herbaceous cover) (<15%)'
esacci_standard_class_descriptions[160] = 'Tree cover, flooded, fresh or brakish water'
esacci_standard_class_descriptions[170] = 'Tree cover, flooded, saline water'
esacci_standard_class_descriptions[180] = 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water'
esacci_standard_class_descriptions[190] = 'Urban areas'
esacci_standard_class_descriptions[200] = 'Bare areas'
esacci_standard_class_descriptions[210] = 'Water bodies'
esacci_standard_class_descriptions[220] = 'Permanent snow and ice'

esacci_extended_classes = [00, 10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140, 150, 151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220]

esacci_extended_class_descriptions = OrderedDict()
esacci_extended_class_descriptions[0] = 'No Data'
esacci_extended_class_descriptions[10] = 'Cropland, rainfed'
esacci_extended_class_descriptions[11] = 'Cropland, rainfed, herbaceous cover'
esacci_extended_class_descriptions[12] = 'Cropland, rainfed, tree or shrub cover'
esacci_extended_class_descriptions[20] = 'Cropland, irrigated or post-flooding'
esacci_extended_class_descriptions[30] = 'Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover)(<50%)'
esacci_extended_class_descriptions[40] = 'Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland(<50%)'
esacci_extended_class_descriptions[50] = 'Tree cover, broadleaved, evergreen, closed to open (>15%)'
esacci_extended_class_descriptions[60] = 'Tree cover, broadleaved, deciduous, closed to open (>15%)'
esacci_extended_class_descriptions[61] = 'Tree cover, broadleaved, deciduous, closed (>40%)'
esacci_extended_class_descriptions[62] = 'Tree cover, broadleaved, deciduous, open (15-40%)'
esacci_extended_class_descriptions[70] = 'Tree cover, needleleaved, evergreen, closed to open (>15%)'
esacci_extended_class_descriptions[71] = 'Tree cover, needleleaved, evergreen, closed to open (>15%)'
esacci_extended_class_descriptions[72] = 'Tree cover, needleleaved, evergreen, open (15-40%)'
esacci_extended_class_descriptions[80] = 'Tree cover, needleleaved, deciduous, closed to open (>15%)'
esacci_extended_class_descriptions[81] = 'Tree cover, needleleaved, deciduous, closed (>40%)'
esacci_extended_class_descriptions[82] = 'Tree cover, needleleaved, deciduous, open (15-40%)'
esacci_extended_class_descriptions[90] = 'Tree cover, mixed leaf type (broadleaved and needleleaved)'
esacci_extended_class_descriptions[100] = 'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)'
esacci_extended_class_descriptions[110] = 'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)'
esacci_extended_class_descriptions[120] = 'Shrubland'
esacci_extended_class_descriptions[121] = 'Evergreen shrubland'
esacci_extended_class_descriptions[122] = 'Deciduous shrubland '
esacci_extended_class_descriptions[130] = 'Grassland'
esacci_extended_class_descriptions[140] = 'Lichens and mosses'
esacci_extended_class_descriptions[150] = 'Sparse vegetation (tree, shrub, herbaceous cover) (<15%)'
esacci_extended_class_descriptions[151] = 'Sparse tree (<15%)'
esacci_extended_class_descriptions[152] = 'Sparse shrub (<15%)'
esacci_extended_class_descriptions[153] = 'Sparse herbaceous cover (<15%)'
esacci_extended_class_descriptions[160] = 'Tree cover, flooded, fresh or brakish water'
esacci_extended_class_descriptions[170] = 'Tree cover, flooded, saline water'
esacci_extended_class_descriptions[180] = 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water'
esacci_extended_class_descriptions[190] = 'Urban areas'
esacci_extended_class_descriptions[200] = 'Bare areas'
esacci_extended_class_descriptions[201] = 'Consolidated bare areas'
esacci_extended_class_descriptions[202] = 'Unconsolidated bare areas'
esacci_extended_class_descriptions[210] = 'Water bodies'
esacci_extended_class_descriptions[220] = 'Permanent snow and ice'


esacci_extended_short_class_descriptions = OrderedDict()
esacci_extended_short_class_descriptions[0] = 'ndv'
esacci_extended_short_class_descriptions[10] = 'crop_rainfed'
esacci_extended_short_class_descriptions[11] = 'crop_rainfed_herb'
esacci_extended_short_class_descriptions[12] = 'crop_rainfed_tree'
esacci_extended_short_class_descriptions[20] = 'crop_irrigated'
esacci_extended_short_class_descriptions[30] = 'crop_natural_mosaic'
esacci_extended_short_class_descriptions[40] = 'natural_crop_mosaic'
esacci_extended_short_class_descriptions[50] = 'tree_broadleaved_evergreen'
esacci_extended_short_class_descriptions[60] = 'tree_broadleaved_deciduous_closed_to_open_15'
esacci_extended_short_class_descriptions[61] = 'tree_broadleaved_deciduous_closed_40'
esacci_extended_short_class_descriptions[62] = 'tree_broadleaved_deciduous_open_15_40'
esacci_extended_short_class_descriptions[70] = 'tree_needleleaved_evergreen_closed_to_open_15'
esacci_extended_short_class_descriptions[71] = 'tree_needleleaved_evergreen_closed_to_open_15_extended'
esacci_extended_short_class_descriptions[72] = 'tree_needleleaved_evergreen_open_15_40'
esacci_extended_short_class_descriptions[80] = 'tree_needleleaved_deciduous_closed_to_open_15'
esacci_extended_short_class_descriptions[81] = 'tree_needleleaved_deciduous_closed_40'
esacci_extended_short_class_descriptions[82] = 'tree_needleleaved_deciduous_open_15_40'
esacci_extended_short_class_descriptions[90] = 'tree_mixed_type'
esacci_extended_short_class_descriptions[100] = 'Mosaic_tree_and_shrub_50_herbaceous_cover_50'
esacci_extended_short_class_descriptions[110] = 'Mosaic_herbaceous_cover_50_tree_and_shrub_50'
esacci_extended_short_class_descriptions[120] = 'Shrubland'
esacci_extended_short_class_descriptions[121] = 'Evergreen_shrubland'
esacci_extended_short_class_descriptions[122] = 'Deciduous_shrubland_'
esacci_extended_short_class_descriptions[130] = 'Grassland'
esacci_extended_short_class_descriptions[140] = 'Lichens_and_mosses'
esacci_extended_short_class_descriptions[150] = 'Sparse_vegetation_tree_shrub_herbaceous_cover_15'
esacci_extended_short_class_descriptions[151] = 'Sparse_tree_15'
esacci_extended_short_class_descriptions[152] = 'Sparse_shrub_15'
esacci_extended_short_class_descriptions[153] = 'Sparse_herbaceous_cover_15'
esacci_extended_short_class_descriptions[160] = 'Tree_cover_flooded_fresh_or_brakish_water'
esacci_extended_short_class_descriptions[170] = 'Tree_cover_flooded_saline_water'
esacci_extended_short_class_descriptions[180] = 'Shrub_or_herbaceous_cover_flooded_fresh_saline_brakish_water'
esacci_extended_short_class_descriptions[190] = 'Urban_areas'
esacci_extended_short_class_descriptions[200] = 'Bare_areas'
esacci_extended_short_class_descriptions[201] = 'Consolidated_bare_areas'
esacci_extended_short_class_descriptions[202] = 'Unconsolidated_bare_areas'
esacci_extended_short_class_descriptions[210] = 'Water_bodies'
esacci_extended_short_class_descriptions[220] = 'Permanent_snow_and_ice'





nlcd_colors = OrderedDict()
nlcd_colors[0] = [0,0,0]
nlcd_colors[1] = [0,249,0]
nlcd_colors[11] = [71,107,160]
nlcd_colors[12] = [209,221,249]
nlcd_colors[21] = [221,201,201]
nlcd_colors[22] = [216,147,130]
nlcd_colors[23] = [237,0,0]
nlcd_colors[24] = [170,0,0]
nlcd_colors[31] = [178,173,163]
nlcd_colors[32] = [249,249,249]
nlcd_colors[41] = [104,170,99]
nlcd_colors[42] = [28,99,48]
nlcd_colors[43] = [181,201,142]
nlcd_colors[51] = [165,140,48]
nlcd_colors[52] = [204,186,124]
nlcd_colors[71] = [226,226,193]
nlcd_colors[72] = [201,201,119]
nlcd_colors[73] = [153,193,71]
nlcd_colors[74] = [119,173,147]
nlcd_colors[81] = [219,216,61]
nlcd_colors[82] = [170,112,40]
nlcd_colors[90] = [186,216,234]
nlcd_colors[91] = [181,211,229]
nlcd_colors[92] = [181,211,229]
nlcd_colors[93] = [181,211,229]
nlcd_colors[94] = [181,211,229]
nlcd_colors[95] = [112,163,186]


nlcd_category_names = OrderedDict()
nlcd_category_names[0] = 'ndv'
nlcd_category_names[11] = 'Open Water'
nlcd_category_names[12] = 'Perennial Ice/Snow'
nlcd_category_names[21] = 'Developed, Open Space'
nlcd_category_names[22] = 'Developed, Low Intensity'
nlcd_category_names[23] = 'Developed, Medium Intensity'
nlcd_category_names[24] = 'Developed High Intensity'
nlcd_category_names[31] = 'Barren Land (Rock/Sand/Clay)'
nlcd_category_names[41] = 'Deciduous Forest'
nlcd_category_names[42] = 'Evergreen Forest'
nlcd_category_names[43] = 'Mixed Forest'
nlcd_category_names[51] = 'Dwarf Scrub'
nlcd_category_names[52] = 'Shrub/Scrub'
nlcd_category_names[71] = 'Grassland/Herbaceous'
nlcd_category_names[72] = 'Sedge/Herbaceous'
nlcd_category_names[73] = 'Lichens'
nlcd_category_names[74] = 'Moss'
nlcd_category_names[81] = 'Pasture/Hay'
nlcd_category_names[82] = 'Cultivated Crops'
nlcd_category_names[90] = 'Woody Wetlands'
nlcd_category_names[95] = 'Emergent Herbaceous Wetlands'

nlcd_category_descriptions = OrderedDict()
nlcd_category_descriptions[0] = 'ndv'
nlcd_category_descriptions[11] = 'areas of open water, generally with less than 25% cover of vegetation or soil.'
nlcd_category_descriptions[12] = 'areas characterized by a perennial cover of ice and/or snow, generally greater than 25% of total cover.'
nlcd_category_descriptions[21] = 'areas with a mixture of some constructed materials, but mostly vegetation in the form of lawn grasses. Impervious surfaces account for less than 20% of total cover. These areas most commonly include large-lot single-family housing units, parks, golf courses, and vegetation planted in developed settings for recreation, erosion control, or aesthetic purposes.'
nlcd_category_descriptions[22] = 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 20% to 49% percent of total cover. These areas most commonly include single-family housing units.'
nlcd_category_descriptions[23] = 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 50% to 79% of the total cover. These areas most commonly include single-family housing units.'
nlcd_category_descriptions[24] = 'highly developed areas where people reside or work in high numbers. Examples include apartment complexes, row houses and commercial/industrial. Impervious surfaces account for 80% to 100% of the total cover.'
nlcd_category_descriptions[31] = 'areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits and other accumulations of earthen material. Generally, vegetation accounts for less than 15% of total cover.'
nlcd_category_descriptions[41] = 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species shed foliage simultaneously in response to seasonal change.'
nlcd_category_descriptions[42] = 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species maintain their leaves all year. Canopy is never without green foliage.'
nlcd_category_descriptions[43] = 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. Neither deciduous nor evergreen species are greater than 75% of total tree cover.'
nlcd_category_descriptions[51] = 'Alaska only areas dominated by shrubs less than 20 centimeters tall with shrub canopy typically greater than 20% of total vegetation. This type is often co-associated with grasses, sedges, herbs, and non-vascular vegetation.'
nlcd_category_descriptions[52] = 'areas dominated by shrubs; less than 5 meters tall with shrub canopy typically greater than 20% of total vegetation. This class includes true shrubs, young trees in an early successional stage or trees stunted from environmental conditions.'
nlcd_category_descriptions[71] = 'areas dominated by gramanoid or herbaceous vegetation, generally greater than 80% of total vegetation. These areas are not subject to intensive management such as tilling, but can be utilized for grazing.'
nlcd_category_descriptions[72] = 'Alaska only areas dominated by sedges and forbs, generally greater than 80% of total vegetation. This type can occur with significant other grasses or other grass like plants, and includes sedge tundra, and sedge tussock tundra.'
nlcd_category_descriptions[73] = 'Alaska only areas dominated by fruticose or foliose lichens generally greater than 80% of total vegetation.'
nlcd_category_descriptions[74] = 'Alaska only areas dominated by mosses, generally greater than 80% of total vegetation.'
nlcd_category_descriptions[81] = 'areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed or hay crops, typically on a perennial cycle. Pasture/hay vegetation accounts for greater than 20% of total vegetation.'
nlcd_category_descriptions[82] = 'areas used for the production of annual crops, such as corn, soybeans, vegetables, tobacco, and cotton, and also perennial woody crops such as orchards and vineyards. Crop vegetation accounts for greater than 20% of total vegetation. This class also includes all land being actively tilled.'
nlcd_category_descriptions[90] = 'areas where forest or shrubland vegetation accounts for greater than 20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.'
nlcd_category_descriptions[95] = 'Areas where perennial herbaceous vegetation accounts for greater than 80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.'




possible_shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.sbn', '.sbx', '.fbn', '.fbx', '.ain', '.aih', '.ixs', '.mxs', '.atx', '.shp.xml', '.cpg', '.qix']
common_gdal_readable_file_extensions = ['.tif', '.bil', '.adf', '.asc', '.hdf', '.nc',]
# gdal_readable_formats = ['AAIGrid', 'ACE2', 'ADRG', 'AIG', 'ARG', 'BLX', 'BAG', 'BMP', 'BSB', 'BT', 'CPG', 'CTG', 'DIMAP', 'DIPEx', 'DODS', 'DOQ1', 'DOQ2', 'DTED', 'E00GRID', 'ECRGTOC', 'ECW', 'EHdr', 'EIR', 'ELAS', 'ENVI', 'ERS', 'FAST', 'GPKG', 'GEORASTER', 'GRIB', 'GMT', 'GRASS', 'GRASSASCIIGrid', 'GSAG', 'GSBG', 'GS7BG', 'GTA', 'GTiff', 'GTX', 'GXF', 'HDF4', 'HDF5', 'HF2', 'HFA', 'IDA', 'ILWIS', 'INGR', 'IRIS', 'ISIS2', 'ISIS3', 'JDEM', 'JPEG', 'JPEG2000', 'JP2ECW', 'JP2KAK', 'JP2MrSID', 'JP2OpenJPEG', 'JPIPKAK', 'KEA', 'KMLSUPEROVERLAY', 'L1B', 'LAN', 'LCP', 'Leveller', 'LOSLAS', 'MBTiles', 'MAP', 'MEM', 'MFF', 'MFF2 (HKV)', 'MG4Lidar', 'MrSID', 'MSG', 'MSGN', 'NDF', 'NGSGEOID', 'NITF', 'netCDF', 'NTv2', 'NWT_GRC', 'NWT_GRD', 'OGDI', 'OZI', 'PCIDSK', 'PCRaster', 'PDF', 'PDS', 'PLMosaic', 'PostGISRaster', 'Rasterlite', 'RIK', 'RMF', 'ROI_PAC', 'RPFTOC', 'RS2', 'RST', 'SAGA', 'SAR_CEOS', 'SDE', 'SDTS', 'SGI', 'SNODAS', 'SRP', 'SRTMHGT', 'USGSDEM', 'VICAR', 'VRT', 'WCS', 'WMS', 'XYZ', 'ZMap',]