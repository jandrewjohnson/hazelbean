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
import inspect

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
        for i in args:
            msg += ', ' + str(i)
        msg, kwargs = self.process(msg, kwargs)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg = str(msg)
        for i in args:
            msg += ', ' + str(i)
        args = []
        msg, kwargs = self.process(msg, kwargs)
        self.logger.info(msg, *args, **kwargs)

    def print(self, msg, *args, **kwargs):
        # Hacky piece of code to report both the names and the values of variables passed to info.
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        names = string[string.find('(') + 1:-1].split(',')
        names = [i.replace(' ', '') for i in names]

        if type(msg) is not str:
            msg = '\n' + names[0] + ':\t' + str(msg)
        else:
            msg = '\n' + msg
        for c, i in enumerate(args):
            # print('names', names, c, i)
            if type(i) is not str:
                msg += '\n' + str(names[c + 1]) + ':\t' + str(i)
            else:
                msg += '\n' + str(i)

        msg = msg.expandtabs(30)

        args = []
        msg, kwargs = self.process(msg, kwargs)

        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        for i in args:
            msg += ', ' + str(i)
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
        for i in args:
            msg += ', ' + str(i)
        msg, kwargs = self.process(msg, kwargs)
        stack_list = traceback.format_stack()
        warning_string = ''
        # stack_list.reverse()
        for i in range(len(stack_list)):
            warning_string += ' ' + stack_list[i].split(', in ')[0] + '\n'
        msg = str(msg) + ' Stacktrace:\n' + warning_string
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

if not os.path.exists(hb.globals.TEMPORARY_DIR):
    try:
        os.makedirs(hb.globals.TEMPORARY_DIR)
    except:
        raise Exception('Could not create temp file at ' + hb.globals.TEMPORARY_DIR + '. Perhaps you do not have permission? Try setting hazelbean/config.TEMPORARY_DIR to something in your user folder.')

uris_to_delete_at_exit = []
plots_to_display_at_exit = []


def general_callback(df_complete, psz_message, p_progress_arg):
    """The argument names come from the GDAL API for callbacks."""
    try:
        current_time = time.time()
        if ((current_time - general_callback.last_time) > 5.0 or
                (df_complete == 1.0 and general_callback.total_time >= 5.0)):
            print(
                "ReprojectImage %.1f%% complete %s, psz_message %s",
                df_complete * 100, p_progress_arg[0], psz_message)
            general_callback.last_time = current_time
            general_callback.total_time += current_time
    except AttributeError:
        general_callback.last_time = time.time()
        general_callback.total_time = 0.0

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






def get_global_geotransform_from_resolution(input_resolution):
    return (-180.0, input_resolution, 0.0, 90.0, 0.0, -input_resolution)

common_bounding_boxes_in_degrees = {
    'global': [-180., -90., 180., 90.]
}

common_projection_wkts = {
    'wgs84': "GEOGCS[\"WGS 84\", DATUM[\"WGS_1984\", SPHEROID[\"WGS 84\", 6378137, 298.257223563, AUTHORITY[\"EPSG\", \"7030\"]], AUTHORITY[\"EPSG\", \"6326\"]],PRIMEM[\"Greenwich\", 0], UNIT[\"degree\", 0.0174532925199433], AUTHORITY[\"EPSG\", \"4326\"]]"
}


luh_data_dir = os.path.join(hb.BASE_DATA_DIR, 'luh2', 'raw_data')
# Corresponds to a directory containing the latest LUH data download of states.nc and management.nc from maryland website
luh_scenario_names = [
    "RCP26_SSP1",
    "RCP34_SSP4",
    "RCP45_SSP2",
    "RCP60_SSP4",
    "RCP70_SSP3",
    "RCP85_SSP5",
    # "historical",
]

luh_scenario_states_paths = OrderedDict()
luh_scenario_states_paths['RCP26_SSP1'] = os.path.join(luh_data_dir, 'RCP26_SSP1', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['RCP34_SSP4'] = os.path.join(luh_data_dir, 'RCP34_SSP4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['RCP45_SSP2'] = os.path.join(luh_data_dir, 'RCP45_SSP2', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['RCP60_SSP4'] = os.path.join(luh_data_dir, 'RCP60_SSP4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['RCP70_SSP3'] = os.path.join(luh_data_dir, 'RCP70_SSP3', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['RCP85_SSP5'] = os.path.join(luh_data_dir, 'RCP85_SSP5', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc")
luh_scenario_states_paths['historical'] = os.path.join(luh_data_dir, 'historical', r"states.nc")

luh_scenario_management_paths = OrderedDict()
luh_scenario_management_paths['RCP26_SSP1'] = os.path.join(luh_data_dir, 'RCP26_SSP1', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['RCP34_SSP4'] = os.path.join(luh_data_dir, 'RCP34_SSP4', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['RCP45_SSP2'] = os.path.join(luh_data_dir, 'RCP45_SSP2', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['RCP60_SSP4'] = os.path.join(luh_data_dir, 'RCP60_SSP4', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['RCP70_SSP3'] = os.path.join(luh_data_dir, 'RCP70_SSP3', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['RCP85_SSP5'] = os.path.join(luh_data_dir, 'RCP85_SSP5', r"multiple-management_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc")
luh_scenario_management_paths['historical'] = os.path.join(luh_data_dir, 'historical', r"management.nc")

luh_state_names = [
    'primf',
    'primn',
    'secdf',
    'secdn',
    'urban',
    'c3ann',
    'c4ann',
    'c3per',
    'c4per',
    'c3nfx',
    'pastr',
    'range',
    'secmb',
    'secma',
]

luh_management_names = [
    'fertl_c3ann',
    'irrig_c3ann',
    'crpbf_c3ann',
    'fertl_c4ann',
    'irrig_c4ann',
    'crpbf_c4ann',
    'fertl_c3per',
    'irrig_c3per',
    'crpbf_c3per',
    'fertl_c4per',
    'irrig_c4per',
    'crpbf_c4per',
    'fertl_c3nfx',
    'irrig_c3nfx',
    'crpbf_c3nfx',
    'fharv_c3per',
    'fharv_c4per',
    'flood',
    'rndwd',
    'fulwd',
    'combf',
    'crpbf_total',
]



worldclim_bioclimatic_variable_names = OrderedDict()
worldclim_bioclimatic_variable_names[1] = 'Annual Mean Temperature'
worldclim_bioclimatic_variable_names[2] = 'Mean Diurnal Range (Mean of monthly (max temp - min temp))'
worldclim_bioclimatic_variable_names[3] = 'Isothermality (BIO2/BIO7) (* 100)'
worldclim_bioclimatic_variable_names[4] = 'Temperature Seasonality (standard deviation *100)'
worldclim_bioclimatic_variable_names[5] = 'Max Temperature of Warmest Month'
worldclim_bioclimatic_variable_names[6] = 'Min Temperature of Coldest Month'
worldclim_bioclimatic_variable_names[7] = 'Temperature Annual Range (BIO5-BIO6)'
worldclim_bioclimatic_variable_names[8] = 'Mean Temperature of Wettest Quarter'
worldclim_bioclimatic_variable_names[9] = 'Mean Temperature of Driest Quarter'
worldclim_bioclimatic_variable_names[10] = 'Mean Temperature of Warmest Quarter'
worldclim_bioclimatic_variable_names[11] = 'Mean Temperature of Coldest Quarter'
worldclim_bioclimatic_variable_names[12] = 'Annual Precipitation'
worldclim_bioclimatic_variable_names[13] = 'Precipitation of Wettest Month'
worldclim_bioclimatic_variable_names[14] = 'Precipitation of Driest Month'
worldclim_bioclimatic_variable_names[15] = 'Precipitation Seasonality (Coefficient of Variation)'
worldclim_bioclimatic_variable_names[16] = 'Precipitation of Wettest Quarter'
worldclim_bioclimatic_variable_names[17] = 'Precipitation of Driest Quarter'
worldclim_bioclimatic_variable_names[18] = 'Precipitation of Warmest Quarter'
worldclim_bioclimatic_variable_names[19] = 'Precipitation of Coldest Quarter'

countries_full_column_names = ['id', 'iso3', 'nev_name', 'fao_name', 'fao_id_c', 'gtap140', 'continent', 'region_un', 'region_wb', 'geom_index', 'abbrev', 'adm0_a3', 'adm0_a3_is', 'adm0_a3_un', 'adm0_a3_us', 'adm0_a3_wb', 'admin', 'brk_a3', 'brk_group', 'brk_name', 'country', 'disp_name', 'economy', 'fao_id', 'fao_reg', 'fips_10_', 'formal_en', 'formal_fr', 'gau', 'gdp_md_est', 'gdp_year', 'geounit', 'gu_a3', 'income_grp', 'iso', 'iso2_cull', 'iso3_cull', 'iso_3digit', 'iso_a2', 'iso_a3', 'iso_a3_eh', 'iso_n3', 'lastcensus', 'name', 'name_alt', 'name_ar', 'name_bn', 'name_cap', 'name_ciawf', 'name_de', 'name_el', 'name_en', 'name_es', 'name_fr', 'name_hi', 'name_hu', 'name_id', 'name_it', 'name_ja', 'name_ko', 'name_long', 'name_nl', 'name_pl', 'name_pt', 'name_ru', 'name_sort', 'name_sv', 'name_tr', 'name_vi', 'name_zh', 'ne_id', 'nev_lname', 'nev_sname', 'note_adm0', 'note_brk', 'official', 'olympic', 'pop_est', 'pop_rank', 'pop_year', 'postal', 'sov_a3', 'sovereignt', 'su_a3',
                               'subregion', 'subunit', 'type', 'un_a3', 'un_iso_n', 'un_vehicle', 'undp', 'uni', 'wb_a2', 'wb_a3', 'wiki1', 'wikidataid', 'wikipedia', 'woe_id', 'woe_id_eh', 'woe_note']

possible_shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.sbn', '.sbx', '.fbn', '.fbx', '.ain', '.aih', '.ixs', '.mxs', '.atx', '.shp.xml', '.cpg', '.qix']
common_gdal_readable_file_extensions = ['.tif', '.bil', '.adf', '.asc', '.hdf', '.nc',]
# gdal_readable_formats = ['AAIGrid', 'ACE2', 'ADRG', 'AIG', 'ARG', 'BLX', 'BAG', 'BMP', 'BSB', 'BT', 'CPG', 'CTG', 'DIMAP', 'DIPEx', 'DODS', 'DOQ1', 'DOQ2', 'DTED', 'E00GRID', 'ECRGTOC', 'ECW', 'EHdr', 'EIR', 'ELAS', 'ENVI', 'ERS', 'FAST', 'GPKG', 'GEORASTER', 'GRIB', 'GMT', 'GRASS', 'GRASSASCIIGrid', 'GSAG', 'GSBG', 'GS7BG', 'GTA', 'GTiff', 'GTX', 'GXF', 'HDF4', 'HDF5', 'HF2', 'HFA', 'IDA', 'ILWIS', 'INGR', 'IRIS', 'ISIS2', 'ISIS3', 'JDEM', 'JPEG', 'JPEG2000', 'JP2ECW', 'JP2KAK', 'JP2MrSID', 'JP2OpenJPEG', 'JPIPKAK', 'KEA', 'KMLSUPEROVERLAY', 'L1B', 'LAN', 'LCP', 'Leveller', 'LOSLAS', 'MBTiles', 'MAP', 'MEM', 'MFF', 'MFF2 (HKV)', 'MG4Lidar', 'MrSID', 'MSG', 'MSGN', 'NDF', 'NGSGEOID', 'NITF', 'netCDF', 'NTv2', 'NWT_GRC', 'NWT_GRD', 'OGDI', 'OZI', 'PCIDSK', 'PCRaster', 'PDF', 'PDS', 'PLMosaic', 'PostGISRaster', 'Rasterlite', 'RIK', 'RMF', 'ROI_PAC', 'RPFTOC', 'RS2', 'RST', 'SAGA', 'SAR_CEOS', 'SDE', 'SDTS', 'SGI', 'SNODAS', 'SRP', 'SRTMHGT', 'USGSDEM', 'VICAR', 'VRT', 'WCS', 'WMS', 'XYZ', 'ZMap',]