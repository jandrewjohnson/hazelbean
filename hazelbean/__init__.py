from __future__ import division, absolute_import, print_function
import logging

IMPORT_LOGGER = logging.getLogger('import_logger')
IMPORT_LOGGER.setLevel(logging.CRITICAL)


# To increase robustness of others using my code (which may not always be thought out well for redistribution), I protect everything in try/except phrases. However, this will suppress useful logging information, so if imports are failing, re-enable standard importing with use_strict_importing=True

use_strict_importing = True

if not use_strict_importing:
    try:
        from hazelbean import config
        from hazelbean.config import *
    except:
        IMPORT_LOGGER.debug('Unable to import config during Hazelbean init.')

    try:
        from hazelbean import utils
        from hazelbean.utils import *
    except:
        IMPORT_LOGGER.debug('Unable to import utils during Hazelbean init.')

    try:
        import hazelbean.file_io
        from hazelbean.file_io import *
    except:
        IMPORT_LOGGER.debug('Unable to import file_io during Hazelbean init.')

    try:
        import hazelbean.project_flow
        from hazelbean.project_flow import *
    except:
        IMPORT_LOGGER.debug('Unable to import project_flow during Hazelbean init.')

    try:
        import hazelbean.os_utils
        from hazelbean.os_utils import *
    except:
        IMPORT_LOGGER.debug('Unable to import os_utils during Hazelbean init.')

    try:
        import hazelbean.cat_ears
        from hazelbean.cat_ears import *
    except:
        IMPORT_LOGGER.debug('Unable to import cat_ears during Hazelbean init.')

    try:
        import hazelbean.spatial_projection
        from hazelbean.spatial_projection import *
    except:
        IMPORT_LOGGER.debug('Unable to import spatial_projection during Hazelbean init.')

    try:
        import hazelbean.spatial_utils
        from hazelbean.spatial_utils import *
    except:
        IMPORT_LOGGER.debug('Unable to import spatial_utils during Hazelbean init.')

    try:
        import hazelbean.stats
        from hazelbean.stats import *
    except:
        IMPORT_LOGGER.debug('Unable to import stats during Hazelbean init.')

    try:
        import hazelbean.geoprocessing
        from hazelbean.geoprocessing import *
    except:
        IMPORT_LOGGER.debug('Unable to import geoprocessing during Hazelbean init.')

    try:
        import hazelbean.watershed_processing
        from hazelbean.watershed_processing import *
    except:
        IMPORT_LOGGER.debug('Unable to import watershed_processing during Hazelbean init.')

    try:
        import hazelbean.calculation_core
        from hazelbean.calculation_core import *
    except:
        IMPORT_LOGGER.debug('Unable to import calculation_core during Hazelbean init.')

    try:
        import hazelbean.calculation_core.cython_functions
        from hazelbean.calculation_core.cython_functions import *
    except:
        IMPORT_LOGGER.debug('Unable to import cython_functions during Hazelbean init.')

    try:
        import hazelbean.calculation_core.aspect_ratio_array_functions
        from hazelbean.calculation_core.aspect_ratio_array_functions import *
    except:
        IMPORT_LOGGER.debug('Unable to import aspect_ratio_array_functions during Hazelbean init.')

    try:
        import hazelbean.ui
        from hazelbean.ui import *
    except:
        IMPORT_LOGGER.debug('Unable to import ui during Hazelbean init.')

    try:
        import hazelbean.ui.auto_ui
        from hazelbean.ui.auto_ui import *
    except:
        IMPORT_LOGGER.debug('Unable to import auto_ui during Hazelbean init.')

    try:
        import hazelbean.arrayframe
        from hazelbean.arrayframe import *
    except:
        IMPORT_LOGGER.debug('Unable to import arrayframe during Hazelbean init.')

    try:
        import hazelbean.arrayframe_numpy_functions
        from hazelbean.arrayframe_numpy_functions import *
    except:
        IMPORT_LOGGER.debug('Unable to import arrayframe_numpy_functions during Hazelbean init.')

    try:
        import hazelbean.visualization
        from hazelbean.visualization import *
    except:
        IMPORT_LOGGER.debug('Unable to import arrayframe_numpy_functions during Hazelbean init.')

else:
    from hazelbean import config
    from hazelbean.config import *

    from hazelbean import utils
    from hazelbean.utils import *

    import hazelbean.file_io
    from hazelbean.file_io import *

    import hazelbean.project_flow
    from hazelbean.project_flow import *

    import hazelbean.os_utils
    from hazelbean.os_utils import *

    import hazelbean.cat_ears
    from hazelbean.cat_ears import *

    import hazelbean.spatial_projection
    from hazelbean.spatial_projection import *

    import hazelbean.spatial_utils
    from hazelbean.spatial_utils import *

    import hazelbean.stats
    from hazelbean.stats import *

    import hazelbean.geoprocessing
    from hazelbean.geoprocessing import *

    import hazelbean.watershed_processing
    from hazelbean.watershed_processing import *

    try:
        import hazelbean.calculation_core
        from hazelbean.calculation_core import *

        import hazelbean.calculation_core.cython_functions
        from hazelbean.calculation_core.cython_functions import *

        import hazelbean.calculation_core.aspect_ratio_array_functions
        from hazelbean.calculation_core.aspect_ratio_array_functions import *
    except:
        print('Unable to import cython-based functions, but this may not be a problem.')

    import hazelbean.ui
    from hazelbean.ui import *

    import hazelbean.ui.auto_ui
    from hazelbean.ui.auto_ui import *

    import hazelbean.arrayframe
    from hazelbean.arrayframe import *

    import hazelbean.arrayframe_numpy_functions
    from hazelbean.arrayframe_numpy_functions import *

    import hazelbean.visualization
    from hazelbean.visualization import *







