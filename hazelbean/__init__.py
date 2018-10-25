from __future__ import division, absolute_import, print_function
import logging

IMPORT_LOGGER = logging.getLogger('import_logger')
IMPORT_LOGGER.setLevel(logging.CRITICAL)

use_strict_importing = True

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

import hazelbean.geoprocessing_extension
from hazelbean.geoprocessing_extension import *

import hazelbean.watershed_processing
from hazelbean.watershed_processing import *

if use_strict_importing:
    import hazelbean.calculation_core
    from hazelbean.calculation_core import *

    import hazelbean.calculation_core.cython_functions
    from hazelbean.calculation_core.cython_functions import *

    import hazelbean.calculation_core.aspect_ratio_array_functions
    from hazelbean.calculation_core.aspect_ratio_array_functions import *
else:
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

import hazelbean.arrayframe_functions
from hazelbean.arrayframe_functions import *

import hazelbean.visualization
from hazelbean.visualization import *







