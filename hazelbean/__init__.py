from __future__ import division, absolute_import, print_function
import logging

IMPORT_LOGGER = logging.getLogger('import_logger')
IMPORT_LOGGER.setLevel(logging.CRITICAL)

use_strict_importing = False
use_strict_importing_for_ui = False


# Must import globals first
import hazelbean.globals
from hazelbean.globals import *

# Must import config next as it contails global logging.
import hazelbean.config
from hazelbean.config import *

import hazelbean.arrayframe
from hazelbean.arrayframe import *

import hazelbean.arrayframe_functions
from hazelbean.arrayframe_functions import *

import hazelbean.cat_ears
from hazelbean.cat_ears import *

import hazelbean.file_io
from hazelbean.file_io import *

import hazelbean.geoprocessing_extension
from hazelbean.geoprocessing_extension import *

import hazelbean.os_utils
from hazelbean.os_utils import *

import hazelbean.project_flow
from hazelbean.project_flow import *

import hazelbean.pyramids
from hazelbean.pyramids import *

import hazelbean.spatial_projection
from hazelbean.spatial_projection import *

import hazelbean.spatial_utils
from hazelbean.spatial_utils import *

import hazelbean.stats
from hazelbean.stats import *

import hazelbean.utils
from hazelbean.utils import *




import hazelbean.raster_vector_interface
from hazelbean.raster_vector_interface import *

# Cython files imported here in TRY statement for users who cant compile
if use_strict_importing:
    import hazelbean.calculation_core
    from hazelbean.calculation_core import *

    sys.path.insert(0, '../../')
    sys.path.insert(0, '../hazelbean')
    sys.path.insert(0, '../hazelbean/calculation_core')

    import hazelbean.calculation_core.cython_functions
    from hazelbean.calculation_core.cython_functions import *

    import hazelbean.calculation_core.aspect_ratio_array_functions
    from hazelbean.calculation_core.aspect_ratio_array_functions import *

    import hazelbean.watershed_processing
    from hazelbean.watershed_processing import *

    import hazelbean.visualization
    from hazelbean.visualization import *

else:
    try:
        import hazelbean.calculation_core.cython_functions
        from hazelbean.calculation_core.cython_functions import *

        import hazelbean.calculation_core.aspect_ratio_array_functions
        from hazelbean.calculation_core.aspect_ratio_array_functions import *

        import hazelbean.watershed_processing
        from hazelbean.watershed_processing import *

        import hazelbean.visualization
        from hazelbean.visualization import *
    except:
        print('Unable to import cython-based functions, but this may not be a problem.')

if use_strict_importing_for_ui:

    import hazelbean.ui
    from hazelbean.ui import *

    import hazelbean.ui.auto_ui
    from hazelbean.ui.auto_ui import *

else:
    try:
        import hazelbean.ui
        from hazelbean.ui import *

        import hazelbean.ui.auto_ui
        from hazelbean.ui.auto_ui import *

        import hazelbean.watershed_processing
        from hazelbean.watershed_processing import *

        import hazelbean.visualization
        from hazelbean.visualization import *
    except:
        pass







