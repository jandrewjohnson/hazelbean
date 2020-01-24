
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os

_REQUIREMENTS = [
    'anytree',
    'geopandas',
    'pygeoprocessing',
    'natcap.invest',
    'rtree',
    'scipy',
    'fiona',
    'netCDF4',
    'statsmodels',
    'sklearn',
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()

packages=find_packages()
include_package_data=True

setup(
  name = 'hazelbean',
  packages = packages,
  version = '1.0.2',
  description = 'Geospatial research tools',
  long_description=long_description,
  author = 'Justin Andrew Johnson',
  url = 'https://github.com/jandrewjohnson/hazelbean',
  download_url = 'https://github.com/jandrewjohnson/hazelbean',
  keywords = ['geospatial', 'raster', 'shapefile', 'sustainability science'],
  classifiers = ["Programming Language :: Python :: 3"],
  install_requires=_REQUIREMENTS,  
  #cmdclass={'build_ext': build_ext},
  #ext_modules=[Extension("cython_functions", ["hazelbean/calculation_core/cython_functions.c"]),
  #             Extension("aspect_ratio_array_functions", ["hazelbean/calculation_core/aspect_ratio_array_functions.c"]),
  #             ]  
  
  ext_modules=cythonize(
    [Extension(
        "hazelbean.calculation_core.cython_functions",
        sources=["hazelbean/calculation_core/cython_functions.pyx"],
        include_dirs=[
            numpy.get_include(),
            'hazelbean/calculation_core/cython_functions'],
        language="c++",
    ),
     Extension(
         "hazelbean.calculation_core.aspect_ratio_array_functions",
         sources=[
             "hazelbean/calculation_core/aspect_ratio_array_functions.pyx"],
         include_dirs=[numpy.get_include()],
         language="c++")],
    )
  
)
