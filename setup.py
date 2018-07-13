
from setuptools import setup, find_packages

packages=find_packages()
include_package_data=True

setup(
  name = 'hazelbean',
  packages = packages,
  version = '0.6.2',
  description = 'Geospatial research tools',
  author = 'Justin Andrew Johnson',
  url = 'https://github.com/jandrewjohnson/hazelbean',
  # download_url = 'https://github.com/jandrewjohnson/hazelbean/releases/hazelbean_x64_py3.6.3/dist/hazelbean-0.3.0_x64_py3.6.3.tar.gz',
  keywords = ['geospatial', 'raster', 'shapefile', 'sustainability science'],
  classifiers = [],
)
