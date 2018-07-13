
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

packages=find_packages()
include_package_data=True

setup(
  name = 'hazelbean',
  packages = packages,
  version = '0.6.5',
  description = 'Geospatial research tools',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Justin Andrew Johnson',
  url = 'https://github.com/jandrewjohnson/hazelbean',
  download_url = 'https://github.com/jandrewjohnson/hazelbean',
  keywords = ['geospatial', 'raster', 'shapefile', 'sustainability science'],
  classifiers = ["Programming Language :: Python :: 3"],
)
