from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from pathlib import Path
from distutils.extension import Extension
import os
import warnings

with open("requirements.txt") as requires_file:
        requirements = requires_file.read().split("\n")


scripts = [
           'make_glade=skyfast.pipelines.create_glade:main',
         
           ]
pymodules = [
             'skyfast/pipelines/create_glade',
             ]




setup(
    name = 'skyfast',
    description = 'skyfast: rapid volume and inclination angle reconstruction',
    author = 'Gabriele Demasi, Giulia Capurri, Walter Del Pozzo',
    url = 'https://github.com/gabrieledemasi/skyfast/tree/main',
    python_requires = '>=3.9',
    packages = ['skyfast'],
    py_modules = pymodules,
    install_requires = requirements,
    entry_points = {
        'console_scripts': scripts,
        },
    version='0.1',
    
    
    )


