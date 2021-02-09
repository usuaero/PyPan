"""PyPan: The greatest thing since PAN AIR"""

from setuptools import setup
import os
import sys

setup(name = 'PyPan',
    version = '1.0.0',
    description = "PyPan: The greatest thing since PAN AIR",
    url = 'https://github.com/usuaero/PyPan',
    author = 'usuaero',
    author_email = 'doug.hunsaker@usu.edu',
    install_requires = ['numpy>=1.18', 'scipy>=1.4', 'pytest', 'matplotlib', 'numpy-stl', 'pyvista'],
    python_requires ='>=3.6.0',
    license = 'MIT',
    packages = ['pypan', 'panair'],
    zip_safe = False)
