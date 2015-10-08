#! /usr/bin/env python
"""
Setup script.

"""
import sys
from   setuptools                 import setup, find_packages, Extension
from   setuptools.command.install import install

import numpy as np

# Version warning
if sys.version_info >= (3,):
    print("Please note that this software was only tested with Python 2.7.")

# Determine whether Cython is available
try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Cython is not available.")
    use_cython = False
else:
    use_cython = True

# Build information
if use_cython:
    ext_modules = [Extension('pycog.euler', ['pycog/euler.pyx'],
                             extra_compile_args=['-Wno-unused-function'],
                             include_dirs=[np.get_include()])]
    cmdclass    = {'build_ext': build_ext}
else:
    ext_modules = [Extension('pycog.euler', ['pycog/euler.c'])]
    cmdclass    = {}

# Setup
setup(
    name='pycog',
    version='0.1',
    license='MIT',
    author='H. Francis Song, G. Robert Yang',
    author_email='song.francis@gmail',
    url='https://github.com/frsong/pycog',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    packages=find_packages(exclude=['examples', 'examples.*', 'paper']),
    setup_requires=['numpy'],
    install_requires=['theano'],
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
        ]
    )
