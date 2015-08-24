#! /usr/bin/env python
"""
Setup script.

"""
import sys

from distutils.core      import setup
from distutils.extension import Extension

import numpy as np

# Determine whether Cython is available
try:
    from Cython.Distutils import build_ext
except ImportError:
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
    cmdclass=cmdclass, 
    ext_modules=ext_modules,
    install_requires=['numpy', 'theano'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
        ]
    )
