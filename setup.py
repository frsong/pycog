#! /usr/bin/env python
"""
Setup script.

"""
import sys
from   setuptools                 import setup, find_packages, Extension
from   setuptools.command.install import install

import numpy as np

# Warn
#if sys.version_info >= (3,):
#    print("Please note that the software was only tested with Python 2.7.")

# Determine whether Cython is available
try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Cython is not available.")
    use_cython = False
else:
    use_cython = True

# Installation class
class pycog_install(install):
    """
    Customized install class to allow 'develop' install only.

    """
    def run(self):
        mode = None
        while mode not in ['', 'develop', 'cancel']:
            if mode != 'develop':
                print("This script is for 'develop' install only.")
            mode = raw_input("Installation mode [develop]/cancel: ").strip()
        if mode in ['', 'develop']:
            self.distribution.run_command('develop')

# Build information
if use_cython:
    ext_modules = [Extension('pycog.euler', ['pycog/euler.pyx'],
                             extra_compile_args=['-Wno-unused-function'],
                             include_dirs=[np.get_include()])]
    cmdclass    = {'build_ext': build_ext}
else:
    ext_modules = [Extension('pycog.euler', ['pycog/euler.c'])]
    cmdclass    = {}
cmdclass['install'] = pycog_install

# Setup
setup(
    name='pycog', 
    version='0.1',
    license='MIT',
    author='H. Francis Song',
    author_email='song.francis@gmail',
    url='https://github.com/frsong/pycog',
    cmdclass=cmdclass, 
    ext_modules=ext_modules,
    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=['theano'],
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
        ]
    )
