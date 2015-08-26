#! /usr/bin/env python
"""
Reproduce every figure in the paper from scratch.

Note
----

Everything means EVERYTHING, i.e., train, run trials, analyze, and make figures.
This is mostly for us to check our work and for you to see what we did; 
we don't recommend that you actually run this script.

"""
import argparse
import os
import subprocess
import sys

#=========================================================================================
# Command line
#=========================================================================================

p = argparse.ArgumentParser()
p.add_argument('args', nargs='*')
p.add_argument('-f', '--format', type=str, default='pdf')
a = p.parse_args()

fmt  = a.format
args = a.args

#=========================================================================================

if 'structure' in args:
    pass

if 'rdm_varstim' in args:
    pass

if 'multisensory' in args:
    pass

if 'mante' in args:
    pass

if 'lee' in args:
    pass
