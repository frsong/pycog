#! /usr/bin/env python
"""
Reproduce every figure in the paper from scratch.

Note
----

Everything means EVERYTHING, i.e., train, run trials, analyze, and make figures.
This is mostly for us to check our work and for you to see exactly what we did; 
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
# Shared steps
#=========================================================================================

def call(x):
    subprocess.call(x, shell=True)

def clean(model):
    call("../examples/do.py models/{} clean".format(model))

def train(model):
    call("../examples/do.py models/{} train".format(model))

def trials(model, analysis=None, ntrials=None):
    if analysis is None:
        analysis = model

    if ntrials is None:
        ntrials = ''
    else:
        ntrials = ' {}'.format(ntrials)
        
    call("../examples/do.py models/{} run analysis/{} trials{}"
         .format(model, analysis, ntrials))

def sort(model, analysis=None):
    if analysis is None:
        analysis = model

    call("../examples/do.py models/{} run analysis/{} sort".format(model, analysis))

def figure(figmaker):
    call("../paper/{}.py".format(figmaker))

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
    clean('lee')
    train('lee')
    trials('lee')
    sort('lee')
    figure('fig_lee')
