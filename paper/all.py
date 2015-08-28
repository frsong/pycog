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
a = p.parse_args()

args = a.args
if not args:
    args = ['structure', 'rdm_varstim', 'rdm_rt', 'mante', 'multisensory', 'lee']

#=========================================================================================
# Shared steps
#=========================================================================================

here = os.path.dirname(os.path.realpath(__file__))
base = os.path.abspath(os.path.join(here, os.pardir))
examples_dir = base + '/examples'
models_dir   = examples_dir + '/models'
analysis_dir = examples_dir + '/analysis'
paper_dir    = base + '/paper'

def call(s):
    print(3*' ' + s)
    #subprocess.call(x, shell=True)

def clean(model):
    call("{}/do.py {}/{} clean".format(examples_dir, models_dir, model))

def train(model):
    call("{}/do.py {}/{} train".format(examples_dir, models_dir, model))

def trials(model, analysis=None, ntrials=None):
    if analysis is None:
        analysis = model

    if ntrials is None:
        ntrials = ''
    else:
        ntrials = ' {}'.format(ntrials)
        
    call("{}/do.py {}/{} run {}/{} trials{}"
         .format(examples_dir, models_dir, model, analysis_dir, analysis, ntrials))

def sort(model, analysis=None):
    if analysis is None:
        analysis = model

    call("{}/do.py {}/{} run {}/{} sort"
         .format(examples_dir, models_dir, model, analysis_dir, analysis))

def figure(fig):
    call("{}/{}.py".format(paper_dir, fig))

#=========================================================================================

if 'structure' in args:
    print("=> Fig. 1")
    clean('rdm_dense')
    train('rdm_dense')
    trials('rdm_dense', 'rdm')
    figure('fig_structure')

if 'rdm_varstim' in args:
    pass

if 'multisensory' in args:
    print("=> Multisensory integration task")
    clean('multisensory')
    train('multisensory')
    sort('multisensory')
    figure('fig_multisensory')

if 'mante' in args:
    print("=> Context-dependent integration task")
    clean('mante')
    train('mante')
    sort('mante')
    call("{}/do.py {}/mante run {}/mante regress"
         .format(examples_dir, models_dir, analysis_dir))
    figure('fig_mante')

if 'lee' in args:
    print("=> Lee sequence generation task")
    clean('lee')
    train('lee')
    trials('lee')
    sort('lee')
    figure('fig_lee')
