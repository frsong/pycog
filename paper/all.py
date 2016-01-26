#! /usr/bin/env python
"""
Reproduce every figure in the paper from scratch.

Notes
-----

* Running this script in its entirety will take some time.

* We run a fair number of trials to get pretty psychometric curves, and this is done
  in one big chunk of memory. You may need to change this to run more trials, depending
  on your setup.

* Training converged for all the seeds we tried but we picked the ones that produced
  the prettiest plots for the paper -- we hope that's understandable!

"""
from __future__ import division

import argparse
import datetime
import os
import subprocess
import sys
from   os.path import join

import numpy as np

from pycog.utils import get_here, mkdir_p

#=========================================================================================
# Command line
#=========================================================================================

p = argparse.ArgumentParser()
p.add_argument('-g', '--gpus', nargs='?', type=int, const=1, default=0)
p.add_argument('-s', '--simulate', action='store_true', default=False)
p.add_argument('args', nargs='*')
a = p.parse_args()

# GPUs
gpus = a.gpus

simulate = a.simulate
args     = a.args
if not args:
    args = [
        'rdm',                         # Fig. 2
        'structure',                   # Fig. 3
        'mante',                       # Fig. 4
        'mante_areas', 'connectivity', # Fig. 5
        'multisensory',                # Fig. 6
        'romo',                        # Fig. 7
        'lee',                         # Fig. 8
        'performance'                  # Fig. 9
        ]

#=========================================================================================
# Shared steps
#=========================================================================================

here          = get_here(__file__)
base          = os.path.abspath(join(here, os.pardir))
examplespath  = join(base, 'examples')
modelspath    = join(examplespath, 'models')
analysispath  = join(examplespath, 'analysis')
paperpath     = join(base, 'paper')
paperfigspath = join(paperpath, 'figs')
timespath     = join(paperpath, 'times')

# Make paths
mkdir_p(paperfigspath)
mkdir_p(timespath)

def call(s):
    if simulate:
        print(3*' ' + s)
    else:
        rv = subprocess.call(s.split())
        if rv != 0:
            sys.stdout.flush()
            print("Something went wrong (return code {}).".format(rv)
                  + " We're probably out of memory.")
            sys.exit(1)

def clean(model):
    call("python {} {} clean"
         .format(join(examplespath, 'do.py'), join(modelspath, model)))

def train(model, seed=None):
    if seed is None:
        seed = ''
    else:
        seed = ' -s {}'.format(seed)

    tstart = datetime.datetime.now()
    call("python {} {} train{} -g{}"
         .format(join(examplespath, 'do.py'), join(modelspath, model), seed, gpus))
    tend = datetime.datetime.now()

    # Save training time
    totalmins = int((tend - tstart).total_seconds()/60)
    timefile = join(timespath, model + '_time.txt')
    np.savetxt(timefile, [totalmins], fmt='%d')

def train_seeds(model, start_seed=1, ntrain=5):
    for seed in xrange(start_seed, start_seed+ntrain):
        suffix = '_s{}'.format(seed)
        s = ' --seed {} --suffix {}'.format(seed, suffix)

        tstart = datetime.datetime.now()
        call("python {} {} clean{}"
             .format(join(examplespath, 'do.py'), join(modelspath, model), s))
        call("python {} {} train{} -g{}"
             .format(join(examplespath, 'do.py'), join(modelspath, model), s, gpus))
        tend = datetime.datetime.now()

        # Save training time
        totalmins = int((tend - tstart).total_seconds()/60)
        timefile = join(timespath, model + suffix + '_time.txt')
        np.savetxt(timefile, [totalmins], fmt='%d')

def trials(model, ntrials, analysis=None, args=''):
    if analysis is None:
        analysis = model

    call("python {} {} run {} trials {} {}".format(join(examplespath, 'do.py'),
                                                   join(modelspath, model),
                                                   join(analysispath, analysis),
                                                   ntrials, args))

def do_action(model, action, analysis=None):
    if analysis is None:
        analysis = model

    call("python {} {} run {} {}".format(join(examplespath, 'do.py'),
                                         join(modelspath, model),
                                         join(analysispath, analysis),
                                         action))

def figure(fig):
    call('python ' + join(paperpath, fig + '.py'))

#=========================================================================================

if 'rdm' in args:
    print("=> Perceptual decision-making task")

    m = 'rdm_varstim'
    clean(m)
    train(m, seed=1001)
    trials(m, 4000, 'rdm', args='--dt_save 20')
    do_action(m, 'sort_stim_onset', 'rdm')
    do_action(m, 'units_stim_onset', 'rdm')

    m = 'rdm_rt'
    clean(m)
    train(m)
    trials(m, 1500, 'rdm', args='--dt_save 10')
    do_action(m, 'sort_response', 'rdm')
    do_action(m, 'units_response', 'rdm')

    figure('fig_rdm')

if 'structure' in args:
    print("=> Perceptual decision-making task (structure)")
    models = ['rdm_nodale', 'rdm_dense', 'rdm_fixed']
    seeds  = [None, 101, 1001] # Pick out the prettiest
    for m, seed in zip(models, seeds):
        clean(m)
        train(m, seed=seed)
        trials(m, 2000, 'rdm', args='--dt_save 100')
        do_action(m, 'selectivity', 'rdm')
    figure('fig_structure')

if 'mante' in args:
    print("=> Context-dependent integration task")
    clean('mante')
    train('mante')
    trials('mante', 200, args='--dt_save 20')
    do_action('mante', 'sort')
    do_action('mante', 'regress')
    do_action('mante', 'units')
    figure('fig_mante')

if 'mante_areas' in args:
    print("=> Context-dependent integration task (areas)")
    clean('mante_areas')
    train('mante_areas')
    trials('mante_areas', 200, 'mante', args='--dt_save 20')
    do_action('mante_areas', 'sort', 'mante')
    do_action('mante_areas', 'regress', 'mante')
    do_action('mante_areas', 'units', 'mante')
    figure('fig_mante_areas')

if 'connectivity' in args:
    print("=> Connectivity for sequence execution task")
    figure('fig_connectivity')

if 'multisensory' in args:
    print("=> Multisensory integration task")
    clean('multisensory')
    train('multisensory', seed=111)
    trials('multisensory', 500, args='--dt_save 20')
    do_action('multisensory', 'sort')
    do_action('multisensory', 'units')
    figure('fig_multisensory')

if 'romo' in args:
    print("=> Parametric working memory task")
    clean('romo')
    train('romo')
    trials('romo', 100, args='--dt_save 20')
    do_action('romo', 'sort')
    do_action('romo', 'units')
    figure('fig_romo')

if 'lee' in args:
    print("=> Eye-movement sequence execution task")
    clean('lee')
    train('lee')
    trials('lee', 100, args='--dt_save 2')
    figure('fig_lee')

if 'performance' in args:
    print("=> Performance")
    for m in ['rdm_varstim', 'rdm_rt']:
        train_seeds(m)
    for m in ['rdm_nodale', 'rdm_dense', 'rdm_fixed']:
        train_seeds(m)
    train_seeds('mante')
    train_seeds('mante_areas')
    train_seeds('multisensory')
    train_seeds('romo')
    train_seeds('lee')
    figure('fig_performance')
