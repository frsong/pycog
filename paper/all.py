#! /usr/bin/env python
"""
Reproduce every figure in the paper from scratch.

Note
----
For some tasks we run many trials to get pretty psychometric curves, and this is
all done in memory.

"""
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
p.add_argument('-s', '--simulate', action='store_true', default=False)
p.add_argument('args', nargs='*')
a = p.parse_args()

simulate = a.simulate
args     = a.args
if not args:
    args = ['structure', 'rdm_varstim', 'rdm_rt', 'mante', 'multisensory',
            'lee', 'lee_areas', 'connectivity', 'performance']

#=========================================================================================
# Shared steps
#=========================================================================================

here         = get_here(__file__)
base         = os.path.abspath(join(here, os.pardir))
examplespath = join(base, 'examples')
modelspath   = join(examplespath, 'models')
analysispath = join(examplespath, 'analysis')
paperpath    = join(base, 'paper')
timespath    = join(paperpath, 'times')

# Make time path
mkdir_p(timespath)

def call(s):
    if simulate:
        print(3*' ' + s)
    else:
        subprocess.call(s, shell=True)

def clean(model):
    call("{} {} clean".format(join(examplespath, 'do.py'), join(modelspath, model)))

def train(model, seed=None):
    if seed is None:
        seed = ''
    else:
        seed = ' -s {}'.format(seed)

    tstart = datetime.datetime.now()
    call("{} {} train{}"
         .format(join(examplespath, 'do.py'), join(modelspath, model), seed))
    tend = datetime.datetime.now()

    # Save training time
    totalmins = int((tend - tstart).total_seconds()/60)
    timefile = join(timespath, model + '_time.txt')
    np.savetxt(timefile, [totalmins], fmt='%d')

def trials(model, ntrials, analysis=None, args=''):
    if analysis is None:
        analysis = model

    rv = call("{} {} run {} trials {} {}".format(join(examplespath, 'do.py'),
                                                 join(modelspath, model),
                                                 join(analysispath, analysis),
                                                 ntrials, args))

def do_action(model, action, analysis=None):
    if analysis is None:
        analysis = model

    call("{} {} run {} {}".format(join(examplespath, 'do.py'), join(modelspath, model),
                                  join(analysispath, analysis), action))

def figure(fig):
    call(join(paperpath, fig + '.py'))

#=========================================================================================

if 'rdm' in args:
    print("=> RDM")
    models = ['rdm_varstim', 'rdm_rt']
    for m in models:
        clean(m)
        train(m)
        trials(m, 2000, 'rdm')
        if m == 'rdm_varstim':
            do_action(m, 'sort_stim_onset', 'rdm')
        elif m == 'rdm_rt':
            do_action(m, 'sort_response', 'rdm')
    figure('fig_rdm')

if 'structure' in args:
    print("=> Structure")
    seeds = {'rdm_nodale': 100, 'rdm_dense': 100, 'rdm_fixed': 100}
    for m, seed in seeds.items():
        #if m != 'rdm_fixed': continue
        clean(m)
        train(m, seed=seed)
        trials(m, 3000, 'rdm')
        do_action(m, 'selectivity', 'rdm')
    figure('fig_structure')

if 'mante' in args:
    print("=> Context-dependent integration task")
    clean('mante')
    train('mante')
    trials('mante', 100, args='--dt_save 10')
    do_action('mante', 'sort')
    do_action('mante', 'regress')
    figure('fig_mante')

if 'multisensory' in args:
    print("=> Multisensory integration task")
    clean('multisensory')
    train('multisensory')
    trials('multisensory', 500, args='--dt_save 10')
    do_action('multisensory', 'sort')
    figure('fig_multisensory')

if 'romo' in args:
    print("=> Parametric working memory task")
    clean('romo')
    train('romo', seed=100)
    trials('romo', 400, args='--dt_save 10')
    do_action('romo', 'sort')
    figure('fig_romo')

if 'lee' in args:
    print("=> Sequence generation task")
    clean('lee')
    train('lee')
    trials('lee', 100)
    figure('fig_lee')

if 'lee_areas' in args:
    print("=> Sequence generation task (with areas)")
    clean('lee_areas')
    train('lee_areas')
    trials('lee_areas', 100, 'lee')
    figure('fig_lee_areas')

if 'connectivity' in args:
    print("=> Connectivity")
    figure('fig_connectivity')

if 'performance' in args:
    print("=> Performance")
    figure('fig_performance')
