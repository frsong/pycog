#! /usr/bin/env python
"""
Reproduce every figure in the paper from scratch.

"""
import argparse
import os
import subprocess
import sys
from   os.path import join

from pycog.utils import get_here

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
    args = ['structure', 'rdm_varstim', 'rdm_rt', 'mante', 'multisensory', 'lee',
            'performance']

#=========================================================================================
# Shared steps
#=========================================================================================

here         = get_here(__file__)
base         = os.path.abspath(join(here, os.pardir))
examplespath = join(base, 'examples')
modelspath   = join(examplespath, 'models')
analysispath = join(examplespath, 'analysis')
paperpath    = join(base, 'paper')

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

    call("{} {} train{}"
         .format(join(examplespath, 'do.py'), join(modelspath, model), seed))

def trials(model, analysis=None, ntrials=None):
    if analysis is None:
        analysis = model

    if ntrials is None:
        ntrials = ''
    else:
        ntrials = ' {}'.format(ntrials)

    call("{} {} run {} trials{}".format(join(examplespath, 'do.py'),
                                        join(modelspath, model),
                                        join(analysispath, analysis),
                                        ntrials))

def do_action(model, action, analysis=None):
    if analysis is None:
        analysis = model

    call("{} {} run {} {}".format(join(examplespath, 'do.py'), join(modelspath, model),
                                  join(analysispath, analysis), action))

def figure(fig):
    call(join(paperpath, fig + '.py'))

#=========================================================================================

if 'structure' in args:
    print("=> Fig. 1")
    seeds   = {'rdm_nodale': 100, 'rdm_dense': 100, 'rdm_fixed': 99}
    ntrials = 22000
    for m, seed in seeds.items():
        clean(m)
        train(m, seed=seed)
        trials(m, 'rdm', ntrials=ntrials)
        do_action(m, 'selectivity', 'rdm')
    figure('fig_structure')

if 'rdm' in args:
    print("=> RDM")
    models  = ['rdm_rt']
    ntrials = 22000
    for m in models:
        clean(m)
        train(m)
        trials(m, 'rdm', ntrials=ntrials)
        if m == 'rdm_varstim':
            do_action(m, 'sort_stim_onset', 'rdm')
        elif m == 'rdm_rt':
            do_action(m, 'sort_response', 'rdm')
    figure('fig_rdm')

if 'multisensory' in args:
    print("=> Multisensory integration task")
    clean('multisensory')
    train('multisensory')
    trials('multisensory', ntrials=24000)
    do_action('multisensory', 'sort')
    figure('fig_multisensory')

if 'mante' in args:
    print("=> Context-dependent integration task")
    clean('mante')
    train('mante')
    trials('mante')
    do_action('mante', 'sort')
    do_action('mante', 'regress')
    figure('fig_mante')

if 'lee' in args:
    print("=> Sequence generation task")
    clean('lee')
    train('lee')
    trials('lee')
    do_action('lee', 'sort')
    figure('fig_lee')
