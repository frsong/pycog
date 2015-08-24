#! /usr/bin/env python
from __future__ import division

import cPickle as pickle
import os

import numpy as np

from pycog             import RNN
from pycog.figtools    import gradient, mpl, Figure
from examples.analysis import rdm

import paper

#=========================================================================================
# Paths
#=========================================================================================

here     = os.path.dirname(os.path.realpath(__file__))
figspath = here + '/figs'

trialsfile  = paper.scratchpath + '/rdm_dense/trials/rdm_dense_trials.pkl'

#=========================================================================================

# Load trial
with open(trialsfile) as f:
    trials = pickle.load(f)

for trial in trials:
    if trial['info']['coh'] == 16:
        break
t = trial['t']

colors = ['orange', 'purple']

# Inputs
for i, clr in enumerate(colors):
    fig  = Figure(w=2, h=1)
    plot = fig.add(None, 'none')

    plot.plot(t, trial['u'][i], color=Figure.colors(clr), lw=1)
    plot.ylim(0, 1.5)

    fig.save(path=figspath, name='fig1a_input{}'.format(i+1))
    fig.close()

# Outputs
for i, clr in enumerate(colors):
    fig  = Figure(w=2, h=1)
    plot = fig.add(None, 'none')

    plot.plot(t, trial['z'][i], color=Figure.colors(clr), lw=1)
    plot.ylim(0, 1.5)

    fig.save(path=figspath, name='fig1a_output{}'.format(i+1))
    fig.close()
