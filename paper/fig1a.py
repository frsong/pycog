#! /usr/bin/env python
from __future__ import division

import cPickle as pickle
import imp
from os.path import join

import numpy as np

from pycog             import RNN
from pycog.figtools    import gradient, mpl, Figure
from pycog.utils       import get_here, get_parent
from examples.analysis import rdm

import paper

#=========================================================================================
# Paths
#=========================================================================================

here     = get_here(__file__)
base     = get_parent(here)
figspath = join(here, 'figs')

modelfile = join(base, 'examples', 'models', 'rdm_dense.py')
savefile  = join(base, 'examples', 'work', 'data', 'rdm_dense', 'rdm_dense.pkl')

#=========================================================================================

m = imp.load_source('model', modelfile)

rnn = RNN(savefile, {'dt': 0.5}, verbose=False)

trial_func = m.generate_trial
trial_args = {
    'name':   'test',
    'catch':  False,
    'coh':    16,
    'in_out': 1
    }
info = rnn.run(inputs=(trial_func, trial_args), seed=10)

colors = ['orange', 'purple']

DT = 15

# Inputs
for i, clr in enumerate(colors):
    fig  = Figure(w=2, h=1)
    plot = fig.add(None, 'none')

    plot.plot(rnn.t[::DT], rnn.u[i][::DT], color=Figure.colors(clr), lw=1)
    plot.ylim(0, 1.5)

    fig.save(path=figspath, name='fig1a_input{}'.format(i+1))
    fig.close()

# Outputs
for i, clr in enumerate(colors):
    fig  = Figure(w=2, h=1)
    plot = fig.add(None, 'none')

    plot.plot(rnn.t[::DT], rnn.z[i][::DT], color=Figure.colors(clr), lw=1)
    plot.ylim(0, 1.5)

    fig.save(path=figspath, name='fig1a_output{}'.format(i+1))
    fig.close()
