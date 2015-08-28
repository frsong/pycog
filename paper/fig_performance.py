#! /usr/bin/env python
from __future__ import division

import argparse
import os

import numpy as np

from pycog             import RNN
from pycog.figtools    import Figure
from examples.analysis import rdm

import paper

#=========================================================================================
# Image format
#=========================================================================================

p = argparse.ArgumentParser()
p.add_argument('-f', '--format', default='pdf')
a = p.parse_args()

Figure.defaults['format'] = p.parse_args().format

#=========================================================================================
# Setup
#=========================================================================================

here     = os.path.dirname(os.path.realpath(__file__))
base     = os.path.abspath(os.path.join(here, os.pardir))
figspath = here + '/figs'

def savefile(model):
    return base + '/examples/work/data/{0}/{0}.pkl'.format(model)

models = [('1', None),
          ('2', None),
          ('3', None),
          ('rdm_varstim', 'Integration (variable stim.)'),
          ('5', None),
          ('mante', 'Context-dependent int.'),
          ('multisensory', 'Multisensory int.'),
          ('romo', 'Parametric working memory'),
          ('lee', 'Lee')]
labels = list('ABCDEFGHI')

#=========================================================================================
# Figure setup
#=========================================================================================

fig = Figure(w=6.1, h=5.4, axislabelsize=7, labelpadx=5, labelpady=5,
             thickness=0.6, ticksize=3, ticklabelsize=6, ticklabelpad=2)

ncols = 3
nrows = 3

w    = 0.245
dx   = w + 0.085
x0   = 0.075
xall = x0 + dx*np.arange(ncols)

h    = 0.22
y0   = 0.72
dy   = h + 0.09
yall = y0 - dy*np.arange(nrows)

plots      = {}
plotlabels = {}
for k in xrange(len(models)):
    i = k//nrows
    j = k%ncols
    if i == 0:
        pady = 0
    else:
        pady = 0.02
    plots[models[k][0]]   = fig.add([xall[j], yall[i]-pady, w, h])
    plotlabels[labels[k]] = (xall[j]-0.06, yall[i]-pady+h+0.01)
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Labels
#=========================================================================================

plot = plots[models[0][0]]
plot.xlabel('Number of trials')
plot.ylabel('Percent correct')

plot = plots[models[-1][0]]
plot.ylabel('Error in eye position')

for k in xrange(len(models)):
    model, desc = models[k]
    if desc is None:
        desc = model
    plots[model].text_upper_center(desc, dy=0.05, fontsize=7)

#=========================================================================================
# Plot performance
#=========================================================================================

for model, _ in models:
    try:
        rnn = RNN(savefile(model), verbose=True)
    except:
        continue

    ntrials = [int(costs[0]) for costs in rnn.costs_history]
    ntrials = np.asarray(ntrials, dtype=int)//int(ntrials[1]-ntrials[0])

    performance = [costs[1][-1] for costs in rnn.costs_history]

    plot = plots[model]
    if model == 'rdm_varstim':
        print(ntrials)
        exit()

        #plot.plot(ntrials, performance, color=Figure.colors('red'), lw=1)
        plot.plot(ntrials[::2], performance[::2], 'o', 
                  mfc=Figure.colors('red'), mec='none', ms=4)

        plot.ylim(0, 100)

#=========================================================================================

fig.save(path=figspath)
