#! /usr/bin/env python
from __future__ import division

import os
from   os.path import join

import numpy as np

from pycog             import RNN
from pycog.figtools    import Figure
from pycog.utils       import get_here, get_parent
from examples.analysis import rdm

import paper

#=========================================================================================
# Setup
#=========================================================================================

here     = get_here(__file__)
base     = get_parent(here)
figspath = join(here, 'figs')

def savefile(model):
    return join(base, 'examples', 'work', 'data', model, model+'.pkl')

models = [('rdm_nodale', '1C: Integration (no Dale)'),
          ('rdm_dense', '1D: Integration (Dale, dense)'),
          ('rdm_fixed', '1E: Integration (Dale, fixed)'),
          ('rdm_varstim', '2A: Integration (variable stim.)'),
          ('5', None),
          ('mante', 'Context-dependent int.'),
          ('multisensory', 'Multisensory int.'),
          ('romo', 'Parametric working memory'),
          ('lee', 'Lee')]
labels = list('ABCDEFGHI')

#=========================================================================================
# Figure setup
#=========================================================================================

fig = Figure(w=6.1, h=5.4, axislabelsize=7, labelpadx=5, labelpady=5, thickness=0.6, 
             ticksize=3, ticklabelsize=6, ticklabelpad=2, format=paper.format)

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
plot.xlabel(r'Number of trials ($\times 10^4$)')
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
    ntrials = np.asarray(ntrials, dtype=int)/int(1e4)
    performance = [costs[1][-1] for costs in rnn.costs_history]

    plot = plots[model]

    if model in ['rdm_nodale', 'rdm_dense', 'rdm_fixed']:
        plot.plot(ntrials, performance, color=Figure.colors('red'), lw=1)
        plot.xlim(ntrials[0], ntrials[-1])
        plot.ylim(40, 100)
        plot.hline(80, color='0.5', lw=0.75)

    if model == 'rdm_varstim':
        plot.plot(ntrials, performance, color=Figure.colors('red'), lw=1)
        plot.xlim(ntrials[0], ntrials[-1])
        plot.ylim(40, 100)
        plot.hline(85, color='0.5', lw=0.75)

#=========================================================================================

fig.save(path=figspath)
