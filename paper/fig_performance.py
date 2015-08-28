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
          ('Integration (variable stim.)', savefile('rdm_varstim')),
          ('5', None),
          ('6', None),
          ('Context-dependent int.', None),
          ('Multisensory int.', None),
          ('Lee', None)]
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
    s, _ = models[k]
    plots[s].text_upper_center(s, dy=0.05, fontsize=7)

#=========================================================================================
# Plot performance
#=========================================================================================

for s, savefile in models:
    if savefile is None:
        continue

    rnn = RNN(savefile, verbose=True)

#=========================================================================================

fig.save(path=figspath)
