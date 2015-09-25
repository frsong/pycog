#! /usr/bin/env python
from __future__ import division

import imp
import os
from   glob    import glob
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

here       = get_here(__file__)
base       = get_parent(here)
modelspath = join(base, 'examples', 'models')
paperpath  = join(base, 'paper')
figspath   = join(here, 'figs')
timespath  = join(paperpath, 'times')

def get_savefile(model):
    return join(base, 'examples', 'work', 'data', model, model+'.pkl')

models = [('rdm_varstim',  '2A: Decision-making (variable stim.)'),
          ('rdm_rt',       '2B: Decision-making (reaction-time)'),
          ('rdm_nodale',   '3A: Decision-making (no Dale)'),
          ('rdm_dense',    '3B: Decision-making (Dale, dense)'),
          ('rdm_fixed',    '3C: Decision-making (Dale, fixed)'),
          ('mante',        '4: Context-dependent int.'),
          ('multisensory', '5: Multisensory int.'),
          ('romo',         '6: Parametric working memory'),
          ('lee',          '7: Lee')]
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
    plotlabels[labels[k]] = (xall[j]-0.062, yall[i]-pady+h+0.01)
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Labels
#=========================================================================================

plot = plots[models[0][0]]
plot.xlabel(r'Number of trials ($\times 10^4$)')
plot.ylabel('Percent correct')

plot = plots['romo']
plot.ylabel('Min. percent correct')

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

clr_target = Figure.colors('red')
clr_actual = '0.2'
clr_seeds  = '0.8'

for model, _ in models:
    plot = plots[model]

    rnn  = RNN(get_savefile(model), verbose=True)
    xall = []

    ntrials     = [int(costs[0]) for costs in rnn.costs_history]
    ntrials     = np.asarray(ntrials, dtype=int)/int(1e4)
    performance = [costs[1][-1] for costs in rnn.costs_history]

    # Get target performance
    modelfile = join(modelspath, model + '.py')
    try:
        m = imp.load_source('model', modelfile)
    except IOError:
        print("Couldn't load model module from {}".format(modelfile))
        sys.exit()
    if 'lee' in model:
        target = m.min_error
    else:
        target = m.TARGET_PERFORMANCE

    plot.plot(ntrials, performance, color=clr_actual, lw=1, zorder=10)
    xall.append(ntrials)

    # y-axis
    if model == 'lee':
        plot.yscale('log')
    else:
        if model == 'romo':
            plot.ylim(0, 100)
        else:
            plot.ylim(40, 100)

    # Number of units
    nunits = '{} units'.format(rnn.p['N'])

    # Training time
    timefile = join(paperpath, 'times', model + '_time.txt')
    if os.path.isfile(timefile):
        time = '{} mins'.format(int(np.loadtxt(timefile)))
    else:
        time = 'X mins'

    # Info
    plot.text_lower_right(nunits, dy=0.13, fontsize=7, color=Figure.colors('green'),
                          zorder=20)
    plot.text_lower_right(time,   dy=0.02, fontsize=7, color=Figure.colors('strongblue'),
                          zorder=20)

    # Other seeds
    gstring = join(base, 'examples', 'work', 'data', model, model+'_s*.pkl')
    savefiles = glob(gstring)
    for savefile in savefiles:
        if 'init' in savefile or 'copy' in savefile:
            continue

        rnnx        = RNN(savefile, verbose=True)
        ntrials     = [int(costs[0]) for costs in rnnx.costs_history]
        ntrials     = np.asarray(ntrials, dtype=int)/int(1e4)
        performance = [costs[1][-1] for costs in rnnx.costs_history]

        plot.plot(ntrials, performance, color=clr_seeds, lw=0.75, zorder=5)
        xall.append(ntrials)

    # x-lim
    xall = np.concatenate(xall)
    plot.xlim(min(xall), max(xall))
    plot.hline(target, color=clr_target, lw=0.75)

#=========================================================================================

fig.save(path=figspath)
