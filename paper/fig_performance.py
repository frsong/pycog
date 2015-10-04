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
          ('lee',          '7: Lee'),
          ('lee_areas',    '8B: Lee (2 areas)')]
labels = list('ABCDEFGHIJ')

#=========================================================================================
# Figure setup
#=========================================================================================

fig = Figure(w=3.8, h=6.75, axislabelsize=6.5, labelpadx=4, labelpady=3, thickness=0.6,
             ticksize=3, ticklabelsize=6, ticklabelpad=2, format=paper.format)

ncols = 2
nrows = 5

w    = 0.36
dx   = w + 0.12
x0   = 0.12
xall = x0 + dx*np.arange(ncols)

h    = 0.13
y0   = 0.82
dy   = h + 0.06
yall = y0 - dy*np.arange(nrows)

plots      = {}
plotlabels = {}
for k in xrange(len(models)):
    i = k//ncols
    j = k%ncols
    if i == 0:
        pady = 0
    else:
        pady = 0.02
    plots[models[k][0]]   = fig.add([xall[j], yall[i]-pady, w, h])
    plotlabels[labels[k]] = (xall[j]-0.08, yall[i]-pady+h+0.015)
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Labels
#=========================================================================================

plot = plots[models[0][0]]
plot.xlabel(r'Number of trials ($\times 10^4$)')
plot.ylabel('Percent correct')

plot = plots['romo']
plot.ylabel('Min. percent correct')

plot = plots[models[-2][0]]
plot.ylabel('Error in eye position')

for k in xrange(len(models)):
    model, desc = models[k]
    if desc is None:
        desc = model
    plots[model].text_upper_center(desc, dy=0.05, fontsize=6.5)

#=========================================================================================
# Plot performance
#=========================================================================================

clr_target = Figure.colors('red')
clr_actual = '0.2'
clr_seeds  = '0.8'

for model, _ in models:
    plot = plots[model]

    try:
        rnn = RNN(get_savefile(model), verbose=True)
    except SystemExit:
        continue

    xall = []

    ntrials     = [int(costs[0]) for costs in rnn.costs_history]
    ntrials     = np.asarray(ntrials, dtype=int)/int(1e4)
    performance = [costs[1][-1] for costs in rnn.costs_history]

    # Because the network is run continuously, the first validation run is meaningless.
    if 'lee' in model:
        ntrials     = ntrials[1:]
        performance = performance[1:]

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

    # Number of units
    nunits = '{} units'.format(rnn.p['N'])

    # Training time
    timefile = join(paperpath, 'times', model + '_time.txt')
    if os.path.isfile(timefile):
        time = '{} mins'.format(int(np.loadtxt(timefile)))
    else:
        time = 'X mins'

    # Info
    dy = 0
    if model in ['lee', 'lee_areas']:
        dy = 0.77
    plot.text_lower_right(nunits, dy=dy+0.105, fontsize=5.5,
                          color=Figure.colors('green'), zorder=20)
    plot.text_lower_right(time, dy=dy+0.005, fontsize=5.5,
                          color=Figure.colors('strongblue'), zorder=20)

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

        if 'lee' in model:
            ntrials     = ntrials[1:]
            performance = performance[1:]
        plot.plot(ntrials, performance, color=clr_seeds, lw=0.75, zorder=5)
        xall.append(ntrials)

    # x-axis
    xall = np.concatenate(xall)
    plot.xlim(min(xall), max(xall))
    plot.hline(target, color=clr_target, lw=0.75)

    # y-axis
    if model == 'lee':
        plot.ylim(0.04, 0.08)
        plot.yticks(np.arange(0.04, 0.09, 0.01))
    elif model == 'lee_areas':
        pass
    elif model == 'romo':
        plot.ylim(0, 100)
    else:
        plot.ylim(40, 100)

#=========================================================================================

fig.save(path=figspath)
