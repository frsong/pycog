#! /usr/bin/env python
from __future__ import division

import cPickle as pickle
import os
from   os.path import join

import numpy as np

from pycog.figtools    import Figure
from pycog.utils       import get_here
from examples.analysis import mante

import paper

#=========================================================================================
# Setup
#=========================================================================================

here     = get_here(__file__)
figspath = join(here, 'figs')

trialsfile  = paper.scratchpath + '/mante/trials/mante_trials.pkl'
sortedfile  = paper.scratchpath + '/mante/trials/mante_sorted.pkl'
betafile    = here + '/../examples/work/data/mante/mante_beta.pkl'

units = [1, 2, 3, 4]

#=========================================================================================
# Figure setup
#=========================================================================================

w   = 7.5
h   = 6.7
r   = w/h
fig = Figure(w=w, h=h, axislabelsize=6.5, labelpadx=5, labelpady=5,
             thickness=0.6, ticksize=3, ticklabelsize=6, ticklabelpad=2)

#-----------------------------------------------------------------------------------------
# Psychometric functions
#-----------------------------------------------------------------------------------------

w_psy  = 0.16
dx_psy = w_psy + 0.07
x_psy  = 0.08

h_psy   = 0.15
dy_psy  = h_psy + 0.07
y_psy_m = 0.8
y_psy_c = y_psy_m - dy_psy

plots = {
    'psy_m': fig.add([x_psy, y_psy_m, w_psy, h_psy]),
    'psy_c': fig.add([x_psy, y_psy_c, w_psy, h_psy])
    }

#-----------------------------------------------------------------------------------------
# State-space diagrams
#-----------------------------------------------------------------------------------------

w  = 0.17
dx = w + 0.06
x0 = 0.33
x1 = x0 + dx
x2 = x1 + dx

h  = h_psy
dy = h + 0.05
y0 = y_psy_m
y1 = y_psy_c

plots['m1'] = fig.add([x0, y0, w, h])
plots['m2'] = fig.add([x1, y0, w, h])
plots['m3'] = fig.add([x2, y0, w, h])
plots['c1'] = fig.add([x0, y1, w, h])
plots['c2'] = fig.add([x1, y1, w, h])
plots['c3'] = fig.add([x2, y1, w, h])

#-----------------------------------------------------------------------------------------
# Single units
#-----------------------------------------------------------------------------------------

w  = 0.105
dx = w + 0.03
x0 = x_psy
x1 = x0 + dx
x2 = x1 + dx
x3 = x2 + dx
xall = [x0, x1, x2, x3]

h  = 0.09
dy = h + 0.02
y0 = 0.4
y1 = y0 - dy
y2 = y1 - dy
y3 = y2 - dy
y_unit = y3

for i, x in enumerate(xall):
    plots[str(i)+'_choice']         = fig.add([x, y0, w, h])
    plots[str(i)+'_motion_choice']  = fig.add([x, y1, w, h])
    plots[str(i)+'_colour_choice']  = fig.add([x, y2, w, h])
    plots[str(i)+'_context_choice'] = fig.add([x, y3, w, h])

#-----------------------------------------------------------------------------------------
# Regression coefficients
#-----------------------------------------------------------------------------------------

w  = 0.1
dx = w + 0.08
x0 = 0.68
x1 = x0 + dx

h  = r*w
dy = h + 0.05
y2 = y_unit
y1 = y2 + dy
y0 = y1 + dy

plots['motion_choice']  = fig.add([x0, y0, w, h])
plots['colour_choice']  = fig.add([x1, y0, w, h])
plots['context_choice'] = fig.add([x0, y1, w, h])
plots['colour_motion']  = fig.add([x1, y1, w, h])
plots['context_motion'] = fig.add([x0, y2, w, h])
plots['context_colour'] = fig.add([x1, y2, w, h])

#-----------------------------------------------------------------------------------------

x0 = 0.02
x1 = 0.27
x2 = 0.6
x3 = 0.64
y0 = 0.96
y1 = 0.5

plotlabels = {
    'A': (x0, y0),
    'B': (x1, y0),
    'C': (x0, y1),
    'D': (x2, y1)
    }
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Psychometric functions
#=========================================================================================

mante_plots = {
    'm': plots['psy_m'],
    'c': plots['psy_c']
    }
mante.psychometric_function(trialsfile, mante_plots, ms=4.5)

plot = plots['psy_m']
plot.xlabel('Motion coherence (\%)')
plot.ylabel('Percent choice 1 (\%)')

plot = plots['psy_c']
plot.xlabel('Color coherence (\%)')
plot.ylabel('Percent choice 1 (\%)')

prop = {'prop': {'size': 5}, 'handlelength': 1.1,
        'handletextpad': 0.8, 'labelspacing': 0.5}
plots['psy_m'].legend(bbox_to_anchor=(0.58, 1.12), **prop)

#=========================================================================================
# State space
#=========================================================================================

mante_plots = {s: plots[s] for s in ['m1', 'm2', 'm3', 'c1', 'c2', 'c3']}
mante.plot_statespace(trialsfile, sortedfile, betafile, mante_plots)

plots['m2'].text_upper_center('Motion context', dy=0.08, fontsize=7, color='k')
plots['c2'].text_upper_center('Color context', dy=0.08, fontsize=7,
                              color=Figure.colors('darkblue'))

#=========================================================================================
# Single-unit activity
#=========================================================================================

#-----------------------------------------------------------------------------------------
# Labels
#-----------------------------------------------------------------------------------------

plot = plots['0_context_choice']
plot.xlabel('Time from stimulus onset (ms)')

#-----------------------------------------------------------------------------------------

# Get stimulus period
with open(trialsfile) as f:
    trials = pickle.load(f)
epochs = trials[0]['info']['epochs']
stimulus_start, stimulus_end  = epochs['stimulus']

# Plot units
yall = []
for i, unit in enumerate(units):
    mante_plots = {
        'choice':         plots[str(i)+'_choice'],
        'motion_choice':  plots[str(i)+'_motion_choice'],
        'colour_choice':  plots[str(i)+'_colour_choice'],
        'context_choice': plots[str(i)+'_context_choice']
        }
    sortby_fontsize = None
    if i == 0:
        sortby_fontsize = 6

    yall.append(mante.plot_unit(unit, sortedfile, mante_plots, t0=stimulus_start,
                                tmin=stimulus_start, tmax=stimulus_end,
                                sortby_fontsize=sortby_fontsize, unit_fontsize=6))

# Shared x tick labels
for i in xrange(len(units)):
    plots[str(i)+'_choice'].xticklabels()
    plots[str(i)+'_motion_choice'].xticklabels()
    plots[str(i)+'_colour_choice'].xticklabels()

# Shared y limits
shared_plots = []
for i in xrange(len(units)):
    shared_plots += [plots[str(i)+'_'+s] for s
                     in ['choice', 'motion_choice', 'colour_choice', 'context_choice']]
ylim = fig.shared_lim(shared_plots, 'y', yall)

# Shared y tick labels
plots['0_choice'].yticks([0])
plots['0_motion_choice'].yticks([0])
plots['0_colour_choice'].yticks([0])
plots['0_context_choice'].yticks([0])
for i in xrange(1, len(units)):
    for s in ['choice', 'motion_choice', 'colour_choice', 'context_choice']:
        plots[str(i)+'_'+s].yticks(plots['0_'+s].get_yticks())
        plots[str(i)+'_'+s].yticklabels()

#=========================================================================================
# Regression coefficients
#=========================================================================================

mante_plots = {s: plots[s] for s
               in ['motion_choice', 'colour_choice', 'context_choice',
                   'colour_motion', 'context_motion', 'context_colour']}
mante.plot_regress(betafile, mante_plots)

#=========================================================================================

fig.save(path=figspath)
