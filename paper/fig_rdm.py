#! /usr/bin/env python
from __future__ import division

import os
from   os.path import join

import numpy as np

from pycog.figtools    import Figure
from pycog.utils       import get_here
from examples.analysis import rdm

import paper

#=========================================================================================
# Setup
#=========================================================================================

here     = get_here(__file__)
figspath = join(here, 'figs')

varstim_trialsfile = join(paper.scratchpath, 'rdm_varstim', 'trials',
                          'rdm_varstim_trials.pkl')
varstim_sortedfile = join(paper.scratchpath, 'rdm_varstim', 'trials',
                          'rdm_varstim_sorted_stim_onset.pkl')

rt_trialsfile = join(paper.scratchpath, 'rdm_rt', 'trials', 'rdm_rt_trials.pkl')
rt_sortedfile = join(paper.scratchpath, 'rdm_rt', 'trials', 'rdm_rt_sorted_response.pkl')

varstim_unit = 10
rt_unit      = 68

#=========================================================================================
# Figure setup
#=========================================================================================

fig = Figure(w=4.3, h=6.6, axislabelsize=7, labelpadx=5, labelpady=5, thickness=0.6,
             ticksize=3, ticklabelsize=6, ticklabelpad=2, format=paper.format)

w  = 0.35
dx = 0.49
x0 = 0.12
x1 = x0 + dx

h  = 0.17
dy = 0.215
y0 = 0.775
y1 = y0 - dy
y2 = y1 - dy - 0.03
y3 = y2 - dy - 0.03

w_inset = 0.12
h_inset = 0.07

plots = {
    'Ain': fig.add([x0, y0+0.72*h, w, 0.3*h]),
    'Bin': fig.add([x1, y0+0.72*h, w, 0.3*h]),
    'A': fig.add([x0, y0, w, 0.65*h]),
    'B': fig.add([x1, y0, w, 0.65*h]),
    'C': fig.add([x0, y1, w, h]),
    'D': fig.add([x1, y1, w, h]),
    'E': fig.add([x0, y2, w, h]),
    'F': fig.add([x1, y2, w, h]),
    'G': fig.add([x0, y3, w, h]),
    'H': fig.add([x1, y3, w, h])
    }
plots['Finset'] = fig.add([x1 + 0.23, y2 + 0.15, w_inset, h_inset])

x0 = 0.025
x1 = x0 + dx + 0.005

y0 = 0.96
y1 = y0 - dy
y2 = y1 - dy - 0.03
y3 = y2 - dy - 0.03

plotlabels = {
    'A': (x0, y0),
    'B': (x1, y0),
    'C': (x0, y1),
    'D': (x1, y1),
    'E': (x0, y2),
    'F': (x1, y2),
    'G': (x0, y3),
    'H': (x1, y3)
    }
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

dashes = [3.5, 1.5]
shift  = 0.008

#=========================================================================================
# Training protocol for outputs
#=========================================================================================

plot = plots['Ain']
plot.text_upper_center('Direction discrimination', dy=0.21, fontsize=7.5)
plot.ylabel('Input', labelpad=7, fontsize=6.5)

plot = plots['A']
plot.xlabel('Time', labelpad=6.5)
plot.ylabel('Target output', labelpad=7, fontsize=6.5)

plot = plots['Bin']
plot.text_upper_center('Reaction time version', dy=0.21, fontsize=7.5)

plot = plots['C']
plot.xlabel(r'\% coherence toward choice 1')
plot.ylabel(r'Percent choice 1')

plot = plots['E']
plot.xlabel('Stimulus duration (ms)')
plot.ylabel('Percent correct')

plot = plots['F']
plot.ylabel('Reaction time (ms)')

plot = plots['F']
plot.xlabel(r'\% coherence toward choice 1')

plot = plots['G']
plot.xlabel('Time from stimulus onset (ms)')
plot.ylabel('Firing rate (a.u.)', labelpad=6)

plot = plots['H']
plot.xlabel('Time from decision (ms)')

#-----------------------------------------------------------------------------------------
# Variable stimulus duration
#-----------------------------------------------------------------------------------------

plot = plots['A']

t_fixation = np.array([0,   200])
t_stimulus = np.array([200, 800])
t_decision = np.array([800, 1000])

hi = 1
lo = 0.1
plot.plot(t_fixation, (lo+shift)*np.ones_like(t_fixation),
          color=Figure.colors('blue'), lw=2)
plot.plot(t_fixation, (lo-shift)*np.ones_like(t_fixation),
          color=Figure.colors('red'), lw=2)
plot.plot(t_decision, hi*np.ones_like(t_decision),
          color=Figure.colors('blue'), lw=2)
plot.plot(t_decision, (lo-shift)*np.ones_like(t_decision),
          color=Figure.colors('red'), lw=2)

plot.xlim(0, t_decision[-1])
plot.xticks()

plot.ylim(0, 1.1)
plot.yticks()

plot.vline(t_fixation[-1], color='0.2', linestyle='--', lw=1, dashes=dashes)
plot.vline(t_decision[0],  color='0.2', linestyle='--', lw=1, dashes=dashes)

# Epochs
plot.text(np.mean(t_fixation), 0.6, 'fix.',
          ha='center', va='center', fontsize=7)
plot.text(np.mean(t_decision), 0.6, 'dec.',
          ha='center', va='center', fontsize=7)

#-----------------------------------------------------------------------------------------
# Inputs
#-----------------------------------------------------------------------------------------

plot = plots['Ain']
plot.axis_off('bottom')
plot.axis_off('left')

rng      = np.random.RandomState(111)
baseline = 0.2
coh      = 8

t  = np.linspace(0, t_decision[1], 201)
u0 = baseline + 0.1*rng.normal(size=len(t))
u1 = baseline + 0.1*rng.normal(size=len(t))
for i in xrange(len(t)):
    if t_stimulus[0] < t[i] <= t_stimulus[1]:
        u0[i] += (1 + 3.2*coh/100)/2
        u1[i] += (1 - 3.2*coh/100)/2

plot.plot(t, u0, color=Figure.colors('blue'), lw=0.5)
plot.plot(t, u1, color=Figure.colors('red'),  lw=0.5)

plot.xticks()
plot.yticks()

plot.text(np.mean(t_stimulus), 0.05, 'variable stimulus',
          ha='center', va='center', fontsize=6.5, color='k')

#-----------------------------------------------------------------------------------------
# Reaction time version
#-----------------------------------------------------------------------------------------

plot = plots['B']

eps = 1e-6
FIX = 200
t   = np.array([0, FIX, 500, 1000])

hi = 1
lo = 0.1
plot.plot(t[0:2], [lo+shift, lo+shift], color=Figure.colors('blue'), lw=2)
plot.plot(t[0:2], [lo-shift, lo-shift], color=Figure.colors('red'), lw=2)
plot.plot(t[2:4], [hi,       hi],       color=Figure.colors('blue'), lw=2)
plot.plot(t[2:4], [lo-shift, lo-shift], color=Figure.colors('red'), lw=2)

plot.xlim(0, t[-1])
plot.xticks()

plot.ylim(0, 1.1)
plot.yticks()

plot.vline(t[1], color='0.2', linestyle='--', lw=1, dashes=dashes)

# Epochs
plot.text(np.mean(t[[0,1]]), 0.6, 'fix.',
          ha='center', va='center', fontsize=7)
plot.text(np.mean(t[[2,3]]), 0.6, 'decision',
          ha='center', va='center', fontsize=7)

#-----------------------------------------------------------------------------------------
# Inputs
#-----------------------------------------------------------------------------------------

plot = plots['Bin']
plot.axis_off('bottom')
plot.axis_off('left')

rng      = np.random.RandomState(222)
baseline = 0.2
coh      = 8

t  = np.linspace(0, t_decision[1], 201)
u0 = baseline + 0.1*rng.normal(size=len(t))
u1 = baseline + 0.1*rng.normal(size=len(t))
for i in xrange(len(t)):
    if t[i] > FIX:
        u0[i] += (1 + 3.2*coh/100)/2
        u1[i] += (1 - 3.2*coh/100)/2

plot.plot(t, u0, color=Figure.colors('blue'), lw=0.5)
plot.plot(t, u1, color=Figure.colors('red'),  lw=0.5)

plot.xticks()
plot.yticks()

plot.text(FIX + 50, 0.05, r'ongoing stimulus',
          ha='left', va='center', fontsize=6.5, color='k')
plot.arrow(780, 0.05, 210, 0, width=0.003, head_width=0.07, head_length=25,
           length_includes_head=True, fc='k', ec='k')

#-----------------------------------------------------------------------------------------
# Psychometric curves
#-----------------------------------------------------------------------------------------

# Variable stimulus
plot = plots['C']
rdm.psychometric_function(varstim_trialsfile, plot, ms=5)

# Reaction time
plot = plots['D']
rdm.psychometric_function(rt_trialsfile, plot, threshold=True, ms=5)

#-----------------------------------------------------------------------------------------
# Proportion correct as a function of stimulus duration
#-----------------------------------------------------------------------------------------

plot = plots['E']
rdm.plot_stimulus_duration(varstim_trialsfile, plot, ms=4.5)

#-----------------------------------------------------------------------------------------
# Reaction time
#-----------------------------------------------------------------------------------------

plot      = plots['F']
plot_dist = plots['Finset']

rdm.chronometric_function(rt_trialsfile, plot, plot_dist, ms=5)
plot.ylim(100, 1000)
plot.yticks(range(200, 1000+100, 200))
plot_dist.xlim(0, 1500)
plot_dist.xticks([0, 1500])
plot_dist.axis_off('left')

#-----------------------------------------------------------------------------------------
# Unit activity
#-----------------------------------------------------------------------------------------

# Variable stimulus
plot = plots['G']
rdm.plot_unit(varstim_unit, varstim_sortedfile, plot, tmin=-200, tmax=600, lw=1.5)
plot.yticks(range(5))

# Legend
prop = {'prop': {'size': 5},
        'handlelength': 1, 'handletextpad': 1.1, 'labelspacing': 0.5}
plot.legend(bbox_to_anchor=(0.34, 1.11), **prop)

# RT
plot = plots['H']
rdm.plot_unit(rt_unit, rt_sortedfile, plot, tmin=-400, lw=1.5)
plot.xticks(range(-400, 100, 100))
plot.yticks([0, 1])

#=========================================================================================

fig.save(path=figspath)
