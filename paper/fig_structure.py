#! /usr/bin/env python
from __future__ import division

import imp
import os
import sys
from   os.path import join

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

nodale_trialsfile = join(paper.scratchpath,
                         'rdm_nodale', 'trials', 'rdm_nodale_trials.pkl')
dense_trialsfile  = join(paper.scratchpath,
                         'rdm_dense', 'trials', 'rdm_dense_trials.pkl')
fixed_trialsfile  = join(paper.scratchpath,
                         'rdm_fixed', 'trials', 'rdm_fixed_trials.pkl')

#-----------------------------------------------------------------------------------------
# Load RNNs to compare
#-----------------------------------------------------------------------------------------

datapath = join(base, 'examples', 'work', 'data')

savefile      = join(datapath, 'rdm_nodale', 'rdm_nodale.pkl')
rnn_nodale    = RNN(savefile, verbose=True)
dprime_nodale = join(datapath, 'rdm_nodale', 'rdm_nodale_dprime.txt')
sortby_nodale = join(datapath, 'rdm_nodale', 'rdm_nodale_selectivity.txt')

savefile     = join(datapath, 'rdm_dense', 'rdm_dense.pkl')
rnn_dense    = RNN(savefile, verbose=True)
dprime_dense = join(datapath, 'rdm_dense', 'rdm_dense_dprime.txt')
sortby_dense = join(datapath, 'rdm_dense', 'rdm_dense_selectivity.txt')

savefile     = join(datapath, 'rdm_fixed', 'rdm_fixed.pkl')
rnn_fixed    = RNN(savefile, verbose=True)
dprime_fixed = join(datapath, 'rdm_fixed', 'rdm_fixed_dprime.txt')
sortby_fixed = join(datapath, 'rdm_fixed', 'rdm_fixed_selectivity.txt')

#=========================================================================================
# Figure setup
#=========================================================================================

w   = 7.5
h   = 4.7
r   = w/h
fig = Figure(w=w, h=h, axislabelsize=7, labelpadx=6, labelpady=6, thickness=0.6,
             ticksize=3, ticklabelsize=6, ticklabelpad=2, format=paper.format)

w_rec = 0.25
h_rec = r*w_rec

w_in  = 0.03
h_in  = h_rec

w_out = w_rec
h_out = r*w_in

hspace = 0.013
vspace = r*hspace

dx = 0.03
x0 = 0.035
x1 = x0 + w_rec + hspace + w_in + dx
x2 = x1 + w_rec + hspace + w_in + dx

offset = 0.03
x0_psy = x0 + offset
x1_psy = x1 + offset
x2_psy = x2 + offset

w_psy = 0.2
h_psy = r*0.8*w_psy

dy = 0.16
y0 = 0.65
y1 = y0 - h_rec - dy

w_mask = 0.12
h_mask = r*w_mask

plots = {
    'Apsy': fig.add([x0_psy,          y0,              w_psy, h_psy]),
    'Arec': fig.add([x0,              y1,              w_rec, h_rec], None),
    'Ain':  fig.add([x0+w_rec+hspace, y1,              w_in,  h_in],  None),
    'Aout': fig.add([x0,              y1-vspace-h_out, w_out, h_out], None,
                    ticklabelpadx=1.5),

    'Bpsy': fig.add([x1_psy,          y0,              w_psy, h_psy]),
    'Brec': fig.add([x1,              y1,              w_rec, h_rec], None),
    'Bin':  fig.add([x1+w_rec+hspace, y1,              w_in,  h_in],  None),
    'Bout': fig.add([x1,              y1-vspace-h_out, w_out, h_out], None),

    'Cpsy': fig.add([x2_psy,          y0,              w_psy, h_psy]),
    'Crec': fig.add([x2,              y1,              w_rec, h_rec], None),
    'Cin':  fig.add([x2+w_rec+hspace, y1,              w_in,  h_in],  None),
    'Cout': fig.add([x2,              y1-vspace-h_out, w_out, h_out], None)
    }

dx2 = x2 - x1
x0  = 0.02
x1  = x0 + dx2
x2  = x1 + dx2
x_mask = 0.525

y = 0.93
plotlabels = {
    'A': (x0, y),
    'B': (x1, y),
    'C': (x2, y)
    }
fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Create color maps for weights
#=========================================================================================

# Colors
white = 'w'
blue  = Figure.colors('strongblue')
red   = Figure.colors('strongred')

def generate_cmap(Ws):
    exc = []
    inh = []
    for W in Ws:
        exc.append( np.ravel(W[np.where(W > 0)]))
        inh.append(-np.ravel(W[np.where(W < 0)]))

    exc = np.sort(np.concatenate(exc))
    inh = np.sort(np.concatenate(inh))

    exc_ignore = int(0.1*len(exc))
    inh_ignore = int(0.1*len(inh))

    cmap_exc = gradient(white, blue)
    norm_exc = mpl.colors.Normalize(vmin=0, vmax=np.round(exc[-exc_ignore], 1), clip=True)
    smap_exc = mpl.cm.ScalarMappable(norm_exc, cmap_exc)

    cmap_inh = gradient(white, red)
    norm_inh = mpl.colors.Normalize(vmin=0, vmax=np.round(inh[-inh_ignore], 1), clip=True)
    smap_inh = mpl.cm.ScalarMappable(norm_inh, cmap_inh)

    cmap_inh_r = gradient(red, white)
    norm_inh_r = mpl.colors.Normalize(vmin=-np.round(inh[-inh_ignore], 1), vmax=0,
                                      clip=True)
    smap_inh_r = mpl.cm.ScalarMappable(norm_inh_r, cmap_inh_r)

    return smap_exc, smap_inh

# Color maps
rnns = [rnn_nodale, rnn_dense, rnn_fixed]
smap_exc_in,  smap_inh_in  = generate_cmap([rnn.Win  for rnn in rnns])
smap_exc_rec, smap_inh_rec = generate_cmap([rnn.Wrec for rnn in rnns])
smap_exc_out, smap_inh_out = generate_cmap([rnn.Wout for rnn in rnns])

#=========================================================================================
# Labels
#=========================================================================================

textprop = dict(dy=0.14, fontsize=7)

plot = plots['Apsy']
plot.xlabel(r'Percent coherence toward choice 1')
plot.ylabel(r'Percent choice 1', labelpad=4)
plot.text_upper_center('No Dale\'s principle', **textprop)

plot = plots['Bpsy']
plot.text_upper_center('Dale, dense initial connectivity', **textprop)

plot = plots['Cpsy']
plot.text_upper_center('Dale, constrained connectivity', **textprop)

#=========================================================================================
# Psychometric curves
#=========================================================================================

# No Dale
plot = plots['Apsy']
rdm.psychometric_function(nodale_trialsfile, plot, ms=5)

# Dale, dense
plot = plots['Bpsy']
rdm.psychometric_function(dense_trialsfile, plot, ms=5)

# Dale, fixed
plot = plots['Cpsy']
rdm.psychometric_function(fixed_trialsfile, plot, ms=5)

#=========================================================================================
# Connection matrices
#=========================================================================================

for rnn, sortbyfile, s, dprimefile in zip([rnn_nodale, rnn_dense, rnn_fixed],
                                          [sortby_nodale, sortby_dense, sortby_fixed],
                                          ['A', 'B', 'C'],
                                          [dprime_nodale, dprime_dense, dprime_fixed]):
    idx = np.loadtxt(sortbyfile, dtype=int)
    RNN.plot_connection_matrix(plots[s+'in'], rnn.Win[idx,:],
                               smap_exc_in, smap_inh_in)
    RNN.plot_connection_matrix(plots[s+'rec'], rnn.Wrec[idx,:][:,idx],
                               smap_exc_rec, smap_inh_rec)
    RNN.plot_connection_matrix(plots[s+'out'], rnn.Wout[:,idx],
                               smap_exc_out, smap_inh_out)

    dprime = np.loadtxt(dprimefile)
    transitions = []
    for i in xrange(1, len(dprime)):
        if dprime[i-1] > 0 and dprime[i] < 0:
            transitions.append(i)

    plot = plots[s+'rec']
    if s == 'A':
        n = transitions[0]
        plot.text(n-0.5, -1.53, '|', ha='center', va='center', fontsize=8)
        plot.text(n-0.5-0.5, -1.45, r'$\leftarrow$\ Selective for 1',
                  ha='right', va='center', fontsize=6)
        plot.text(n-0.5+0.5, -1.45, r'Selective for 2 $\rightarrow$',
                  ha='left', va='center', fontsize=6)
    else:
        # Excitatory units
        n = transitions[0]
        plot.text(n-0.5, -3.3, 'Excitatory units', ha='center', va='center',
                  color=Figure.colors('blue'), fontsize=6)
        plot.text(n-0.5, -1.53, '|', ha='center', va='center',
                  color=Figure.colors('blue'), fontsize=8)
        plot.text(n-0.5-0.5, -1.63, r'$\leftarrow$', color=Figure.colors('blue'),
                  ha='right', va='center', fontsize=6)
        plot.text(n-0.5+0.5, -1.63, r'$\rightarrow$', color=Figure.colors('blue'),
                  ha='left', va='center', fontsize=6)

        # Inhibitory units
        try:
            n = transitions[1]
            plot.text(n-0.5, -3.3, 'Inhibitory units', ha='center', va='center',
                      color=Figure.colors('red'), fontsize=6)
            plot.text(n-0.5, -1.53, '|', ha='center', va='center',
                      color=Figure.colors('red'), fontsize=8)
            plot.text(n-0.5-0.5, -1.63, r'$\leftarrow$', color=Figure.colors('red'),
                      ha='right', va='center', fontsize=6)
            plot.text(n-0.5+0.5, -1.63, r'$\rightarrow$', color=Figure.colors('red'),
                      ha='left', va='center', fontsize=6)
        except:
            pass

plot = plots['Ain']
plot.xaxis.set_tick_params(pad=-3)
plot.xticks(np.arange(2)+0.06)
plot.xaxis.set_tick_params(pad=-2)
plot.xticklabels(['Choice 1', 'Choice 2'], rotation='vertical', fontsize=5.5)

plot = plots['Arec']
plot.xaxis.set_label_position('top')
plot.xlabel('From', labelpad=15)
plot.ylabel('To', labelpad=6)

plot = plots['Aout']
plot.yaxis.set_tick_params(pad=-2)
plot.yticks(np.arange(2)+0.01)
plot.yticklabels(['Choice 1', 'Choice 2'], fontsize=5.5)

#=========================================================================================

fig.save(path=figspath)
