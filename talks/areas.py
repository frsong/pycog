#! /usr/bin/env python
from __future__ import division

import imp
import os
import sys
from   os.path import join

import numpy as np

from pycog           import RNN
from pycog.figtools  import gradient, mpl, Figure
from pycog.utils     import get_here, get_parent

import paper

#=========================================================================================
# Paths
#=========================================================================================

here     = get_here(__file__)
base     = get_parent(here)
figspath = join(here, 'figs')

#-----------------------------------------------------------------------------------------
# Load RNNs to compare
#-----------------------------------------------------------------------------------------

datapath = join(base, 'examples', 'work', 'data')

savefile = join(datapath, 'mante', 'mante.pkl')
rnn1     = RNN(savefile, verbose=True)

savefile = join(datapath, 'mante_areas', 'mante_areas.pkl')
rnn2     = RNN(savefile, verbose=True)

# Load model
modelfile = join(base, 'examples', 'models', 'mante_areas.py')
m         = imp.load_source('model', modelfile)

#=========================================================================================
# Figure setup
#=========================================================================================

w   = 7.5
h   = 3.8
r   = w/h
fig = Figure(w=w, h=h, axislabelsize=7, labelpadx=6, labelpady=6, thickness=0.6,
             ticksize=3, ticklabelsize=6, ticklabelpad=2, format=paper.format)

w_rec = 0.38
h_rec = r*w_rec

w_in  = 0.04
h_in  = h_rec

w_out = w_rec
h_out = r*w_in

hspace = 0.022
vspace = r*hspace

dx = 0.04
x0 = 0.035
x1 = x0 + w_rec + hspace + w_in + dx

w_psy = 0.2
h_psy = 0.7

y = 0.16

plots = {
    'Arec': fig.add([x0,              y,              w_rec, h_rec], None),
    'Ain':  fig.add([x0+w_rec+hspace, y,              w_in,  h_in],  None),
    'Aout': fig.add([x0,              y-vspace-h_out, w_out, h_out], None,
                    ticklabelpadx=1.5),

    'Brec': fig.add([x1,              y,              w_rec, h_rec], None),
    'Bin':  fig.add([x1+w_rec+hspace, y,              w_in,  h_in],  None),
    'Bout': fig.add([x1,              y-vspace-h_out, w_out, h_out], None)
    }

dx2 = x1 - x0
x0  = 0.01
x1  = x0 + dx2

y = 0.93

#plotlabels = {
#    'A': (x0, y),
#    'B': (x1, y)
#    }
#fig.plotlabels(plotlabels, fontsize=paper.plotlabelsize)

#=========================================================================================
# Labels
#=========================================================================================

plot = plots['Arec']
plot.text_upper_center('Context-dependent integration, unstructured connectivity',
                       dy=0.06, fontsize=7)
plot.xaxis.set_label_position('top')
plot.xlabel('From', labelpad=4)
plot.ylabel('To', labelpad=4)

plot = plots['Brec']

units  = m.EXC_SENSORY
label  = '"Sensory" area (exc)'

groups  = [m.EXC_SENSORY, m.EXC_MOTOR, m.INH_SENSORY, m.INH_MOTOR]
labels  = [r"``Sensory'' area ($\text{E}_\text{S}$)",
           r"``Motor'' area ($\text{E}_\text{M}$)",
           r"($\text{I}_\text{S}$)",
           r"($\text{I}_\text{M}$)"]
colors  = [Figure.colors('green'), Figure.colors('orange')]
colors += colors
for group, label, color in zip(groups, labels, colors):
    extent = (np.array([group[0], group[-1]]) + 0.5)/m.N
    plot.plot(extent, 1.03*np.ones(2), lw=2, color=color, transform=plot.transAxes)
    plot.text(np.mean(extent), 1.05, label, ha='center', va='bottom',
              fontsize=6, color=color, transform=plot.transAxes)

    plots['Bin'].plot(1.2*np.ones(2), 1-extent, lw=2, color=color,
                      transform=plot.transAxes)

E_S = m.EXC_SENSORY
E_M = m.EXC_MOTOR
x_S = np.mean([E_S[0], E_S[-1]])
x_M = np.mean([E_M[0], E_M[-1]])

plot = plots['Brec']
plot.text(x_S, x_S,
          r"$\mathbf{E}_\mathbf{S} \boldsymbol{\leftarrow}"
          + r" \mathbf{E}_\mathbf{S}$",
          ha='center', va='center', fontsize=10, color='k')
plot.text(x_S, x_M,
          r"$\mathbf{E}_\mathbf{M} \boldsymbol{\leftarrow}"
          + r" \mathbf{E}_\mathbf{S}$",
          ha='center', va='center', fontsize=10, color='k')
plot.text(x_M, x_S,
          r"$\mathbf{E}_\mathbf{S} \boldsymbol{\leftarrow}"
          + r" \mathbf{E}_\mathbf{M}$",
          ha='center', va='center', fontsize=10, color='k')
plot.text(x_M, x_M,
          r"$\mathbf{E}_\mathbf{M} \boldsymbol{\leftarrow}"
          + r" \mathbf{E}_\mathbf{M}$",
          ha='center', va='center', fontsize=10, color='k')

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

    if len(inh) > 0:
        cmap_inh = gradient(white, red)
        norm_inh = mpl.colors.Normalize(vmin=0, vmax=np.round(inh[-inh_ignore], 1),
                                        clip=True)
        smap_inh = mpl.cm.ScalarMappable(norm_inh, cmap_inh)

        cmap_inh_r = gradient(red, white)
        norm_inh_r = mpl.colors.Normalize(vmin=-np.round(inh[-inh_ignore], 1), vmax=0,
                                          clip=True)
        smap_inh_r = mpl.cm.ScalarMappable(norm_inh_r, cmap_inh_r)
    else:
        smap_inh = None

    return smap_exc, smap_inh

# Color maps
rnns = [rnn1, rnn2]
smap_exc_in,  smap_inh_in  = generate_cmap([rnn.Win  for rnn in rnns])
smap_exc_rec, smap_inh_rec = generate_cmap([rnn.Wrec for rnn in rnns])
smap_exc_out, smap_inh_out = generate_cmap([rnn.Wout for rnn in rnns])

#=========================================================================================
# Connection matrices
#=========================================================================================

for rnn, s in zip([rnn1, rnn2], ['A', 'B']):
    RNN.plot_connection_matrix(plots[s+'in'], rnn.Win,
                               smap_exc_in, smap_inh_in)
    RNN.plot_connection_matrix(plots[s+'rec'], rnn.Wrec,
                               smap_exc_rec, smap_inh_rec)
    RNN.plot_connection_matrix(plots[s+'out'], rnn.Wout,
                               smap_exc_out, smap_inh_out)

plot = plots['Ain']
plot.xaxis.set_tick_params(pad=-3)

plot = plots['Aout']
plot.yaxis.set_tick_params(pad=-2)
plot.yticks(np.arange(2)+0.02)
plot.yticklabels(['Choice 1', 'Choice 2'], fontsize=6)

#=========================================================================================

fig.save(path=figspath)
