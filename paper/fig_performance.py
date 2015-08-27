#! /usr/bin/env python
from __future__ import division

import argparse
import os

import numpy as np

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
figspath = here + '/figs'

varstim_trialsfile = paper.scratchpath + '/rdm_varstim/trials/rdm_varstim_trials.pkl'
varstim_sortedfile = (paper.scratchpath 
                      + '/rdm_varstim/trials/rdm_varstim_sorted_stim_onset.pkl')

rt_trialsfile = paper.scratchpath + '/rdm_rt/trials/rdm_rt_trials.pkl'
rt_sortedfile = paper.scratchpath + '/rdm_rt/trials/rdm_rt_sorted_response.pkl'

varstim_unit = 11
rt_unit      = 11

#=========================================================================================
# Figure setup
#=========================================================================================

fig = Figure(w=6, h=5, axislabelsize=7, labelpadx=5, labelpady=5,
             thickness=0.6, ticksize=3, ticklabelsize=6, ticklabelpad=2)

plots = {
    'rdm_dale': fig.add()
    }

'''
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
'''

#=========================================================================================



#=========================================================================================

fig.save(path=figspath)
