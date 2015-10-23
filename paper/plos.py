#! /usr/bin/env python
import os
import subprocess
import sys
from   os.path import join

from pycog.utils import get_here, mkdir_p

#=========================================================================================
# Shared steps
#=========================================================================================

here      = get_here(__file__)
base      = os.path.abspath(join(here, os.pardir))
paperpath = join(base, 'paper')

mkdir_p(join(paperpath, 'figs', 'plos'))

def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)

def figure(fig, n):
    call('python ' + join(paperpath, fig + '.py'))
    call('mv {} {}'.format(join(paperpath, 'figs', fig + '.eps'),
                           join(paperpath, 'figs', 'plos', 'fig{}.eps'.format(n))))

#=========================================================================================

figs = {
    'fig_rdm':          2,
    'fig_structure':    3,
    'fig_mante':        4,
    'fig_connectivity': 5,
    'fig_multisensory': 6,
    'fig_romo':         7,
    'fig_lee':          8,
    'fig_performance':  9
    }

for f, n in figs.items():
    figure(f, n)
