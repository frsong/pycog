"""
A more convenient interface to matplotlib.

  Note
  ----
  
  It appears that matplotlib's savefig has a bug in certain versions. Update your
  matplotlib to the latest version and this problem will go away.

"""
from __future__ import absolute_import
from __future__ import division

import os
import subprocess
import sys
from   os.path import join

import numpy                      as np
import matplotlib                 as mpl; mpl.use('Agg') # For compatibility on cluster
import matplotlib.pyplot          as plt
import mpl_toolkits.mplot3d.art3d as art3d

from matplotlib.colors    import colorConverter
from matplotlib.mlab      import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.special        import cbrt

from .utils import mkdir_p

THIS = "pycog.figtools"

#=========================================================================================
# Font, LaTeX
#=========================================================================================

mpl.rcParams['font.family']        = 'sans-serif'
mpl.rcParams['ps.useafm']          = True
mpl.rcParams['pdf.use14corefonts'] = True

# Setup LaTeX if available
try:
    FNULL = open(os.devnull, 'w')
    subprocess.check_call('latex --version', shell=True,
                          stdout=FNULL, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError:
    latex = False
    print("[ {} ] Warning: Couldn't find LaTeX. Your figures will look ugly."
          .format(THIS))
else:
    latex = True
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = (
        '\usepackage{sfmath}'
        '\usepackage[T1]{fontenc}'
        '\usepackage{amsmath}'
        '\usepackage{amssymb}'
        )

#=========================================================================================
# Global defaults
#=========================================================================================

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

#=========================================================================================
# Simple color map
#=========================================================================================

def gradient(cmin, cmax):
    if isinstance(cmin, str):
        cmin = colorConverter.to_rgb(cmin)
    if isinstance(cmax, str):
        cmax = colorConverter.to_rgb(cmax)

    cdict = {
        'red':   [(0, 0,       cmin[0]),
                  (1, cmax[0], 1)],
        'green': [(0, 0,       cmin[1]),
                  (1, cmax[1], 1)],
        'blue':  [(0, 0,       cmin[2]),
                  (1, cmax[2], 1)]
        }
    
    return mpl.colors.LinearSegmentedColormap('cmap', cdict, N=1000)

#=========================================================================================
# Colors
#=========================================================================================

def apply_alpha(color, alpha=0.7):
    fg = np.asarray(colorConverter.to_rgb(color))
    bg = np.ones(3)

    return tuple(alpha*fg + (1-alpha)*bg)

colors = {
    'strongblue':   '#2171b5',
    'strongred':    '#cb181d',
    'blue':         apply_alpha('#447294'),
    'green':        apply_alpha('#30701e'),
    'red':          apply_alpha('#bf2121'),
    'salmon':       '#ee9572',
    'lightred':     '#bf6d6b',
    'darkred':      '#8f2a2a',
    'lightblue':    '#8fbcdb',
    'orange':       apply_alpha('#e58c2c'),
    'magenta':      apply_alpha('#c42d95'),
    'purple':       apply_alpha('#8064a2'),    
    'lightgreen':   '#78cd71',
    'darkblue':     '#084594',#'#315d7d',
    'gray':         '0.5',
    'darkgray':     '0.3',
    'lightgray':    '0.7',
    'lightlightgray': '0.9',
    'black':        '#000000',
    'white':        '#ffffff'
    }

#=========================================================================================
# Subplot
#=========================================================================================

class Subplot(object):
    """
    Interface to Axes.

    You can access any attribute that belongs to Axes through Subplot.

    """
    def __getattr__(self, name):
        if hasattr(self.ax, name):
            return getattr(self.ax, name)
        if hasattr(self.ax, 'set_'+name):
            return getattr(self.ax, 'set_'+name)
        raise NotImplementedError("Subplot." + name)

    def __init__(self, fig, p, rect):
        """
        rect : [left, bottom, width, height]

        """
        self.p     = p
        self.ax    = fig.add_axes(rect)
        self.axes  = [self.xaxis, self.yaxis]

    def set_thickness(self, thickness):
        for v in self.ax.spines.values():
            v.set_linewidth(thickness)
        for ax in self.axes:
            ax.set_tick_params(width=thickness)

    def set_tick_params(self, ticksize, ticklabelsize, ticklabelpad):
        for ax in self.axes:
            ax.set_tick_params(size=ticksize, labelsize=ticklabelsize, pad=ticklabelpad)

    def format(self, style, p=None):
        if style == 'bottomleft':
            for s in ['top', 'right']:
                self.spines[s].set_visible(False)
            self.xaxis.tick_bottom()
            self.yaxis.tick_left()

            self.set_thickness(p['thickness'])
            self.set_tick_params(p['ticksize'], p['ticklabelsize'], p['ticklabelpad'])
        elif style == 'none':
            for s in self.spines.values():
                s.set_visible(False)
            
            self.xticks()
            self.yticks()

    def axis_off(self, axis=None):
        self.spines[axis].set_visible(False)
        self.yticks()
        self.yticklabels()

    #/////////////////////////////////////////////////////////////////////////////////////

    def plot(self, *args, **kwargs):
        kwargs.setdefault('clip_on', False)
        kwargs.setdefault('zorder', 10)
        
        return self.ax.plot(*args, **kwargs)

    def xlabel(self, *args, **kwargs):
        kwargs.setdefault('fontsize', self.p['axislabelsize'])
        kwargs.setdefault('labelpad', self.p['labelpadx'])
        
        return self.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        kwargs.setdefault('fontsize', self.p['axislabelsize'])
        kwargs.setdefault('labelpad', self.p['labelpady'])
        
        return self.set_ylabel(*args, **kwargs)

    def xticks(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_xticks(*args, **kwargs)

    def yticks(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_yticks(*args, **kwargs)

    def xticklabels(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_xticklabels(*args, **kwargs)

    def yticklabels(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_yticklabels(*args, **kwargs)

    def equal(self):
        return self.set_aspect('equal')

    #/////////////////////////////////////////////////////////////////////////////////////
    # Annotation

    def legend(self, *args, **kargs):
        kargs.setdefault('bbox_transform', self.transAxes)
        kargs.setdefault('frameon', False)
        kargs.setdefault('numpoints', 1)

        return self.ax.legend(*args, **kargs)

    def text_upper_center(self, s, dx=0, dy=0, fontsize=7.5, color='k', **kwargs):
        return self.text(0.5+dx, 1+dy, s, ha='center', va='bottom',
                         fontsize=fontsize, color=color, 
                         transform=self.transAxes, **kwargs)

    def text_upper_left(self, s, dx=0, dy=0, fontsize=7.5, color='k', **kwargs):
        return self.text(0.04+dx, 0.97+dy, s, ha='left', va='top',
                         fontsize=fontsize, color=color, 
                         transform=self.transAxes, **kwargs)

    def text_upper_right(self, s, dx=0, dy=0, fontsize=7.5, color='k', **kwargs):
        return self.text(0.97+dx, 0.97+dy, s, ha='right', va='top',
                         fontsize=fontsize, color=color, 
                         transform=self.transAxes, **kwargs)

    #/////////////////////////////////////////////////////////////////////////////////////

    def hline(self, y, **kwargs):
        kwargs.setdefault('zorder', 0)
        kwargs.setdefault('color', '0.2')

        return self.plot(self.get_xlim(), 2*[y], **kwargs)

    def vline(self, x, **kwargs):
        kwargs.setdefault('zorder', 0)
        kwargs.setdefault('color', '0.2')

        return self.plot(2*[x], self.get_ylim(), **kwargs)

    def circle(self, center, r, **kwargs):
        kwargs.setdefault('zorder', 0)
        circle = mpl.patches.Circle(center, r, **kwargs)

        return self.add_patch(circle)

    def highlight(self, x1, x2, **kwargs):
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()

        kwargs.setdefault('facecolor', '0.9')
        kwargs.setdefault('edgecolor', 'none')
        kwargs.setdefault('alpha',     0.7)
        kwargs.setdefault('zorder',    1)
        fill = self.fill_between([x1, x2], ymin*np.ones(2), ymax*np.ones(2), **kwargs)

        # Restore axis limits
        self.xlim(xmin, xmax)
        self.ylim(ymin, ymax)

        return fill

    #/////////////////////////////////////////////////////////////////////////////////////

    def lim(self, axis, data, lower=None, upper=None, margin=0.05, relative=True):
        if isinstance(data[0], (tuple, list)):
            data = np.concatenate(data)

        amin = min(data)
        amax = max(data)

        # Add margin
        if relative:
            da = margin*(amax - amin)
        else:
            da = margin

        amin -= da
        amax += da

        # Bounds
        if lower is not None:
            amin = lower
        if upper is not None:
            amax = upper

        # Set
        alim = amin, amax
        if axis == 'x':
            self.xlim(*alim)
        elif axis == 'y':
            self.ylim(*alim)
        else:
            raise ValueError("Invalid axis.")

        return alim

    #/////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def sturges(data):
        return int(np.ceil(np.log2(len(data)) + 1))

    @staticmethod
    def scott(data, ddof=0):
        if np.std(data, ddof=ddof) == 0:
            return sturges_rule(data)

        h = 3.5*np.std(data, ddof=ddof)/cbrt(len(data))
        
        return int((np.max(data) - np.min(data))/h)

    def hist(self, data, bins=None, get_binedges=False, **kwargs):
        defaults = {
            'color':    Figure.colors('blue'),
            'normed':   True,
            'rwidth':   1,
            'histtype': 'stepfilled'
            }
        
        # Fill parameters
        for k in defaults:
            kwargs.setdefault(k, defaults[k])

        # Determine number of bins
        if bins is None:
            if len(data) < 200:
                bins = Subplot.sturges(data)
            else:
                bins = Subplot.scott(data)

        # Plot histogram
        pdf, binedges, patches = self.ax.hist(data, bins, **kwargs)

        # Modify appearance
        plt.setp(patches, 'facecolor', kwargs['color'], 'linewidth', 0)
        
        if get_binedges:
            return pdf, binedges
        return pdf

#=========================================================================================
# Subplot (3D)
#=========================================================================================

class Subplot3D(object):
    def __getattr__(self, name):
        if hasattr(self.ax, name):
            return getattr(self.ax, name)
        if hasattr(self.ax, 'set_'+name):
            return getattr(self.ax, 'set_'+name)
        raise NotImplementedError("Subplot3D." + name)

    def __init__(self, fig, p, rect):
        self.p     = p
        self.ax    = fig.add_axes(rect, projection='3d')
    
    #/////////////////////////////////////////////////////////////////////////////////////

    def xlabel(self, *args, **kwargs):
        kwargs.setdefault('fontsize', self.p['axislabelsize'])
        kwargs.setdefault('labelpad', self.p['labelpadx'])

        return self.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        kwargs.setdefault('fontsize', self.p['axislabelsize'])
        kwargs.setdefault('labelpad', self.p['labelpady'])

        return self.set_ylabel(*args, **kwargs)

    def zlabel(self, *args, **kwargs):
        kwargs.setdefault('fontsize', self.p['axislabelsize'])
        kwargs.setdefault('labelpad', self.p['labelpadz'])

        return self.set_zlabel(*args, **kwargs)

    def xticks(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_xticks(*args, **kwargs)

    def yticks(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_yticks(*args, **kwargs)

    def zticks(self, *args, **kwargs):
        if len(args) == 0:
            args = [],

        return self.set_zticks(*args, **kwargs)

    #/////////////////////////////////////////////////////////////////////////////////////

    def zcircle(self, center, r, z0, **kwargs):
        circle = mpl.patches.Circle(center, r, **kwargs)
        patch  = self.add_patch(circle)
        art3d.pathpatch_2d_to_3d(patch, z=z0)

#=========================================================================================
# Figure
#=========================================================================================

class Figure(object):
    defaults = {
        'w':             6.5,
        'h':             5.5,
        'rect':          [0.12, 0.12, 0.8, 0.8],
        'thickness':     0.7,
        'ticksize':      3.5,
        'axislabelsize': 8.5,
        'ticklabelsize': 7,
        'ticklabelpad':  2.5,
        'labelpadx':     7.5,
        'labelpady':     8.5,
        'labelpadz':     7.5,
        'format':        'pdf'
        }

    @staticmethod
    def colors(name):
        if name in colors:
            return colors[name]
        return name

    #/////////////////////////////////////////////////////////////////////////////////////

    def __getattr__(self, name):
        if hasattr(self.fig, name):
            return getattr(self.fig, name)
        raise NotImplementedError("Figure." + name)

    def __init__(self, **kwargs):
        self.p = kwargs.copy()
        for k in Figure.defaults:
            self.p.setdefault(k, Figure.defaults[k])

        self.fig   = plt.figure(figsize=(self.p['w'], self.p['h']))
        self.plots = []

    #/////////////////////////////////////////////////////////////////////////////////////

    def add(self, rect=None, style='bottomleft', projection=None, **kwargs):
        if rect is None:
            rect = self.p['rect']

        # Override figure defaults for this subplot
        p = self.p.copy()
        for k, v in kwargs.items():
            if k in p:
                p[k] = v

        # 3D subplot
        if projection == '3d':
            return Subplot3D(self.fig, p, rect)

        plot = Subplot(self.fig, p, rect)
        if style is not None:
            plot.format(style, p)

        # List of plots in this figure
        self.plots.append(plot)

        return plot

    #/////////////////////////////////////////////////////////////////////////////////////

    def plotlabels(self, labels, **kwargs):
        plot = self.plots[0]
        for label, (x, y) in labels.items():
            plot.text(x, y, label, ha='left', va='bottom', 
                      transform=self.transFigure, **kwargs)

    #/////////////////////////////////////////////////////////////////////////////////////

    def shared_lim(self, plots, axis, data, **kwargs):
        """
        Make the axis scale the same in all the plots.
        
        """
        try:
            data = np.concatenate(data)
        except:
            pass
        data = np.ravel(data)

        for plot in plots:
            lim = plot.lim(axis, data, **kwargs)
                
        return lim

    #/////////////////////////////////////////////////////////////////////////////////////

    def save(self, name=None, path=None, fmt=None, transparent=True, **kwargs):
        if path is None:
            path = os.path.dirname(os.path.realpath(sys.argv[0])) + '/work/figs'
            mkdir_p(path)
        if name is None:
            name = os.path.splitext(sys.argv[0].split('/')[-1])[0]

        fname = join(path, name + '.' + self.p['format'])
        plt.figure(self.fig.number)
        plt.savefig(fname, transparent=transparent, **kwargs)
        print("[ {}.Figure.save ] ".format(THIS) + fname)

    def close(self):
        plt.close(self.fig)
