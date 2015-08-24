"""
Analyze variants of the random dots motion task with target input and saccade output.

"""
from __future__ import division

import cPickle as pickle
import sys

import numpy as np

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure

#=========================================================================================
# Setup
#=========================================================================================

# For messages
THIS = "examples.analysis.rdm_saccade"

# File to store trials in
def get_trialsfile(p):
    return '{}/{}_trials.pkl'.format(p['trialspath'], p['name'])

# Load trials
def load_trials(trialsfile):
    with open(trialsfile) as f:
        trials = pickle.load(f)

    return trials, len(trials)

# File to store sorted trials in
def get_sortedfile_stim_onset(p):
    return '{}/{}_sorted_stim_onset.pkl'.format(p['trialspath'], p['name'])

# File to store sorted trials in
def get_sortedfile_response(p):
    return '{}/{}_sorted_response.pkl'.format(p['trialspath'], p['name'])

# File to store selectivity
def get_selectivityfile(p):
    return '{}/{}_selectivity.pkl'.format(p['datapath'], p['name'])

# Valid saccade radius
r_saccade = 0.5
r_valid   = 0.1

# Choice function
def get_choice(trial):
    # Decision epoch
    e = trial['info']['epochs']['decision']
    decision, = np.where((e[0] <= t) & (t <= e[1]))

    # Tin
    T = tasktools.deg2rad(info['Tin'])
    C = np.tile(r_saccade*np.array([np.cos(T), np.sin(T)]), (len(decision), 1)).T
    saccaded_to_Tin = np.sum(1*(np.sum((z[:,decision] - C)**2), axis=0) <= r_valid**2)

    # Tout
    T = tasktools.deg2rad(info['Tout'])
    C = np.tile(r_saccade*np.array([np.cos(T), np.sin(T)]), (len(decision), 1)).T
    saccaded_to_Tin = np.sum(1*(np.sum((z[:,decision] - C)**2), axis=0) <= r_valid**2)

    # Invalid trial
    if saccaded_to_Tin > 0 and saccaded_to_Tout > 0:
        return 'both'
    elif saccaded_to_Tin == 0 and saccaded_to_Tout == 0:
        return 'neither'

    # Choice
    if saccaded_to_Tin > 0:
        return 0
    return 1

def safe_divide(x):
    if x == 0:
        return 0
    return 1/x

# Nice colors to represent coherences
colors = {
        0:  '#c6dbef',
        1:  '#9ecae1',
        2:  '#6baed6',
        4:  '#4292c6',
        8:  '#2171b5',
        16: '#084594'
        }

# Integration threshold
threshold = 0.8

#=========================================================================================

def run_trials(p, args):
    """
    Run trials.

    """
    # Model
    m = p['model']

    # Number of trials
    try:
        ntrials = int(args[0])
    except:
        ntrials = 100*(m.nconditions + 1)

    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in xrange(ntrials):
            b = i % (m.nconditions + 1)
            if b == 0:
                # Zero-coherence condition
                coh    = 0
                in_out = rng.choice(m.in_outs)
            else:
                # All other conditions
                k0, k1 = tasktools.unravel_index(b-1, (len(m.cohs), len(m.in_outs)))
                coh    = m.cohs[k0]
                in_out = m.in_outs[k1]

            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':   'test',
                'catch':  False, 
                'coh':    coh, 
                'in_out': in_out
                }
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            if coh == 0:
                s = "Trial {:>{}}/{}: {:>3}".format(i+1, w, ntrials, info['coh'])
            else:
                s = ("Trial {:>{}}/{}: {:>+3}"
                     .format(i+1, w, ntrials, info['in_out']*info['coh']))
            sys.stdout.write(backspaces*'\b' + s)
            sys.stdout.flush()
            backspaces = len(s)

            # Save
            dt    = rnn.t[1] - rnn.t[0]
            step  = int(p['dt_save']/dt)
            trial = {
                't':    rnn.t[::step],
                'u':    rnn.u[:,::step],
                'r':    rnn.r[:,::step],
                'z':    rnn.z[:,::step],
                'info': info
                }
            trials.append(trial)
    except KeyboardInterrupt:
        pass
    print("")

    # Save all
    filename = get_trialsfile(p)
    with open(filename, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)
    print("[ {}.run_trials ] Trials saved to {}".format(THIS, filename))

    # Compute the psychometric function
    psychometric_function(filename)

#=========================================================================================

def psychometric_function(trialsfile, plot=None, **kwargs):
    """
    Compute and plot the sychometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric function
    #-------------------------------------------------------------------------------------

    choice_by_coh = {}
    ncorrect = 0
    nskip    = 0
    for i, trial in enumerate(trials):
        info = trial['info']
    
        # Signed coherence
        coh = info['in_out']*info['coh']

        # Choice
        choice = get_choice(trial)
        if isinstance(choice, str):
            if choice == 'both':
                print("[ {}.psychometric_function ]".format(THIS)
                      + "Skipping trial {}/{}: saccaded to both Tin and Tout."
                      .format(i+1, ntrials))
            elif choice == 'neither':
                print("[ {}.psychometric_function ]".format(THIS)
                      + "Skipping trial {}/{}: saccaded to neither Tin nor Tout."
                      .format(i+1, ntrials))
            nskip += 1
            continue
        choice_by_coh.setdefault(coh, []).append(choice)

        # Correct
        if choice == info['choice']:
            ncorrect += 1

    # Report overall performance
    pcorrect = 100*ncorrect/ntrials
    print("[ {}.psychometric_function ] {:.2f}% correct,".format(THIS, pcorrect)
          + " skipped {}/{} indeterminate trials.".format(nskip, ntrials))

    cohs = np.sort(np.array(choice_by_coh.keys()))
    p0   = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        choices = np.array(choice_by_coh[coh])
        p0[i]   = 1 - np.sum(choices)/len(choices)
    scaled_cohs = 3.2*cohs

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plot is not None:
        # Data
        prop = {'ms':     kwargs.get('ms',  6),
                'mfc':    kwargs.get('mfc', '0.2'),
                'mew':    kwargs.get('mew', 0),
                'zorder': 10}
        plot.plot(scaled_cohs, 100*p0, 'o', **prop)

        # Fit
        prop = {'color':  kwargs.get('mfc', '0.2'),
                'lw':     kwargs.get('lw',  1),
                'zorder': 5}
        try:
            popt, func = fittools.fit_psychometric(scaled_cohs, p0)

            fit_cohs = np.linspace(min(scaled_cohs), max(scaled_cohs), 201)
            fit_p0   = func(fit_cohs, **popt)
            plot.plot(fit_cohs, 100*fit_p0, **prop)
        except RuntimeError:
            print("[ {}.psychometric_function ]".format(THIS)
                  + " Unable to fit, drawing a line through the points.")
            plot.plot(scaled_cohs, 100*p0, **prop)

        plot.ylim(0, 100)
        plot.yticks([0, 25, 50, 75, 100])
        plot.lim('x', scaled_cohs)

#=========================================================================================

def plot_stimulus_duration(trialsfile, plot, **kwargs):
    """
    Percent correct as a function of stimulus duration.

    """
    from pycog.datatools import partition

    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric performance by stimulus duration
    #-------------------------------------------------------------------------------------

    correct_duration_by_coh = {}
    for i, trial in enumerate(trials):
        info = trial['info']

        # Coherence
        coh = info['coh']
        if coh == 0:
            continue

        # Correct, stimulus duration
        correct = 1*(get_choice(trial) == info['choice'])
        correct_duration_by_coh.setdefault(coh, ([], []))[0].append(correct)
        correct_duration_by_coh[coh][1].append(np.ptp(info['epochs']['stimulus']))

        correct_by_coh = {}

    correct_by_coh = {}
    for coh, (correct, duration) in correct_duration_by_coh.items():
        Xbins, Ybins, Xedges, _ = partition(np.asarray(duration), np.asarray(correct),
                                            nbins=10)
        correct_by_coh[coh] = ((Xedges[:-1] + Xedges[1:])/2,
                               [100*np.sum(Ybin > 0)*safe_divide(len(Ybin))
                                for Ybin in Ybins])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    lineprop = {'lw':  kwargs.get('lw', 1)}
    dataprop = {'ms':  kwargs.get('ms', 6),
                'mew': kwargs.get('mew', 0)}

    cohs = sorted(correct_by_coh)
    xall = []
    for coh in cohs:
        stim, correct = correct_by_coh[coh]

        plot.plot(stim, correct, color=colors[coh], label='{}\%'.format(3.2*coh),
                  **lineprop)
        plot.plot(stim, correct, 'o', mfc=colors[coh], **dataprop)
        xall.append(stim)

    plot.lim('x', xall)
    plot.ylim(50, 100)

#=========================================================================================

def chronometric_function(trialsfile, plot, plot_dist=None, **kwargs):
    """
    Chronometric function.

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    #-------------------------------------------------------------------------------------
    # Compute psychometric performance by RT 
    #-------------------------------------------------------------------------------------

    correct_rt_by_coh = {}
    error_rt_by_coh   = {}
    n_below_threshold = 0
    ntot = 0
    for trial in trials:
        info   = trial['info']
        choice = get_choice(trial)
        coh    = info['coh']

        # Reaction time
        w, = np.where(trial['z'][choice] > threshold)
        if len(w) == 0:
            n_below_threshold += 1
            continue
        rt = trial['t'][w[0]] - info['epochs']['stimulus'][0]

        if choice == info['choice']:
            correct_rt_by_coh.setdefault(coh, []).append(rt)
        else:
            error_rt_by_coh.setdefault(coh, []).append(rt)
    print("[ {}.chronometric_function ] {}/{} trials did not reach threshold."
          .format(THIS, n_below_threshold, ntrials))

    # Correct trials
    cohs = np.sort(np.array(correct_rt_by_coh.keys()))
    cohs = cohs[np.where(cohs > 0)]
    correct_rt = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        if len(correct_rt_by_coh[coh]) > 0:
            correct_rt[i] = np.mean(correct_rt_by_coh[coh])
    scaled_cohs = 3.2*cohs

    # Error trials
    cohs = np.sort(np.array(error_rt_by_coh.keys()))
    cohs = cohs[np.where(cohs > 0)]
    error_rt = np.zeros(len(cohs))
    for i, coh in enumerate(cohs):
        if len(error_rt_by_coh[coh]) > 0:
            error_rt[i] = np.mean(error_rt_by_coh[coh])
    error_scaled_cohs = 3.2*cohs

    #-------------------------------------------------------------------------------------
    # Plot RTs
    #-------------------------------------------------------------------------------------

    # Correct
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    plot.plot(scaled_cohs, correct_rt, **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6),
            'mfc':       '0.2',
            'mew':       0}
    plot.plot(scaled_cohs, correct_rt, **prop)

    # Error
    prop = {'color': kwargs.get('color', '0.2'),
            'lw':    kwargs.get('lw', 1)}
    plot.plot(error_scaled_cohs, error_rt, **prop)
    prop = {'marker':    'o',
            'linestyle': 'none',
            'ms':        kwargs.get('ms',  6) - 1,
            'mfc':       'w',
            'mec':       '0.2',
            'mew':       kwargs.get('mew', 1)}
    plot.plot(error_scaled_cohs, error_rt, **prop)

    plot.xscale('log')
    plot.xticks([1, 10, 100])
    plot.xticklabels([1, 10, 100])

    #plot.ylim(300, 1000)

    #-------------------------------------------------------------------------------------
    # Plot RT distribution
    #-------------------------------------------------------------------------------------

    correct_rt = []
    for rt in correct_rt_by_coh.values():
        correct_rt += rt

    error_rt = []
    for rt in error_rt_by_coh.values():
        error_rt += rt

    pdf = plot_dist.hist(correct_rt, color='0.2')
    plot_dist.lim('y', pdf, lower=0)

    print("[ {}.chronometric_function ]".format(THIS))
    print("  Mean RT, correct trials: {:.2f} ms".format(np.mean(correct_rt)))
    print("  Mean RT, error trials:   {:.2f} ms".format(np.mean(error_rt)))

#=========================================================================================

def sort_trials_stim_onset(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get unique conditions
    conds = []
    cohs  = []
    for trial in trials:
        info = trial['info']
        conds.append((info['coh'], info['in_out']))
        cohs.append(info['coh'])
    conds = list(set(conds))
    cohs  = list(set(cohs))

    #-------------------------------------------------------------------------------------
    # Prepare for averaging
    #-------------------------------------------------------------------------------------

    # Number of units
    nunits = trials[0]['r'].shape[0]
    
    # Number of time points
    stimulus = [np.ptp(trial['info']['epochs']['stimulus']) for trial in trials]
    idx      = np.argmax(stimulus)
    trial    = trials[idx]
    t        = trial['t']
    w        = np.where(t <= trial['info']['epochs']['stimulus'][1])[0][-1] + 1
    t        = t[:w] - trial['info']['epochs']['stimulus'][0]
    ntime    = len(t)

    #-------------------------------------------------------------------------------------
    # Average across conditions
    #-------------------------------------------------------------------------------------

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: np.zeros(ntime)           for c in conds}

    for trial in trials:
        info = trial['info']
        
        # Include only correct trials
        coh    = info['coh']
        choice = get_choice(trial)

        # Include only correct trials
        if choice != info['choice']:
            continue

        t_i = trial['t']
        w_i = np.where(t_i <= info['epochs']['stimulus'][1])[0][-1] + 1

        c = (info['coh'], info['in_out'])
        sorted_trials[c][:,:w_i] += trial['r'][:,:w_i]
        ntrials_by_cond[c][:w_i]  += 1
    for c in conds:
        sorted_trials[c] *= np.array([safe_divide(x) for x in ntrials_by_cond[c]])
    sorted_trials['t'] = t

    # Save
    with open(sortedfile, 'wb') as f:
        pickle.dump(sorted_trials, f, pickle.HIGHEST_PROTOCOL)
        print(("[ {}.sort_trials_stim_onset ]"
               " Trials sorted and aligned to stimulus onset, saved to {}")
              .format(THIS, sortedfile))

#=========================================================================================

def sort_trials_response(trialsfile, sortedfile):
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    # Get unique conditions
    conds = []
    cohs  = []
    for trial in trials:
        info = trial['info']
        conds.append((info['coh'], info['in_out']))
        cohs.append(info['coh'])
    conds = list(set(conds))
    cohs  = list(set(cohs))

    #-------------------------------------------------------------------------------------
    # Determine reaction time on each trial
    #-------------------------------------------------------------------------------------

    valid_trial_idx   = []
    valid_rt_idx      = []
    n_below_threshold = 0
    for i, trial in enumerate(trials):
        info   = trial['info']
        choice = get_choice(trial)

        # Reaction time
        w, = np.where(trial['z'][choice] > threshold)
        if len(w) == 0:
            n_below_threshold += 1
            continue

        # Include only correct trials
        if choice != info['choice']:
            continue
        
        valid_trial_idx.append(i)
        valid_rt_idx.append(w[0]+1)
    print("[ {}.sort_trials_response ] {}/{} trials did not reach threshold."
          .format(THIS, n_below_threshold, ntrials))

    #-------------------------------------------------------------------------------------
    # Prepare for averaging
    #-------------------------------------------------------------------------------------

    # Number of units
    nunits = trials[0]['r'].shape[0]
    
    # Number of time points
    idx    = np.argmax(valid_rt_idx)
    w      = valid_rt_idx[idx]
    t      = trials[valid_trial_idx[idx]]['t'][:w]
    t     -= t[-1]
    ntime  = len(t)

    #-------------------------------------------------------------------------------------
    # Average across conditions
    #-------------------------------------------------------------------------------------

    sorted_trials   = {c: np.zeros((nunits, ntime)) for c in conds}
    ntrials_by_cond = {c: np.zeros(ntime) for c in conds}
    for i, w in zip(valid_trial_idx, valid_rt_idx):
        trial = trials[i]
        info  = trial['info']

        c = (info['coh'], info['in_out'])
        sorted_trials[c][:,-w:] += trial['r'][:,:w]
        ntrials_by_cond[c][-w:] += 1
    for c in conds:
        sorted_trials[c] *= np.array([safe_divide(x) for x in ntrials_by_cond[c]])
    sorted_trials['t'] = t

    # Save
    with open(sortedfile, 'wb') as f:
        pickle.dump(sorted_trials, f, pickle.HIGHEST_PROTOCOL)
        print(("[ {}.sort_trials_response ]"
               " Trials sorted and aligned to response, saved to {}")
              .format(THIS, sortedfile))

#=========================================================================================

def plot_unit(unit, sortedfile, plot, t0=0, tmin=-np.inf, tmax=np.inf, **kwargs):
    # Load sorted trials
    with open(sortedfile) as f:
        sorted_trials = pickle.load(f)

    # Time
    t  = sorted_trials.pop('t')
    w, = np.where((tmin <= t) & (t <= tmax))
    t  = t[w] - t0

    conds = sorted_trials.keys()

    all = []
    for c in sorted(conds, key=lambda c: c[0]):
        coh, in_out = c

        prop = {'color': colors[coh],
                'lw':    kwargs.get('lw', 1.5)}

        if in_out == +1:
            prop['label'] = '{:.1f}\%'.format(3.2*coh)
        else:
            prop['linestyle'] = '--'
            prop['dashes'] = kwargs.get('dashes', [3.5, 2.5])

        r = sorted_trials[c][unit][w]
        plot.plot(t, r, **prop)
        all.append(r)

    plot.xlim(t[0], t[-1])
    plot.xticks([t[0], 0, t[-1]])
    plot.lim('y', all, lower=0)

    return np.concatenate(all)

#=========================================================================================

def get_choice_selectivity(trialsfile):
    """
    Get total firing rate during the stimulus period for 0 vs 1 trials

    """
    # Load trials
    trials, ntrials = load_trials(trialsfile)

    r_tot = np.zeros(trials[0]['r'].shape[0])
    for trial in trials:
        t = trial['t']
        start, end = trial['info']['epochs']['stimulus']
        stimulus,  = np.where((start < t) & (t <= end))

        choice = get_choice(trial)
        if choice == 0:
            sign = +1
        else:
            sign = -1
        r_tot += sign*np.sum(trial['r'][:,stimulus], axis=1)
    
    return r_tot

#=========================================================================================

def do(action, args, p):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #-------------------------------------------------------------------------------------
    # Trials
    #-------------------------------------------------------------------------------------

    if action == 'trials':
        run_trials(p, args)

    #-------------------------------------------------------------------------------------
    # Psychometric function
    #-------------------------------------------------------------------------------------

    elif action == 'psychometric':
        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------------------------

        trialsfile = get_trialsfile(p)
        psychometric_function(trialsfile, plot)

        plot.xlabel(r'\% coherence toward $T_\text{in}$')
        plot.ylabel(r'Percent $T_\text{in}$')

        #---------------------------------------------------------------------------------

        fig.save(path=p['figspath'], name=p['name']+'_'+action)
        fig.close()

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    elif action == 'sort_stim_onset':
        sort_trials_stim_onset(get_trialsfile(p), get_sortedfile_stim_onset(p))

    elif action == 'sort_response':
        sort_trials_response(get_trialsfile(p), get_sortedfile_response(p))

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to stimulus onset
    #-------------------------------------------------------------------------------------

    elif action == 'units_stim_onset':
        # Clear existing files
        #    fig.save(path=p['figspath'], 
        #             name=p['name']+'_stim_onset_unit{:03d}'.format(i))

        # Load sorted trials
        sortedfile = get_sortedfile_stim_onset(p)
        with open(sortedfile) as f:
            sorted_trials = pickle.load(f)

        for i in xrange(p['model'].N):
            # Check if the unit does anything
            active = False
            for condition_averaged in sorted_trials.values():
                if np.std(condition_averaged[i]) > 0.05:
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'], 
                     name=p['name']+'_stim_onset_unit{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to response
    #-------------------------------------------------------------------------------------

    elif action == 'units_response':
        # Load sorted trials
        sortedfile = get_sortedfile_response(p)
        with open(sortedfile) as f:
            sorted_trials = pickle.load(f)

        for i in xrange(p['model'].N):
            # Check if the unit does anything
            active = False
            for condition_averaged in sorted_trials.values():
                if np.std(condition_averaged[i]) > 0.05:
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'], 
                     name=p['name']+'_response_unit_{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Selectivity
    #-------------------------------------------------------------------------------------

    elif action == 'selectivity':
        m = p['model']
        
        trialsfile = get_trialsfile(p)
        r_tot = get_choice_selectivity(trialsfile)

        units = np.arange(r_tot.shape[0])
        try:
            idx = np.argsort(r_tot[m.EXC])[::-1]
            exc = list(units[m.EXC][idx])

            idx = np.argsort(r_tot[m.INH])[::-1]
            inh = list(units[m.INH][idx])

            units = exc + inh
        except:
            idx   = np.argsort(r_tot)[::-1]
            units = list(units[idx])

        # Save
        filename = get_selectivityfile(p)
        np.savetxt(filename, units, fmt='%d')
        print("[ {}.do ] Choice selectivity saved to {}".format(THIS, filename))

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action.".format(THIS))
