from __future__ import absolute_import

import os

from .utils import mkdir_p

def write_jobfile(cmd, jobname, pbspath, scratchpath, 
                  nodes=1, ppn=1, gpus=0, mem=4, ndays=1, queue=''):
    """
    Create a job file.
    
    Parameters
    ----------

    cmd : str
          Command to execute.

    jobname : str
              Name of the job.

    pbspath : str
              Directory to store PBS file in.

    scratchpath : str
                  Directory to store output files in.

    nodes : int, optional
            Number of compute nodes.

    ppn : int, optional
          Number of processors per node.

    gpus : int, optional
           Number of GPU cores.

    mem : int, optional
          Amount, in GB, of memory.

    ndays : int, optional
            Running time, in days.

    queue : str, optional
            Queue name.

    Returns
    -------

    jobfile : str
              Path to the job file.

    """
    if gpus > 0:
        gpus = ':gpus={}:titan'.format(gpus)
    else:
        gpus = ''

    if queue != '':
        queue = '#PBS -q {}\n'.format(queue)

    if ppn > 1:
        threads = '#PBS -v OMP_NUM_THREADS={}\n'.format(ppn)
    else:
        threads = ''

    mkdir_p(pbspath)
    jobfile = '{}/{}.pbs'.format(pbspath, jobname)

    with open(jobfile, 'w') as f:
        f.write(
            '#! /bin/bash\n'
            + '\n'
            + '#PBS -l nodes={}:ppn={}{}\n'.format(nodes, ppn, gpus)
            + '#PBS -l mem={}GB\n'.format(mem)
            + '#PBS -l walltime={}:00:00\n'.format(24*ndays)
            + queue
            + '#PBS -N {}\n'.format(jobname[0:16])
            + ('#PBS -e localhost:{}/${{PBS_JOBNAME}}.e${{PBS_JOBID}}\n'
               .format(scratchpath))
            + ('#PBS -o localhost:{}/${{PBS_JOBNAME}}.o${{PBS_JOBID}}\n'
               .format(scratchpath))
            + threads
            + '\n'
            + 'cd {}\n'.format(scratchpath)
            + 'pwd > {}.log\n'.format(jobname)
            + 'date >> {}.log\n'.format(jobname)
            + 'which python >> {}.log\n'.format(jobname)
            + '{} >> {}.log 2>&1\n'.format(cmd, jobname)
            + '\n'
            + 'exit 0;\n'
            )

    return jobfile
