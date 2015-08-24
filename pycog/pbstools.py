import os

from utils import mkdir_p

def write_jobfile(cmd, jobname, pbspath, scratchpath, nodes=1, ppn=1, gpus=0, mem=4,
                  ndays=1, queue='s48'):
    """
    Create a job file.
    
    Args
    ----
    cmd : str
          Command to execute.

    Returns
    -------

    jobfile : str

    """
    cores = ''
    if gpus > 0:
        ppn   = gpus
        cores = ':gpus={}:titan'.format(gpus)

    threads = ''
    if ppn > 1:
        threads = '#PBS -v OMP_NUM_THREADS={}\n'.format(ppn)

    mkdir_p(pbspath)
    jobfile = '{}/{}.pbs'.format(pbspath, jobname)

    with open(jobfile, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '\n'
            + '#PBS -l nodes={}:ppn={}{}\n'.format(nodes, ppn, cores)
            + '#PBS -l mem={}GB\n'.format(mem)
            + '#PBS -l walltime={}:00:00\n'.format(24*ndays)
            + '#PBS -q {}\n'.format(queue)
            + '#PBS -N {}\n'.format(jobname[0:16])
            + '#PBS -e localhost:{}/${{PBS_JOBNAME}}.e${{PBS_JOBID}}\n'.format(scratchpath)
            + '#PBS -o localhost:{}/${{PBS_JOBNAME}}.o${{PBS_JOBID}}\n'.format(scratchpath)
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
