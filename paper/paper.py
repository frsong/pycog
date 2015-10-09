import os
from   os.path import join

scratchroot   = os.environ.get('SCRATCH', join(os.path.expanduser('~'), 'scratch'))
scratchpath   = join(scratchroot, 'work', 'examples')
plotlabelsize = 11
format        = 'pdf'
