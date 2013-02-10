#! /usr/bin/env python

#from .ooVersionNumber import __version__

import os, sys
curr_dir = ''.join([elem + os.sep for elem in __file__.split(os.sep)[:-1]])
sys.path += [curr_dir, curr_dir + 'kernel']

from ooVersionNumber import __version__
from oo import *

#from kernel.GUI import manage
#from kernel.oologfcn import OpenOptException
#from kernel.nonOptMisc import oosolver

from GUI import manage
from oologfcn import OpenOptException
from nonOptMisc import oosolver
from mfa import MFA


isE = False
try:
    import enthought
    isE = True
except ImportError:
    pass
try:
    import envisage
    import mayavi
    isE = True
except ImportError:
    pass
try:
    import xy
    isE = False
except ImportError:
    pass
  
if isE:
    s = """
    Seems like you are using OpenOpt from 
    commercial Enthought Python Distribution;
    consider using free GPL-licensed alternatives
    PythonXY (http://www.pythonxy.com) or
    Sage (http://sagemath.org) instead.
    """
    print(s)

    
#__all__ = filter(lambda s:not s.startswith('_'),dir())

#from numpy.testing import NumpyTest
#test = NumpyTest().test


