from numpy import isfinite, any, hstack
from openopt.kernel.baseSolver import *
from sqlcp import sqlcp as SQLCP
from numpy.linalg import LinAlgError
from LCPSolve import LCPSolve

class lcpsolve(baseSolver):
    __name__ = 'lcp'
    __license__ = "MIT"
    __authors__ = "Rob Dittmar, Enzo Michelangeli and IT Vision Ltd"
    __alg__ = "Lemke's Complementary Pivot algorithm"
    __optionalDataThatCanBeHandled__ = []
    #iterfcnConnected = True
    #_canHandleScipySparse = True
    __info__ = '''  '''
    pivtol = 1e-8

    def __init__(self): pass
    def __solver__(self, p):
        w, z, retcode = LCPSolve(p.M,p.q, pivtol=self.pivtol)
        p.xf = hstack((w, z))
        if retcode[0] == 1:
            p.istop = 1000
            p.msg = 'success'
        elif retcode[0] == 2:
            p.istop = -1000
            p.msg = 'ray termination'
