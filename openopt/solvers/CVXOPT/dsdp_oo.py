from openopt.kernel.baseSolver import baseSolver
from CVXOPT_SDP_Solver import CVXOPT_SDP_Solver

class dsdp(baseSolver):
    __name__ = 'dsdp'
    __license__ = "GPL"
    __authors__ = "Steven J. Benson and Yinyu Ye, Mathematics and Computer Science Division, Argonne National Laboratory, IL"
    __homepage__ = 'http://www-unix.mcs.anl.gov/DSDP/'
    #__alg__ = ""
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'S', 'd']
    gaptol = 1e-5
    properTextOutput = True
    _canHandleScipySparse = True
    
    def __init__(self): pass

    def __solver__(self, p):
        from cvxopt import solvers
        solvers.options['DSDP_Monitor'] = p.iprint
        solvers.options['DSDP_MaxIts'] = p.maxIter
        solvers.options['DSDP_GapTolerance'] = self.gaptol
        
        return CVXOPT_SDP_Solver(p, 'dsdp')
