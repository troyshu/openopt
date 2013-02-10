from openopt.kernel.baseSolver import baseSolver
from openopt import OpenOptException

class glpk(baseSolver):
    __name__ = 'glpk'
    __license__ = "GPL v.2"
    __authors__ = "http://www.gnu.org/software/glpk + Python bindings from http://abel.ee.ucla.edu/cvxopt"
    __homepage__ = 'http://www.gnu.org/software/glpk'
    #__alg__ = ""
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'binVars']
    _canHandleScipySparse = True
    
    def __init__(self): 
        try:
            import cvxopt
        except ImportError:
            raise OpenOptException('for solver glpk cvxopt is required, but it was not found')

    def __solver__(self, p):
        from CVXOPT_LP_Solver import CVXOPT_LP_Solver
        return CVXOPT_LP_Solver(p, 'glpk')
