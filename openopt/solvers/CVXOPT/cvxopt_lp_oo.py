from openopt.kernel.baseSolver import baseSolver
from CVXOPT_LP_Solver import CVXOPT_LP_Solver

class cvxopt_lp(baseSolver):
    __name__ = 'cvxopt_lp'
    __license__ = "LGPL"
    __authors__ = "http://abel.ee.ucla.edu/cvxopt"
    __alg__ = "see http://abel.ee.ucla.edu/cvxopt"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    properTextOutput = True
    _canHandleScipySparse = True
    def __init__(self): pass
    def __solver__(self, p):
        return CVXOPT_LP_Solver(p, 'native_CVXOPT_LP_Solver')
