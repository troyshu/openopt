from openopt.kernel.baseSolver import baseSolver
from CVXOPT_SOCP_Solver import CVXOPT_SOCP_Solver

class cvxopt_socp(baseSolver):
    __name__ = 'cvxopt_socp'
    __license__ = "LGPL"
    __authors__ = "http://abel.ee.ucla.edu/cvxopt"
    __alg__ = "see http://abel.ee.ucla.edu/cvxopt"
    __optionalDataThatCanBeHandled__ = ['A','b','Aeq', 'beq', 'lb', 'ub']
    properTextOutput = True
    _canHandleScipySparse = True
    def __init__(self): pass
    def __solver__(self, p):
        return CVXOPT_SOCP_Solver(p, 'native_CVXOPT_SOCP_Solver')

