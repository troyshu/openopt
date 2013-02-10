from openopt.kernel.baseSolver import baseSolver
from CVXOPT_SDP_Solver import CVXOPT_SDP_Solver

class cvxopt_sdp(baseSolver):
    __name__ = 'cvxopt_sdp'
    __license__ = "LGPL"
    __authors__ = "http://abel.ee.ucla.edu/cvxopt"
    __alg__ = "see http://abel.ee.ucla.edu/cvxopt"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'S', 'd']
    properTextOutput = True
    _canHandleScipySparse = True
    def __init__(self): pass
    def __solver__(self, p):
        return CVXOPT_SDP_Solver(p, 'native_CVXOPT_SDP_Solver')

