from openopt.kernel.baseSolver import baseSolver
from CVXOPT_QP_Solver import CVXOPT_QP_Solver

class cvxopt_qp(baseSolver):
    __name__ = 'cvxopt_qp'
    __license__ = "LGPL"
    __authors__ = "http://abel.ee.ucla.edu/cvxopt"
    __alg__ = "see http://abel.ee.ucla.edu/cvxopt"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    properTextOutput = True
    _canHandleScipySparse = True
    def __init__(self): pass
    def __solver__(self, p):
        return CVXOPT_QP_Solver(p, 'native_CVXOPT_QP_Solver')
