from openopt.kernel.baseSolver import baseSolver
import asa as ASA
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F

class asa(baseSolver):
    __name__ = 'asa'
    __license__ = "BSD"
    __authors__ = 'Lester Ingber, connected to Python by Robert Jordens, connected to OO by Dmitrey'
    __alg__ = "Adaptive Simulated Annealing"
    iterfcnConnected = False
    _requiresFiniteBoxBounds = True
    __homepage__ = 'http://pypi.python.org/pypi/pyasa'

    __info__ = "non-default parameters are not implemented yet"
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = True#lambda self, p: p.__isNoMoreThanBoxBounded__()
    funcForIterFcnConnection = 'f'
    f_iter = 10000000

    def __init__(self):pass
    def __solver__(self, p):

        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)

        xf, ff, code, opt_struct = ASA.asa(p.f, p.x0, p.lb, p.ub, full_output=True)
        if p.istop == 0: p.istop = 1000


