from scipy.optimize import fsolve
from numpy import asfarray
from openopt.kernel.baseSolver import baseSolver


class scipy_fsolve(baseSolver):
    __name__ = 'scipy_fsolve'
    __license__ = "BSD"
    #__authors__ =
    #__alg__ = ""
    __info__ = """
    solves system of n non-linear equations with n variables.
    """

    def __init__(self):pass
    def __solver__(self, p):
        #xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals, warning = (p.iprint>=0))
        # "warning" has been removed in latest scipy version
        xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals)
        p.istop = 1000
        p.iterfcn(xf)


