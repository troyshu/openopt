from scipy.optimize import fminbound
from openopt.kernel.baseSolver import baseSolver

class scipy_fminbound(baseSolver):
    __name__ = 'scipy_fminbound'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __license__ = "BSD"
    __authors__ = "Travis E. Oliphant"
    __alg__ = "Brent's method (golden section + parabolic fit)"
    __info__ = '1-dimensional minimizer for finite box-bound problems'
    __isIterPointAlwaysFeasible__ = lambda self, p: True
    def __init__(self):pass
    def __solver__(self, p):
        if p.n != 1: p.err('the solver ' + self.__name__ +' can handle singe-variable problems only')
        if not p.__isFiniteBoxBounded__(): p.err('the solver ' + self.__name__ +' requires finite lower and upper bounds')
        xf, ff, ierr, numfunc= fminbound(p.f, p.lb, p.ub, disp = 0, xtol = 0.999*p.xtol, maxfun = p.maxFunEvals, full_output = 1)

        p.xk = p.xf = xf
        p.fk = p.ff = ff
        if ierr==0: p.istop = 1000
        p.iterfcn()

