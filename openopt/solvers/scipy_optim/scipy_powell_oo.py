from scipy.optimize import fmin_powell
from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.ooMisc import isSolved

class scipy_powell(baseSolver):
    __name__ = 'scipy_powell'
    __license__ = "BSD"
    #__authors__ =
    __alg__ = ""
    __info__ = 'unconstrained NLP solver, cannot handle user-supplied gradient'
    iterfcnConnected = True

    def __init__(self):pass
    def __solver__(self, p):

        def iterfcn(x):
            p.xk, p.fk = x, p.f(x)
            p.iterfcn()
            iter = p.iter - 1
            if p.istop: raise isSolved

        try:
            iterfcn(p.x0)
            xf = fmin_powell(p.f, p.x0, xtol=p.xtol, ftol = p.ftol, disp = 0, maxiter=p.maxIter, maxfun = p.maxFunEvals, callback=iterfcn)
        except isSolved:
            xf = p.xk

        ff = p.f(p.xk)

        p.xk = p.xf = xf
        p.fk = p.ff = ff
        p.istop = 1000
        p.iterfcn()
