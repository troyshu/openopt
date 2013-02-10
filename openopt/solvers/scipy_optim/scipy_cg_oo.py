from scipy.optimize import fmin_cg
from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.ooMisc import isSolved

class scipy_cg(baseSolver):
    __name__ = 'scipy_cg'
    __license__ = "BSD"
    #__authors__ =
    __alg__ = "nonlinear conjugate gradient algorithm of Polak and Ribiere See Wright, and Nocedal 'Numerical Optimization', 1999, pg. 120-122"
    __info__ = 'unconstrained NLP solver'
    iterfcnConnected = True

    def __init__(self):pass
    def __solver__(self, p):

        def iterfcn(x):
            p.xk, p.fk = x, p.f(x)
            p.iterfcn()
            if p.istop: raise isSolved

        try:
            xf = fmin_cg(p.f, p.x0, fprime=p.df, gtol=p.gtol, disp = 0, maxiter=p.maxIter, callback=iterfcn)
        except isSolved:
            xf = p.xk

        ff = p.f(xf)
        p.xk = p.xf = xf
        p.fk = p.ff = ff
        p.istop = 1000
        #p.iterfcn()

