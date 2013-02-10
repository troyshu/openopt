from scipy.optimize import broyden2
from numpy import asfarray
from openopt.kernel.baseSolver import baseSolver

class scipy_broyden2(baseSolver):
    __name__ = 'scipy_broyden2'
    __license__ = "BSD"
    #__authors__ = 
    __alg__ = "a quasi-Newton-Raphson method, updates the inverse Jacobian directly"  
    __info__ = """
    solves system of n non-linear equations with n variables. 
    """

    def __init__(self):pass
    def __solver__(self, p):
        
        p.xk = p.x0.copy()
        p.fk = asfarray(max(abs(p.f(p.x0)))).flatten()
        
        p.iterfcn()
        if p.istop:
            p.xf, p.ff = p.xk, p.fk
            return 
        
        try: xf = broyden2(p.f, p.x0, iter = p.maxIter)
        except: 
            p.istop = -1000
            return

        p.xk = p.xf = asfarray(xf)
        p.fk = p.ff = asfarray(max(abs(p.f(xf)))).flatten()
        p.istop = 1000
        p.iterfcn()
