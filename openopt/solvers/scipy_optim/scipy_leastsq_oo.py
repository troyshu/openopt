from scipy.optimize import leastsq
from numpy import asfarray
from openopt.kernel.baseSolver import baseSolver


class scipy_leastsq(baseSolver):
    __name__ = 'scipy_leastsq'
    __license__ = "BSD"
    #__authors__ = 
    #__alg__ = ""
    __info__ = """
    MINPACK's lmdif and lmder algorithms
    """
    #__constraintsThatCannotBeHandled__ = (all)
    
    def __init__(self): pass
    def __solver__(self, p):
        
        p.xk = p.x0.copy()
        p.fk = asfarray((p.f(p.x0)) ** 2).sum().flatten()
            
        p.iterfcn()
        if p.istop:
            p.xf, p.ff = p.xk, p.fk
            return 
        
        if p.userProvided.df:
            xf, cov_x, infodict, mesg, ier = leastsq(p.f, p.x0, Dfun=p.df, xtol = p.xtol, ftol = p.ftol, maxfev = p.maxFunEvals, full_output = 1)
        else:
            xf, cov_x, infodict, mesg, ier = leastsq(p.f, p.x0, xtol = p.xtol, maxfev = p.maxFunEvals, epsfcn = p.diffInt, ftol = p.ftol, full_output = 1)
        
        if ier == 1: p.istop = 1000
        else: p.istop = -1000
        p.msg = mesg
            
        ff = asfarray((p.f(xf)) ** 2).sum().flatten()
        p.xk = xf
        p.fk = ff
        
        p.xf = xf
        p.ff = ff        
        p.iterfcn()
        


