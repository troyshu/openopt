from scipy.optimize import fmin_slsqp
import openopt
from openopt.kernel.setDefaultIterFuncs import *
from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from openopt.kernel.baseSolver import baseSolver
from numpy import *

class EmptyClass: pass

class scipy_slsqp(baseSolver):
    __name__ = 'scipy_slsqp'
    __license__ = "BSD"
    __authors__ = """Dieter Kraft, connected to scipy by Rob Falck, connected to OO by Dmitrey"""
    __alg__ = "Sequential Least SQuares Programming"
    __info__ = 'constrained NLP solver'
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']

    def __init__(self): pass
    def __solver__(self, p):
        bounds = []
        if any(isfinite(p.lb)) or any(isfinite(p.ub)):
            ind_inf = where(p.lb==-inf)[0]
            p.lb[ind_inf] = -1e50
            ind_inf = where(p.ub==inf)[0]
            p.ub[ind_inf] = 1e50
            for i in range(p.n):
                bounds.append((p.lb[i], p.ub[i]))
        empty_arr = array(())
        empty_arr_n = array(()).reshape(0, p.n)
        if not p.userProvided.c:
            p.c = lambda x: empty_arr.copy()
            p.dc = lambda x: empty_arr_n.copy()
        if not p.userProvided.h:
            p.h = lambda x: empty_arr.copy()        
            p.dh = lambda x: empty_arr_n.copy()
        C = lambda x: -hstack((p.c(x), p.matmult(p.A, x) - p.b))
        fprime_ieqcons = lambda x: -vstack((p.dc(x), p.A))
        #else: C,  fprime_ieqcons = None,  None
        #if p.userProvided.h:
        H = lambda x: hstack((p.h(x), p.matmult(p.Aeq, x) - p.beq))
        fprime_eqcons = lambda x: vstack((p.dh(x), p.Aeq))
        #else: H,  fprime_eqcons = None,  None
       # fprime_cons = lambda x: vstack((p.dh(x), p.Aeq, p.dc(x), p.A))

        x, fx, its, imode, smode = fmin_slsqp(p.f, p.x0, bounds=bounds,  f_eqcons = H, f_ieqcons = C, full_output=1, iprint=-1, fprime = p.df, fprime_eqcons = fprime_eqcons, fprime_ieqcons = fprime_ieqcons, acc = p.contol, iter = p.maxIter)
        p.msg = smode
        if imode == 0: p.istop = 1000
        #elif imode == 9: p.istop = ? CHECKME that OO kernel is capable of handling the case
        else: p.istop = -1000
        p.xk, p.fk = array(x), fx

