from numpy import asfarray, argmax, sign, inf, log10, dot
from openopt.kernel.ooMisc import norm
from openopt.kernel.baseSolver import baseSolver
from openopt import NSP
from openopt.kernel.setDefaultIterFuncs import IS_MAX_FUN_EVALS_REACHED, FVAL_IS_ENOUGH, SMALL_DELTA_X, SMALL_DELTA_F

class nssolve(baseSolver):
    __name__ = 'nssolve'
    __license__ = "BSD"
    __authors__ = 'Dmitrey Kroshko'
    __alg__ = "based on Naum Z. Shor r-alg"
    iterfcnConnected = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    __isIterPointAlwaysFeasible__ = lambda self, p: p.isUC
    
    nspSolver = 'autoselect'
    __info__ = """
    Solves system of non-smooth or noisy equations
    via (by default) minimizing max residual using NSP solver (default UkrOpt.ralg).

    Can handle user-supplied gradient/subradient (p.df field)
    If the one is not available -
    splitting equations to separate functions is recommended
    (to speedup calculations):
    f = [func1, func2, ...] or f = ([func1, func2, ...)

    ns- can be interpreted as
    NonSmooth
    or NoiSy
    or Naum Shor (Ukrainian academician, my teacher, r-algorithm inventor)
    """

    def __init__(self):pass
    def __solver__(self, p):
        if SMALL_DELTA_X in p.kernelIterFuncs.keys(): p.kernelIterFuncs.pop(SMALL_DELTA_X)
        if SMALL_DELTA_F in p.kernelIterFuncs.keys(): p.kernelIterFuncs.pop(SMALL_DELTA_F)

        if self.nspSolver == 'autoselect':
            nspSolver = 'amsg2p' if p.isUC else 'ralg'
        else:
            nspSolver = self.nspSolver
        
#        nspSolver = 'ralg'

        way = 3 if nspSolver == 'ralg' else 2
       
        if way == 1:
            use2 = False
            f = lambda x: sum(abs(p.f(x)))
            def df(x):
                return dot(p.df(x),  sign(p.f(x)))
        elif way == 2:
            use2 = True
            f = lambda x: sum(p.f(x)**2)
            def df(x):
                return 2.0*dot(p.f(x),  p.df(x))
        elif way == 3:
            use2 = False
            f = lambda x: max(abs(p.f(x)))
            def df(x):
                F = p.f(x)
                ind = argmax(abs(F))
                return p.df(x, ind) * sign(F[ind])

        FTOL = p.ftol**2 if use2 else p.ftol
        def iterfcn(*args,  **kwargs):
            p2.primalIterFcn(*args,  **kwargs)
            p.xk = p2.xk.copy()
            Fk = norm(p.f(p.xk), inf)
            p.rk = p.getMaxResidual(p.xk)

            #TODO: ADD p.rk

            if p.nEvals['f'] > p.maxFunEvals:
                p.istop = p2.istop = IS_MAX_FUN_EVALS_REACHED
            elif p2.istop!=0:
                if Fk < FTOL and p.rk < p.contol:
                    p.istop = 15
                    msg_contol = '' if p.isUC else 'and contol '
                    p.msg = 'solution with required ftol ' + msg_contol+ 'has been reached'
                else: p.istop = p2.istop

            p.iterfcn()
            
            return p.istop


        p2 = NSP(f, p.x0, df=df, xtol = p.xtol/1e16, gtol = p.gtol/1e16,\
        A=p.A,  b=p.b,  Aeq=p.Aeq,  beq=p.beq,  lb=p.lb,  ub=p.ub, \
        maxFunEvals = p.maxFunEvals, fEnough = FTOL, maxIter=p.maxIter, iprint = -1, \
        maxtime = p.maxTime, maxCPUTime = p.maxCPUTime,  noise = p.noise, fOpt = 0.0)

        if p.userProvided.c:
            p2.c,  p2.dc = p.c,  p.dc
        if p.userProvided.h:
            p2.h,  p2.dh = p.h,  p.dh

        p2.primalIterFcn,  p2.iterfcn = p2.iterfcn, iterfcn

        if p.debug: p2.iprint = 1
        
        if nspSolver == 'ralg':
            if p.isUC: p2.ftol = p.ftol / 1e16
            else: p2.ftol = 0.0
        else:
            p2.ftol = 0.0
            p2.xtol = 0.0
            p2.gtol = 0.0
        if use2:
            p2.fTol = 0.5*p.ftol ** 2
        else:
            p2.fTol = 0.5*p.ftol
        r2 = p2.solve(nspSolver)
        #xf = fsolve(p.f, p.x0, fprime=p.df, xtol = p.xtol, maxfev = p.maxFunEvals)
        xf = r2.xf
        p.xk = p.xf = xf
        p.fk = p.ff = asfarray(norm(p.f(xf), inf)).flatten()
        #p.istop is defined in iterfcn
