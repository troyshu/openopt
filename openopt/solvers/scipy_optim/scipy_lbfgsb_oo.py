from scipy.optimize.lbfgsb import fmin_l_bfgs_b
import openopt
from openopt.kernel.setDefaultIterFuncs import *
from openopt.kernel.ooMisc import WholeRepr2LinConst
from openopt.kernel.baseSolver import baseSolver

class scipy_lbfgsb(baseSolver):
    __name__ = 'scipy_lbfgsb'
    __license__ = "BSD"
    __authors__ = """Ciyou Zhu, Richard Byrd, and Jorge Nocedal <nocedal@ece.nwu.edu>,
    connected to scipy by David M. Cooke <cookedm@physics.mcmaster.ca> and Travis Oliphant,
    connected to openopt by Dmitrey"""
    __alg__ = "l-bfgs-b"
    __info__ = 'box-bounded limited-memory NLP solver, can handle lb<=x<=ub constraints, some lb-ub coords can be +/- inf'#TODO: add '__info__' field to other solvers
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True

    def __init__(self):pass

    def __solver__(self, p):
        #WholeRepr2LinConst(p)#TODO: remove me

        bounds = []

        # don't work in Python ver < 2.5
        # BOUND = lambda x: x if isfinite(x) else None

        def BOUND(x):
            if isfinite(x): return x
            else: return None

        for i in range(p.n): bounds.append((BOUND(p.lb[i]), BOUND(p.ub[i])))

        xf, ff, d = fmin_l_bfgs_b(p.f, p.x0, fprime=p.df,
                  approx_grad=0,  bounds=bounds,
                  iprint=p.iprint, maxfun=p.maxFunEvals)

        if d['warnflag'] in (0, 2):
            # if 2 - some problems can be present, but final check from RunProbSolver will set negative istop if solution is unfeasible
            istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
            if d['warnflag'] == 0: msg = 'converged'
        elif d['warnflag'] == 1:  istop = IS_MAX_FUN_EVALS_REACHED

        p.xk = p.xf = xf
        p.fk = p.ff = ff
        p.istop = istop
        p.iterfcn()

