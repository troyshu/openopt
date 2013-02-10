from lp_solve import lp_solve as lps
#import lp_solve
#lp_solve.PRESOLVE_DUALS = 0#524288 + 1048576#10000

from openopt.kernel.baseSolver import baseSolver
from numpy import asarray, inf, ones, nan, ravel

from openopt.kernel.ooMisc import LinConst2WholeRepr

def List(x):
    if x == None or x.size == 0: return None
    else: return x.tolist()

class lpSolve(baseSolver):
    __name__ = 'lpSolve'
    __license__ = "LGPL"
    __authors__ = "Michel Berkelaar, michel@es.ele.tue.nl"
    __homepage__ = 'http://sourceforge.net/projects/lpsolve, http://www.cs.sunysb.edu/~algorith/implement/lpsolve/implement.shtml, http://www.nabble.com/lp_solve-f14350i70.html'
    __alg__ = "lpsolve"
    __info__ = 'use p.scale = 1 or True to turn scale mode on'
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars']
    def __init__(self): pass
    def __solver__(self, p):

        LinConst2WholeRepr(p)
        f = - asarray(p.f) # sign '-' because lp_solve by default searches for maximum, not minimum
        scalemode = False
        if p.scale in [1, True]:
            scalemode = 1
        elif not (p.scale in [None, 0, False]):
            p.warn(self.__name__ + ' requires p.scale from [None, 0, False, 1, True], other value obtained, so scale = 1 will be used')
            scalemode = 1
        [obj, x_opt, duals] = lps(List(f.flatten()), List(p.Awhole), List(p.bwhole.flatten()), List(p.dwhole.flatten()), \
        List(p.lb.flatten()), List(p.ub.flatten()), (1+asarray(p.intVars)).tolist(), scalemode)
        if obj != []:
            # ! don't involve p.ff  - it can be different because of goal and additional constant from FuncDesigner
            p.xf = ravel(x_opt)
            p.duals = duals
            p.istop = 1
        else:
            p.ff = nan
            p.xf = nan*ones(p.n)
            p.duals = []
            p.istop = -1

