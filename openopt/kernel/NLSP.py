from baseProblem import NonLinProblem

from ooMisc import norm
from numpy import inf, asfarray, atleast_1d, dot, abs, ndarray
from setDefaultIterFuncs import FVAL_IS_ENOUGH, SMALL_DELTA_F
from nonOptMisc import getSolverFromStringName
import NLP
try:
    import scipy
    solver = 'scipy_fsolve'
except ImportError:
    solver = 'nssolve'

class NLSP(NonLinProblem):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    probType = 'NLSP'
    goal = 'solution'
    allowedGoals = ['solution']
    isObjFunValueASingleNumber = False
    showGoal = False
    expectedArgs = ['f', 'x0']
    solver = solver
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        if 'is_oovar' in dir(self.f):
            self.f = [self.f]

    def objFuncMultiple2Single(self, fv):
        return norm(atleast_1d(asfarray(fv)), inf)

    def nlsp2nlp(self, solver, **solver_params):
        ff = lambda x: sum(asfarray(self.f(x))**2)
        p = NLP.NLP(ff, self.x0)
        self.inspire(p, sameConstraints=True)
        if self.userProvided.df:
            p.df = lambda x: dot(2*asfarray(self.f(x)), asfarray(self.df(x)))
        p.f = ff


        def nlsp_iterfcn(*args,  **kwargs):
            if len(args) != 0 and type(args[0]) != ndarray: # hence Point
                p.primalIterFcn(args[0].x, max(abs(self.f(args[0].x))), args[0].mr(),  **kwargs)
                # TODO: add nNaNs
            elif len(args) > 1:
                p.primalIterFcn(args[0], max(abs(self.f(args[0]))), *args[2:],  **kwargs)
            elif 'fk' in kwargs.keys():
                kwargs['fk'] = max(abs(self.f(args[0])))
                p.primalIterFcn(*args, **kwargs)
            else:
                p.primalIterFcn(*args,  **kwargs)
            p.xk = self.xk
            p.fk = p.f(p.xk)
            p.rk = self.rk
            # TODO: add nNaNs

            p.istop = self.istop
            return p.istop

        ftol_init = self.ftol
        contol_init = self.contol
        
        msg_contol = '' if self.isUC else 'and contol '
        
        def nlsp_callback(nlsp):
            # nlsp = self
            if all(abs(asfarray(self.f(nlsp.xk))) < ftol_init)  and (self.isUC or self.getMaxResidual(nlsp.xk) < contol_init):
                self.msg = 'solution with required ftol ' + msg_contol+ 'has been reached'
                return (15, self.msg)
            else:
                return False

        self.callback = [nlsp_callback]
        if SMALL_DELTA_F in self.kernelIterFuncs.keys():self.kernelIterFuncs.pop(SMALL_DELTA_F)
        p.primalIterFcn,  p.iterfcn = self.iterfcn, nlsp_iterfcn
        p.goal = 'min'
        #self.fEnough = self.ftol


        p.iprint = -1

        Multiplier = 1e16

        #self.ftol /= Multiplier
        self.xtol /= Multiplier
        self.gtol /= Multiplier

        p.show = False
        
        p.fOpt = 0.0
        p.fTol = p.ftol**2
        
        r = p.solve(solver, **solver_params)

        #self.ftol *= Multiplier
        self.xtol *= Multiplier
        self.gtol *= Multiplier

        if self.istop == FVAL_IS_ENOUGH:
            self.msg = 'solution with required ftol ' + msg_contol+ 'has been reached'
            self.istop = 15

        #self.iterfcn(xk = r.xk, fk = r.fk, rk = r.rk)
        #self.show = show

        # TODO: fix it!
        #r.iterValues.f = self.iterValues.f

        #r.ff = max(abs(asfarray(self.f(r.xf))))
        return r
