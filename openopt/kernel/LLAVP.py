from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, isfinite, sign
from ooMisc import norm
import NSP

class LLAVP(MatrixProblem):
    _optionalData = ['damp', 'X', 'c']
    allowedGoals = ['minimum', 'min']
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for LLAVP constructor, must be 0..2 + (optionaly) some kwargs')
        if len(args) > 0: kwargs['C'] = args[0]
        if len(args) > 1: kwargs['d'] = args[1]

        MatrixProblem.__init__(self)
        llavp_init(self, kwargs)

    def objFunc(self, x):
        r = norm(dot(self.C, x) - self.d, 1)
        if not self.damp is None:
            r += self.damp * norm(x-self.X, 1)
        #if any(isfinite(self.f)): r += dot(self.f, x)
        return r

    def llavp2nsp(self, solver, **solver_params):
        if hasattr(self,'x0'): p = NSP.NSP(ff, self.x0, df=dff)
        else: p = NSP.NSP(ff, zeros(self.n), df=dff)
        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
        self.inspire(p)
        self.iprint = -1
        # for LLAVP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0
        #p.checkdf()
        #p.solver.__optionalDataThatCanBeHandled__ += ['damp', 'X', 'c']
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
        return r

    def _Prepare(self):
        MatrixProblem._Prepare(self)
        if not self.damp is None and not any(isfinite(self.X)):
            self.X = zeros(self.n)




def llavp_init(prob, kwargs):

    prob.probType = 'LLAVP'
    prob.goal = 'minimum'
    prob.allowedGoals = ['minimum', 'min']
    prob.showGoal = False

    kwargs['C'] = asfarray(kwargs['C'])

    prob.n = kwargs['C'].shape[1]
    prob.lb = -inf * ones(prob.n)
    prob.ub =  inf * ones(prob.n)
    if 'damp' not in kwargs.keys(): kwargs['damp'] = None
    if 'X' not in kwargs.keys(): kwargs['X'] = nan*ones(prob.n)

    if prob.x0 is None: prob.x0 = zeros(prob.n)

    return assignScript(prob, kwargs)

ff = lambda x, LLAVprob: LLAVprob.objFunc(x)
def dff(x, LLAVprob):
    r = dot(sign(dot(LLAVprob.C, x) - LLAVprob.d), LLAVprob.C)
    #r = (LLAVprob.C * sign(dot(LLAVprob.C, x) - LLAVprob.d)).sum(0)
    if not LLAVprob.damp is None: r += LLAVprob.damp * (sign(x - LLAVprob.X)).sum(0)
    return r
