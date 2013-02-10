from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye, hstack, vstack, asarray, atleast_2d
from ooMisc import norm
import LP

class LUNP(MatrixProblem):
    probType = 'LUNP'
    goal = 'minimum'
    allowedGoals = ['minimum', 'min']
    showGoal = False
    _optionalData = []
    expectedArgs = ['C', 'd']
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        self.n = self.C.shape[1]
    #    if 'damp' not in kwargs.keys(): kwargs['damp'] = None
    #    if 'X' not in kwargs.keys(): kwargs['X'] = nan*ones(self.n)

        if self.x0 is None: self.x0 = zeros(self.n)
        
        #lunp_init(self, kwargs)

    def objFunc(self, x):
        r = norm(dot(self.C, x) - self.d, inf)
#        if not self.damp is None:
#            r += self.damp * norm(x-self.X)**2 / 2.0
#        if any(isfinite(self.f)): r += dot(self.f, x)
        return r

    def lunp2lp(self, solver, **solver_params):
        shapeC = atleast_2d(self.C).shape
        nVars = shapeC[1] + 1
        nObj = shapeC[0]
        f = hstack((zeros(nVars)))
        f[-1] = 1
        p = LP.LP(f)
        # TODO: check - is it performed in self.inspire(p)?
        if hasattr(self,'x0'): p.x0 = self.x0
        #p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
        self.inspire(p)
        p.x0 = hstack((p.x0, [0]))
        p.A = vstack((hstack((self.A, zeros((atleast_2d(self.A).shape[0], 1)))), \
                      hstack((self.C, -ones((nObj, 1)))), \
                      hstack((-self.C, -ones((nObj, 1))))))
        p.b = hstack((p.b, self.d, -self.d))
        
        p.lb = hstack((p.lb, -inf))
        p.ub = hstack((p.ub, inf))
        
        p.Aeq = hstack((self.Aeq, zeros((atleast_2d(self.Aeq).shape[0], 1))))
        
        #p.iprint = -1
        self.iprint = -1
        # for LLSP plot is via NLP
        #p.show = self.show
        #p.plot, self.plot = self.plot, 0
        #p.checkdf()
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf[:-1], r.ff, r.rf
        self.istop, self.msg = p.istop, p.msg
        return r

#    def _Prepare(self):
#        MatrixProblem._Prepare(self)
#        if not self.damp is None and not any(isfinite(self.X)):
#            self.X = zeros(self.n)




#def ff(x, LLSPprob):
#    r = dot(LLSPprob.C, x) - LLSPprob.d
#    return dot(r, r)
#ff = lambda x, LLSPprob: LLSPprob.objFunc(x)
#def dff(x, LLSPprob):
#    r = dot(LLSPprob.C.T, dot(LLSPprob.C,x)  - LLSPprob.d)
#    if not LLSPprob.damp is None: r += LLSPprob.damp*(x - LLSPprob.X)
#    if all(isfinite(LLSPprob.f)) : r += LLSPprob.f
#    return r
#
#def d2ff(x, LLSPprob):
#    r = dot(LLSPprob.C.T, LLSPprob.C)
#    if not LLSPprob.damp is None: r += LLSPprob.damp*eye(x.size)
#    return r
