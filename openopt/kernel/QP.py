import NLP

from ooMisc import isspmatrix
from baseProblem import MatrixProblem
from numpy import asfarray, dot, nan, zeros, isfinite, all, ravel


class QP(MatrixProblem):
    probType = 'QP'
    goal = 'minimum'
    allowedGoals = ['minimum', 'min']
    showGoal = False
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'QC', 'intVars']
    expectedArgs = ['H', 'f']
    
    def _Prepare(self):
        # TODO: handle cvxopt sparse matrix case here
        self.n = self.H.shape[0]
        if not hasattr(self, 'x0') or self.x0 is None or self.x0[0] == nan:
            self.x0 = zeros(self.n)
        MatrixProblem._Prepare(self)
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        if len(args) > 1 or 'f' in kwargs.keys():
            self.f = ravel(self.f)
            self.n = self.f.size
        if len(args) > 0 or 'H' in kwargs.keys():
            # TODO: handle sparse cvxopt matrix H unchanges
            # if not ('cvxopt' in str(type(H)) and 'cvxopt' in p.solver): 
            if not isspmatrix(self.H):
                self.H = asfarray(self.H, float) # TODO: handle the case in runProbSolver()
        

    def objFunc(self, x):
        return asfarray(0.5*dot(x, self.matMultVec(self.H, x)) + dot(self.f, x).sum()).flatten()

    def qp2nlp(self, solver, **solver_params):
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff, d2f=d2ff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff, d2f=d2ff)
        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
        p.iprint = self.iprint
        self.inspire(p)
        self.iprint = -1
        

        # for QP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0

        #p.checkdf()
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
        return r


ff = lambda x, QProb: QProb.objFunc(x)
def dff(x, QProb):
    r = dot(QProb.H, x)
    if all(isfinite(QProb.f)) : r += QProb.f
    return r

def d2ff(x, QProb):
    r = QProb.H
    return r

