from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, asfarray, nan, zeros, isfinite, all


class SDP(MatrixProblem):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'S', 'd']
    expectedArgs = ['f']
    goal = 'minimum'
    #TODO: impolement goal = max, maximum for SDP
    #allowedGoals = ['minimum', 'min', 'maximum', 'max']
    allowedGoals = ['minimum', 'min']
    showGoal = True    
    def __init__(self, *args, **kwargs):
        self.probType = 'SDP'
        self.S = {}
        self.d = {}
        MatrixProblem.__init__(self, *args, **kwargs)
        self.f = asfarray(self.f)
        self.n = self.f.size
        if self.x0 is None: self.x0 = zeros(self.n)
        
    def _Prepare(self):
        MatrixProblem._Prepare(self)
        if self.solver.__name__ in ['cvxopt_sdp', 'dsdp']:
            try:
                from cvxopt.base import matrix
                matrixConverter = lambda x: matrix(x, tc='d')
            except:
                self.err('cvxopt must be installed')
        else:
            matrixConverter = asfarray
        for i in self.S.keys(): self.S[i] = matrixConverter(self.S[i])
        for i in self.d.keys(): self.d[i] = matrixConverter(self.d[i])
#        if len(S) != len(d): self.err('semidefinite constraints S and d should have same length, got '+len(S) + ' vs '+len(d)+' instead')
#        for i in range(len(S)):
#            d[i] = matrixConverter(d[i])
#            for j in range(len(S[i])):
#                S[i][j] = matrixConverter(S[i][j])
            
        
        
    def __finalize__(self):
        MatrixProblem.__finalize__(self)
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
                if hasattr(self, fn):
                    setattr(self, fn, -getattr(self, fn))
        

    def objFunc(self, x):
        return asfarray(dot(self.f, x).sum()).flatten()
