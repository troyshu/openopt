from ooMisc import assignScript
from nonOptMisc import isspmatrix
from baseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros, isnan, any, vstack, array, asfarray
from ooMisc import norm

class LCP(MatrixProblem):
    _optionalData = []
    expectedArgs = ['M', 'q']
    goal = 'solve'
    probType = 'LCP'
    allowedGoals = ['solve']
    showGoal = False

    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        self.x0 = zeros(2*len(self.q))
        
    def objFunc(self, x):
        return norm(dot(self.M, x[x.size/2:]) +self.q - x[:x.size/2], inf)

