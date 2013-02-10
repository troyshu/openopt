from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, asfarray, nan, zeros, isfinite, all, asscalar


class SOCP(MatrixProblem):
    probType = 'SOCP'
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    goal = 'minimum'
    allowedGoals = ['minimum', 'min']
    #TODO: add goal=max, maximum
    showGoal = True
    expectedArgs = ['f', 'C', 'd']
    # required are f, C, d
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        self.f = asfarray(self.f)
        self.n = self.f.size # for p.n to be available immediately after assigning prob
        if self.x0 is None: self.x0 = zeros(self.n)


    def objFunc(self, x):
        return asscalar(dot(self.f, x))
