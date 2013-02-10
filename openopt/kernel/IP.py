from baseProblem import NonLinProblem
#from numpy.linalg import norm
from numpy import inf

class IP(NonLinProblem):
    probType = 'IP'
    goal = 'solution'
    allowedGoals = ['solution']
    showGoal = False
    _optionalData = []
    expectedArgs = ['f', 'domain']
    ftol = None
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        domain = args[1]
        self.x0 = dict([(v, 0.5*(val[0]+val[1])) for v, val in domain.items()])
        self.constraints = [v>bounds[0] for v,  bounds in domain.items()] + [v<bounds[1] for v,  bounds in domain.items()]
        #self.data4TextOutput = ['objFunVal', 'residual']
        self._Residual = inf

    def objFunc(self, x):
        return 0
        #raise 'unimplemented yet'
        
        #r = norm(dot(self.C, x) - self.d) ** 2  /  2.0
        #return r
