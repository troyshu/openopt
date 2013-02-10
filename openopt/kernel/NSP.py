from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf

class NSP(NonLinProblem):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    expectedArgs = ['f', 'x0']
    probType = 'NSP'
    JacobianApproximationStencil = 3
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True    
    def __init__(self, *args, **kwargs):
        self.goal = 'minimum'
        NonLinProblem.__init__(self, *args, **kwargs)
       #TODO: set here default tolx, tolcon, diffInt etc for NS Problem







