from baseProblem import NonLinProblem
from numpy import asarray, ones, inf
from setDefaultIterFuncs import MAX_NON_SUCCESS 

class GLP(NonLinProblem):
    probType = 'GLP'
    _optionalData = ['lb', 'ub', 'c', 'A', 'b']
    expectedArgs = ['f', 'x0']
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    goal = 'minimum'
    showGoal = False
    isObjFunValueASingleNumber = True
    plotOnlyCurrentMinimum= True
    _currentBestPoint = None
    _nonSuccessCounter = 0
    maxNonSuccess = 15
    
    def __init__(self, *args, **kwargs):
        #if len(args) > 1: self.err('incorrect args number for GLP constructor, must be 0..1 + (optionaly) some kwargs')

        NonLinProblem.__init__(self, *args, **kwargs)
        
        def maxNonSuccess(p):
            newPoint = p.point(p.xk)
            if self._currentBestPoint is None:
                self._currentBestPoint = newPoint
                return False
            elif newPoint.betterThan(self._currentBestPoint):
                self._currentBestPoint = newPoint
                self._nonSuccessCounter = 0
                return False
            self._nonSuccessCounter += 1
            if self._nonSuccessCounter > self.maxNonSuccess:
                return (True, 'Non-Success Number > maxNonSuccess = ' + str(self.maxNonSuccess))
            else:
                return False
        
        self.kernelIterFuncs[MAX_NON_SUCCESS] = maxNonSuccess
        if 'lb' in kwargs.keys():
            self.n = len(kwargs['lb'])
        elif 'ub' in kwargs.keys():
            self.n = len(kwargs['ub'])
        if hasattr(self, 'n'):
            if not hasattr(self, 'lb'):
                self.lb = -inf * ones(self.n)
            if not hasattr(self, 'ub'):
                self.ub =  inf * ones(self.n)
            if 'x0' not in kwargs.keys(): self.x0 = (asarray(self.lb) + asarray(self.ub)) / 2.0
