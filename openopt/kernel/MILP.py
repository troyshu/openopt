from numpy import ceil, floor, argmax, ndarray, copy
from setDefaultIterFuncs import SMALL_DELTA_X, SMALL_DELTA_F
from LP import LP

class MILP(LP):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'boolVars']
    probType = 'MILP'
    expectedArgs = ['f', 'x0']
    allowedGoals = ['minimum', 'min', 'max', 'maximum']
    showGoal = True
    _milp_prepared = False
    
    def _Prepare(self):
        if self._milp_prepared: return
        self._milp_prepared = True
        LP._Prepare(self)
        r = []
        if type(self.intVars) not in [list, tuple, set] and not isinstance(self.intVars, ndarray):
            self.intVars = [self.intVars]
        if self.isFDmodel:
            
            ########### obsolete, to be removed in future versions
            if self.intVars != []:
                s = '''
                For FuncDesigner models prob parameter intVars is deprecated
                and will be removed in future versions, use oovar(..., domain = int) instead'''
                self.pWarn(s)
            for iv in self.intVars:
                if self.fixedVars is not None and iv in self.fixedVars or\
                self.freeVars is not None and iv not in self.freeVars:
                    continue
                r1, r2 = self._oovarsIndDict[iv]
                r += range(r1, r2)
            ########### obsolete, to be removed in future versions
        
            for v in self._freeVarsList:
                if isinstance(v.domain, (tuple, list, ndarray, set)):
                    self.err('for FuncDesigner MILP models only variables with domains int, bool or None (real) are implemented for now')
                if v.domain in (int, 'int', bool, 'bool'):
                    r1, r2 = self._oovarsIndDict[v]
                    r += range(r1, r2)
            
            self.intVars, self._intVars = r, self.intVars
        self._intVars_vector = self.intVars
                
        if SMALL_DELTA_X in self.kernelIterFuncs: self.kernelIterFuncs.pop(SMALL_DELTA_X)
        if SMALL_DELTA_F in self.kernelIterFuncs: self.kernelIterFuncs.pop(SMALL_DELTA_F)
        def getMaxResidualWithIntegerConstraints(x, retAll = False):
            r, fname, ind = self.getMaxResidual2(x, True)
            if len(self.intVars) != 0:
                intV = x[self.intVars]
                intDifference = abs(intV-intV.round())
                intConstraintNumber = argmax(intDifference)
                intConstraint = intDifference[intConstraintNumber]
                #print 'intConstraint:', intConstraint
                if intConstraint > r:
                    intConstraintNumber = self.intVars[intConstraintNumber]
                    r, fname, ind = intConstraint, 'int', intConstraintNumber 
            if retAll:
                return r, fname, ind
            else:
                return r
        self.getMaxResidual, self.getMaxResidual2 = getMaxResidualWithIntegerConstraints, self.getMaxResidual
        
        # TODO: 
        # 1) ADD BOOL VARS
        self.lb, self.ub = copy(self.lb), copy(self.ub)
        self.lb[self.intVars] = ceil(self.lb[self.intVars])
        self.ub[self.intVars] = floor(self.ub[self.intVars])
        
#        if self.goal in ['max', 'maximum']:
#            self.f = -self.f
    def __finalize__(self):
        LP.__finalize__(self)
        if self.isFDmodel: self.intVars = self._intVars
#    def __finalize__(self):
#        MatrixProblem.__finalize__(self)
#        if self.goal in ['max', 'maximum']:
#            self.f = -self.f
#            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
#                if hasattr(self, fn):
#                    setattr(self, fn, -getattr(self, fn))
    
#    def __init__(self, *args, **kwargs):
#        LP.__init__(self, *args, **kwargs)
