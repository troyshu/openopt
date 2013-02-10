from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt
from numpy import isinf

class stogo(NLOPT_BASE):
    __name__ = 'stogo'
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True
    _requiresFiniteBoxBounds = True
    #properTextOutput = True
    
    #TODO: check it!
    #_canHandleScipySparse = True
    
    funcForIterFcnConnection = 'f'
#    _requiresBestPointDetection = True
    useRand = True
    
    def __init__(self): pass
    def __solver__(self, p):
        #p.f_iter = 1
        p.maxNonSuccess = 1e10
        p.maxIter = 1e10
        if isinf(p.maxTime):
            s= """currently due to some Python <-> C++ code connection issues 
            the solver stogo requires finite user-defined maxTime; 
            since you have not provided it, 15 sec will be used"""
            p.pWarn(s)
            p.maxTime = 15
        solver = nlopt.GD_STOGO_RAND if self.useRand else nlopt.GD_STOGO
        NLOPT_AUX(p, solver)
        
        #NLOPT_AUX(p, nlopt.GD_MLSL_LDS)
