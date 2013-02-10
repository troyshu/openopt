from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS
#import numpy as np
from .tsp import main

class sa(baseSolver):
    __name__ = 'sa'
    __license__ = "BSD"
    __authors__ = "John Montgomery, connected to OO by Dmitrey"
    __alg__ = "simulated annealing"
    __license__ = 'Creative Commons Attribution 3.0 Unported'
    iterfcnConnected = True
    __homepage__ = ''
    #__optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'b', 'Aeq','beq','c','h']
    __info__ = ''
    

    # Check it!
    __isIterPointAlwaysFeasible__ = True

        
    def __solver__(self, p):
        
        #self.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        #self.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        #self.kernelIterFuncs.pop(MAX_NON_SUCCESS, None)
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        p.kernelIterFuncs.pop(MAX_NON_SUCCESS, None)
        
        p._bestPoint = p.point(p.x0)
        p.solver._requiresBestPointDetection = True
        
        M = p.M
        iterations, score, best = main(M, p)
        
        p.ff = p.fk = score
        p.xk = p.xf = best
        if p.istop == 0: 
            p.istop = 1000
        
