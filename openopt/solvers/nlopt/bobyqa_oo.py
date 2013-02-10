from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class bobyqa(NLOPT_BASE):
    __name__ = 'bobyqa'
    __alg__ = 'Bound constrained Optimization BY Quadratic Approximation'
    __authors__ = 'Michael JD Powell'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = True
    
    funcForIterFcnConnection = 'f'
    
    def __solver__(self, p):
#        if p.n < 15:
#            p.f_iter = 4
#        else:
#            p.f_iter = p.n/4

        
        NLOPT_AUX(p, nlopt.LN_BOBYQA)
