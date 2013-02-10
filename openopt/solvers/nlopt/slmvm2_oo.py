from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class slmvm2(NLOPT_BASE):
    __name__ = 'slmvm2'
    __alg__ = 'Shifted limited-memory variable-metric, rank 2'
    __authors__ = 'Prof. Ladislav Luksan'
    __isIterPointAlwaysFeasible__ = True
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
   
    def __init__(self): pass
    def __solver__(self, p):
        NLOPT_AUX(p, nlopt.LD_VAR2)
