from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class ptn(NLOPT_BASE):
    __name__ = 'ptn'
    __alg__ = 'Preconditioned truncated Newton'
    __authors__ = 'Prof. Ladislav Luksan'
    __isIterPointAlwaysFeasible__ = True
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    
    def __init__(self): pass
    def __solver__(self, p):
        NLOPT_AUX(p, nlopt.LD_TNEWTON_PRECOND_RESTART)
        #NLOPT_AUX(p, nlopt.LD_TNEWTON)
        #NLOPT_AUX(p, nlopt.LD_TNEWTON_PRECOND)
