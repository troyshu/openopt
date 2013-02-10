from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class sbplx(NLOPT_BASE):
    __name__ = 'sbplx'
    __alg__ = 'a variant of Nelder-Mead that uses Nelder-Mead on a sequence of subspaces'
    __authors__ = 'Steven G. Johnson'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True
    
    funcForIterFcnConnection = 'f'
    
    def __init__(self): pass
    def __solver__(self, p):
        #NLOPT_AUX(p, nlopt.LN_BOBYQA)
        #NLOPT_AUX(p, nlopt.LN_COBYLA)
        NLOPT_AUX(p, nlopt.LN_SBPLX)
        #NLOPT_AUX(p, nlopt.LD_MMA)
