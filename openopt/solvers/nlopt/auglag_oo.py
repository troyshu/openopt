from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class auglag(NLOPT_BASE):
    __name__ = 'auglag'
    __alg__ = "Augmented Lagrangian"
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'c', 'h','A', 'b', 'Aeq', 'beq']
    
    #TODO: check it!
    #_canHandleScipySparse = True
    
    #funcForIterFcnConnection = 'f'
    
    
    def __init__(self): pass
    def __solver__(self, p):
        NLOPT_AUX(p, nlopt.LD_AUGLAG)
