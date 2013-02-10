from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class isres(NLOPT_BASE):
    __name__ = 'isres'
    __alg__ = 'Improved Stochastic Ranking Evolution Strategy'
    __authors__ = 'S. G. Johnson'
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    funcForIterFcnConnection = 'f'
    population = 0
    #__isIterPointAlwaysFeasible__ = lambda self, p: True
    __isIterPointAlwaysFeasible__ = True
    _requiresFiniteBoxBounds = True
    
    def __init__(self): pass
    def __solver__(self, p):
        nlopt_opts = {'set_population':self.population} if self.population != 0 else {}
        if self.population != 0: p.f_iter = self.population
        NLOPT_AUX(p, nlopt.GN_ISRES, nlopt_opts)
