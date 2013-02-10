from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class mlsl(NLOPT_BASE):
    __name__ = 'mlsl'
    __alg__ = 'Multi-Level Single-Linkage'
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    #funcForIterFcnConnection = 'f'
    population = 0
    __isIterPointAlwaysFeasible__ = True
    _requiresFiniteBoxBounds = True
    
    def __init__(self): pass
    def __solver__(self, p):
        nlopt_opts = {'set_population':self.population} if self.population != 0 else {}
        #if self.population != 0: p.f_iter = self.population
        #p.f_iter = 4*p.n
        NLOPT_AUX(p, nlopt.G_MLSL_LDS, nlopt_opts)
