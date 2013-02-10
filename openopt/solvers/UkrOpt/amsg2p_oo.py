from openopt.kernel.baseSolver import *
#from openopt.kernel.Point import Point
#from openopt.kernel.setDefaultIterFuncs import *
from numpy import isfinite
from amsg2p import amsg2p as Solver

class amsg2p(baseSolver):
    __name__ = 'amsg2p'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Petro I. Stetsyuk, amsg2p"
    __optionalDataThatCanBeHandled__ = []
    iterfcnConnected = True
    #_canHandleScipySparse = True

    #default parameters
#    T = float64
    
    showRes = False
    show_nnan = False
    gamma = 1.0
#    approach = 'all active'

    def __init__(self): pass
    def __solver__(self, p):
        #assert self.approach == 'all active'
        if not p.isUC: p.warn('Handling of constraints is not implemented properly for the solver %s yet' % self.__name__)
        if p.fOpt is None: 
            if not isfinite(p.fEnough):
                p.err('the solver %s requires providing optimal value fOpt')
            else:
                p.warn("you haven't provided optimal value fOpt for the solver %s; fEnough = %0.2e will be used instead" %(self.__name__, p.fEnough))
                p.fOpt = p.fEnough
        if p.fTol is None: 
            s = '''
            the solver %s requires providing required objective function tolerance fTol
            15*ftol = %0.1e will be used instead
            ''' % (self.__name__, p.ftol)
            p.pWarn(s)
            fTol = 15*p.ftol
        else: fTol = p.fTol
        
        def itefcn(*args, **kwargs):
            p.iterfcn(*args, **kwargs)
            return p.istop
        x, itn = Solver(p.f, p.df, p.x0, fTol, p.fOpt, self.gamma, itefcn)

        if p.f(x) < p.fOpt + fTol:
            p.istop = 10
        #p.iterfcn(x)
