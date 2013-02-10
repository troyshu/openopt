#from numpy import asfarray, argmax, sign, inf, log10
from openopt.kernel.baseSolver import baseSolver
from numpy import isfinite, array, asfarray, ravel #asfarray,  inf,  atleast_1d
from pswarm_py import pswarm as PSWARM
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F

#from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F

class pswarm(baseSolver):
    __name__ = 'pswarm'
    __license__ = "LGPL"
    __authors__ = 'A. I. F. Vaz (http://www.norg.uminho.pt/aivaz), connected to OO by Dmitrey'
    __alg__ = "A. I. F. Vaz and L. N. Vicente, A particle swarm pattern search method for bound constrained global optimization, Journal of Global Optimization, 39 (2007) 197-219. The algorithm combines pattern search and particle swarm. Basically, it applies a directional direct search in the poll step (coordinate search in the pure simple bounds case) and particle swarm in the search step."
    iterfcnConnected = True
    __homepage__ = 'http://www.norg.uminho.pt/aivaz/pswarm/'

    __info__ = "parameters: social (default = 0.5), cognitial (0.5), fweight (0.4), iweight (0.9), size (42), tol (1e-5), ddelta (0.5), idelta (2.0). Can handle constraints lb <= x <= ub (values beyond 1e20 are treated as 1e20), A x <= b. Documentation says pswarm is capable of using parallel calculations (via MPI) but I don't know is it relevant to Python API."
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'b']
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()

    social = 0.5
    cognitial = 0.5
    fweight = 0.4
    iweight = 0.9
    size = 42
    tol = 1e-5
    ddelta = 0.5
    idelta = 2.0

    def __init__(self):pass
    def __solver__(self, p):

        #if not p.__isFiniteBoxBounded__(): p.err('this solver requires finite lb, ub: lb <= x <= ub')
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
                
        lb, ub = p.lb, p.ub
        lb[lb < -1e20] = -1e20
        ub[ub > 1e20] = 1e20

        def f(x):
            if len(x) == 1:
                y = ravel(x)
            else:
                y = asfarray(x)
            return array(p.f(y), 'float')

        Problem = {
            'Variables':  p.n,
            'objf': f,
            'lb': lb.tolist(),
            'ub': ub.tolist(),
            }

        if any(isfinite(p.b)):
               Problem['A'] = p.A.tolist()
               Problem['b'] = p.b.tolist()
        if hasattr(p,'x0') and p.x0 is not None:
               Problem['x0'] = p.x0.tolist()

        def pswarm_iterfcn(it, leader, fx, x):
            p.xk = array(x)
            p.fk = array(fx)
            #print x, fx
            p.iterfcn()
            if p.istop != 0:
                p.debugmsg('istop:'+str(p.istop))
                return -1.0
            else: return 1.0

        Options = {
            'maxf': 2*p.maxFunEvals, # to provide correct istop triggered  from OO kernel
            'maxit': p.maxIter+15, # to provide correct istop triggered  from OO kernel
            'social': self.social,
            'cognitial': self.cognitial,
            'fweight': self.fweight,
            'iweight': self.iweight,
            'size': self.size,
            'tol': self.tol,
            'ddelta': self.ddelta,
            'idelta': self.idelta,
            'iprint': 1,
            'outputfcn': pswarm_iterfcn,
            'vectorized': 1
            }

        result = PSWARM(Problem,Options)

        #print 'results:', xf, ff
        p.xf, p.ff = result['x'], result['f']
        if p.istop == 0: p.istop = 1000
        #if p.istop == 0 : p.istop = 1000


