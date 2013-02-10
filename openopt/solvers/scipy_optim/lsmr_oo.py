from scipy.sparse.linalg import lsmr as scipy_lsmr
from openopt.kernel.setDefaultIterFuncs import IS_MAX_ITER_REACHED
from openopt.kernel.baseSolver import baseSolver
from numpy import asfarray, atleast_1d

class lsmr(baseSolver):
    __name__ = 'lsmr'
    __license__ = "BSD"
    __authors__ = 'D. C.-L. Fong and M. A. Saunders'
    __alg__ = '"LSMR: An iterative algorithm for sparse least-squares problems", SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011. http://arxiv.org/abs/1006.0758'
    __info__ = 'requires scipy version >= 0.11b installed'
    atol=1e-06
    btol=1e-06
    conlim=100000000.0
    _canHandleScipySparse = True
    __optionalDataThatCanBeHandled__ = ['damp']
    
    def __init__(self):
        pass
    
    def __solver__(self, p):
        x, istop, itn, normr, normar, norma, conda, normx = \
        scipy_lsmr(p.C, p.d, p.damp if p.damp is not None else 0.0, self.atol, self.btol, self.conlim, p.maxIter)
        xf = x[:p.C.shape[1]]
        ff = atleast_1d(asfarray(p.F(xf)))
        p.xf = p.xk = xf
        p.ff = p.fk = ff
        
        if istop == 0:
            p.istop, p.msg = 0, '0 is solution'
        elif istop == 1:
            p.istop, p.msg = 1000, 'x is an approximate solution to A*x = B according to atol and btol'
        elif istop == 2:
            p.istop, p.msg = 1000, 'x approximately solves the least-squares problem according to atol'
        elif istop == 3:
            p.istop, p.msg = 0, 'COND(A) seems to be greater than CONLIM'
        elif istop == 4:
            p.istop, p.msg =  1000, 'x is an approximate solution to A*x = B with atol = btol = eps (machine precision)'
        elif istop == 5:
            p.istop, p.msg = 1000, 'x approximately solves the least-squares problem according to atol = eps (machine precision)'
        elif istop == 6:
            p.istop, p.msg = 0, 'COND(A) seems to be greater than CONLIM = 1/eps (machine precision)'
        elif istop == 7:
            p.istop, p.msg = IS_MAX_ITER_REACHED, 'max iter (%d) has been reached' % p.maxIter
        

