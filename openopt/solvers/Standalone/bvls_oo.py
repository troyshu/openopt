from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf
from openopt.kernel.baseSolver import baseSolver

try:
    import bvls as BVLS
except:
    from openopt.kernel.oologfcn import OpenOptException
    raise OpenOptException('You should have bvls.f compiled via f2py, see OO LLSP doc webpage for details')

class bvls(baseSolver):
    __name__ = 'bvls'
    __license__ = "BSD"
    __authors__ = 'Robert L. Parker rlparker[at]ucsd.edu, Philip B. Stark stark[at]stat.berkeley.edu'
    __alg__ = '"Bounded Variable Least Squares:  An Algorithm and Applications" by P.B. Stark and R.L. Parker, in the journal "Computational Statistics", vol.10(2), 1995'
    __info__ = 'requires manual compilation of bvls.f by f2py, see OO online doc for details'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']

    __bvls_inf__ = 1e100
    T = float64
    def __init__(self): pass

    def __solver__(self, p):
        key = 0
        T = self.T
        istate = zeros(p.n+1,  int)
        m, n = p.C.shape[0],  p.n
        zz = zeros(m, T)
        act = zeros((m, m+2),  T)
        a, b = T(p.C),  T(p.d)
        bl,  bu = p.lb.copy(), p.ub.copy()
        bl[where(bl==-inf)[0]] = -self.__bvls_inf__
        bu[where(bu==inf)[0]] = self.__bvls_inf__

        xf, w, istop, msg,  iter = BVLS.bvls(key, a, b, bl, bu, p.maxIter, act, zz,  istate)

        p.istop, p.msg, p.iter = istop, msg.rstrip(), iter
        p.xf = xf
        #p.ff = p.fk = w[0]

