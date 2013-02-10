from scipy.linalg.flapack import dgelss
from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d
from openopt.kernel.baseSolver import baseSolver

class lapack_dgelss(baseSolver):
    __name__ = 'lapack_dgelss'
    __license__ = "BSD"
    __authors__ = 'Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., Courant Institute, Argonne National Lab, and Rice University'
    #__alg__ = ""
    __info__ = 'wrapper to LAPACK dgelss routine (double precision), requires scipy & LAPACK 3.0 or newer installed'
    def __init__(self):pass
    def __solver__(self, p):
        res = dgelss(p.C, p.d)
        x,info = res[1], res[-1]
        xf = x[:p.C.shape[1]]
        ff = atleast_1d(asfarray(p.F(xf)))
        p.xf = p.xk = xf
        p.ff = p.fk = ff
        if info == 0: p.istop = 1000
        else: p.istop = -1000


