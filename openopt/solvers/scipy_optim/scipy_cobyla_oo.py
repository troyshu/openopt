from scipy.optimize import fmin_cobyla
import openopt
from openopt.kernel.setDefaultIterFuncs import *
from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from openopt.kernel.baseSolver import baseSolver
from numpy import inf, array, copy
#from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F

class EmptyClass: pass

class scipy_cobyla(baseSolver):
    __name__ = 'scipy_cobyla'
    __license__ = "BSD"
    __authors__ = """undefined"""
    __alg__ = "Constrained Optimization BY Linear Approximation"
    __info__ = 'constrained NLP solver, no user-defined derivatives are handled'#TODO: add '__info__' field to other solvers
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    funcForIterFcnConnection = 'f'

    def __init__(self): pass
    def __solver__(self, p):
        #p.kernelIterFuncs.pop(SMALL_DELTA_X)
        #p.kernelIterFuncs.pop(SMALL_DELTA_F)
        xBounds2Matrix(p)
        p.cobyla = EmptyClass()
        if p.userProvided.c: p.cobyla.nc = p.c(p.x0).size
        else: p.cobyla.nc = 0
        if p.userProvided.h: p.cobyla.nh = p.h(p.x0).size
        else: p.cobyla.nh = 0

        det_arr = cumsum(array((p.cobyla.nc, p.cobyla.nh, p.b.size, p.beq.size, p.cobyla.nh, p.beq.size)))

        cons = []
        for i in range(det_arr[-1]):
            if i < det_arr[0]:
                c = lambda x, i=i: - p.c(x)[i] # cobyla requires positive constraints!
            elif det_arr[0] <= i < det_arr[1]:
                j = i - det_arr[0]
                c = lambda x, j=j: p.h(x)[j]
            elif det_arr[1] <= i < det_arr[2]:
                j = i - det_arr[1]
                #assert 0<= j <p.cobyla.nb
                c = lambda x, j=j: p.b[j] - p.dotmult(p.A[j], x).sum() # cobyla requires positive constraints!
            elif det_arr[2] <= i < det_arr[3]:
                j = i - det_arr[2]
                #assert 0<= j <p.cobyla.nbeq
                c = lambda x, j=j: p.dotmult(p.Aeq[j], x).sum() - p.beq[j]
            elif det_arr[3] <= i < det_arr[4]:
                j = i - det_arr[3]
                c = lambda x, j=j: - p.h(x)[j]
            elif det_arr[4] <= i < det_arr[5]:
                j = i - det_arr[4]
                #assert 0<= j <p.cobyla.nbeq
                c = lambda x, j=j: p.dotmult(p.Aeq[j], x).sum() - p.beq[j]
            else:
                p.err('error in connection cobyla to openopt')
            cons.append(c)
##        def oo_cobyla_cons(x):
##            c0 = -p.c(x)
##            c1 = p.h(x)
##            c2 = -(p.matmult(p.A, x) - p.b)
##            c3 = p.matmult(p.Aeq, x) - p.beq
##            return hstack((c0, c1, -c1, c2, c3, -c3))


#        p.xk = p.x0.copy()
#        p.fk = p.f(p.x0)
#
#        p.iterfcn()
#        if p.istop:
#            p.xf = p.xk
#            p.ff = p.fk
#            return


        xf = fmin_cobyla(p.f, p.x0, cons = tuple(cons), iprint = 0, maxfun = p.maxFunEvals, rhoend = p.xtol )

        p.xk = xf
        p.fk = p.f(xf)
        p.istop = 1000
#        p.iterfcn()
#        p.xf = xf
#        p.ff = p.fk



