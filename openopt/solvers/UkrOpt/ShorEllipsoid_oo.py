if __name__ == '__main__':
    import sys, os.path as pth
    sys.path.insert(0,pth.split(pth.split(pth.split(pth.split(pth.realpath(pth.dirname(__file__)))[0])[0])[0])[0])
    from openopt import NSP

    from numpy import ones, arange
    N = 4
    p = NSP(lambda x: abs(x).sum(), ones([N,1]), maxFunEvals = 1e6, plot = False)
    p.r0 = N
    r = p.solve('ShorEllipsoid')

from numpy import diag, ones, inf, any, copy, sqrt
from openopt.kernel.baseSolver import baseSolver
class ShorEllipsoid(baseSolver):
    __name__ = 'ShorEllipsoid'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Naum Z. Shor modificated method of ellipsoids"
    iterfcnConnected = True
    #__optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']

    def __init__(self): pass

    def __solver__(self, p):

        #if p.useScaling: p.err('ShorEllipsoid + scaling isn''t implemented yet')


        n = p.n

        B = diag(ones(n))

        x0 = copy(p.x0)

        ########################
        # Shor MME engine                         #
        # //by Dmitrey Kroshko                     #
        # //icq 275976670(inviz)                    #
        ########################

        x = x0.copy()
        xPrev = x0.copy()
        xf = x0.copy()
        xk = x0.copy()
        p.xk = x0.copy()

        if not hasattr(p, 'r0'):  p.err('ShorEllipsoid solver requires r0')
        r = p.r0

        f = p.f(x)

        fk = f
        ff = f
        p.fk = fk

        g = p.df(x)
        p._df = g

        p.iterfcn()
        if p.istop:
            p.xf = x
            p.ff = f
            #r.advanced.ralg.hs = hs
            return


        multiplier_R = sqrt(1.+1./n**2)
        beta = multiplier_R - 1. / n
        alfa = 1. / beta

        for k in range(p.maxIter):
            BTG = p.matmult(B.T, g.reshape(-1,1))
            dzeta_k = (BTG / p.norm(BTG))
            hk = r*beta/n
            x  -= hk * p.matmult(B, dzeta_k).flatten()
            B += p.matmult(B, p.matmult((1-alfa)*dzeta_k, dzeta_k.T))
            r *= multiplier_R
            f = p.f(x)
            g = p.df(x)
            xk = x.copy()
            fk = f


            if fk < ff:
                ff, xf = fk, xk.copy()

            p.fk = fk
            p.xk = xk
            p._df = g#CHECK ME! - MODIFIED!!
            p.iterfcn()
            if p.istop:
                p.xf = xf
                p.ff = ff
                #p.advanced.ralg.hs = max(norm(xPrev - x), hsmin)
                p._df = g
                return
