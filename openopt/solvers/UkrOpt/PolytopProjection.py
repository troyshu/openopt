from openopt import QP
from numpy import  dot, asfarray, ones, zeros, max
from numpy.linalg import norm

def PolytopProjection(data, T = 1.0, isProduct = False, solver = None):
    if solver is None: 
        solver = 'cvxopt_qp'
        
    #data = float128(data)
    if isProduct:
        H = data
        n = data.shape[0]
        m = len(T)
    else:
        H = dot(data, data.T)
        n, m = data.shape
    #print H.shape
    #print 'PolytopProjection: n=%d, m=%d, H.shape[0]= %d, H.shape[1]= %d ' %(n, m, H.shape[0], H.shape[1])
    #T = abs(dot(H, ones(n)))
    f = -asfarray(T) *ones(n)
    p = QP(H, f, lb = zeros(n), iprint = -1, maxIter = 150)

    xtol = 1e-6
    if max(T) < 1e5*xtol: xtol = max(T)/1e5
    r = p._solve(solver, ftol = 1e-16, xtol = xtol, maxIter = 10000)
    sol = r.xf

    if isProduct:
        return r.xf
    else:
        s = dot(data.T, r.xf)
        return s.flatten(), r.xf        
