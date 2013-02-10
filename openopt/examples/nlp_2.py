from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros
N = 30
M = 5
ff = lambda x: ((x-M)**2).sum()
p = NLP(ff, cos(arange(N)))
p.df =  lambda x: 2*(x-M)
p.c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]

def dc(x):
    r = zeros((2, p.n))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2]
    return r
p.dc = dc

h1 = lambda x: 1e1*(x[-1]-1)**4
h2 = lambda x: (x[-2]-1.5)**4
p.h = lambda x: (h1(x), h2(x))

def dh(x):
    r = zeros((2, p.n))
    r[0,-1] = 1e1*4*(x[-1]-1)**3
    r[1,-2] = 4*(x[-2]-1.5)**3
    return r
p.dh = dh

p.lb = -6*ones(N)
p.ub = 6*ones(N)
p.lb[3] = 5.5
p.ub[4] = 4.5

#r = p.solve('ipopt', showLS=0, xtol=1e-7, maxIter = 1504)
#solver = 'ipopt'
solver = 'ralg'
#solver = 'scipy_slsqp'
#solver = 'algencan'
r = p.solve(solver, maxIter = 1504, plot=1)
#!! fmin_cobyla can't use user-supplied gradient
#r = p.solve('scipy_cobyla')

