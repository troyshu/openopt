from openopt import GLP
from numpy import *
N = 100
aN = arange(N)
f = lambda x: ((x-aN)**2).sum()
p = GLP(f, lb = -ones(N),  ub = N*ones(N),  maxIter = 1e3,  maxFunEvals = 1e5,  maxTime = 10,  maxCPUTime = 300)

#optional: graphic output
#p.plot = 1

r = p.solve('de', plot=1, debug=1, iprint=0)
x_opt,  f_opt = r.xf,  r.ff
