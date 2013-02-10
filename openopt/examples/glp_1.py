from openopt import GLP
from numpy import *

f = lambda x: (x[0]-1.5)**2 + sin(0.8 * x[1] ** 2 + 15)**4 + cos(0.8 * x[2] ** 2 + 15)**4 + (x[3]-7.5)**4
p = GLP(f, lb = -ones(4),  ub = ones(4),  maxIter = 1e3,  maxFunEvals = 1e5,  maxTime = 3,  maxCPUTime = 3)

#optional: graphic output
#p.plot = 1 or p.solve(..., plot=1) or p = GLP(..., plot=1)

r = p.solve('de', plot=1)
x_opt,  f_opt = r.xf,  r.ff
