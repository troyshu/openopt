from openopt import GLP
from numpy import *

f = lambda x: (x*x - 10*cos(2*pi*x) + 10).sum() #Rastrigin function
p = GLP(f, lb = -ones(10)*5.12,  ub = ones(10)*5.12,  maxIter = 1e3,  maxFunEvals = 1e5,  maxTime = 10,  maxCPUTime = 10)

r = p.solve('de', plot=0)
x_opt,  f_opt = r.xf,  r.ff
print x_opt
