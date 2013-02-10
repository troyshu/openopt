from openopt import GLP
from numpy import *

# objective function
# (x0 - 1.5)^2 + sin(0.8 * x1^2 + 15)^4 + cos(0.8 * x2^2 + 15)^4 + (x3 - 7.5)^4 -> min
f = lambda x: (x[0]-1.5)**2 + sin(0.8 * x[1] ** 2 + 15)**4 + cos(0.8 * x[2] ** 2 + 15)**4 + (x[3]-7.5)**4

# box-bound constraints lb <= x <= ub
lb, ub = -ones(4),  ones(4)

# linear inequality constraints
# x0 + x3 <= 0.15
# x1 + x3 <= 1.5
# as Ax <= b

A = mat('1 0 0 1; 0 1 0 1') # tuple, list, numpy array etc are OK as well
b = [0.15, 1.5] # tuple, list, numpy array etc are OK as well


# non-linear constraints 
# x0^2 + x2^2 <= 0.15
# 1.5 * x0^2 + x1^2 <= 1.5

c = lambda x: (x[0] ** 2 + x[2] ** 2 - 0.15,  1.5 * x[0] ** 2 + x[1] ** 2 - 1.5)


p = GLP(f, lb=lb, ub=ub, A=A, b=b, c=c, maxIter = 250,  maxFunEvals = 1e5,  maxTime = 30,  maxCPUTime = 30)

r = p.solve('de', mutationRate = 0.15, plot=1)
x_opt,  f_opt = r.xf,  r.ff
