"""
Modifying of some solver default parameters is performed via
either kwargs for p.solve() (they can be solver or prob attributes)
or using oosolver (see examples/oosolver.py for more details).
"""
from numpy import *
from openopt import *
f = lambda x: (x[0]-1.5)**2 + sin(0.8 * x[1] ** 2 + 15)**4 + cos(0.8 * x[2] ** 2 + 15)**4 + (x[3]-7.5)**4
lb, ub = -ones(4), ones(4)

# example 1
p = GLP(f, lb=lb, ub=ub,  maxIter = 1e3, maxCPUTime = 3,  maxFunEvals=1e5,  fEnough = 80)
# solve() kwargs can include some prob settings (like maxTime) as well
r = p.solve('galileo', crossoverRate = 0.80, maxTime = 3,  population = 15,  mutationRate = 0.15)

# example 2, via oosolver
solvers = [oosolver('ralg', h0 = 0.80, alp = 2.15, show = False), oosolver('ralg', h0 = 0.15, alp = 2.80, color = 'k')]
for i, solver in enumerate(solvers):
    p = NSP(f, [0]*4, lb=lb, ub=ub, legend='ralg'+str(i+1))
    r = p.solve(solver, plot=True)

